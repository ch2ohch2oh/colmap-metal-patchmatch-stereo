// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>

#ifndef COLMAP_METAL_EMBEDDED_SOURCE
#include "patch_match_metal_types.h"
#endif

using namespace metal;

using Camera = MetalPatchMatchCamera;
using Params = MetalPatchMatchParams;
using Sweep = MetalPatchMatchSweep;
using Images = texture2d_array<float>;
using Depths = texture2d_array<float, access::read>;

// Hypotheses store the reference-camera normal in xyz and z-depth in w.
constexpr sampler kLinear(coord::pixel, address::clamp_to_edge, filter::linear);
constant float kMaxPhotoCost = 2.0f;

uint Hash(uint value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  return value ^ (value >> 16);
}

float Random(thread uint& state) {
  state = Hash(state + 0x9e3779b9u);
  return (float(state >> 9) + 0.5f) * 0x1p-23f;
}

float3 Ray(float2 pixel, constant Params& p) {
  return float3(fma(pixel, p.inv_intrinsics.xy, p.inv_intrinsics.zw), 1.0f);
}

float3 RandomNormal(float3 ray, thread uint& state) {
  const float z = 2.0f * Random(state) - 1.0f;
  const float phi = 6.283185307f * Random(state);
  const float radius = sqrt(max(0.0f, 1.0f - z * z));
  const float3 normal = float3(radius * cos(phi), radius * sin(phi), z);
  return dot(normal, ray) > 0.0f ? -normal : normal;
}

bool ValidPlane(float4 plane, float3 ray, constant Params& p) {
  return plane.w > 0.0f && plane.w >= p.depth_min && plane.w <= p.depth_max &&
         abs(dot(plane.xyz, plane.xyz) - 1.0f) < 0.01f &&
         dot(plane.xyz, ray) < -1e-6f;
}

float3 Rotate(float3 point, const device Camera& camera) {
  return float3(dot(camera.from_ref[0].xyz, point),
                dot(camera.from_ref[1].xyz, point),
                dot(camera.from_ref[2].xyz, point));
}

float3 Translation(const device Camera& camera) {
  return float3(
      camera.from_ref[0].w, camera.from_ref[1].w, camera.from_ref[2].w);
}

float2 Project(float3 point, float4 intrinsics) {
  return fma(point.xy / point.z, intrinsics.xy, intrinsics.zw);
}

bool Inside(float2 pixel, const device Camera& camera) {
  return all(pixel >= 0.0f) &&
         all(pixel <= float2(camera.width - 1, camera.height - 1));
}

float PhotoCost(int2 pixel,
                float3 ray,
                float4 plane,
                uint source,
                const device Camera& camera,
                constant Params& p,
                Images images) {
  // The plane-induced homography maps a ray r to R*r + t*(n*r)/(n*X).
  const float plane_offset = dot(plane.xyz, ray) * plane.w;
  if (plane_offset >= -1e-8f) {
    return kInvalidPhotoCost;
  }
  const float3 center = Rotate(ray * plane.w, camera) + Translation(camera);
  if (center.z <= 0.0f || !Inside(Project(center, camera.intrinsics), camera)) {
    return kInvalidPhotoCost;
  }
  const float3 scaled_t = Translation(camera) / plane_offset;
  const float3 warp_x = p.inv_intrinsics.x * (float3(camera.from_ref[0].x,
                                                     camera.from_ref[1].x,
                                                     camera.from_ref[2].x) +
                                              scaled_t * plane.x);
  const float3 warp_y = p.inv_intrinsics.y * (float3(camera.from_ref[0].y,
                                                     camera.from_ref[1].y,
                                                     camera.from_ref[2].y) +
                                              scaled_t * plane.y);
  const float3 warp_center = center / plane.w;
  const float ref_center = images.sample(kLinear, float2(pixel) + 0.5f, 0).r;

  float weight_sum = 0.0f;
  float2 sum = 0.0f;
  float3 squares = 0.0f;
  int ref_samples = 0;
  int valid_samples = 0;
  for (int y = -p.radius; y <= p.radius; y += p.step) {
    const int row = pixel.y + y;
    if (row < 0 || row >= p.height) {
      continue;
    }
    const float3 warp_row = warp_center + float(y) * warp_y;
    for (int x = -p.radius; x <= p.radius; x += p.step) {
      const int col = pixel.x + x;
      if (col < 0 || col >= p.width) {
        continue;
      }
      ++ref_samples;
      const float3 warped = warp_row + float(x) * warp_x;
      if (warped.z <= 0.0f ||
          dot(plane.xyz, ray) +
                  dot(plane.xy, float2(x, y) * p.inv_intrinsics.xy) >=
              0.0f) {
        // A grazing plane may intersect part of the patch behind a camera.
        // Reject it instead of computing NCC on the surviving subset.
        return kInvalidPhotoCost;
      }
      const float2 src_pixel = Project(warped, camera.intrinsics);
      if (!Inside(src_pixel, camera)) {
        continue;
      }
      ++valid_samples;
      const float ref = images.sample(kLinear, float2(col, row) + 0.5f, 0).r;
      const float src = images.sample(kLinear, src_pixel + 0.5f, source + 1).r;
      const float delta = ref - ref_center;
      const float weight = fast::exp(float(x * x + y * y) * p.spatial_weight +
                                     delta * delta * p.color_weight);
      weight_sum += weight;
      sum += weight * float2(ref, src);
      squares += weight * float3(ref * ref, src * src, ref * src);
    }
  }
  if (valid_samples < max(3, (ref_samples + 1) / 2) || weight_sum <= 0.0f) {
    return kInvalidPhotoCost;
  }
  sum /= weight_sum;
  squares /= weight_sum;
  const float2 variance = squares.xy - sum * sum;
  if (any(variance < 1e-5f)) {
    return kInvalidPhotoCost;
  }
  return clamp(
      1.0f - (squares.z - sum.x * sum.y) / sqrt(variance.x * variance.y),
      0.0f,
      kMaxPhotoCost);
}

float GeomError(float2 pixel,
                float3 point,
                uint source,
                const device Camera& camera,
                constant Params& p,
                Depths depths) {
  const float3 src_point = Rotate(point, camera) + Translation(camera);
  if (src_point.z <= 0.0f) {
    return INFINITY;
  }
  const float2 src_pixel = Project(src_point, camera.intrinsics);
  if (!Inside(src_pixel, camera)) {
    return INFINITY;
  }
  const float depth = depths.read(uint2(floor(src_pixel + 0.5f)), source).r;
  if (!(depth > 0.0f) || !isfinite(depth)) {
    return INFINITY;
  }
  const float3 lifted =
      depth *
      float3((src_pixel - camera.intrinsics.zw) / camera.intrinsics.xy, 1.0f);
  const float3 back =
      camera.center_in_ref.xyz + lifted.x * camera.from_ref[0].xyz +
      lifted.y * camera.from_ref[1].xyz + lifted.z * camera.from_ref[2].xyz;
  if (back.z <= 0.0f) {
    return INFINITY;
  }
  const float2 ref_pixel =
      (back.xy / back.z - p.inv_intrinsics.zw) / p.inv_intrinsics.xy;
  return length(ref_pixel - pixel);
}

float2 ViewingCosines(float3 point,
                      float3 normal,
                      const device Camera& camera) {
  const float3 to_source = normalize(camera.center_in_ref.xyz - point);
  return float2(dot(normalize(-point), to_source), dot(normal, to_source));
}

float ViewWeight(float cost,
                 float3 ray,
                 float4 plane,
                 const device Camera& camera,
                 constant Params& p) {
  const float3 point = ray * plane.w;
  const float3 src = Rotate(point, camera) + Translation(camera);
  if (src.z <= 0.0f || !Inside(Project(src, camera.intrinsics), camera)) {
    return 0.0f;
  }
  const float2 angles = ViewingCosines(point, plane.xyz, camera);
  const float tri_scale = max(0.0f, angles.x - p.min_triangulation_cos) /
                          max(1e-7f, 1.0f - p.min_triangulation_cos);
  const float tri_weight = max(0.0f, 1.0f - tri_scale * tri_scale);
  const float incidence = 1.0f - clamp(angles.y, 0.0f, 1.0f);
  // Local area change under the plane homography, using its determinant.
  const float3 center = camera.center_in_ref.xyz;
  const float plane_offset = dot(plane.xyz, point);
  const float z_ratio = plane.w / src.z;
  const float area = abs(1.0f - dot(plane.xyz, center) / plane_offset) *
                     camera.intrinsics.x * camera.intrinsics.y *
                     p.inv_intrinsics.x * p.inv_intrinsics.y * z_ratio *
                     z_ratio * z_ratio;
  const float resolution = min(area, 1.0f / max(area, 1e-7f));
  return (0.01f + fast::exp(p.ncc_weight * min(cost, kMaxPhotoCost) *
                            min(cost, kMaxPhotoCost))) *
         tri_weight * fast::exp(p.incident_weight * incidence * incidence) *
         resolution;
}

kernel void Initialize(device float4* planes [[buffer(0)]],
                       device half* costs [[buffer(1)]],
                       const device Camera* cameras [[buffer(2)]],
                       constant Params& p [[buffer(3)]],
                       Images images [[texture(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
  const uint index = gid.y * p.width + gid.x;
  const float3 ray = Ray(float2(gid), p);
  uint state = Hash(index);
  const float initial_depth = mix(p.depth_min, p.depth_max, Random(state));
  float4 plane = float4(RandomNormal(ray, state), initial_depth);
  if (p.geom_consistency && ValidPlane(planes[index], ray, p)) {
    plane = planes[index];
  }
  planes[index] = plane;
  const uint num_pixels = uint(p.width) * p.height;
  for (int source = 0; source < p.num_sources; ++source) {
    costs[source * num_pixels + index] = half(
        PhotoCost(int2(gid), ray, plane, source, cameras[source], p, images));
  }
}

float4 Refine(int2 pixel,
              float4 plane,
              float4 previous,
              constant Sweep& sweep,
              device half* costs,
              const device Camera* cameras,
              constant Params& p,
              Images images,
              Depths depths) {
  const uint index = uint(pixel.y) * p.width + pixel.x;
  const uint num_pixels = uint(p.width) * p.height;
  const float3 ray = Ray(float2(pixel), p);
  const float3 prev_ray = Ray(float2(pixel - int2(sweep.dx, sweep.dy)), p);
  const float denom = dot(previous.xyz, ray);
  previous.w =
      denom < -1e-6f ? previous.w * dot(previous.xyz, prev_ray) / denom : 0.0f;
  if (!ValidPlane(previous, ray, p)) {
    previous = plane;
  }

  uint state = Hash(index ^ sweep.seed);
  const float depth = clamp(
      plane.w * (1.0f + (2.0f * Random(state) - 1.0f) * sweep.perturbation),
      p.depth_min,
      p.depth_max);
  float3 normal =
      normalize(plane.xyz + RandomNormal(ray, state) * sweep.perturbation);
  if (dot(normal, ray) >= -1e-6f || !all(isfinite(normal))) {
    normal = plane.xyz;
  }
  const float4 candidates[] = {plane,
                               previous,
                               float4(normal, depth),
                               float4(normal, plane.w),
                               float4(plane.xyz, depth)};

  float weight_sum = 0.0f;
  for (int source = 0; source < p.num_sources; ++source) {
    weight_sum += ViewWeight(float(costs[source * num_pixels + index]),
                             ray,
                             plane,
                             cameras[source],
                             p);
  }

  // Systematic Monte Carlo sampling. Score each view at most once, weighting
  // its cost by how many samples landed in its interval of the CDF.
  const float phase = Random(state);
  const float sample_count = float(p.num_samples);
  float cumulative = 0.0f;
  float evaluated = 0.0f;
  float totals[5] = {};
  for (int source = 0; source < p.num_sources; ++source) {
    const float cached = float(costs[source * num_pixels + index]);
    float count = 1.0f;
    if (weight_sum > 0.0f) {
      cumulative += ViewWeight(cached, ray, plane, cameras[source], p);
      const float end =
          clamp(ceil(cumulative / weight_sum * sample_count - phase),
                0.0f,
                sample_count);
      count = end - evaluated;
      evaluated = end;
      if (count <= 0.0f) {
        continue;
      }
    }
    for (int i = 0; i < 5; ++i) {
      const float4 candidate = candidates[i];
      float cost = min(
          kMaxPhotoCost,
          i == 0
              ? cached
              : PhotoCost(
                    pixel, ray, candidate, source, cameras[source], p, images));
      if (p.geom_consistency) {
        cost += p.geom_regularizer * min(p.geom_max_cost,
                                         GeomError(float2(pixel),
                                                   ray * candidate.w,
                                                   source,
                                                   cameras[source],
                                                   p,
                                                   depths));
      }
      totals[i] += count * cost;
    }
  }

  int best = 0;
  for (int i = 1; i < 5; ++i) {
    // The cached cost is half precision. Require an improvement greater than
    // its rounding error so identical proposals do not replace the incumbent.
    const float tolerance =
        0.0005f * (weight_sum > 0.0f ? evaluated : p.num_sources);
    if (totals[i] + tolerance < totals[best]) {
      best = i;
    }
  }
  if (best != 0) {
    for (int source = 0; source < p.num_sources; ++source) {
      costs[source * num_pixels + index] = half(PhotoCost(
          pixel, ray, candidates[best], source, cameras[source], p, images));
    }
  }
  return candidates[best];
}

kernel void Propagate(const device float4* input [[buffer(0)]],
                      device half* costs [[buffer(1)]],
                      const device Camera* cameras [[buffer(2)]],
                      constant Params& p [[buffer(3)]],
                      device float4* output [[buffer(4)]],
                      constant Sweep& sweep [[buffer(5)]],
                      Images images [[texture(0)]],
                      Depths depths [[texture(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
  const bool horizontal = sweep.dx != 0;
  const int length = horizontal ? p.width : p.height;
  const int stride = horizontal ? sweep.dx : sweep.dy;
  const int start = int(gid.x) * sweep.stripe_length;
  const int end = min(start + sweep.stripe_length, length);
  const int coordinate = stride > 0 ? start : length - 1 - start;
  const int2 direction = int2(sweep.dx, sweep.dy);
  int2 pixel = horizontal ? int2(coordinate, gid.y) : int2(gid.y, coordinate);
  // Within a stripe the preceding hypothesis is already refined. At a stripe
  // boundary it comes from the immutable input of this sweep.
  const int2 neighbor = pixel - direction;
  float4 previous =
      start == 0 ? float4(0.0f) : input[neighbor.y * p.width + neighbor.x];
  for (int i = start; i < end; ++i, pixel += direction) {
    const uint index = uint(pixel.y) * p.width + pixel.x;
    previous = Refine(pixel,
                      input[index],
                      previous,
                      sweep,
                      costs,
                      cameras,
                      p,
                      images,
                      depths);
    output[index] = previous;
  }
}

kernel void Filter(device float4* planes [[buffer(0)]],
                   const device half* costs [[buffer(1)]],
                   const device Camera* cameras [[buffer(2)]],
                   constant Params& p [[buffer(3)]],
                   device uchar* graph [[buffer(4)]],
                   Depths depths [[texture(1)]],
                   uint2 gid [[thread_position_in_grid]]) {
  const uint index = gid.y * p.width + gid.x;
  const uint num_pixels = uint(p.width) * p.height;
  const float4 plane = planes[index];
  const float3 point = Ray(float2(gid), p) * plane.w;
  int consistent = 0;
  for (int source = 0; source < p.num_sources; ++source) {
    const uint offset = source * num_pixels + index;
    const float2 angles = ViewingCosines(point, plane.xyz, cameras[source]);
    bool valid = float(costs[offset]) < kInvalidPhotoCost &&
                 float(costs[offset]) <= p.filter_max_photo_cost &&
                 angles.x <= p.filter_triangulation_cos && angles.y > 0.0f;
    if (valid && p.geom_consistency) {
      valid =
          GeomError(float2(gid), point, source, cameras[source], p, depths) <=
          p.filter_max_geom_cost;
    }
    consistent += int(valid);
    if (p.write_graph) {
      graph[offset] = uchar(valid);
    }
  }
  if (p.filter && consistent < p.min_consistent) {
    planes[index] = float4(0.0f);
  }
}
