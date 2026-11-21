// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_metal.h"

#include "colmap/math/math.h"
#include "colmap/mvs/patch_match_metal_types.h"
#include "colmap/util/logging.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace colmap::mvs {
namespace {

#include "patch_match_metal_source.h"

struct Pipelines {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> initialize;
  id<MTLComputePipelineState> propagate;
  id<MTLComputePipelineState> filter;
};

id<MTLDevice> GetSupportedDevice() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return [device supportsFamily:MTLGPUFamilyApple7] ? device : nil;
}

const Pipelines& GetPipelines() {
  static const Pipelines pipelines = [] {
    @autoreleasepool {
      id<MTLDevice> device = GetSupportedDevice();
      THROW_CHECK(device != nil) << "Metal PatchMatch requires an accessible Apple Silicon GPU.";
      id<MTLCommandQueue> queue = [device newCommandQueue];
      THROW_CHECK(queue != nil) << "Could not create the Metal command queue.";
      NSError* error = nil;
      MTLCompileOptions* options = [MTLCompileOptions new];
      options.languageVersion = MTLLanguageVersion2_3;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
      if (@available(macOS 15.0, *)) {
        options.mathMode = MTLMathModeSafe;
      } else
#endif
      {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
      }
      id<MTLLibrary> library = [device newLibraryWithSource:@(kPatchMatchMetalSource)
                                                    options:options
                                                      error:&error];
      THROW_CHECK(library != nil) << "Could not compile Metal PatchMatch shaders: "
                                  << error.localizedDescription.UTF8String;
      const auto load = [&](NSString* name) {
        NSError* pipeline_error = nil;
        id<MTLFunction> function = [library newFunctionWithName:name];
        THROW_CHECK(function != nil) << "Missing Metal kernel " << name.UTF8String;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&pipeline_error];
        THROW_CHECK(pipeline != nil) << "Could not create Metal kernel " << name.UTF8String << ": "
                                     << pipeline_error.localizedDescription.UTF8String;
        return pipeline;
      };
      LOG(INFO) << "PatchMatch stereo using Metal on " << device.name.UTF8String;
      return Pipelines{device, queue, load(@"Initialize"), load(@"Propagate"), load(@"Filter")};
    }
  }();
  return pipelines;
}

id<MTLBuffer> AllocateBuffer(id<MTLDevice> device,
                             size_t num_bytes,
                             MTLResourceOptions options,
                             const char* name) {
  THROW_CHECK_LE(num_bytes, device.maxBufferLength)
      << "Metal buffer " << name
      << " exceeds the device limit. Reduce PatchMatchStereo.max_image_size "
         "or the number of source images.";
  id<MTLBuffer> buffer = [device newBufferWithLength:num_bytes options:options];
  THROW_CHECK(buffer != nil) << "Could not allocate the Metal buffer " << name << " (" << num_bytes
                             << " bytes). Reduce image size or the number of source images.";
  buffer.label = @(name);
  return buffer;
}

id<MTLTexture> AllocateTexture(id<MTLDevice> device,
                               MTLPixelFormat format,
                               size_t width,
                               size_t height,
                               size_t layers,
                               NSString* label) {
  THROW_CHECK(width <= 16384 && height <= 16384 && layers <= 2048)
      << "Metal PatchMatch texture exceeds device dimensions. Reduce image "
         "size or the number of source images.";
  MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
  descriptor.textureType = MTLTextureType2DArray;
  descriptor.pixelFormat = format;
  descriptor.width = width;
  descriptor.height = height;
  descriptor.arrayLength = layers;
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = MTLTextureUsageShaderRead;
  id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
  THROW_CHECK(texture != nil) << "Could not allocate Metal texture " << label.UTF8String
                              << ". Reduce image size or the number of source images.";
  texture.label = label;
  return texture;
}

void Upload(id<MTLTexture> texture,
            size_t layer,
            size_t width,
            size_t height,
            const void* data,
            size_t row_bytes) {
  [texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
             mipmapLevel:0
                   slice:layer
               withBytes:data
             bytesPerRow:row_bytes
           bytesPerImage:row_bytes * height];
}

MetalPatchMatchCamera MakeCamera(const Image& ref, const Image& source) {
  MetalPatchMatchCamera camera{};
  float R[9], t[3], center[3];
  ComputeRelativePose(ref.GetR(), ref.GetT(), source.GetR(), source.GetT(), R, t);
  ComputeProjectionCenter(R, t, center);
  const float* K = source.GetK();
  // Match the integer pixel centers used by CUDA PatchMatch and StereoFusion.
  camera.intrinsics = {K[0], K[4], K[2] - 0.5f, K[5] - 0.5f};
  for (int row = 0; row < 3; ++row) {
    camera.from_ref[row] = {R[3 * row], R[3 * row + 1], R[3 * row + 2], t[row]};
  }
  camera.center_in_ref = {center[0], center[1], center[2], 0.0f};
  camera.width = source.GetWidth();
  camera.height = source.GetHeight();
  return camera;
}

float GaussianWeight(double sigma) { return -0.5 / (sigma * sigma); }

MetalPatchMatchParams MakeParams(const PatchMatchOptions& options,
                                 const Image& ref,
                                 size_t num_sources) {
  const float* K = ref.GetK();
  MetalPatchMatchParams p{};
  p.inv_intrinsics = {1.0f / K[0], 1.0f / K[4], (0.5f - K[2]) / K[0], (0.5f - K[5]) / K[4]};
  p.width = ref.GetWidth();
  p.height = ref.GetHeight();
  p.num_sources = num_sources;
  p.radius = options.window_radius;
  p.step = options.window_step;
  p.num_samples = options.num_samples;
  p.geom_consistency = options.geom_consistency;
  p.filter = options.filter;
  p.write_graph = options.write_consistency_graph;
  p.min_consistent = options.filter_min_num_consistent;
  p.depth_min = options.depth_min;
  p.depth_max = options.depth_max;
  p.spatial_weight =
      GaussianWeight(options.sigma_spatial > 0 ? options.sigma_spatial : options.window_radius);
  p.color_weight = GaussianWeight(options.sigma_color);
  p.ncc_weight = GaussianWeight(options.ncc_sigma);
  p.incident_weight = GaussianWeight(options.incident_angle_sigma);
  p.min_triangulation_cos = std::cos(DegToRad(options.min_triangulation_angle));
  p.geom_regularizer = options.geom_consistency_regularizer;
  p.geom_max_cost = options.geom_consistency_max_cost;
  p.filter_max_photo_cost = 1.0 - options.filter_min_ncc;
  p.filter_triangulation_cos = std::cos(DegToRad(options.filter_min_triangulation_angle));
  p.filter_max_geom_cost = options.filter_geom_consistency_max_cost;
  return p;
}

void Dispatch(id<MTLComputeCommandEncoder> encoder,
              id<MTLComputePipelineState> pipeline,
              size_t width,
              size_t height) {
  const size_t group_width = std::min(width, pipeline.threadExecutionWidth);
  const size_t group_height =
      std::min({height, size_t{4}, pipeline.maxTotalThreadsPerThreadgroup / group_width});
  [encoder dispatchThreads:MTLSizeMake(width, height, 1)
      threadsPerThreadgroup:MTLSizeMake(group_width, group_height, 1)];
  [encoder endEncoding];
}

}  // namespace

struct PatchMatchMetal::Impl {
  Impl(const PatchMatchOptions& options, const PatchMatch::Problem& problem)
      : options(options), problem(problem) {}

  void Run() {
    const auto& gpu = GetPipelines();
    const Image& ref = problem.images->at(problem.ref_image_idx);
    const auto& sources = problem.src_image_idxs;
    const auto p = MakeParams(options, ref, sources.size());
    THROW_CHECK(p.depth_min >= 0.0f && p.depth_max > 0.0f && std::isfinite(p.depth_max))
        << "Metal PatchMatch requires an explicit or estimated depth range.";
    const size_t num_pixels = ref.GetWidth() * ref.GetHeight();
    const size_t num_costs = num_pixels * sources.size();
    THROW_CHECK_LE(num_costs, std::numeric_limits<uint32_t>::max())
        << "Reduce PatchMatchStereo.max_image_size or the number of source images.";

    size_t src_width = 0, src_height = 0;
    for (int idx : sources) {
      const Image& image = problem.images->at(idx);
      src_width = std::max(src_width, image.GetWidth());
      src_height = std::max(src_height, image.GetHeight());
    }
    id<MTLTexture> images = AllocateTexture(gpu.device,
                                            MTLPixelFormatR8Unorm,
                                            std::max(src_width, ref.GetWidth()),
                                            std::max(src_height, ref.GetHeight()),
                                            sources.size() + 1,
                                            @"Images");
    id<MTLTexture> depths = AllocateTexture(gpu.device,
                                            MTLPixelFormatR32Float,
                                            p.geom_consistency ? src_width : 1,
                                            p.geom_consistency ? src_height : 1,
                                            p.geom_consistency ? sources.size() : 1,
                                            @"Source depths");
    Upload(images, 0, p.width, p.height, ref.GetBitmap().RowMajorData().data(), p.width);
    id<MTLBuffer> cameras = AllocateBuffer(gpu.device,
                                           sources.size() * sizeof(MetalPatchMatchCamera),
                                           MTLResourceStorageModeShared,
                                           "Source cameras");
    auto* camera_data = static_cast<MetalPatchMatchCamera*>(cameras.contents);
    for (size_t i = 0; i < sources.size(); ++i) {
      const Image& image = problem.images->at(sources[i]);
      const size_t width = image.GetWidth(), height = image.GetHeight();
      camera_data[i] = MakeCamera(ref, image);
      Upload(images, i + 1, width, height, image.GetBitmap().RowMajorData().data(), width);
      if (p.geom_consistency) {
        const auto& depth = problem.depth_maps->at(sources[i]);
        Upload(depths, i, width, height, depth.GetPtr(), width * sizeof(float));
      }
    }

    id<MTLBuffer> input = AllocateBuffer(
        gpu.device, num_pixels * sizeof(MetalFloat4), MTLResourceStorageModeShared, "Planes A");
    id<MTLBuffer> output =
        AllocateBuffer(gpu.device, input.length, MTLResourceStorageModeShared, "Planes B");
    id<MTLBuffer> cost_buffer =
        AllocateBuffer(gpu.device, num_costs * sizeof(__fp16), MTLResourceStorageModeShared, "NCC");
    id<MTLBuffer> graph_buffer = AllocateBuffer(gpu.device,
                                                p.write_graph ? num_costs : 1,
                                                MTLResourceStorageModeShared,
                                                "Consistency mask");
    if (p.geom_consistency) {
      const auto& depth = problem.depth_maps->at(problem.ref_image_idx);
      const auto& normal = problem.normal_maps->at(problem.ref_image_idx);
      auto* planes = static_cast<MetalFloat4*>(input.contents);
      for (size_t i = 0; i < num_pixels; ++i) {
        const auto* n = normal.GetPtr() + i;
        planes[i] = {n[0], n[num_pixels], n[2 * num_pixels], depth.GetPtr()[i]};
      }
    }

    id<MTLCommandBuffer> commands = [gpu.queue commandBuffer];
    THROW_CHECK(commands != nil) << "Could not create the Metal command buffer.";
    commands.label = @"PatchMatch";
    const auto encode = [&](id<MTLComputePipelineState> pipeline) {
      id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
      THROW_CHECK(encoder != nil) << "Could not create the Metal compute encoder.";
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:input offset:0 atIndex:0];
      [encoder setBuffer:cost_buffer offset:0 atIndex:1];
      [encoder setBuffer:cameras offset:0 atIndex:2];
      [encoder setBytes:&p length:sizeof(p) atIndex:3];
      [encoder setTexture:images atIndex:0];
      [encoder setTexture:depths atIndex:1];
      return encoder;
    };
    Dispatch(encode(gpu.initialize), gpu.initialize, p.width, p.height);
    constexpr int kDirections[][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    constexpr int kStripeLength = 4;
    for (int iteration = 0; iteration < options.num_iterations; ++iteration) {
      for (int sweep_idx = 0; sweep_idx < 4; ++sweep_idx) {
        const MetalPatchMatchSweep sweep = {kDirections[sweep_idx][0],
                                            kDirections[sweep_idx][1],
                                            kStripeLength,
                                            1u + 4u * iteration + sweep_idx,
                                            std::exp2(-float(iteration) - 0.25f * sweep_idx)};
        id<MTLComputeCommandEncoder> encoder = encode(gpu.propagate);
        [encoder setBuffer:output offset:0 atIndex:4];
        [encoder setBytes:&sweep length:sizeof(sweep) atIndex:5];
        const int length = sweep.dx != 0 ? p.width : p.height;
        const int lines = sweep.dx != 0 ? p.height : p.width;
        Dispatch(encoder, gpu.propagate, (length + kStripeLength - 1) / kStripeLength, lines);
        std::swap(input, output);
      }
    }
    if (p.filter || p.write_graph) {
      id<MTLComputeCommandEncoder> encoder = encode(gpu.filter);
      [encoder setBuffer:graph_buffer offset:0 atIndex:4];
      Dispatch(encoder, gpu.filter, p.width, p.height);
    }
    [commands commit];
    [commands waitUntilCompleted];
    THROW_CHECK(commands.status == MTLCommandBufferStatusCompleted)
        << "Metal PatchMatch command buffer failed: "
        << commands.error.localizedDescription.UTF8String;

    const size_t allocated_bytes = input.allocatedSize + output.allocatedSize +
                                   cost_buffer.allocatedSize + graph_buffer.allocatedSize +
                                   cameras.allocatedSize + images.allocatedSize +
                                   depths.allocatedSize;
    LOG(INFO) << "Metal PatchMatch GPU time: " << commands.GPUEndTime - commands.GPUStartTime
              << " s, per-view Metal resources: " << allocated_bytes / (1024.0 * 1024.0) << " MiB";
    width = p.width;
    height = p.height;
    planes = input;
    costs = cost_buffer;
    graph = p.write_graph ? graph_buffer : nil;
  }

  const PatchMatchOptions options;
  const PatchMatch::Problem problem;
  size_t width = 0;
  size_t height = 0;
  id<MTLBuffer> planes;
  id<MTLBuffer> costs;
  id<MTLBuffer> graph;
};

PatchMatchMetal::PatchMatchMetal(const PatchMatchOptions& options,
                                 const PatchMatch::Problem& problem)
    : impl_(std::make_unique<Impl>(options, problem)) {}

PatchMatchMetal::~PatchMatchMetal() = default;

bool PatchMatchMetal::IsDeviceAvailable() {
  @autoreleasepool {
    return GetSupportedDevice() != nil;
  }
}

void PatchMatchMetal::CheckDevice() { GetPipelines(); }

void PatchMatchMetal::Run() {
  @autoreleasepool {
    impl_->Run();
  }
}

DepthMap PatchMatchMetal::GetDepthMap() const {
  const auto& m = *impl_;
  DepthMap depth(m.width, m.height, m.options.depth_min, m.options.depth_max);
  const auto* planes = static_cast<const MetalFloat4*>(m.planes.contents);
  for (size_t i = 0; i < m.width * m.height; ++i) {
    depth.GetPtr()[i] = planes[i].w;
  }
  return depth;
}

NormalMap PatchMatchMetal::GetNormalMap() const {
  const auto& m = *impl_;
  NormalMap normal(m.width, m.height);
  const auto* planes = static_cast<const MetalFloat4*>(m.planes.contents);
  const size_t num_pixels = m.width * m.height;
  for (size_t i = 0; i < num_pixels; ++i) {
    for (size_t channel = 0; channel < 3; ++channel) {
      normal.GetPtr()[channel * num_pixels + i] = planes[i][channel];
    }
  }
  return normal;
}

Mat<float> PatchMatchMetal::GetSelProbMap() const {
  const auto& m = *impl_;
  Mat<float> probability(m.width, m.height, m.problem.src_image_idxs.size());
  const auto* costs = static_cast<const __fp16*>(m.costs.contents);
  const float weight = GaussianWeight(m.options.ncc_sigma);
  for (size_t i = 0; i < probability.GetData().size(); ++i) {
    const float cost = costs[i];
    probability.GetPtr()[i] =
        cost < kInvalidPhotoCost ? std::exp(weight * cost * cost) : 0.0f;
  }
  return probability;
}

std::vector<int> PatchMatchMetal::GetConsistentImageIdxs() const {
  const auto& m = *impl_;
  const auto* mask = static_cast<const uint8_t*>(m.graph.contents);
  std::vector<int> indices;
  if (mask == nullptr) {
    return indices;
  }
  const auto* planes = static_cast<const MetalFloat4*>(m.planes.contents);
  const auto& sources = m.problem.src_image_idxs;
  const size_t num_pixels = m.width * m.height;
  for (size_t i = 0; i < num_pixels; ++i) {
    if (planes[i].w <= 0.0f) {
      continue;
    }
    const size_t start = indices.size();
    indices.insert(indices.end(),
                   {static_cast<int>(i % m.width), static_cast<int>(i / m.width), 0});
    for (size_t source = 0; source < sources.size(); ++source) {
      if (mask[source * num_pixels + i]) {
        indices.push_back(sources[source]);
      }
    }
    if (indices.size() == start + 3) {
      indices.resize(start);
    } else {
      indices[start + 2] = indices.size() - start - 3;
    }
  }
  return indices;
}

}  // namespace colmap::mvs
