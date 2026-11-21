// SPDX-License-Identifier: BSD-3-Clause

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using MetalFloat4 = metal::float4;
#else
#pragma once
#include <simd/simd.h>
using MetalFloat4 = simd_float4;
#endif

enum { kInvalidPhotoCost = 3 };

// Host/shader buffer layout. Keep scalars 32-bit and vectors 16-byte aligned.
struct MetalPatchMatchCamera {
  MetalFloat4 intrinsics;   // fx, fy, cx, cy
  MetalFloat4 from_ref[3];  // row-major [R | t]
  MetalFloat4 center_in_ref;
  int width;
  int height;
};

struct MetalPatchMatchParams {
  MetalFloat4 inv_intrinsics;  // 1/fx, 1/fy, -cx/fx, -cy/fy
  int width;
  int height;
  int num_sources;
  int radius;
  int step;
  int num_samples;
  int geom_consistency;
  int filter;
  int write_graph;
  int min_consistent;
  float depth_min;
  float depth_max;
  float spatial_weight;
  float color_weight;
  float ncc_weight;
  float incident_weight;
  float min_triangulation_cos;
  float geom_regularizer;
  float geom_max_cost;
  float filter_max_photo_cost;
  float filter_triangulation_cos;
  float filter_max_geom_cost;
};

struct MetalPatchMatchSweep {
  int dx;
  int dy;
  int stripe_length;
  unsigned int seed;
  float perturbation;
};
