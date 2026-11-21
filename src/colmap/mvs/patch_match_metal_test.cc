// SPDX-License-Identifier: BSD-3-Clause

#include "colmap/mvs/patch_match_metal.h"

#include "colmap/mvs/consistency_graph.h"

#include <algorithm>
#include <cmath>
#include <string>

#include <gtest/gtest.h>

namespace colmap::mvs {
namespace {

constexpr float kDepth = 4.0f;
constexpr float kFocalLength = 80.0f;
constexpr int kWidth = 79;
constexpr int kHeight = 61;

Image MakeImage(int width, int height, float tx, float principal_offset = 0) {
  const float K[] = {kFocalLength,
                     0,
                     kWidth / 2.0f + principal_offset,
                     0,
                     kFocalLength,
                     kHeight / 2.0f,
                     0,
                     0,
                     1};
  const float R[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const float T[] = {tx, 0, 0};
  return Image("", width, height, K, R, T);
}

float Texture(float x, float y) {
  return 128.0f + 33.0f * std::sin(0.77f * x + 0.32f * y) +
         28.0f * std::cos(0.13f * x - 0.88f * y) +
         21.0f * std::sin(1.39f * x - 0.65f * y);
}

void Render(Image& image, float extra_signal = 0) {
  Bitmap bitmap(image.GetWidth(), image.GetHeight(), /*as_rgb=*/false);
  const float shift =
      kFocalLength * image.GetT()[0] / kDepth + image.GetK()[2] - kWidth / 2.0f;
  for (int y = 0; y < bitmap.Height(); ++y) {
    for (int x = 0; x < bitmap.Width(); ++x) {
      const float value =
          Texture(x - shift, y) + extra_signal * std::sin(1.2f * x + 1.9f * y);
      bitmap.SetPixel(x, y, BitmapColor<uint8_t>(std::lround(value)));
    }
  }
  image.SetBitmap(std::move(bitmap));
}

struct PlaneScene {
  PlaneScene()
      : images{MakeImage(kWidth, kHeight, 0.0f),
               MakeImage(kWidth - 3, kHeight, 0.25f),
               MakeImage(kWidth, kHeight - 3, -0.25f)} {
    for (Image& image : images) {
      Render(image);
      depth_maps.emplace_back(image.GetWidth(), image.GetHeight(), 2.0f, 6.0f);
      depth_maps.back().Fill(kDepth);
      NormalMap& normals =
          normal_maps.emplace_back(image.GetWidth(), image.GetHeight());
      std::fill_n(normals.GetPtr() + 2 * image.GetWidth() * image.GetHeight(),
                  image.GetWidth() * image.GetHeight(),
                  -1.0f);
    }
    problem.ref_image_idx = 0;
    problem.src_image_idxs = {1, 2};
    problem.images = &images;
    problem.depth_maps = &depth_maps;
    problem.normal_maps = &normal_maps;
  }

  std::vector<Image> images;
  std::vector<DepthMap> depth_maps;
  std::vector<NormalMap> normal_maps;
  PatchMatch::Problem problem;
};

class PatchMatchMetalTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!PatchMatchMetal::IsDeviceAvailable()) {
      GTEST_SKIP() << "No accessible Apple Silicon GPU";
    }
  }
};

TEST_F(PatchMatchMetalTest, BilateralNccMatchesScalarReference) {
  PlaneScene scene;
  scene.images[1] = MakeImage(kWidth - 3, kHeight, 0.0f, 0.375f);
  Render(scene.images[1], /*extra_signal=*/21.0f);
  scene.problem.src_image_idxs = {1};
  PatchMatchOptions options;
  options.depth_min = options.depth_max = kDepth;
  options.geom_consistency = false;
  options.filter = false;
  options.num_iterations = 1;
  options.window_radius = 3;
  options.window_step = 2;
  options.sigma_spatial = 2.0;
  options.sigma_color = 0.16;
  options.ncc_sigma = 0.3;
  PatchMatch matcher(options, scene.problem);
  matcher.Run();

  constexpr int kRow = 24;
  constexpr int kCol = 31;
  const auto ref = [&](int x, int y) {
    return scene.images[0].GetBitmap().GetPixel(x, y)->r / 255.0;
  };
  const auto src = [&](int x, int y) {
    const auto& bitmap = scene.images[1].GetBitmap();
    return (0.625 * bitmap.GetPixel(x, y)->r +
            0.375 * bitmap.GetPixel(x + 1, y)->r) /
           255.0;
  };
  double w_sum = 0, a_sum = 0, b_sum = 0, aa_sum = 0, bb_sum = 0, ab_sum = 0;
  for (int dy = -3; dy <= 3; dy += 2) {
    for (int dx = -3; dx <= 3; dx += 2) {
      const double a = ref(kCol + dx, kRow + dy);
      const double b = src(kCol + dx, kRow + dy);
      const double delta = a - ref(kCol, kRow);
      const double w = std::exp(
          -0.5 * ((dx * dx + dy * dy) / 4.0 + delta * delta / (0.16 * 0.16)));
      w_sum += w;
      a_sum += w * a;
      b_sum += w * b;
      aa_sum += w * a * a;
      bb_sum += w * b * b;
      ab_sum += w * a * b;
    }
  }
  const double ncc = (w_sum * ab_sum - a_sum * b_sum) /
                     std::sqrt((w_sum * aa_sum - a_sum * a_sum) *
                               (w_sum * bb_sum - b_sum * b_sum));
  const double cost = 1.0 - ncc;
  EXPECT_NEAR(matcher.GetSelProbMap().Get(kRow, kCol),
              std::exp(-0.5 * cost * cost / (0.3 * 0.3)),
              0.002);
}

TEST_F(PatchMatchMetalTest, ReconstructsTexturedPlaneFromPhotometry) {
  PlaneScene scene;
  PatchMatchOptions options;
  options.depth_min = 2.0;
  options.depth_max = 6.0;
  options.geom_consistency = false;
  options.window_radius = 4;
  options.num_samples = 4;
  options.filter_min_ncc = 0.8;
  options.filter_min_triangulation_angle = 1.0;
  PatchMatch matcher(options, scene.problem);
  matcher.Run();

  const DepthMap depths = matcher.GetDepthMap();
  const NormalMap normals = matcher.GetNormalMap();
  EXPECT_EQ(depths.GetWidth(), kWidth);
  EXPECT_EQ(normals.GetHeight(), kHeight);
  size_t total = 0;
  size_t facing = 0;
  double error_sum = 0.0;
  size_t valid = 0;
  for (int y = 10; y < kHeight - 10; ++y) {
    for (int x = 10; x < kWidth - 10; ++x) {
      ++total;
      const float depth = depths.Get(y, x);
      if (depth <= 0) {
        continue;
      }
      ++valid;
      error_sum += std::abs(depth - kDepth);
      float normal[3];
      normals.GetSlice(y, x, normal);
      facing += normal[2] < -0.9f;
      EXPECT_NEAR(std::hypot(normal[0], normal[1], normal[2]), 1.0f, 1e-5f);
    }
  }
  ASSERT_GT(valid, 0);
  RecordProperty("valid_depth_fraction", std::to_string(double(valid) / total));
  RecordProperty("mean_absolute_depth_error",
                 std::to_string(error_sum / valid));
  RecordProperty("normal_alignment_fraction",
                 std::to_string(double(facing) / valid));
  EXPECT_GT(double(valid) / total, 0.95);
  EXPECT_LT(error_sum / valid, 0.08);
  EXPECT_GT(double(facing) / valid, 0.85);
}

TEST_F(PatchMatchMetalTest, GeometricFilterAndGraph) {
  PlaneScene scene;
  PatchMatchOptions options;
  options.depth_min = options.depth_max = kDepth;
  options.num_iterations = 1;
  options.filter_min_triangulation_angle = 1.0;
  options.filter_geom_consistency_max_cost = 0.5;
  options.geom_consistency_max_cost = 0.1;
  options.write_consistency_graph = true;

  // At this baseline, z=5 in a source map produces one pixel of reprojection
  // error. Filtering uses the uncapped error rather than the search penalty.
  scene.depth_maps[1].Fill(5.0f);
  PatchMatch reject(options, scene.problem);
  reject.Run();
  EXPECT_EQ(reject.GetDepthMap().Get(30, 39), 0.0f);
  EXPECT_EQ(reject.GetNormalMap().Get(30, 39, 2), 0.0f);

  options.filter_min_num_consistent = 1;
  PatchMatch accept(options, scene.problem);
  accept.Run();
  EXPECT_EQ(accept.GetDepthMap().Get(30, 39), kDepth);
  EXPECT_NEAR(accept.GetNormalMap().Get(30, 39, 2), -1.0f, 1e-5f);
  const auto graph = accept.GetConsistencyGraph();
  int num_images = 0;
  const int* image_idxs = nullptr;
  graph.GetImageIdxs(30, 39, &num_images, &image_idxs);
  ASSERT_EQ(num_images, 1);
  EXPECT_EQ(image_idxs[0], 2);
}

}  // namespace
}  // namespace colmap::mvs
