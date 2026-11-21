// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "colmap/mvs/patch_match.h"

namespace colmap::mvs {

class PatchMatchMetal {
 public:
  PatchMatchMetal(const PatchMatchOptions& options,
                  const PatchMatch::Problem& problem);
  ~PatchMatchMetal();

  static bool IsDeviceAvailable();
  static void CheckDevice();
  void Run();

  DepthMap GetDepthMap() const;
  NormalMap GetNormalMap() const;
  Mat<float> GetSelProbMap() const;
  std::vector<int> GetConsistentImageIdxs() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace colmap::mvs
