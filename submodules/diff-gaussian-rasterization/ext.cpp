/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 */
#include <torch/extension.h>
#include "rasterize_points.h"
#include "iekf_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_contributors", &RasterizeGaussiansContributorsCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("rasterize_gaussian_features", &RasterizeGaussianFeaturesCUDA);
  m.def("rasterize_gaussian_features_backward", &RasterizeGaussianFeaturesBackwardCUDA);
  m.def("rasterize_gaussian_positions", &RasterizeGaussianPositionsCUDA);
  m.def("rasterize_gaussian_positions_backward", &RasterizeGaussianPositionsBackwardCUDA);
  m.def("rasterize_gaussian_flow_stats", &RasterizeGaussianFlowStatsCUDA);
  m.def("rasterize_gaussian_pixel_feature_stats", &RasterizeGaussianPixelFeatureStatsCUDA);
  m.def("mark_visible", &markVisible);
  // Sparse alpha-aware IEKF observation operators (additive; the bindings above
  // are unchanged so the legacy path is unaffected).
  m.def("iekf_build_cache", &IEKFBuildCacheCUDA);
  m.def("iekf_render", &IEKFRenderCUDA);
  m.def("iekf_apply_H", &IEKFApplyHCUDA);
  m.def("iekf_apply_HT", &IEKFApplyHTCUDA);
  m.def("iekf_build_fisher", &IEKFBuildFisherCUDA);
  // Grouped / supernode state (V2 section 3.3).
  m.def("iekf_expand_group_tangent", &IEKFExpandGroupTangentCUDA);
  m.def("iekf_reduce_group_adjoint", &IEKFReduceGroupAdjointCUDA);
  m.def("iekf_reduce_group_fisher", &IEKFReduceGroupFisherCUDA);
}
