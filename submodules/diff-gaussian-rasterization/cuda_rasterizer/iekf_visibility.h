/*
 * Visible-object-alpha operators on a frozen IEKF linearization cache
 * (see iekf_visibility.cu). Additive: nothing in iekf.h changes.
 */
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "iekf.h"

namespace CudaRasterizer { namespace IEKFVis {

void visibleAlpha(int P, int W, int H, const IEKF::CachePointers cache,
                  const uint2* ranges, const uint32_t* point_list,
                  const int* ray_mode, const float* z_obs, float delta_z,
                  float* out_avis, float* out_Tf, uint32_t* out_n_contrib,
                  int* out_front, int* out_behind, bool debug);

void applyJVP(int P, int W, int H, const IEKF::CachePointers cache,
              const uint2* ranges, const uint32_t* point_list,
              const float* means3D, const float* viewmatrix, const float* projmatrix,
              float tan_fovx, float tan_fovy,
              const int* ray_mode, const float* z_obs, float delta_z,
              const float* pose_delta, float* out_davis, bool debug);

void applyVJP(int P, int W, int H, const IEKF::CachePointers cache,
              const uint2* ranges, const uint32_t* point_list,
              const float* means3D, const float* viewmatrix, const float* projmatrix,
              float tan_fovx, float tan_fovy,
              const int* ray_mode, const float* z_obs, float delta_z,
              const float* z_alpha, const float* final_Tf, const uint32_t* n_contrib,
              float* out_adjoint, bool debug);

void buildFisher(int P, int W, int H, const IEKF::CachePointers cache,
                 const uint2* ranges, const uint32_t* point_list,
                 const float* means3D, const float* viewmatrix, const float* projmatrix,
                 float tan_fovx, float tan_fovy,
                 const int* ray_mode, const float* z_obs, float delta_z,
                 const float* weight_alpha, const float* final_Tf, const uint32_t* n_contrib,
                 float* out_fisher, bool debug);

}} // namespace CudaRasterizer::IEKFVis
