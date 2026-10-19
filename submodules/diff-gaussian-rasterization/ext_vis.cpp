/*
 * Visible-alpha IEKF operators, built as a SEPARATE module
 * (diff_gaussian_rasterization._C_vis, setup_vis.py) so the production _C is
 * not rebuilt; the operators read caches built by _C.iekf_build_cache.
 */
#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
IEKFVisibleAlphaCUDA(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                     const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                     const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                     const int, const int, const torch::Tensor&, const torch::Tensor&, const float, const bool);
torch::Tensor
IEKFVisApplyHCUDA(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                  const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                  const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                  const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const float, const float,
                  const int, const int, const torch::Tensor&, const torch::Tensor&, const float,
                  const torch::Tensor&, const bool);
torch::Tensor
IEKFVisApplyHTCUDA(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                   const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                   const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
                   const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const float, const float,
                   const int, const int, const torch::Tensor&, const torch::Tensor&, const float,
                   const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const bool);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("iekf_visible_alpha", &IEKFVisibleAlphaCUDA);
  m.def("iekf_vis_apply_H", &IEKFVisApplyHCUDA);
  m.def("iekf_vis_apply_HT", &IEKFVisApplyHTCUDA);
}
