# Conda CUDA 12.8 env `rigeuler3dgs` for RTX 4070, vanilla 3DGS, Euler dynamic 3DGS, and DINOv3

This is the conda translation of the CUDA 12.8 Docker setup.

Key differences from the RTX 5090 Dockerfile:

```bash
export TORCH_CUDA_ARCH_LIST="8.9"
```

because RTX 4070 / Ada uses compute capability 8.9, not 12.0.

## Create the env

```bash
conda env create -f environment_rigeuler3dgs_cuda128_py310.yml
conda activate rigeuler3dgs
```

## Install PyTorch nightly cu128 + build 3DGS extensions

Run from the Gaussian Splatting repo root, or pass the repo root as the first argument:

```bash
bash install_after_conda_create.sh /path/to/gaussian-splatting
```

## DINOv3 submodule

```bash
cd /path/to/gaussian-splatting
git submodule add https://github.com/facebookresearch/dinov3 submodules/dinov3
git submodule update --init --recursive
python -m pip install -e submodules/dinov3
```

## Test

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
import diff_gaussian_rasterization
import simple_knn._C
print('OK')
PY
```
