#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$HOME/Joint3DGS/RigEuler3DGS}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rigeuler3dgs

if [[ "${CONDA_DEFAULT_ENV:-}" != "rigeuler3dgs" ]]; then
  echo "[ERROR] Expected conda env rigeuler3dgs, got ${CONDA_DEFAULT_ENV:-none}" >&2
  exit 1
fi

cd "$REPO_ROOT"

echo "[INFO] Repo: $REPO_ROOT"
echo "[INFO] Python: $(which python)"
python -V

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export FORCE_CUDA=1
# RTX 4070 / Ada Lovelace = compute capability 8.9
export TORCH_CUDA_ARCH_LIST="8.9"

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "[ERROR] nvcc not found at $CUDA_HOME/bin/nvcc. Check /usr/local/cuda-12.8 installation." >&2
  exit 1
fi

python -m pip install --upgrade pip setuptools wheel ninja packaging

# Stable CUDA 12.8 PyTorch stack. Do not use nightly here: nightly produced mismatched
# torch/torchvision dates in the failed base install.
python -m pip install --no-cache-dir \
  torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu128

python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
PY

# Build 3DGS CUDA extensions against the active env + CUDA 12.8.
# diff-gaussian-rasterization is NOT a submodule: it carries this project's own
# CUDA kernels and its sources are tracked in this repository. Its GLM
# dependency is not -- 1530 unmodified upstream headers -- so fetch it here.
# Without this the build fails on a fresh clone with a missing <glm/glm.hpp>.
GLM_DIR=submodules/diff-gaussian-rasterization/third_party/glm
if [[ ! -f "$GLM_DIR/glm/glm.hpp" ]]; then
  echo "fetching GLM for the rasterizer"
  rm -rf "$GLM_DIR"
  git clone https://github.com/g-truc/glm.git "$GLM_DIR"
  git -C "$GLM_DIR" checkout 5c46b9c07008ae65cb81ab79cd677ecc1934b903
fi

python -m pip install --no-build-isolation -v --force-reinstall submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation -v --force-reinstall submodules/simple-knn
if [[ -d submodules/fused-ssim ]]; then
  python -m pip install --no-build-isolation -v --force-reinstall submodules/fused-ssim
fi

# DINOv3 as local submodule. If already present, only install editable.
if [[ ! -d submodules/dinov3 ]]; then
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git submodule add https://github.com/facebookresearch/dinov3 submodules/dinov3 || git clone https://github.com/facebookresearch/dinov3 submodules/dinov3
    git submodule update --init --recursive
  else
    git clone https://github.com/facebookresearch/dinov3 submodules/dinov3
  fi
fi
python -m pip install -e submodules/dinov3

python - <<'PY'
import torch
import diff_gaussian_rasterization
import simple_knn._C
print('3DGS CUDA extensions: OK')
try:
    import dinov3
    print('DINOv3 import: OK')
except Exception as exc:
    print('DINOv3 import warning:', repr(exc))
PY
