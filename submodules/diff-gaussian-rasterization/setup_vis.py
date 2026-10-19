#
# Builds ONLY the visible-alpha IEKF operators as diff_gaussian_rasterization._C_vis,
# in place, without touching the production _C module:
#     python setup_vis.py build_ext --inplace
#
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

HERE = os.path.dirname(os.path.abspath(__file__))
setup(
    name="diff_gaussian_rasterization_vis",
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C_vis",
            sources=["cuda_rasterizer/iekf_visibility.cu", "iekf_visibility_points.cu", "ext_vis.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(HERE, "third_party/glm/"),
                                         "-I" + os.path.join(HERE, "cuda_rasterizer")]})
    ],
    cmdclass={"build_ext": BuildExtension},
)
