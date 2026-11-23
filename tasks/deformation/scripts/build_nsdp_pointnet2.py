from __future__ import annotations

import argparse
import glob
import hashlib
import os
import shutil
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build NSDP PointNet2 FPS into an external runtime overlay"
    )
    parser.add_argument("--nsdp-root", required=True, type=Path)
    parser.add_argument("--overlay-root", required=True, type=Path)
    parser.add_argument("--build-root", required=True, type=Path)
    parser.add_argument("--cuda-home", required=True, type=Path)
    parser.add_argument("--cuda-arch", default="8.6")
    args = parser.parse_args()

    nsdp_root = args.nsdp_root.resolve()
    source_package = nsdp_root / "pointnet2_ops_lib" / "pointnet2_ops"
    source_root = source_package / "_ext-src"
    if not source_root.is_dir():
        raise FileNotFoundError(f"PointNet2 sources not found: {source_root}")
    if shutil.which("cl.exe") is None:
        raise RuntimeError("Run this script from a Visual Studio x64 developer environment")

    args.overlay_root.mkdir(parents=True, exist_ok=True)
    args.build_root.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_HOME"] = str(args.cuda_home)
    os.environ["TORCH_CUDA_ARCH_LIST"] = args.cuda_arch
    os.environ["DISTUTILS_USE_SDK"] = "1"

    import torch
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected a CUDA 12.8 Torch build, found {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the selected Torch environment")

    sources = sorted(glob.glob(str(source_root / "src" / "*.cpp")))
    sources += sorted(glob.glob(str(source_root / "src" / "*.cu")))
    if not sources:
        raise FileNotFoundError(f"No PointNet2 C++/CUDA sources found under {source_root}")

    extension = CUDAExtension(
        name="pointnet2_ops._ext",
        sources=sources,
        include_dirs=[str(source_root / "include")],
        extra_compile_args={
            "cxx": ["/O2"],
            "nvcc": ["-O3", "-Xfatbin", "-compress-all", "--allow-unsupported-compiler"],
        },
    )

    sys.argv = [
        sys.argv[0],
        "build_ext",
        "--build-lib",
        str(args.overlay_root),
        "--build-temp",
        str(args.build_root / "objects"),
    ]
    setup(
        name="nsdp-pointnet2-runtime-overlay",
        ext_modules=[extension],
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
    )

    overlay_package = args.overlay_root / "pointnet2_ops"
    built_extensions = list(overlay_package.glob("_ext*.pyd"))
    if len(built_extensions) != 1:
        raise RuntimeError(
            f"Expected one built extension in {overlay_package}, found {built_extensions}"
        )
    for filename in (
        "__init__.py",
        "_version.py",
        "pointnet2_modules.py",
        "pointnet2_utils.py",
    ):
        shutil.copy2(source_package / filename, overlay_package / filename)
    overlay_extension = built_extensions[0]

    print(f"extension: {overlay_extension}")
    print(f"extension_sha256: {sha256_file(overlay_extension)}")
    print(f"cuda_arch: {args.cuda_arch}")
    print(f"torch: {torch.__version__}")
    print(f"torch_cuda: {torch.version.cuda}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
