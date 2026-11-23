from __future__ import annotations

import argparse
from pathlib import Path

from gcnf_deformation.normalization import prepare_off_mesh


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize a triangular OFF mesh for NSDP")
    parser.add_argument("input_off", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--scale", type=float, default=0.35)
    args = parser.parse_args()

    mesh_path, matrix_path = prepare_off_mesh(args.input_off, args.output_dir, args.scale)
    print(f"mesh: {mesh_path}\ntransform: {matrix_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
