#!/usr/bin/env python3
"""Render upright Front & 180-deg +Y Rear comparison sheets via Open3D raycasting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
from warp_core import camera_from_npz, extract_surface, load_json

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def get_fonts() -> tuple[Any, Any, Any]:
    try:
        return (
            ImageFont.truetype(FONT_PATH, 16),
            ImageFont.truetype(FONT_PATH, 13),
            ImageFont.truetype(FONT_PATH, 11),
        )
    except Exception:
        f = ImageFont.load_default()
        return f, f, f


def rear_camera(cam: dict[str, Any]) -> dict[str, Any]:
    """Proper 180-deg yaw around world +Y keeping the camera upright above the floor."""
    c = dict(cam)
    for k in ("position", "right", "up", "forward"):
        v = np.asarray(cam[k], dtype=np.float64)
        c[k] = np.array([-v[0], v[1], -v[2]], dtype=np.float64)
    return c


def render_mesh(
    verts: np.ndarray, faces: np.ndarray, cam: dict[str, Any], res: int = 280
) -> Image.Image:
    u = np.linspace(-float(cam["half_extent"]), float(cam["half_extent"]), res, dtype=np.float64)
    uu, vv = np.meshgrid(u, u, indexing="xy")
    pos = np.asarray(cam["position"], dtype=np.float64)
    right = np.asarray(cam["right"], dtype=np.float64)
    up = np.asarray(cam["up"], dtype=np.float64)
    fwd = np.asarray(cam["forward"], dtype=np.float64)
    origins = pos[None, None, :] + uu[..., None] * right + vv[..., None] * up
    dirs = np.broadcast_to(fwd, origins.shape)

    v32 = np.ascontiguousarray(verts, dtype=np.float32)
    f32 = np.ascontiguousarray(faces, dtype=np.uint32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(v32), o3d.core.Tensor(f32))
    rays = np.concatenate([origins, dirs], axis=-1).reshape(-1, 6).astype(np.float32)
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    t = ans["t_hit"].numpy()
    prim = ans["primitive_ids"].numpy()
    valid = np.isfinite(t) & (prim >= 0) & (prim < len(f32))
    rgb = np.full((res, res, 3), 255, dtype=np.uint8)
    idx = np.flatnonzero(valid)
    if len(idx):
        tri = f32[prim[idx]]
        v0 = v32[tri[:, 0]].astype(np.float64)
        v1 = v32[tri[:, 1]].astype(np.float64)
        v2 = v32[tri[:, 2]].astype(np.float64)
        n = np.cross(v1 - v0, v2 - v0)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
        n[np.einsum("ij,j->i", n, fwd) > 0] *= -1
        key = -0.35 * right + 0.55 * up - 0.76 * fwd
        key /= np.linalg.norm(key)
        fill = 0.65 * right + 0.15 * up - 0.74 * fwd
        fill /= np.linalg.norm(fill)
        diffuse = 0.56 + 0.34 * np.maximum(n @ key, 0.0) + 0.10 * np.maximum(n @ fill, 0.0)
        shade = np.clip(0.76 * diffuse, 0.0, 1.0)
        rgb.reshape(-1, 3)[idx] = np.rint(shade[:, None] * 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(rgb[::-1]), "RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render comparison sheets from run_eval.py outputs.")
    ap.add_argument("--run-dir", type=Path, required=True, help="Output directory of run_eval.py")
    ap.add_argument("--manifest", type=Path, required=True, help="Manifest JSON (e.g. test.json)")
    ap.add_argument("--data-root", type=Path, required=True, help="Root dir for manifest NPZ paths")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    res_data = load_json(run_dir / "results.json")
    per = res_data["per_shape"]
    entry_by_oid = {e["object_id"]: e for e in load_json(args.manifest)}
    ftitle, fcol, fsmall = get_fonts()

    def load_arm(oid: str, arm: str) -> tuple[np.ndarray, np.ndarray]:
        with np.load(run_dir / "inference" / f"{oid}.npz", allow_pickle=False) as z:
            return np.asarray(z[f"{arm}_verts"], np.float64), np.asarray(z[f"{arm}_faces"], np.int64)

    def load_gt_cam(oid: str) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
        with np.load(args.data_root / entry_by_oid[oid]["path"], allow_pickle=False) as z:
            m = extract_surface(np.asarray(z["target_sdf"], np.float32), -1.0, 1.0, 64)
            assert m is not None
            return m, camera_from_npz(z)

    # Worst-6 Front + Upright Rear sheet
    worst6 = sorted(per, key=lambda x: -x["full_chamfer"]["hy3d_cnf_normal_warp"])[:6]
    cell, pad, hdr = 250, 26, 44
    cols = [
        "GT (Front)",
        "hy3d_raw (Front)",
        "Normal-Only (Front)",
        "GT (Rear)",
        "hy3d_raw (Rear)",
        "Normal-Only (Rear)",
    ]
    sheet = Image.new("RGB", (6 * cell, hdr + len(worst6) * (cell + pad)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 6), "Worst 6 by Normal-Only Full Chamfer (Front & Rear)", fill=(0, 0, 0), font=ftitle)
    for ci, ct in enumerate(cols):
        draw.rectangle([ci * cell, 24, (ci + 1) * cell - 1, hdr - 1], fill=(235, 238, 244))
        draw.text((ci * cell + 8, 27), ct, fill=(0, 0, 0), font=fcol)
    for ri, r in enumerate(worst6):
        oid = r["object_id"]
        gt_m, cam = load_gt_cam(oid)
        rcam = rear_camera(cam)
        y0 = hdr + ri * (cell + pad)
        for ci, m in enumerate((gt_m, load_arm(oid, "hy3d_raw"), load_arm(oid, "hy3d_cnf_normal_warp"))):
            sheet.paste(render_mesh(m[0], m[1], cam, cell), (ci * cell, y0))
            sheet.paste(render_mesh(m[0], m[1], rcam, cell), ((ci + 3) * cell, y0))
        fr = r["full_chamfer"]["hy3d_raw"]
        fn = r["full_chamfer"]["hy3d_cnf_normal_warp"]
        mj = r["folding"]["hy3d_cnf_normal_warp"]["surface_min_detJ"]
        draw.text(
            (8, y0 + cell + 5),
            f"{oid[:10]}  Full: {fr:.5f} -> {fn:.5f}  minJ={mj:.3f}",
            fill=(20, 20, 20),
            font=fcol,
        )
    out_p = run_dir / "artifacts" / "comparison_worst6_normal_warp.png"
    sheet.save(out_p)
    print("Saved", out_p)

    # Full overview grids (32 chairs per sheet: raw | Normal-Only)
    for part_i, sub in enumerate((per[:32], per[32:]), start=1):
        if not sub:
            continue
        c2, p2, h2 = 180, 20, 36
        n_rows = (len(sub) + 3) // 4
        ov = Image.new("RGB", (8 * c2, h2 + n_rows * (c2 + p2)), (255, 255, 255))
        dr = ImageDraw.Draw(ov)
        dr.text((10, 8), f"Overview Part {part_i}: Left=hy3d_raw, Right=Normal-Only", fill=(0, 0, 0), font=ftitle)
        for k, r in enumerate(sub):
            oid = r["object_id"]
            _, cam = load_gt_cam(oid)
            x0, y0 = (k % 4) * 2 * c2, h2 + (k // 4) * (c2 + p2)
            ov.paste(render_mesh(*load_arm(oid, "hy3d_raw"), cam, c2), (x0, y0))
            ov.paste(render_mesh(*load_arm(oid, "hy3d_cnf_normal_warp"), cam, c2), (x0 + c2, y0))
            fr = r["full_chamfer"]["hy3d_raw"]
            fn = r["full_chamfer"]["hy3d_cnf_normal_warp"]
            mj = r["folding"]["hy3d_cnf_normal_warp"]["surface_min_detJ"]
            dr.text((x0 + 4, y0 + c2 + 3), f"{oid[:8]} {fr:.4f}->{fn:.4f} J={mj:.2f}", fill=(25, 25, 25), font=fsmall)
        out_ov = run_dir / "artifacts" / f"grid_part{part_i}_front.png"
        ov.save(out_ov)
        print("Saved", out_ov)


if __name__ == "__main__":
    main()
