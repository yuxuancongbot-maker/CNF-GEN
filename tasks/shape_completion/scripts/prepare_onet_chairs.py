#!/usr/bin/env python3
"""Reproducible preparation of the ONet watertight chair SDF records.

Reads frozen chair IDs from a split config, extracts ONLY those members from
the verified ``03001627.zip`` archive, and runs the vendored data-preparation
pipeline (camera, normalisation, 64^3 TSDF, 32 anchors, first-hit normals).

Usage (from the task directory or repo root)::

  python scripts/prepare_onet_chairs.py \
      --archive /path/to/03001627.zip \
      --out /path/to/output/data \
      --config configs/onet_chair_split.json \
      --workers 8

The archive must already have been obtained with scripts/download_onet_chairs.py.
No download, training, or GPU work is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = TASK_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gcnf_shape_completion import data_prep as prep  # noqa: E402


def _libegl_from_arg(value: str | None) -> None:
    if value:
        os.environ["LD_LIBRARY_PATH"] = value + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    # Common local locations for a headless libEGL.so.1.
    for candidate in (TASK_ROOT / ".lib", TASK_ROOT / "src" / "gcnf_shape_completion" / ".lib"):
        if candidate.is_dir():
            os.environ["LD_LIBRARY_PATH"] = (
                str(candidate) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
            )


def discover_off_map(zf: zipfile.ZipFile) -> dict[str, str]:
    off_map: dict[str, str] = {}
    for name in zf.namelist():
        if not name.endswith(".off"):
            continue
        parts = Path(name).parts
        if "4_watertight_scaled" not in parts:
            continue
        if name.startswith("/") or ".." in parts:
            continue
        oid = Path(name).stem
        if not (29 <= len(oid) <= 32):
            continue
        if any(c not in "0123456789abcdef" for c in oid):
            continue
        off_map[oid] = name
    return off_map


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if "ids" not in cfg:
        raise ValueError("split config missing 'ids'")
    for split in ("train", "val", "test"):
        if split not in cfg["ids"]:
            raise ValueError(f"split config missing ids.{split}")
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True, type=Path, help="path to verified 03001627.zip")
    ap.add_argument("--out", required=True, type=Path, help="output data directory")
    ap.add_argument("--config", type=Path, default=TASK_ROOT / "configs" / "onet_chair_split.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true", help="recompute existing records")
    ap.add_argument("--libegl", default=None, help="directory containing libEGL.so.1")
    ap.add_argument(
        "--limit", type=int, default=0, help="process only first N ids per split (debug)"
    )
    args = ap.parse_args()

    _libegl_from_arg(args.libegl)

    t_start = time.time()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = read_config(args.config.resolve())
    ids_cfg = cfg["ids"]
    counts = cfg.get("selection", {}).get("counts", {})
    N = {s: int(counts.get(s, len(ids_cfg[s]))) for s in ("train", "val", "test")}

    # Apply limit (debug) by trimming IDs.
    if args.limit and args.limit > 0:
        for s in ("train", "val", "test"):
            ids_cfg[s] = ids_cfg[s][: args.limit]
            N[s] = len(ids_cfg[s])

    # Frozen IDs must be unique and disjoint.
    all_ids = [i for s in ("train", "val", "test") for i in ids_cfg[s]]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("frozen split IDs are not disjoint")
    print(f"[prep] loaded {sum(len(v) for v in ids_cfg.values())} frozen ids", flush=True)

    archive = args.archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found: {archive}")

    with zipfile.ZipFile(archive) as zf:
        off_map = discover_off_map(zf)
        print(f"[prep] archive has {len(off_map)} chair OFF meshes", flush=True)

        # Validate all frozen IDs are available.
        missing = [oid for oid in all_ids if oid not in off_map]
        if missing:
            raise RuntimeError(f"{len(missing)} frozen IDs missing from archive: {missing[:10]}")

        selections = {s: list(ids_cfg[s]) for s in ("train", "val", "test")}
        needed = set(all_ids)
        off_paths: dict[str, Path] = {}
        for oid in sorted(needed):
            name = off_map[oid]
            zf.extract(name, out)
            p = (out / name).resolve()
            if out not in p.parents or not p.is_file():
                raise RuntimeError(f"unsafe/missing extraction: {name}")
            off_paths[oid] = p
        print(f"[prep] extracted {len(off_paths)} meshes", flush=True)

    prep_mod = prep

    # Benchmark on three real OFFs (matches original runner behaviour).
    bench = list(selections["train"])[:3]
    b0 = time.time()
    for oid in bench:
        prep_mod.prepare_one(
            prep_mod.PrepareTask(
                mesh_path=str(off_paths[oid]),
                object_id=oid,
                dataset="ShapeNet-chair-ONet-watertight",
                source_split="train",
                out_dir=str(out),
                seed=prep_mod._seed_from_text(oid),
                overwrite=args.overwrite,
            )
        )
    avg_s = (time.time() - b0) / max(1, len(bench))
    projected_wall = avg_s * len(all_ids) / max(1, args.workers)
    print(
        f"[prep] benchmark avg={avg_s:.1f}s/mesh projected_wall={projected_wall / 60:.1f}min",
        flush=True,
    )

    simplify_record: dict = {}
    use_simplified = projected_wall > 90 * 60
    if use_simplified:
        print("[prep] enabling bounded Open3D quadric simplification at 80000 faces", flush=True)
        # reuse original simplify_all via a local implementation
        import open3d as o3d

        dest_dir = out / "simplified_off"
        dest_dir.mkdir(exist_ok=True)
        new_paths = {}
        for oid, src in off_paths.items():
            dst = dest_dir / f"{oid}.off"
            mesh = o3d.io.read_triangle_mesh(str(src))
            before = len(mesh.triangles)
            if before > 80000:
                mesh = mesh.simplify_quadric_decimation(80000)
                o3d.io.write_triangle_mesh(str(dst), mesh)
                changed = True
            else:
                shutil.copyfile(src, dst)
                changed = False
            new_paths[oid] = dst
            simplify_record[oid] = dict(
                before_faces=before, after_faces=len(mesh.triangles), simplified=changed
            )
        off_paths = new_paths
        atomic_json(
            out / "simplification.json",
            dict(
                enabled=True,
                method="Open3D simplify_quadric_decimation",
                target_faces=80000,
                applied_to="all records identically",
                reason=f"projected_wall={projected_wall:.0f}s > 5400s",
                per_mesh=simplify_record,
            ),
        )
    else:
        atomic_json(
            out / "simplification.json",
            dict(
                enabled=False,
                reason=f"projected_wall={projected_wall:.0f}s within budget",
                benchmark_avg_s=avg_s,
            ),
        )

    def make_tasks(ids, split):
        src_split = "test" if split == "test" else "train"
        return [
            dict(
                mesh_path=str(off_paths[oid]),
                object_id=oid,
                dataset="ShapeNet-chair-ONet-watertight",
                source_split=src_split,
                out_dir=str(out),
                seed=prep_mod._seed_from_text(oid),
                overwrite=args.overwrite,
            )
            for oid in ids
        ]

    results: list[dict] = []
    round_no = 0
    pending = {s: list(selections[s]) for s in selections}
    failures: list[dict] = []
    winner_ids = {s: [] for s in selections}
    used_backups = {s: [] for s in selections}

    while True:
        round_no += 1
        tasks = []
        for split in ("train", "val", "test"):
            src = "test" if split == "test" else "train"
            tasks.extend(make_tasks(pending[split], split))
        if not tasks:
            break
        print(
            f"[prep] round {round_no}: preparing {len(tasks)} records with {args.workers} workers",
            flush=True,
        )
        r0 = time.time()
        rnd = prep_mod.run_tasks(tasks, args.workers)
        results.extend(rnd)
        for r in rnd:
            split = (
                "test"
                if r.get("source_split") == "test"
                else "val"
                if r["object_id"] in set(selections["val"])
                else "train"
            )
            if r.get("status") == "ok":
                winner_ids[split].append(r["object_id"])
                pending[split] = [x for x in pending[split] if x != r["object_id"]]
            else:
                failures.append(r)
                pending[split] = [x for x in pending[split] if x != r["object_id"]]
        print(
            f"[prep] round {round_no} done in {time.time() - r0:.1f}s: "
            "winners train/val/test="
            f"{len(winner_ids['train'])}/{len(winner_ids['val'])}/{len(winner_ids['test'])}",
            flush=True,
        )
        atomic_json(
            out / "prep_progress.json",
            dict(
                status="preparing",
                round=round_no,
                winners={k: len(v) for k, v in winner_ids.items()},
                target=N,
                failures=len(failures),
                elapsed_s=time.time() - t_start,
                updated_at=time.time(),
            ),
        )

        if all(len(winner_ids[s]) >= N[s] for s in ("train", "val", "test")):
            break
        if not any(pending.values()):
            print("[prep] no pending IDs left and targets not met", flush=True)
            break

    # Trim to exact frozen sizes.
    final_ids = {s: sorted(winner_ids[s])[: N[s]] for s in ("train", "val", "test")}
    by_id = {r["object_id"]: r for r in results}
    final_tasks_by_split = {
        s: [t for t in make_tasks(final_ids[s], s)] for s in ("train", "val", "test")
    }
    final_results = [
        by_id[oid] for s in ("train", "val", "test") for oid in final_ids[s] if oid in by_id
    ]

    # Verify all frozen IDs succeeded.
    for s in ("train", "val", "test"):
        if len(final_ids[s]) != N[s]:
            raise RuntimeError(f"{s}: only {len(final_ids[s])} succeeded, need {N[s]}")
        missing_res = [
            oid for oid in selections[s] if oid not in by_id or by_id[oid].get("status") != "ok"
        ]
        if missing_res:
            raise RuntimeError(f"{s}: failed records {missing_res[:5]}")

    split_meta = {
        "source": (
            "Official author-released ONet ShapeNet v1 watertight chairs (public research release)"
        ),
        "category": "03001627",
        "category_name": "chair",
        "member": "03001627.zip",
        "archive_sha256": cfg["source_release"]["member_sha256"],
        "official_lists": "frozen in configs/onet_chair_split.json",
        "selection_seed": cfg["selection"]["seed"],
        "val_fraction": cfg["selection"].get("val_fraction", 0.10),
        "val_carve_out_of": cfg["selection"].get(
            "val_carve_out_of", "official train (disjoint from test)"
        ),
        "subset": {s: N[s] for s in ("train", "val", "test")},
        "final_ids": final_ids,
        "backup_ids_used": used_backups,
        "excluded_counts": {
            "failed_records": len(failures),
        },
        "note": (
            "Frozen IDs from configs/onet_chair_split.json. "
            "NPZ/axes/camera/mask logic from vendored data_prep "
            "(prepare_chairs)."
        ),
    }
    summary = prep_mod.write_outputs(
        out,
        final_tasks_by_split,
        final_results,
        split_meta,
        "ShapeNet-chair-ONet-watertight (official public release)",
    )
    print(f"[prep] manifests written: {summary['counts']}", flush=True)

    # Strict validation (arrays, manifests, camera/depth reproducibility).
    report = prep_mod.verify_root(out, strict=True)
    atomic_json(out / "QA" / "strict_validation.json", report)
    atomic_json(
        out / "prep_progress.json",
        dict(
            status="validating" if report["ok"] else "validation_failed",
            winners={k: len(v) for k, v in final_ids.items()},
            validation_ok=report["ok"],
            problems=report["problems"][:50],
            elapsed_s=time.time() - t_start,
            updated_at=time.time(),
        ),
    )
    if not report["ok"]:
        print(f"[prep] STRICT VALIDATION FAILED: {report['problems'][:10]}", flush=True)
        atomic_json(
            out / "prep_status.json",
            dict(
                status="blocked_strict_validation",
                problems=report["problems"][:50],
                updated_at=time.time(),
            ),
        )
        return 2

    manifest_hashes = {s: sha256(out / f"{s}.json") for s in ("train", "val", "test")}
    # Verify manifest hashes match frozen config? (informational)
    print(f"[prep] manifest hashes: {manifest_hashes}", flush=True)

    atomic_json(
        out / "READY.json" if False else out / "prep_status.json",
        dict(
            status="READY",
            generated_at=time.time(),
            counts=summary["counts"],
            elapsed_s=time.time() - t_start,
            updated_at=time.time(),
        ),
    )
    print(f"[prep] READY-ish published in {(time.time() - t_start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
