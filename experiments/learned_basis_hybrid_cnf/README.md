# Wan 1.3B action-conditioned world model — input-level v2 (learned-basis Hybrid-CNF)

This directory is a faithful, path-configurable migration of the 2026-11-14
**input-level v2** experiment (learned-basis Hybrid-CNF, 16 scenes). It does
**not** re-derive the reported numbers; it migrates the code so a collaborator
can install it, run the CPU checks, and reproduce the full
data → training → sampling → control rendering → Wan → scoring → statistics →
PSNR chain against their own copies of the data/model/LoRA/evaluator.

The already-produced 256-video results remain pre-existing evidence (see
[docs/historical_results_20261114.md](docs/historical_results_20261114.md) and
[docs/reproduction.md](docs/reproduction.md)).

## What this package contains

```text
experiments/learned_basis_hybrid_cnf/
  pyproject.toml                       installable package "learned-basis-hybrid-cnf"
  src/learned_basis_hybrid_cnf/
    config.py                          central path resolution (CNF_V2_* env / JSON config)
    basis.py                           NeuralKernelBasis (Legendre + neural kernel residual + QR)
    projection.py                      exact float64 hard projection (no ridge)
    data.py                            v2 reference repair + dataset loading
    train.py                           fixed/learned basis training (6000 steps)
    sampling.py                        50-step Euler sampling, SLSQP safety repair, certificates
    util_eval.py                       rendering, hashing, protocol, shared constants
    video.py                           Wan 1.3B + LoRA pipeline and GT-replay gate
    scoring.py                         frozen RGB segmenter (SmallUNet) scoring
    statistics.py                      scene-paired bootstrap statistics
    evaluate.py                        orchestration and report writing
    tests/                             pure-CPU numerics regression tests
  scripts/
    run_learned_basis_hybrid_cnf.py    main entry (--stage train|motion|video|score|stats + --run-all)
    finalize_learned_basis_hybrid_cnf.py  CPU-only stats/report finalizer
    psnr_eval.py                       supplemental CPU PSNR evaluation
    generate_dataset.py                base pick-and-place dataset generation
    prepare_v2_references.py           16 v2 eval references + targets
    train_evaluator.py                 train/calibrate the frozen RGB segmenter
  configs/paths.example.json           example config (placeholders)
  docs/                                reproduction guide + historical results
```

The shared `cnf_motion` package (Legendre basis, flow prior, PyBullet scene,
soft field, metrics) lives at the repository root in `src/cnf_motion/` and is
installed together with the root package.

## Install

From the repository root, in a fresh Python 3.10+ virtualenv:

```bash
python3 -m venv .venv-wan
source .venv-wan/bin/activate
python -m pip install -U pip
# installs gcnf + cnf_motion and the motion dependencies (torch, pybullet, ...)
python -m pip install -e '.[motion]'
# installs this experiment package (plus the optional Wan inference deps)
python -m pip install -e 'experiments/learned_basis_hybrid_cnf[wan]'
```

`[wan]` adds `diffusers`, `omegaconf`, `transformers`, `accelerate`,
`safetensors`, `Pillow`. VideoX-Fun itself is an external repository pointed to
via `CNF_V2_VIDEOX`; it is not vendored here.

## Configuration

Every external resource and the output directory are resolved by
`learned_basis_hybrid_cnf.config` from, in order of precedence:

1. environment variables prefixed `CNF_V2_`;
2. a JSON config file at `CNF_V2_CONFIG` (default
   `experiments/learned_basis_hybrid_cnf/configs/paths.json`);
3. defaults (only the output directory `outputs/learned_basis_hybrid_cnf_v2`).

| Variable | Meaning |
|---|---|
| `CNF_V2_OUT` | output directory (default `outputs/learned_basis_hybrid_cnf_v2`) |
| `CNF_V2_TRAIN_DS` | train dataset dir (`pickplace256`: `q/*.npy`, `metadata.json`) |
| `CNF_V2_EVAL_DS` | eval dataset dir (`pickplace256_eval`: `q/`, `ref/`, `train/`, `metadata.json`) |
| `CNF_V2_V2_COND` | dir containing v2 eval `references/*.npy` and `targets/` |
| `CNF_V2_WAN_MODEL` | Wan 1.3B Control model dir |
| `CNF_V2_WAN_LORA` | Control LoRA `.safetensors` |
| `CNF_V2_VIDEOX` | VideoX-Fun repository dir |
| `CNF_V2_EVALUATOR_PT/SPEC/CAL` | frozen RGB evaluator checkpoint/spec/calibration |
| `CNF_V2_BASE_1_3B` | input-baseline dir with the 8 GT-replay `control/*_gt.mp4`, `videos/*_gt.mp4` |
| `CNF_V2_OLD_V2_COND` | legacy 8-scene v2 references (only for `prepare_v2_references.py`) |

Example (using a JSON config, from the repository root):

```bash
cp experiments/learned_basis_hybrid_cnf/configs/paths.example.json \
   experiments/learned_basis_hybrid_cnf/configs/paths.json    # then edit it
export CNF_V2_CONFIG="$PWD/experiments/learned_basis_hybrid_cnf/configs/paths.json"
python experiments/learned_basis_hybrid_cnf/scripts/run_learned_basis_hybrid_cnf.py --verify-config
```

## Data, model, and weight preparation

Binary artifacts (dataset, Wan model, LoRA, evaluator, v2 references) are **not
committed to Git**. Their sources and whether they are regenerable here are in
[docs/reproduction.md](docs/reproduction.md); the key facts:

- **Base dataset** is regenerable and deterministic:
  ```bash
  python experiments/learned_basis_hybrid_cnf/scripts/generate_dataset.py \
    --output datasets/pickplace256 --count 128 --seed 0
  python experiments/learned_basis_hybrid_cnf/scripts/generate_dataset.py \
    --output datasets/pickplace256_eval --count 16 --seed 1
  ```
  (requires the `franka_panda/panda.urdf` from `pybullet_data`.)

- **16 v2 references + targets**: `prepare_v2_references.py` generates scenes
  8–15 with the repair rule but **reuses scenes 0–7 from the legacy 8-scene v2
  references** (`CNF_V2_OLD_V2_COND`), which are **not regenerable from this
  repo**.
  ```bash
  python experiments/learned_basis_hybrid_cnf/scripts/prepare_v2_references.py
  ```

- **Frozen RGB evaluator** (`SmallUNet`, 2000 steps) is regenerable:
  ```bash
  python experiments/learned_basis_hybrid_cnf/scripts/train_evaluator.py
  ```

- **Wan 1.3B Control model + LoRA**, **VideoX-Fun**, and the **8 GT-replay
  control/video pairs** (`CNF_V2_BASE_1_3B`) are external/pre-existing and are
  **not regenerable here**; the LoRA training itself is not migrated.

All four preparation scripts and `--stage train` refuse to overwrite existing
outputs unless `--force` is passed.

## Reproducible commands (staged; run in order from the repository root)

```bash
# CPU-only numerical verification (no GPU, no Wan): needs eval dataset, v2 refs,
# evaluator; writes into CNF_V2_OUT.
python experiments/learned_basis_hybrid_cnf/scripts/run_learned_basis_hybrid_cnf.py --verify-ready

# 1. train fixed + learned bases (also repairs the 128 train trajectories into CNF_V2_OUT).
#    Use --force only to overwrite an existing training run.
python experiments/learned_basis_hybrid_cnf/scripts/run_learned_basis_hybrid_cnf.py --stage train

# 2. sample 128 trajectories + render semantic controls (CNF / safety / certificates).
python experiments/learned_basis_hybrid_cnf/scripts/run_learned_basis_hybrid_cnf.py --stage motion

# 3. Wan 1.3B + LoRA: 256-video matrix (GPU; the expensive step, ~36 min on one A100).
python experiments/learned_basis_hybrid_cnf/scripts/run_learned_basis_hybrid_cnf.py --stage video

# 4. scoring + statistics.
python experiments/learned_basis_hybrid_cnf/scripts/run_learned_basis_hybrid_cnf.py --stage score
python experiments/learned_basis_hybrid_cnf/scripts/run_learned_basis_hybrid_cnf.py --stage stats

# 5. CPU-only PSNR over the generated videos.
python experiments/learned_basis_hybrid_cnf/scripts/psnr_eval.py
```

`--run-all` is shorthand for stages 2–4 plus report writing; it **does not**
train (stage 1) and **does not** compute PSNR (stage 5). All outputs go under
`CNF_V2_OUT`; point `CNF_V2_OUT` at a fresh directory to avoid touching existing
results. The entry also refuses to start GPU work while another process holds
the GPU (`check_gpu_free_for_eval`).

## Tests

Pure-CPU numerics regression tests (QR orthonormality, exact projection,
finite-difference gradients through the real training path) need no dataset,
model, or GPU:

```bash
python -m unittest discover -s experiments/learned_basis_hybrid_cnf/src/learned_basis_hybrid_cnf/tests -v
```

These cover the numerical core (basis, projection, training-step losses). The
PyBullet/Wan stages are exercised by `--verify-ready` and the staged runs.

## Provenance and result scope

- Historical results, artifact SHA256, and the verification scope are in
  [docs/historical_results_20261114.md](docs/historical_results_20261114.md) and
  [docs/reproduction.md](docs/reproduction.md); they are marked historical and
  are not presented as newly re-run.
- The 16-scene v2 protocol is distinct from the historical 8-scene A/C
  experiment; this package implements only the v2 protocol.
- RGB adherence, simulator-domain segmenter calibration, and single-training-seed
  limitations are as stated in the historical record; no new claims are made here.
