# Prior-Preserving Hunyuan3D-2mini DiT LoRA + 1-Step IFT Differentiable CNF

Controlled 2-arm ablation investigating whether **differentiating through the Matérn-5/2 Normal-Only CNF warp during DiT LoRA fine-tuning** improves occluded 3D shape completion over **Plain LoRA + the exact same test-time CNF solver** and **Zero-Shot Hunyuan3D + CNF**.

## Method

1. **Frozen `ShapeVAE` + Rank-8 DiT LoRA**: Preserves `ShapeVAE`'s glass-smooth decoded surface; attaches rank-8 LoRA (`2.62M` params) to `double_blocks` and `single_blocks` (`qkv`, `proj`, `linear1`, `linear2`).
2. **Exact 1-Step Implicit Function Theorem (IFT) Differentiable CNF** ([`lora_ift_core.py`](lora_ift_core.py)):
   - Runs a fast 4-step detached GPU normal-only Matérn-5/2 Newton solve to reach equilibrium `f_0(p_i + u_det(p_i)) ≈ 0`.
   - Applies a single differentiable `32×32` SPD linear solve `A d_alpha = -f_0(p + u_det; theta) / ||grad f_0||` to backpropagate geometry loss gradients `dL/dtheta` cleanly through the CNF warp (verified against finite-difference full nonlinear re-solving: `rel_err = 2.386e-7`).
3. **Prior-Preservation Velocity Anchoring + 10% CFG Dropout**:
   - Because the public `hunyuan3d-vae-v2-mini` release omits `encoder.*` weights, we avoid noisy latent inversion (which caused Swiss-cheese surface holes in an earlier v1 probe) by sampling clean pretrained DiT priors `x1_prior`, aligning per-chair canonical `32³` bounding boxes, and anchoring LoRA velocity to the frozen pretrained DiT (`MSE(v_lora, v_ref)` with zero extra VRAM via `disable_lora`).

## 64-Chair Held-Out Results (Identical FP64 Matérn-5/2 Normal-Only Test-Time CNF)

| Method | Raw Full / Hidden CD | +CNF Full CD | +CNF Hidden CD | `min det(J)` (med / min) | Folding | `max\|f0\|` (FP64) | Mean Dihedral |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot + CNF** | `0.00425 / 0.02540` | `0.00186` | `0.01348` | `0.872 / 0.080` | `0/64` | `1.53e-13` | `6.59°` |
| **Plain LoRA v2 + CNF** | `0.00389 / 0.02456` | `0.00167` (**-10.0%**) | **`0.01311`** (**-2.8%**) | **`0.882`** / `0.082` | `0/64` | `1.21e-13` | `6.47°` |
| **CNF-Aware LoRA v2 + CNF** | `0.00398 / 0.02435` | **`0.00165`** (**-11.3%**) | `0.01338` (-0.8%) | `0.880` / **`0.155`** | `0/64` | **`1.20e-13`** | `6.49°` |

### Paired Bootstrap (10,000 Resamples, `CNF-Aware LoRA v2 + CNF` vs `Plain LoRA v2 + CNF`)

- **Hidden Chamfer**: `0.01338` vs `0.01311` (`+2.1%`, 95% CI `[-0.00041, +0.00109]`, **statistically tied**; `cnf_lora_v2` wins `35/64` individual chairs).
- **Full Chamfer**: `0.00165` vs `0.00167` (`-1.4%`, 95% CI `[-0.00013, +0.00008]`, statistically tied; `-11.3%` vs Zero-Shot `0.00186`).
- **Worst-Case Jacobian Safety**: `cnf_lora_v2` nearly doubles worst-case `min det(J)` (`0.155` vs `0.082` / `0.080`), and rescues the hardest Zero-Shot failure (`59fd3d1409`: Hidden CD `0.03611 → 0.00300`, `12×` reduction, `minJ 0.08 → 0.77`).

### Visual Comparison (Front & Rear Views)

- Front: [`artifacts/comparison_lora_v2_front.png`](artifacts/comparison_lora_v2_front.png)
- Rear: [`artifacts/comparison_lora_v2_rear.png`](artifacts/comparison_lora_v2_rear.png)

## Quick Start & Offline Tests

```bash
# Offline CPU unit tests (zero weights/GPU required; includes exact IFT vs FD gradient check)
python -m unittest discover -s tasks/shape_completion/hunyuan3d_dit_lora_cnf -p "test_*.py" -v

# 1. Cache clean pretrained DiT priors & per-chair 32^3 alignment
python tasks/shape_completion/hunyuan3d_dit_lora_cnf/prep_prior_cache.py \
  --data-root /path/to/shapenet_chairs \
  --rgba-dir /path/to/train_rgba \
  --hy3d-repo /path/to/Hunyuan3D-2 \
  --hy3d-weights /path/to/hunyuan3d-2mini-weights \
  --out-dir ./cache

# 2. Train matched Plain LoRA and CNF-Aware LoRA arms
python tasks/shape_completion/hunyuan3d_dit_lora_cnf/train_dit_lora.py \
  --cache-dir ./cache \
  --hy3d-repo /path/to/Hunyuan3D-2 \
  --hy3d-weights /path/to/hunyuan3d-2mini-weights \
  --out-dir ./checkpoints
```
