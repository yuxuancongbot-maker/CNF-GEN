# 入口约定

核心逻辑在 `src/gcnf_shape_completion/`，本目录只有薄入口。

| 脚本 | 作用 |
|---|---|
| `train.py` | `preflight`（validation-only，无优化器步进、无 checkpoint）与 `train`（显式 `--run` 才步进）；外部 SDFusion/权重由 `--sdfusion-root`/`--weights-dir` 或环境变量配置 |
| `evaluate.py` | 六方法留出评测；接受 `--manifest`、`--checkpoint-dir`、`--out`、`--count`、`--ddim-steps`、`--seed` |
| `download_onet_chairs.py` | 数据侧：按固定清单下载官方归档成员（由数据工作项维护） |
| `prepare_onet_chairs.py` | 数据侧：生成 64³ xyz NPZ 与 manifest（由数据工作项维护） |

```bash
python tasks/shape_completion/scripts/train.py --help
python tasks/shape_completion/scripts/train.py preflight --manifest <list.json> --out <report.json>
python tasks/shape_completion/scripts/evaluate.py --help
```

脚本自行把 `src/` 加入模块搜索路径；已 `pip install -e tasks/shape_completion` 时不依赖这一行。不在脚本中嵌入机器绝对路径或旧工作区路径。完整命令见 [README](../README.md)。

## 固定椅子数据

从仓库根目录运行，归档和处理产物写入运行目录。Open3D 的 headless 环境需要可用的 `libEGL.so.1`。 [环境安装](../README.md#安装)

```bash
shape_task=tasks/shape_completion
shape_run=runs/shape_completion/onet-chair
python "$shape_task/scripts/download_onet_chairs.py" \
  --out "$shape_run/raw/03001627.zip"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python "$shape_task/scripts/prepare_onet_chairs.py" \
  --archive "$shape_run/raw/03001627.zip" \
  --out "$shape_run/data" --workers 8
```

脚本取得 Occupancy Networks 发布的归档中的水密 ShapeNet v1 椅子成员，按冻结的模型 ID 清单生成 1,024 条训练、64 条验证和 64 条测试记录。归档成员 SHA-256 为 `9d13d6ce86f308b5f13ba56cf534c300e9c279a5addfa1ea000d39f341652310`。 [数据发布说明](https://github.com/raphaelsulzer/dsrv-data#shapenetv1-13-class-subset-of-choy-et-al) · [固定 ID 清单](../configs/onet_chair_split.json)
