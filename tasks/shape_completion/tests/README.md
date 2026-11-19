# 离线测试

默认测试离线、确定性、无 GPU，不下载数据、不要求官方权重：

```bash
python -m unittest discover -s tasks/shape_completion/tests -v
```

- `test_cpu.py`：各向异性轴约定、三次样条解析导数与 autograd、掩码降采样/腐蚀、掩码噪声共享、严格数据契约、FP64 CNF 精确值/导数求解与 batched 列右端项、梯度路径。
- `test_seeded_sampling.py`：六方法固定表面采样偏移、CNF 与 plain_post 配对采样偏移一致、显式 seed 的可复现性与全局 RNG 不被读写。

测试用 `src/` 相对路径导入包，不依赖旧工作区 `PYTHONPATH`。真实数据与模型实验走 `scripts/train.py` 与 `scripts/evaluate.py`，不进入默认测试。
