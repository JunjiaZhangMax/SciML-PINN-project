# 如何选配置

- **验证环境：** `smoke_forward.json` 或 `smoke_inverse.json`，只跑短训练。
- **复现实验：** `forward_full_adam_20261109.json` 是已完成的全时段正问题配置；11 月 8 日的配置用于追溯旧运行。
- **开始反问题：** `pilot_inverse_n5000_noise0_seed0.json` 使用 5,000 条无噪声速度观测，Adam 50,000 步。pilot 表示不计入正式统计，并非短训练。
- **从已有模型继续做优化实验：** `pilot2_inverse_n5000_noise0_seed0.json` 需要上一项 pilot 的检查点，在新运行中做 20,000 步变学习率 Adam，再执行 L-BFGS。

新实验可以复制一份配置，换一个 `name`。正问题可改 `time_end` 缩短时间窗口；反问题可改 `n_observations` 和 `noise` 比较数据量和噪声影响。训练时记录的损失分项是未加权值，只有 total 是加权总和。

正式实验的 25 组组合由 [suite()](../sciml/config.py) 定义。在仓库根目录执行 `python scripts/export_configs.py` 可一次生成配置和顺序清单到 `configs/generated/`，不会启动训练，也不会覆盖已有配置。

自己设计反问题对照时，只需选一份基础配置并列出要变化的值。例如固定 100 条速度观测，比较三档噪声，每档重复三个种子：

```powershell
python scripts/run_inverse.py --observations 100 --noise 0 0.01 0.05 --seeds 0 1 2 --steps 2000 --runs runs/noise_pilot
```

默认仅预览，末尾加 `--run` 才逐组训练。列表取所有组合；每个种子在不同条件下配对抽样。脚本使用固定额外 PDE 点，数量、测试集、网络和学习率来自同一份基础配置（用 `--base` 指定）。2,000 步用于初筛。每组完整配置和指标随训练保存，全组结束后 `summary.csv` 汇总本次选中的各组。重复同一条命令会跳过已完成组、从已有检查点续训。不同预算请换一个 `--runs` 目录。
