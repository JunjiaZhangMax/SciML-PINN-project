# 如何选配置

- **验证环境：** `smoke_forward.json` 或 `smoke_inverse.json`，只跑短训练。
- **复现实验：** `forward_full_adam_20261109.json` 是已完成的全时段正问题配置；11 月 8 日的配置用于追溯旧运行。
- **开始反问题：** `pilot_inverse_n5000_noise0_seed0.json` 使用 5,000 条无噪声速度观测，Adam 50,000 步。pilot 表示不计入正式统计，并非短训练。

新实验可以复制一份配置，换一个 `name`。正问题可改 `time_end` 缩短时间窗口；反问题可改 `n_observations` 和 `noise` 比较数据量和噪声影响。训练时记录的损失分项是未加权值，只有 total 是加权总和。

正式实验的 25 组组合由 [suite()](../sciml/config.py) 定义。在仓库根目录执行 `python scripts/export_configs.py` 可一次生成配置和顺序清单到 `configs/generated/`，不会启动训练，也不会覆盖已有配置。
