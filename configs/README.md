# 完整实验配置

基线按问题和实验用途命名：`forward_baseline`、`inverse_baseline`。JSON 文件名、`name` 和默认运行目录名一致。做对照实验时，按研究问题建目录，再用观测数、噪声和种子区分各组，例如 `runs/inverse_noise/n100_noise0.05_seed0/`。

- forward_baseline.json：已完成的正问题基线，结果见同名 runs 子目录。
- inverse_baseline.json：5,000 个无噪声观测、seed 0、50,000 步 Adam。训练完成后结果位于 `runs/inverse_baseline/`。
- [inverse_adam5k_lbfgs50k_seed0.json](inverse_adam5k_lbfgs50k_seed0.json)：相同观测设置，从头训练 5,000 步 Adam（学习率从 0.001 降至 0.0001），随后最多 50,000 次 L-BFGS 迭代。

自己设计反问题对照时，只需选一份基础配置并列出要变化的值。例如固定 100 条速度观测，比较三档噪声，每档重复三个种子：

```powershell
python scripts/run_inverse.py --observations 100 --noise 0 0.01 0.05 --seeds 0 1 2 --steps 2000 --runs runs/noise_pilot
```

默认仅预览，末尾加 `--run` 才逐组训练。列表取所有组合；每个种子在不同条件下配对抽样。脚本使用固定额外 PDE 点，数量、测试集、网络和学习率来自同一份基础配置（用 `--base` 指定）。2,000 步用于初筛。每组完整配置和指标随训练保存，全组结束后 `summary.csv` 汇总本次选中的各组。重复同一条命令会跳过已完成组、从已有检查点续训。不同预算请换一个 `--runs` 目录。
