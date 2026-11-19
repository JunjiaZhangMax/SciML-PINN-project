# 实验清单

每次训练占一行。完成训练并核对 `runs/<运行名>/metrics.json` 后，把 `[ ]` 改成 `[x]`，在行末记下结果路径和一句观察。误差大也照实勾选；中断或报错则保留空框，在行末写明中断步数或原因。若结果保存在组员机器上，注明保存位置。

## 基线与试运行

- [x] 正问题 · `forward_baseline` · 已知系数 1、0.01，全时间窗，5,000 个内部 PDE 点，Adam 50,000 步。测试集横向速度相对误差 53.9%。本地结果：`runs/forward_baseline/metrics.json`。
- [ ] 反问题试运行 · `inverse_baseline` · N=5,000，噪声 0%，seed 0，在观测点计算 PDE 残差。已有历史摘要，待取得完整结果或在本机复现。[配置](../../configs/inverse_baseline.json) 的 purpose 为 pilot，单独记录。

- [ ] 短 Adam 预热试运行 · [inverse_adam5k_lbfgs50k_seed0](../../configs/inverse_adam5k_lbfgs50k_seed0.json) · N=5,000，噪声 0%，seed 0。5,000 步 Adam 后切换到 L-BFGS（最多 50,000 次），比较切换时与最终的系数、速度、压力误差和耗时。

## 正式反问题实验

N 表示时空速度观测总数。下列运行名与 `python scripts/export_configs.py` 导出的配置对应。计划使用完整时间窗、8×20 tanh、float64、Adam 50,000 步、学习率 0.001。正式运行前统一确认预算。每组查看两个系数及其相对误差、测试集速度和去均值压力误差。

### 参考实验：5,000 条观测，PDE 残差在观测点计算

- [ ] N=5000，噪声 0%，seed 0 · `reference_n5000_noise0_seed0`
- [ ] N=5000，噪声 0%，seed 1 · `reference_n5000_noise0_seed1`
- [ ] N=5000，噪声 0%，seed 2 · `reference_n5000_noise0_seed2`
- [ ] N=5000，噪声 1%，seed 0 · `reference_n5000_noise0.01_seed0`
- [ ] N=5000，噪声 1%，seed 1 · `reference_n5000_noise0.01_seed1`
- [ ] N=5000，噪声 1%，seed 2 · `reference_n5000_noise0.01_seed2`

### 敏感性实验：各组共用 5,000 个固定 PDE 配点

N=100 的四档噪声用于比较噪声影响；噪声 5% 的三档 N 用于比较稀疏性。两项比较共用 N=100、噪声 5% 的三次运行。

- [ ] N=100，噪声 0%，seed 0 · `sensitivity_n100_noise0_seed0`
- [ ] N=100，噪声 0%，seed 1 · `sensitivity_n100_noise0_seed1`
- [ ] N=100，噪声 0%，seed 2 · `sensitivity_n100_noise0_seed2`
- [ ] N=100，噪声 1%，seed 0 · `sensitivity_n100_noise0.01_seed0`
- [ ] N=100，噪声 1%，seed 1 · `sensitivity_n100_noise0.01_seed1`
- [ ] N=100，噪声 1%，seed 2 · `sensitivity_n100_noise0.01_seed2`
- [ ] N=100，噪声 5%，seed 0 · `sensitivity_n100_noise0.05_seed0`
- [ ] N=100，噪声 5%，seed 1 · `sensitivity_n100_noise0.05_seed1`
- [ ] N=100，噪声 5%，seed 2 · `sensitivity_n100_noise0.05_seed2`
- [ ] N=100，噪声 10%，seed 0 · `sensitivity_n100_noise0.1_seed0`
- [ ] N=100，噪声 10%，seed 1 · `sensitivity_n100_noise0.1_seed1`
- [ ] N=100，噪声 10%，seed 2 · `sensitivity_n100_noise0.1_seed2`
- [ ] N=50，噪声 5%，seed 0 · `sensitivity_n50_noise0.05_seed0`
- [ ] N=50，噪声 5%，seed 1 · `sensitivity_n50_noise0.05_seed1`
- [ ] N=50，噪声 5%，seed 2 · `sensitivity_n50_noise0.05_seed2`
- [ ] N=200，噪声 5%，seed 0 · `sensitivity_n200_noise0.05_seed0`
- [ ] N=200，噪声 5%，seed 1 · `sensitivity_n200_noise0.05_seed1`
- [ ] N=200，噪声 5%，seed 2 · `sensitivity_n200_noise0.05_seed2`

## 备选与新增

- [ ] 备选：短时间窗正问题 · 只训练到 t=5，与完整时间窗模型在相同的前 5 个时间单位、相同验证点上比较横向速度误差。执行前补齐样点和训练预算。
- 新增时复制：`- [ ] 研究问题 · 运行名 · 改动参数及取值 · seed · 训练预算 · 对照组 · 结果位置`
