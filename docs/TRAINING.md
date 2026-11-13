# 训练代码使用说明

在仓库根目录运行短训练：

```powershell
python -m sciml train --config configs/smoke_forward.json
python -m sciml train --config configs/smoke_inverse.json
```

两份 smoke 配置仅用于检查流程，不代表已经得到准确的科学结果。正式实验请复制配置、修改 name，并设置训练预算、采样数量和时间范围。每次运行使用新的 name，避免覆盖已有结果。

## 反问题试运行（识别 lambda1、lambda2）

新机器上先准备数据；已存在的文件只校验 SHA256，不会被覆盖：

```powershell
python -m sciml download
```

然后运行试运行配置。它与 reference_n5000_noise0_seed0 的抽样和模型设置相同，
但 purpose 为 pilot，不计入正式统计：

```powershell
python -m sciml train --config configs/pilot_inverse_n5000_noise0_seed0.json
python -m sciml plot --run runs/pilot_inverse_n5000_noise0_seed0
```

第二条命令读取 history.csv，在 output/figures/ 生成各项损失和两个系数随步数变化的曲线，
不重新训练或评估。训练为全量 Adam；该配置在 CPU 上约 0.36 秒/步，在 RTX 5070 Ti 上实测约 0.22 秒/步（50000 步 3.1 小时）。

### 第一次试运行的结果与第二次试运行

第一次试运行（50000 步恒定学习率 Adam，GPU 上约 0.22 秒/步，共 3.1 小时）没有收敛：
lambda1 误差 1.9%，lambda2 误差 26%，去均值压力误差 21%，损失结束时仍在下降。

第二次试运行从第一次的模型继续，不重头训练：先做 20000 步学习率从 1e-3 衰减到 1e-4 的 Adam，
再做最多 30000 次 L-BFGS 迭代。运行前 runs/ 下必须有第一次试运行的目录：

```powershell
python -m sciml train --config configs/pilot2_inverse_n5000_noise0_seed0.json
python -m sciml plot --run runs/pilot2_inverse_n5000_noise0_seed0
```

这组步数是估计值，尚未在真实数据上验证。建议的达标线：lambda1 误差低于 1%、lambda2 误差低于 10%，
且两条系数曲线走平；达标后再用同一设置跑正式实验。

## 三个可选训练设置

默认都不启用，旧配置的行为和检查点指纹不变。

- init_from：同一 runs 目录下另一个运行的名字。新运行加载它的模型和 Adam 状态作为起点，
  自己的步数从 0 计。两个运行必须使用同一数据文件、同一 mode、网络结构和时间范围。
  来源运行不会被修改；来源的名字、检查点 SHA256、步数和耗时记录在新运行的
  checkpoint.pt、status.json 和 metrics.json 的 init 字段里。metrics.json 的步数和耗时只算新运行自己的。
- adam_lr_final：Adam 学习率从 adam_lr（第 1 步）按等比方式衰减到该值（最后一步）。
  学习率只由步数决定，所以中断恢复后的结果与一次跑完逐位一致。
- use_lbfgs：为 true 时，Adam 结束后运行最多 lbfgs_steps 次 L-BFGS 迭代（强 Wolfe 线搜索，
  记忆长度 lbfgs_history）。每 log_every 次迭代记录并保存一次。达到收敛容差会提前结束，
  status.json 的 reason 会写明实际迭代数。只写 lbfgs_steps 而 use_lbfgs 为 false 时不会运行 L-BFGS：
  suite() 生成的配置和早期的配置文件带有 lbfgs_steps=10000，它们保持纯 Adam。

history.csv 中 L-BFGS 行的 stage 为 lbfgs，step 接在 Adam 步数之后；曲线图用虚线标出分界。
L-BFGS 阶段中断后 --resume 从最近一次保存的权重继续，但曲率记忆会重建，因此不保证与一次跑完逐位一致。

## 阅读 train.py

文件只包含一个数据容器和两个函数，建议先阅读 train()：

1. TrainingData：保存训练张量、抽样记录和设备。
2. prepare_training_data()：读取数据，根据正反问题准备训练输入，并转换为 float64 张量。
3. train()：准备数据和模型 → 直接调用 losses.py → Adam 更新 → 保存和评估。

mode=known 使用已知系数、初始速度和矩形窗口边界速度；mode=inverse 使用速度观测识别两个系数。压力真值只用于评估。

## 损失与优化

正问题调用 forward_objective()，反问题调用 inverse_objective()，权重来自配置。每步使用全部训练数据计算损失，再执行 total.backward() 和 adam.step()。不使用分块计算；全量计算的显存占用更高。L-BFGS 默认关闭，见上文 use_lbfgs。

## 日志与输出

log_every 控制 Adam 日志间隔。日志也记录初始状态和最后一步；损失在更新后的当前模型参数下计算。

runs/name/ 保存：

- config.json：实验设置。
- samples.npz：训练数据、采样索引和坐标范围，不包含压力标签。
- history.csv：各项损失、两个系数及耗时。
- checkpoint.pt：最近保存的模型、Adam 状态、训练步数、耗时及用于评估的文件指纹。
- metrics.json：独立测试集上的速度、压力和系数误差。
- predictions.npz：测试预测和三个时刻的完整空间快照。
- status.json：running/completed/failed/interrupted、原因、最近检查点的 Adam 与 L-BFGS 步数，以及 init 来源。

损失出现 NaN/Inf 时训练立即停止并抛出 FloatingPointError，status.json 记为 failed。
含非有限权重的模型不会写入 checkpoint.pt，因此最近一次正常的检查点仍可用于 --resume 或排查。

checkpoint_every 控制定期保存间隔，训练结束也会保存。中断后使用同一份配置和数据恢复：

```powershell
python -m sciml train --config configs/forward_full_adam_20261109.json --resume
```

恢复从最近检查点接续，检查点之后的未保存步骤需要重算。旧运行若没有 checkpoint.pt，无法恢复。当前网络无随机层，恢复会加载模型和 Adam 状态。physics_batch_size 不再使用；lbfgs_steps 和 lbfgs_history 只在 use_lbfgs 为 true 时生效。恢复时配置不能改动，包括 adam_steps；要在已完成的运行上继续训练，用新名字加 init_from。

## 单独评估

```powershell
python -m sciml evaluate --run runs/smoke_forward
```

评估使用保存的测试索引。压力预测和参考值在每个测试时刻分别减去空间均值，再计算相对误差。误差以小数保存，转换为百分比时乘以 100。
