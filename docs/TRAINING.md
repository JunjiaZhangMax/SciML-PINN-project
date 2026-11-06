# 训练代码使用说明

在仓库根目录运行短训练：

```powershell
python -m sciml train --config configs/smoke_forward.json
python -m sciml train --config configs/smoke_inverse.json
```

两份 smoke 配置仅用于检查流程，不代表已经得到准确的科学结果。正式实验请复制配置、修改 name，并设置训练预算、采样数量和时间范围。每次运行使用新的 name，避免覆盖已有结果。

## 阅读 train.py

文件只包含一个数据容器和两个函数，建议先阅读 train()：

1. TrainingData：保存训练张量、抽样记录和设备。
2. prepare_training_data()：读取数据，根据正反问题准备训练输入，并转换为 float64 张量。
3. train()：准备数据和模型 → 直接调用 losses.py → Adam 更新 → 保存和评估。

mode=known 使用已知系数、初始速度和矩形窗口边界速度；mode=inverse 使用速度观测识别两个系数。压力真值只用于评估。

## 损失与优化

正问题调用 forward_objective()，反问题调用 inverse_objective()，权重来自配置。每步使用全部训练数据计算损失，再执行 total.backward() 和 adam.step()。不再使用分块计算和 L-BFGS；全量计算的显存占用更高。

## 日志与输出

log_every 控制 Adam 日志间隔。日志也记录初始状态和最后一步；损失在更新后的当前模型参数下计算。

runs/name/ 保存：

- config.json：实验设置。
- samples.npz：训练数据、采样索引和坐标范围，不包含压力标签。
- history.csv：各项损失、两个系数及耗时。
- checkpoint.pt：最终模型、训练步数、耗时及用于评估的文件指纹。
- metrics.json：独立测试集上的速度、压力和系数误差。
- predictions.npz：测试预测和三个时刻的完整空间快照。

简化版仅在训练结束时保存模型，不支持断点恢复；不再生成 status.json 和 provenance.json。旧配置中的 lbfgs_steps、lbfgs_history、physics_batch_size 和 checkpoint_every 为兼容保留，当前训练不使用它们。实际步数由 adam_steps 决定，学习率由 adam_lr 决定。

## 单独评估

```powershell
python -m sciml evaluate --run runs/smoke_forward
```

评估使用保存的测试索引。压力预测和参考值在每个测试时刻分别减去空间均值，再计算相对误差。误差以小数保存，转换为百分比时乘以 100。
