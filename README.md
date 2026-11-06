# SciML Final Project: Cylinder-Wake PINN

供小组自己实现的项目骨架。目标：通过连续时间 PINN 联合识别 Navier–Stokes
系数并重建速度/压力，研究速度观测噪声与稀疏性的影响。

**首次了解项目，建议先读 [从零理解项目：NS 方程、PINN 原理与实验设计](docs/PROJECT_GUIDE.md)。**

English: [Project guide — NS equations, PINN principles, and experiment design](docs/PROJECT_GUIDE.en.md).

**状态：数据准备、正反问题训练、日志、最终模型保存和训练后评估可用；快照与参考动画脚本可用。当前训练为简化版，不支持断点恢复。批量调度与结果汇总绘图仍待实现。短训练用于验证流程，不代表正式研究结论。**

训练说明见 [运行训练与检查结果](docs/TRAINING.md)。

## 目录与职责

```text
sciml/
  config.py       配置、JSON 读写、25 次实验定义（已实现）
  cli.py          参数解析、训练与评估入口
  data.py         数据加载、初边界准备、抽样与加噪
  model.py        网络与固定/待识别参数
  physics.py      自动微分、速度与 PDE 残差
  losses.py       正反问题各项损失
  train.py        Adam/L-BFGS、分块梯度、日志和恢复
  evaluate.py     固定测试集、压力去均值评估与预测导出
  experiments.py  分阶段批量实验与基线检查接口
  plotting.py     单次实验和统计汇总图接口
configs/          默认配置与 25 次实验清单
data/             原始数据与处理产物
notebooks/        探索性分析，正式逻辑放 sciml/
tests/            科学正确性验收占位
docs/             接口约定、实施清单、报告和展示骨架
runs/             将来每次训练的模型、索引、日志与指标
output/           将来汇总图和最终交付材料
```

## 使用

在仓库根目录执行。建议创建独立环境；已有合适的 PyTorch 环境也可直接使用。
按硬件选择 PyTorch 安装方式，普通 pip 安装不保证得到 CUDA 构建。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m sciml --help
python -m sciml plan
```

`configs/generated/` 已提供 25 份配置和顺序清单。另导出一份可用：

```powershell
python -m sciml init-configs --output configs/my_protocol
```

先用短训练配置验证运行环境：

```powershell
python -m sciml train --config configs/smoke_forward.json
python -m sciml evaluate --run runs/smoke_forward
```

下载命令、批量调度和汇总绘图仍是待实现接口：

```powershell
python -m sciml download
python -m sciml plot --run runs/known_n5000_noise0_seed0
python -m sciml batch --stage reference
python -m sciml plot --runs runs
```

相对路径以当前工作目录为基准，执行前应切换到仓库根目录。

## 建议实现顺序

1. `data.py`：读取并查看真值，确认轴顺序、单位和范围。
2. `model.py`、`physics.py`：网络、流函数速度、导数、动量残差。
3. `losses.py`、`train.py`：已知系数检查，然后无噪声反问题。
4. `evaluate.py`：固定测试集与逐时刻去均值压力误差。
5. `experiments.py`、`plotting.py`：基线核验后进行正式实验与统计。

详见 [接口约定](docs/DESIGN.md)、[实施清单](docs/ROADMAP.md)、
[报告骨架](docs/report_outline.md)、[展示骨架](docs/presentation_outline.md)。

## 检查

```powershell
python -m compileall -q sciml tests
python -m unittest discover -s tests -v
```

已实现的单元与集成测试覆盖数据、损失、训练恢复和评估。
test_acceptance.py 中七项早期占位仍标记 skipped，不能视为验证通过。

## 参考

- Raissi, Perdikaris & Karniadakis (2019), *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*.
- 方法说明：https://maziarraissi.github.io/PINNs/
- 参考代码与数据：https://github.com/maziarraissi/PINNs

这是自行编写的骨架，没有复制原仓库实现。以后借用代码或数据时请保留来源和适用许可。
