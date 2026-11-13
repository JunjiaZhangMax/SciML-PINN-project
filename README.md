# SciML：圆柱尾流 PINN

给定圆柱下游的一部分速度观测，我们希望恢复速度、压力，并识别 Navier–Stokes 方程的对流与黏性系数。正问题固定系数，用初始和边界速度检验实现；反问题从散布的速度观测学习系数。

**第一次看项目？** 先读[中文原理说明](docs/PROJECT_GUIDE.md)或[英文版](docs/PROJECT_GUIDE.en.md)。想看效果，下载后用浏览器打开[预测与真值动画](output/forward_comparison.html)。

## 按任务找入口

- **运行单个实验**：在 [configs/](configs/README.md) 选配置，按[训练说明](docs/TRAINING.md)启动、恢复或评估。
- **比较反问题配置**：用 [run_inverse.py](scripts/run_inverse.py) 指定观测数、噪声和种子，预览组合后加 `--run` 执行。各组结果和汇总表保存在指定的 `runs/` 子目录。
- **查看数据与结果**：[data/](data/README.md) 存参考数据，`runs/` 存本地训练日志、模型与指标，[output/](output/) 存展示文件。[脚本索引](scripts/README.md) 说明如何生成快照和动画。
- **准备课程提交**：使用[报告提纲](docs/deliverables/report_outline.md)和[展示提纲](docs/deliverables/presentation_outline.md)。[原始实验约定](docs/planning/DESIGN.md)与[实施清单](docs/planning/ROADMAP.md)用于回顾小组计划。

## 跑通一次

在仓库根目录、已安装适合本机的 PyTorch 的 Python 环境中执行：

```powershell
python -m pip install -e .
python -m sciml download
python -m sciml train --config configs/smoke_inverse.json
python -m sciml plot --run runs/smoke_inverse
```

短训练用于验证流程。训练结束会自动评估，指标在 `runs/<name>/metrics.json`。如需重新评估，运行 `python -m sciml evaluate --run runs/<name>`。恢复中断的训练时，用原配置运行 train 并加上 `--resume`。

## 从哪里读代码

1. [data.py](sciml/data.py)：读入参考数据，准备初始条件、边界或散布观测。
2. [model.py](sciml/model.py) → [physics.py](sciml/physics.py) → [losses.py](sciml/losses.py)：从坐标得到流函数与压力，通过求导得到速度，再计算数据误差和方程残差。
3. [train.py](sciml/train.py) → [evaluate.py](sciml/evaluate.py)：Adam 更新参数、保存检查点，再用独立测试点计算误差。

[训练与输出说明](docs/TRAINING.md) · [选择实验配置](configs/README.md) · [生成快照与动画](scripts/README.md) · [原作者的代码与数据](https://github.com/maziarraissi/PINNs)

运行测试：`python -m unittest discover -s tests`。
