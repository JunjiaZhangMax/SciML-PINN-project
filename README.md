# SciML：圆柱尾流 PINN

给定圆柱下游的一部分速度观测，我们希望恢复速度、压力，并识别 Navier–Stokes 方程的对流与黏性系数。正问题固定系数，用初始和边界速度检验实现；反问题从散布的速度观测学习系数。

**第一次看项目？** 先读[中文原理说明](docs/PROJECT_GUIDE.md)或[英文版](docs/PROJECT_GUIDE.en.md)。想看效果，下载后用浏览器打开[预测与真值动画](output/forward_comparison.html)。

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
