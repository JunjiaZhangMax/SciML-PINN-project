# 实验与可视化脚本

从仓库根目录运行。核心训练使用 `python -m sciml train`，以下脚本处理实验组合和展示。

- [run_inverse.py](run_inverse.py) 按观测数、噪声和种子组合运行反问题，具体例子见[配置说明](../configs/README.md)。
- [export_configs.py](export_configs.py) 导出课程计划中的 25 组配置。
- [testing_data_plot.py](testing_data_plot.py) 查看指定时刻的参考速度和压力快照。
- [diagnose_forward.py](diagnose_forward.py) 分析已有正问题结果中的入口误差和尾流衰减。
- [animate_comparison.py](animate_comparison.py) 并排播放预测与真值，[animate_reference.py](animate_reference.py) 播放参考场。

## 训练结果与真值并排动画

```powershell
python scripts/animate_comparison.py --run runs/forward_full_adam_20261109
```

生成 `output/forward_comparison.html`，左侧为最终 PINN 预测、右侧为参考值，三行分别为 u、v、p。
两侧同步播放全部 200 个时刻；每个变量使用跨时刻、跨预测和真值统一的色标。
压力逐时刻分别去全空间均值。原始规则网格直接显示，不进行时间插值。
这是最终模型随物理时间变化的动画，不是模型训练过程动画。

同时保存静态预览 `.png`、来源记录 `.source.json`，以及运行目录中的全时空数值 `animation_fields.npz`。
无 GPU 时可加 `--device cpu`。用 `--stride 2` 可减少输出帧数。

## 仅真值动画

脚本使用同一个 `cylinder_nektar_wake.mat`，不涉及 PINN 训练。
输出同步变化的 u、v、逐时刻去空间均值的 p；色标在整个时间范围内固定。
HTML 自带播放、暂停、进度滑块、逐帧和调速按钮，可离线打开。

在项目根目录执行：

```powershell
python scripts/animate_reference.py
```

默认生成 `output/reference_animation.html`，用浏览器打开，点击 ▶ 播放。
使用全部 200 帧，默认 10 fps，时间从 0 到 19.9；空间通过原坐标三角剖分插值显示。

```powershell
# 调快播放，不改变原数据
python scripts/animate_reference.py --fps 20

# 每两帧取一帧，可减小输出；实际时间写在每帧标题中
python scripts/animate_reference.py --stride 2 --output output/reference_light.html

# 导出 GIF（使用 Pillow，无需 FFmpeg）
python scripts/animate_reference.py --output output/reference_animation.gif

# 导出 MP4（需先单独安装并配置 FFmpeg）
python scripts/animate_reference.py --output output/reference_animation.mp4
```

Notebook 内显示已生成的 HTML（从 notebooks/ 目录运行时）：

```python
from IPython.display import IFrame, display
display(IFrame('../output/reference_animation.html', width='100%', height=1100))
```

每个动画同时保存 `.source.json`，记录数据哈希、时间索引、色标范围和显示变换。
不同场的数值含义不同，各有自己的固定色标；颜色不能跨场直接比较数值大小。
