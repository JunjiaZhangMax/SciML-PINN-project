# 参考流场动画

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
