# 配置

generated/ 是 config.suite() 导出的 25 份 JSON 和 manifest.json，尚未运行。
顺序为 known -> reference -> sensitivity。字段见 sciml/config.py 和 docs/DESIGN.md。

可运行 `python -m sciml init-configs --output configs/my_protocol` 另导出一份。
正式实验前冻结配置；不能延长单个条件后将结果与不同预算的运行混在一起。
purpose 区分 production/pilot/smoke，后两种不纳入正式统计。

## 损失权重

Config 和各份 JSON 可设置 w_observation、w_initial、w_boundary、w_residual、w_gauge。
默认均为 1.0，必须有限且非负。0 表示该项不贡献到总损失，但当前仍计算该项，
所以所需的数据仍需提供。这些权重不等于待识别的 lambda1/lambda2。

独立调用损失函数时，需要将配置权重传入。train.py 已自动读取这些权重：

```python
from sciml.losses import forward_objective, inverse_objective

# 正问题：初始条件 + 边界条件 + PDE + 压力参考
losses = forward_objective(
    model,
    initial_coordinates, initial_velocity,
    boundary_coordinates, boundary_velocity,
    collocation_coordinates, gauge_coordinates,
    **config.forward_loss_weights(),
)

# 反问题：观测 + PDE；也可额外传入初始/边界数据
losses = inverse_objective(
    model,
    observation_coordinates, observed_velocity,
    collocation_coordinates,
    **config.inverse_loss_weights(),
)
```

各分项是未加权的原始损失，total 才是加权总和。
w_gauge 用于当前正问题入口；反问题入口暂未加入压力参考项。
旧 JSON 即使没有权重字段，加载时也会使用 1.0 默认值。
权重随配置保存，并计入配置指纹，便于追溯和比较实验。

## 数据准备与时间窗口

`time_end` 默认是 `null`（Python 中为 `None`），使用完整参考时间范围。
可以在 JSON 中添加 `"time_end": 1.0`，先用 t=0 到 t=1 的原始帧。
不做时间插值；归一化上界取最后一个实际选中的参考时刻。

```python
from sciml.config import Config
from sciml.train import prepare_training_data
from sciml.model import PINN
from sciml.losses import forward_objective

config = Config(mode="known", time_end=1.0, device="auto")
prepared = prepare_training_data(config)
print({key: tuple(value.shape) for key, value in prepared.loss_inputs.items()})

model = PINN(prepared.samples.lower, prepared.samples.upper, config).to(prepared.device)
losses = forward_objective(model, **prepared.loss_inputs, **config.forward_loss_weights())
```

prepare_training_data 只准备数据，不会启动训练。train 入口直接调用 losses.py，仅执行 Adam、
日志/检查点保存和最终测试评估。先运行 smoke_forward.json 或 smoke_inverse.json 验证流程，
完整说明见 docs/TRAINING.md。

此训练入口中 known 表示窗口正问题：初始速度取首帧全部空间点，边界速度取后续帧的
四条边，角点不重复，不把流动边界设为零。测试集仅取后续帧内部点，按时刻分层；
PDE 配点从其余内部时空记录中按 n_collocation 抽取，只使用坐标。
因此这个分支不使用 n_observations 或 collocation='observations' 的旧观测设置。
压力参考点固定在窗口中心，各选中时刻均设参考压力为零，不使用压力真值。

inverse 分支提供嵌套观测、配对噪声、固定测试集，按 collocation 选择观测配点或固定额外配点。
原始扁平索引和噪声记录保留在 prepared.samples，供之后的保存/评估实现使用。
若需固定系数但仍用散布观测做重建检查，可直接调用 data.prepare_data，而不是正问题准备函数。
