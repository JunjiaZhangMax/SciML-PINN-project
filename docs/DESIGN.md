# 接口与实验约定

本文件定义模块如何衔接，不提供算法实现。配置是研究起点，正式实验前冻结。

## 数据流

`Config + MAT -> WakeData -> PreparedData -> PINN + objective -> checkpoint/logs -> evaluate -> plotting`

- 原始数组：`X_star=(S,2)`、`t=(T,1)`、`U_star=(S,2,T)`、`p_star=(S,T)`。
- 内部坐标统一 `[x,y,t]`，网络输出 `[psi,p]`。
- 展平索引：`i=s*T+time_index`，必须保存索引以复现抽样。
- x=[1,8]、y=[-2,2]、t=[0,19.9]，是圆柱下游尾流窗口。
- data 模块返回 NumPy 数组；训练入口转换为同设备 float64 张量。
- 边界来自坐标元数据，不从测试标签估计。训练损失不接收压力标签。

## 抽样与加噪

- 按时刻分层划分测试点，默认每时刻 500 个位置，split_seed=2026。
- 观测数是时空记录数；相同 observation_seed 从同一训练池排列取前缀。
- `noisy=clean+noise*std(clean,axis=0,ddof=0)*epsilon`。
- 相同 noise_seed 共用二维基础噪声序列，不同 N 取前缀。
- 噪声对照固定观测坐标、初始化、基础噪声，只改变噪声比例。
- 稀疏对照使用嵌套的 50/100/200 条记录；各 N 分别计算 clean 标准差。
- reference 配点等于观测坐标；sensitivity 用固定的 5000 个训练池配点。
- 配点种子 2027，不随观测/初始化/噪声变化，不读取配点速度标签。
- 不同重复种子同时改变初始化、观测和噪声，代表整体重复性。

## 模型与物理

- 默认 8 个隐藏层、每层 20 个 tanh 单元，内部归一化输入到 [-1,1]。
- `u=psi_y`、`v=-psi_x`，连续性由流函数表示满足。
- `f=u_t+lambda1*(u*u_x+v*u_y)+p_x-lambda2*(u_xx+u_yy)`。
- `g=v_t+lambda1*(u*v_x+v*v_y)+p_y-lambda2*(v_xx+v_yy)`。
- lambda1 是对流项系数，lambda2 是黏性项系数，真值为 1 和 0.01。
- objective 返回可微张量，不 backward、不调用优化器、不写日志。
- 损失为 `mean(du²+dv²)+mean(f²+g²)`，两个 mean 分别按自身样本数归一化。
- PDE 涉及 psi 的三阶空间导数，评估速度也需要坐标梯度。

## 训练与恢复

- 默认 Adam 50000 步、lr=0.001，后接 L-BFGS 最多 10000 次迭代。
- L-BFGS closure 清梯度并重算损失，闭包调用数不等于优化器迭代数。
- 如实现 PDE 分块，按块样本占比累积梯度，保持原目标不变。
- 恢复时核验配置指纹、数据 SHA256、索引；保存优化器、阶段、步数、随机状态。
- 完成已知参数检查与首个无噪声反问题审阅后，再进入 sensitivity。
- 基线通过标准由小组在正式敏感性实验前确定并记录，骨架不暗定准确率门槛。
- smoke/pilot 与 production 分开保存，不混入正式统计；同组训练预算一致。
- 崩溃、中断、非物理系数、未收敛都留下记录，不静默过滤失败。

建议每次运行保存：

```text
runs/<config.name>/
  config.json       实际配置与 purpose
  provenance.json   数据哈希、代码版本/哈希、软件与设备信息
  samples.npz       观测/配点/测试索引、基础噪声、噪声标准差
  history.csv       阶段、步数、闭包调用数、损失、系数、耗时
  checkpoint.pt     网络、优化器、阶段进度、随机状态
  status.json       running/completed/failed/interrupted 与原因
  metrics.json      参数误差、场误差、耗时与评估约定
  predictions.npz   测试集预测与固定时刻快照
```

## 评估与图

- 参数相对误差为 `abs(lambda1-1)/1`、`abs(lambda2-0.01)/0.01`。
- u、v 分别计算测试集相对 L2。
- 压力按 time_index 分组，对预测与真值分别减去该时刻空间均值，再算整体相对 L2。
- 分母为零时显式记录 undefined 及原因，不能记为零误差。
- 快照预设 time_index=0,100,199，全域图与留出测试指标区别标注。
- 汇总显示均值、样本 SD、单次结果、实际 n 和失败数。
- 只汇总相同预算/协议的正式结果，不在统计时重训或重新抽样。
- 真值/预测共用色标，压力采用同一去均值规则，误差图单独注明色标。

## 25 次实验

- known：N=5000、0% 噪声、固定系数，1 次。
- reference：N=5000、0%/1%、种子 0/1/2，6 次。
- sensitivity：N=100 的 0/1/5/10% 加 N=50/200 的 5%，各三次，18 次。
- N=100、5% 在两组分析中共用，不重复训练。
