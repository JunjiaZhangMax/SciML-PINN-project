# FlyWheel 初学者代码导览

阅读日期：2026-11-16。面向已有 PyTorch 模型实现经验、正在入门 TPU kernel 编程的读者。

代码基准：读取时 main 指向提交 [2ba7d19](https://github.com/heavyball-research/flywheel-tpu/commit/2ba7d193fb4c4d46940d86927f85047c67665ea0)，提交时间为 2026-11-14；以下源码链接均固定到该提交。本文基于 README、完整文件树、关键源码和 JAX 官方 TPU/Pallas 文档的静态阅读；没有运行内核、测试或性能基准，没有发送邮件或修改 GitHub。学习顺序与任务安排是我的建议。

## 1. 项目解决什么问题

FlyWheel 把 attention 的计算、数据搬运和状态更新组织成适合 TPU 的低层算子。它使用 JAX Pallas，覆盖 softmax attention 的 MHA、MQA/GQA，以及混合模型中的 GDN/KDA 线性 attention；目标场景包括读入提示词的 prefill、逐步生成的 decode 和部分 softmax attention 训练。README 说明面向 TPU v6e、v7x。依据：[README.md:19–70](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/README.md#L19-L70)。

从熟悉的数学出发，softmax attention 仍然计算 `softmax(QKᵀ / √d)V`；优化空间在于如何分块、避免保存完整 attention 矩阵、复用数据，并让搬运与计算交叠。朴素数学版本可见 [tests/softmax_attention/util.py:181–273](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/tests/softmax_attention/util.py#L181-L273)；分块实现可见 [flywheel_tpu/softmax_attention/fwd_blocks.py:633–699](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/fwd_blocks.py#L633-L699)、[flywheel_tpu/softmax_attention/fwd_blocks.py:812–939](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/fwd_blocks.py#L812-L939)，流水线的连接可见 [flywheel_tpu/softmax_attention/kernels/fwd_dense.py:321–436](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/kernels/fwd_dense.py#L321-L436)。这是从源码得到的实现理解。

对 PyTorch 使用者最重要的接口区别是：flash-attn 的名字和参数习惯被沿用，但输入输出是 JAX 数组；dense prefill 默认采用 `[B,H,T,D]`，KV-cache 接口的 query 采用 `[B,T,H,D]`。常规输入要求 bf16；paged cache 另有 fp8 支持。dropout、ALiBi 等兼容参数的非默认值会被拒绝。依据：[docs/api.md:5–43](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/docs/api.md#L5-L43)、[flywheel_tpu/softmax_attention/decode.py:218–255](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/decode.py#L218-L255)。

README 的加速比是特定硬件、形状和对照实现下的项目报告。本文不把它当作自己复现的结果，也不推断为所有模型的端到端加速。benchmark 文档明确区分包含投影的 block 时间与 profiler 提取的 kernel 时间：[docs/benchmarks.md:3–40](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/docs/benchmarks.md#L3-L40)。

## 2. 目录怎样读

以下为按入门需要裁剪的目录树，非逐文件清单：

```text
flywheel_tpu/
  __init__.py
  core/                       TPU 常量与分块小工具
  softmax_attention/
    prefill.py, decode.py      上层入口
    dispatch.py, config.py     内核选择与分块配置
    schedule.py, variant.py    遍历范围、mask 与计算变体
    fwd_blocks.py, dma.py      计算片段与搬运辅助
    autograd.py               自定义反向规则
    kernels/                  dense / varlen / paged / decode / MLA / backward
  linear_attention/
    gdn/, kda/                融合卷积与递推状态更新
tests/                        正确性与接口行为
benchmarks/                   性能实验和对照实现
docs/                         API、性能与端到端实验说明
eval/                         完整模型任务评估
third_party/, patches/        外部实现与本仓库的集成补丁
scripts/                      MLIR 检查、性能门禁等工具
```

- **先看公开入口。** 顶层只把三个 softmax API 列为公开导出；线性 attention 从自己的子包导入。临时兼容的内部导出带警告，不宜据此设计新调用。依据：[flywheel_tpu/__init__.py:20–60](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/__init__.py#L20-L60)、[docs/api.md:414–431](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/docs/api.md#L414-L431)。
- **core 是理解硬件约束的短入口。** `tpu.py` 定义 128 lanes、8 sublanes，并解释 bf16 动态窗口的对齐；`tiling.py` 只有向上取整工具。依据：[flywheel_tpu/core/tpu.py:16–27](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/core/tpu.py#L16-L27)、[flywheel_tpu/core/tiling.py:15–20](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/core/tiling.py#L15-L20)。
- **tests 可当作可执行规格阅读。** softmax 的朴素 reference 把公式、GQA head 复制、mask 和 dtype 写清楚，主测试比较前向、LSE 和梯度；线性 attention 也有逐 token 的递推 reference。依据：[tests/softmax_attention/util.py:181–273](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/tests/softmax_attention/util.py#L181-L273)、[tests/softmax_attention/test_prefill.py:240–334](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/tests/softmax_attention/test_prefill.py#L240-L334)、[tests/linear_attention/reference.py:100–157](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/tests/linear_attention/reference.py#L100-L157)。
- **benchmarks 与 eval 分开。** 前者测 attention block/kernel；后者通过 vLLM 与 tpu-inference 跑完整模型任务。该仓库的端到端复现路径使用固定子模块加补丁，不等同于只安装内核包。依据：[docs/benchmarks.md:16–40](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/docs/benchmarks.md#L16-L40)、[docs/e2e_evaluation.md:24–60](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/docs/e2e_evaluation.md#L24-L60)、[justfile:1–17](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/justfile#L1-L17)。
- **第三方源码和辅助脚本放在后面。** 对照代码在 [benchmarks/](https://github.com/heavyball-research/flywheel-tpu/tree/2ba7d193fb4c4d46940d86927f85047c67665ea0/benchmarks) 和 [third_party/](https://github.com/heavyball-research/flywheel-tpu/tree/2ba7d193fb4c4d46940d86927f85047c67665ea0/third_party)，编译检查、性能门禁入口在 [scripts/](https://github.com/heavyball-research/flywheel-tpu/tree/2ba7d193fb4c4d46940d86927f85047c67665ea0/scripts)。初学阶段先读本仓库 API→kernel→reference，暂时无需初始化完整模型评估环境；这是基于上述依赖关系的学习建议。

## 3. 最关键的一条调用链：dense causal prefill

推荐第一次只跟踪：`B=1, Hq=4, Hkv=2, Tq=Tk=128, D=128, causal=True`，其余选项保持默认。这保留 GQA 的 head 映射，又避开变长、分页与 RoPE 分支；形状约束来自 [flywheel_tpu/softmax_attention/prefill.py:87–125](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/prefill.py#L87-L125)。可先从 README Quick Start 理解用户侧调用：[README.md:106–146](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/README.md#L106-L146)。

**① API 整理输入：prefill.py → flash_attn_func。**

它检查兼容参数与输入布局，解析 Q/K/V 的维度，把序列长度补齐，设置默认 `1/√d` 缩放和 mask 变体。causal 对齐使用原始长度的 `Tk-Tq`，随后把 batch 与 head 折叠成 kernel 所用的轴；计算结束再恢复形状并裁掉 padding。依据：[flywheel_tpu/softmax_attention/prefill.py:127–241](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/prefill.py#L127-L241)、[flywheel_tpu/softmax_attention/prefill.py:242–341](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/prefill.py#L242-L341)。

这里的 padding 具有语义约束：未设 causal、右窗口边界也不是 0 时，dense 路径不能隐藏补齐后的 KV 尾部，因此非 128 整数倍的 Tk 会报错，代码建议改用 varlen 接口。依据：[flywheel_tpu/softmax_attention/prefill.py:176–191](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/prefill.py#L176-L191)。不能把“所有长度自动补零”理解成“所有 mask 都安全”。

**② 调度与配置：dispatch.py → get_kernel / make_flash_attn_mha。**

`get_kernel` 在适用条件下查调优表；未命中时用默认分块规则。它选择搬运块 `block_q/block_kv`、计算块 `block_q_compute/block_kv_compute`、stage 数、物理布局和 head folding，然后建立内核。依据：[flywheel_tpu/softmax_attention/dispatch.py:1120–1263](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/dispatch.py#L1120-L1263)。所以源码里同时出现“大块”和“小块”并不重复：前者决定 staging，后者决定一轮计算的片段；字段定义见 [flywheel_tpu/softmax_attention/config.py:211–238](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/config.py#L211-L238)。

`make_flash_attn_mha` 把配置组织成 FwdTraits、Variant 和 schedule，再选择 dense 或 varlen forward。schedule 计算某个 Q block 需要访问的 KV block 区间，并标记边界处是否需要 mask。依据：[flywheel_tpu/softmax_attention/dispatch.py:1068–1116](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/dispatch.py#L1068-L1116)、[flywheel_tpu/softmax_attention/schedule.py:190–217](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/schedule.py#L190-L217)、[flywheel_tpu/softmax_attention/schedule.py:275–311](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/schedule.py#L275-L311)。

**③ Pallas 调用边界：kernels/fwd_dense.py。**

`flash_attn_forward` 进入 `dense_pallas_call`。后者检查可整除性、声明输入输出的 memory space 和形状、分配 scratch，然后调用 `pl.pallas_call`；`interpret=True` 被转成 TPU 的 InterpretParams。依据：[flywheel_tpu/softmax_attention/kernels/fwd_dense.py:1023–1058](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/kernels/fwd_dense.py#L1023-L1058)、[flywheel_tpu/softmax_attention/kernels/fwd_dense.py:916–982](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/kernels/fwd_dense.py#L916-L982)。

Scratch 中包括 VMEM 的 Q/K/V staging、float32 累加器、row max/row sum、SMEM 标记及 DMA semaphore。这一段把“张量公式”具体落到了“哪些数据留在片上、需要多少缓冲”。依据：[flywheel_tpu/softmax_attention/kernels/fwd_dense.py:797–879](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/kernels/fwd_dense.py#L797-L879)。

**④ Kernel 内部：搬运 → QK → mask/softmax → PV → 写回。**

`dense_fwd_body` 中的 block step 等待当前块、预取后续块，并把 `qk`、`online_softmax` 与 `fragment_pv` 连接起来。依据：[flywheel_tpu/softmax_attention/kernels/fwd_dense.py:321–436](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/kernels/fwd_dense.py#L321-L436)。`dma.advance` 则是可独立阅读的 ring-buffer 例子：等待所需块，再往下一个 slot 预取后继块；不要误认为所有 dense 搬运都必须经由这个 helper。依据：[flywheel_tpu/softmax_attention/dma.py:28–67](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/dma.py#L28-L67)。

`qk` 用 `lax.dot_general` 产生 float32 logits；`online_softmax` 维护每行统计并产生当前片段的概率，PV 消费片段而无需先存一张完整 T×T attention 矩阵。依据：[flywheel_tpu/softmax_attention/fwd_blocks.py:633–699](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/fwd_blocks.py#L633-L699)、[flywheel_tpu/softmax_attention/fwd_blocks.py:812–939](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/fwd_blocks.py#L812-L939)、[flywheel_tpu/softmax_attention/kernels/fwd_dense.py:378–436](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/kernels/fwd_dense.py#L378-L436)。

需要特别留意该版本的数值实现：正常路径使用首片段选定的 static anchor，检测到风险时才用 recovery/replay 重新定标；logits 以 log2 单位进入 exp2。阅读时应按代码理解，不能直接套成“每块都更新最大值”的简化叙述。依据：[flywheel_tpu/softmax_attention/fwd_blocks.py:882–938](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/fwd_blocks.py#L882-L938)、[flywheel_tpu/softmax_attention/softmax.py:25–58](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/softmax.py#L25-L58)。

**⑤ 训练分支：autograd.py。**

默认支持的 dense 前向通过 `jax.custom_vjp` 接入 backward；反向需要时才保存输出与 LSE，再进入 Pallas backward kernel。局部窗口、softcap、RoPE 等前向能力并不意味着有对应梯度；GDN/KDA 在文档中也明确为 inference-only。依据：[flywheel_tpu/softmax_attention/autograd.py:27–72](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/autograd.py#L27-L72)、[flywheel_tpu/softmax_attention/prefill.py:124–125](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/prefill.py#L124-L125)、[docs/api.md:426–431](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/docs/api.md#L426-L431)。

## 4. 另外两条路径如何协作

**KV cache / decode。** `decode.py::flash_attn_with_kvcache` 接受历史 K/V 和可选新 token。连续 cache 的形状为 `[B,capacity,Hkv,D]`；分页路径使用合并的 K/V page pool 与 block table。普通单 token decode、短 verify window 与长/packed query 会走不同分支，后者可进入 paged varlen prefill。依据：[flywheel_tpu/softmax_attention/decode.py:213–284](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/decode.py#L213-L284)、[flywheel_tpu/softmax_attention/decode.py:469–596](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/decode.py#L469-L596)。

这解释了 prefill 和 decode 能共享变体、配置与片段计算的设计，同时保留不同内核调度。cache 会被 donation 并更新，调用后必须接住返回的新 cache，不能继续拿旧缓冲当快照。依据：[flywheel_tpu/softmax_attention/decode.py:280–284](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/decode.py#L280-L284)、[flywheel_tpu/softmax_attention/decode.py:597–625](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/decode.py#L597-L625)。

**GDN/KDA。** 两者是 softmax attention 之外的另一族算子：把 packed QKV 上的 causal depthwise conv1d、SiLU 与 delta-rule 状态更新融合。它们管理的是 convolution state 和 recurrent state slot cache。依据：[docs/api.md:414–431](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/docs/api.md#L414-L431)。

GDN 子目录中，`compute_conv1d.py` 实现卷积，`compute_gdn.py` 实现递推数学，`memory_ref.py` 组织权重/metadata 引用及 HBM↔VMEM 搬运，`vmem_ldst.py` 从原生布局中取出各 head 的激活。依据：[flywheel_tpu/linear_attention/gdn/compute_conv1d.py:15–38](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/linear_attention/gdn/compute_conv1d.py#L15-L38)、[flywheel_tpu/linear_attention/gdn/compute_gdn.py:888–942](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/linear_attention/gdn/compute_gdn.py#L888-L942)、[flywheel_tpu/linear_attention/gdn/memory_ref.py:15–39](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/linear_attention/gdn/memory_ref.py#L15-L39)、[flywheel_tpu/linear_attention/gdn/vmem_ldst.py:15–34](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/linear_attention/gdn/vmem_ldst.py#L15-L34)。

先看 GDN wrapper 的参数与状态形状，再看 metadata 怎样映射 request/token/slot；`GDNMode.BATCHED` 将多个短请求装进一个 tile，`PER_SEQ` 让一个长请求跨多个 tile。依据：[flywheel_tpu/linear_attention/gdn/wrapper.py:1087–1179](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/linear_attention/gdn/wrapper.py#L1087-L1179)、[flywheel_tpu/linear_attention/gdn/metadata.py:24–81](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/linear_attention/gdn/metadata.py#L24-L81)、[flywheel_tpu/linear_attention/gdn/config.py:62–71](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/linear_attention/gdn/config.py#L62-L71)。实际计算顺序在 [flywheel_tpu/linear_attention/gdn/wrapper.py:414–475](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/linear_attention/gdn/wrapper.py#L414-L475)：卷积和 SiLU → 拆出激活 → chunked_gdn → 输出与状态存储。

若数学不熟，先读 [tests/linear_attention/reference.py:100–157](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/tests/linear_attention/reference.py#L100-L157) 的逐 token 状态更新，再读优化后的 chunked 版本。KDA 的关键区别之一是每个 key channel 都有 decay gate，状态按行衰减；代码直接说明于 [flywheel_tpu/linear_attention/kda/wrapper.py:1112–1127](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/linear_attention/kda/wrapper.py#L1112-L1127)。我的建议是先打通 softmax 路径，再进入这套状态与分块数学。

## 5. 需要补齐的 TPU / AI hardware programming 知识

按“能帮助读懂哪段代码”的顺序补：

1. **JAX 的执行模型。** 学会 jit 的 tracing/static 参数、显式随机 key、grad/vjp，以及异步执行与 buffer donation。项目中的落点是 [flywheel_tpu/softmax_attention/autograd.py:42–71](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/autograd.py#L42-L71)、[benchmarks/softmax_attention/attention_block.py:196–208](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/benchmarks/softmax_attention/attention_block.py#L196-L208) 和 [flywheel_tpu/softmax_attention/decode.py:280–284](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/decode.py#L280-L284)。先做到能解释“为什么形状/配置变化可能触发编译，为什么计时要等待设备完成”。可从 [JAX JIT 官方文档](https://docs.jax.dev/en/latest/jit-compilation.html) 与下述项目计时代码配合阅读。

2. **HBM、VMEM、SMEM 与 MXU/VPU。** HBM 保存大数组；计算需要把数据搬到片上存储和寄存器，矩阵乘与向量运算由不同单元承担。对应项目的 VMEM scratch、SMEM 元数据和 DMA semaphore：[flywheel_tpu/softmax_attention/kernels/fwd_dense.py:824–876](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/kernels/fwd_dense.py#L824-L876)。官方入门：[TPU Pipelining 的内存与计算单元说明](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html#tpu-and-its-memory-spaces)。

3. **Pallas Ref、BlockSpec 与显式内存操作。** 理解 Ref 代表可读写缓冲，BlockSpec 描述输入输出窗口，kernel 怎样分配 scratch、拷贝并同步。官方：[Grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)。回看 [flywheel_tpu/softmax_attention/kernels/fwd_dense.py:942–982](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/kernels/fwd_dense.py#L942-L982) 时要注意：这个 dense 实现把大数组声明为 ANY，并在 kernel 内管理遍历与搬运；不能假定每个 FlyWheel 分块都由 pallas_call 的外层 grid 自动生成。

4. **Layout、对齐和分块资源预算。** 理解最后两轴、lane/sublane 排布、bf16 打包，及扩大块尺寸带来的复用收益与 VMEM 开销。官方：[Writing TPU kernels with Pallas](https://docs.jax.dev/en/latest/pallas/tpu/details.html#array-layouts)。项目中的具体证据是 [flywheel_tpu/core/tpu.py:18–25](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/core/tpu.py#L18-L25)、[flywheel_tpu/softmax_attention/config.py:42–77](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/config.py#L42-L77)。8×128 是理解 float32 向量布局的起点；不能把所有 dtype、硬件代际和 HBM/VMEM 窗口都当成同一种 tile。

5. **Software pipeline 与同步正确性。** 学会多缓冲、预取、start/wait、slot 生命周期，才能判断某个 slot 是否被过早覆盖。[flywheel_tpu/softmax_attention/config.py:80–94](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/config.py#L80-L94) 明确说明 stage 至少为 2 的正确性原因；[flywheel_tpu/softmax_attention/dma.py:39–67](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/dma.py#L39-L67) 给出搬运流不变量。CPU interpret 能帮助验证小例子；compiled race 测试在解释模式下会跳过，见 [tests/softmax_attention/test_prefill.py:827–850](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/tests/softmax_attention/test_prefill.py#L827-L850)。

6. **数值与性能实验。** 学习 bf16 输入/float32 累加、softmax 定标、误差容忍，以及算术强度、带宽瓶颈、预热与同步计时。官方 [Pallas 矩阵乘教程](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html#matrix-multiplication-performance) 把 FLOPs、搬运量和分块联系起来。项目用 float32 reference 与重排运算的 bf16 reference 衡量误差：[tests/softmax_attention/test_prefill.py:240–283](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/tests/softmax_attention/test_prefill.py#L240-L283)；计时则单独记录首次调用、预热后多轮取中位数：[benchmarks/common/timing.py:40–74](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/benchmarks/common/timing.py#L40-L74)。

上述知识足以支持第一轮阅读；多芯片通信、MLA、fp8、ReplaySSM 可以等到需要处理相应路径时再深入。这是学习范围建议，依据是这些能力各有独立入口，而最小 dense 路径已经可以单独运行：[flywheel_tpu/softmax_attention/kernels/](https://github.com/heavyball-research/flywheel-tpu/tree/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/kernels)、[README.md:106–146](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/README.md#L106-L146)。

## 6. 三个可执行的第一步任务

这些是下一步建议，尚未执行或改动源码。预计耗时仅为安排学习的参考。

### 任务一：做一个 128-token GQA 正确性小实验（约 1–2 小时，CPU 可做）

- **准备：** 在独立源码副本中使用 Python ≥3.12、JAX ≥0.11.2；按 README 在非 TPU 环境执行 `uv sync`，TPU VM 才加 `--extra tpu`。版本依据：[pyproject.toml:6–34](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/pyproject.toml#L6-L34)；安装依据：[README.md:77–93](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/README.md#L77-L93)。本仓库的 CPU CI 使用 Ubuntu；本文没有验证本机 Windows 安装，环境遇到差异时可选 Linux/WSL。依据：[.github/workflows/ci.yml:24–32](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/.github/workflows/ci.yml#L24-L32)。
- **动作：** 新建本地脚本，固定随机 seed，生成 bf16 Q=`[1,4,128,128]`、K/V=`[1,2,128,128]`，调用 `flash_attn_func(..., causal=True, interpret=True)`。把 Q/K/V 和输出交换 T/H 轴后，与 `tests.softmax_attention.util.attention_ref` 比较。依据：[tests/softmax_attention/test_prefill.py:124–128](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/tests/softmax_attention/test_prefill.py#L124-L128)、[tests/softmax_attention/test_prefill.py:240–280](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/tests/softmax_attention/test_prefill.py#L240-L280)。
- **验收：** 输出形状正确；记录 max/mean absolute error；另算 `upcast=False,reorder_ops=True` 的 bf16 reference，并沿用项目中 `kernel_error <= 2*bf16_reference_error + arithmetic_atol` 的前向判据。依据：[tests/softmax_attention/test_prefill.py:139–141](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/tests/softmax_attention/test_prefill.py#L139-L141)、[tests/softmax_attention/test_prefill.py:248–280](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/tests/softmax_attention/test_prefill.py#L248-L280)。
- **产出：** 一个可重复运行的小脚本，附一页 Q/K/V→padding→dispatch→Pallas→output 的形状记录。解释模式只用于正确性和学习，不记录为 TPU 性能。

### 任务二：用三个边界例子弄懂 causal 与 padding（约 2–3 小时，CPU 可做）

- **动作：** 沿用任务一，新增本地 `tests/softmax_attention/test_beginner_contracts.py`，其余维度保持不变，构造三例：
  1. Tq=128、Tk=129、`causal=False`：断言 dense 接口抛出 ValueError。
  2. 同样长度、`causal=True`：应成功，并和 bottom-right causal reference 比较。
  3. Tq=129、Tk=128、`causal=True,return_softmax_lse=True`：第一行没有可见 key，应 out=0、LSE=-inf，其余行匹配 reference。
- **为什么选这些：** 它们分别触及补齐后的 KV 尾部、原始长度的 causal_offset，以及全 mask 行的恢复逻辑，直接对应 [flywheel_tpu/softmax_attention/prefill.py:176–219](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/prefill.py#L176-L219)、[flywheel_tpu/softmax_attention/prefill.py:309–337](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/prefill.py#L309-L337)；reference 的对齐语义见 [tests/softmax_attention/util.py:194–209](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/tests/softmax_attention/util.py#L194-L209)。
- **运行与验收：** 新文件写好后，在副本根目录运行 `uv run python -m pytest tests/softmax_attention/test_beginner_contracts.py -q`；三个断言通过，并能手画解释 Tk-Tq 为 +1 与 -1 时的可见区域。LSE 比较沿用 [tests/softmax_attention/test_prefill.py:144–155](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/tests/softmax_attention/test_prefill.py#L144-L155) 的判据。所有新调用显式传 `interpret=True`，不需要运行整个测试矩阵。

### 任务三：在 TPU 上复现一个小范围性能基线（约 1–2 小时，需 TPU）

- **前置：** 任务一正确性通过，且有 TPU VM；从源码环境按 README 安装 TPU extra。没有 TPU 时先完成前两项，本任务等待硬件。
- **动作：** 只用 FlyWheel backend，分别测 T=1024 和 T=4096，其他参数相同。在源码副本根目录运行以下单行命令，再把 `--seq 1024` 改成 `--seq 4096`、输出文件相应改名：

```bash
uv run --no-sync python -m benchmarks.softmax_attention.attention_block --impl flywheel --heads 4 --heads-k 2 --head-dim 128 --batch 1 --mask causal --seq 1024 --output beginner-1024.jsonl
```

- **命令来源：** 参数定义见 [benchmarks/softmax_attention/attention_block.py:494–534](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/benchmarks/softmax_attention/attention_block.py#L494-L534)；该 backend 把 QKV 投影、attention、输出投影包在同一个 jit 中，见 [benchmarks/softmax_attention/attention_block.py:188–208](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/benchmarks/softmax_attention/attention_block.py#L188-L208)。这是 block 基线，不能称作纯 attention kernel 时间。
- **验收：** 两个 JSON 均为 `status=ok`，记录 TPU 型号、源码提交、Python/JAX/libtpu 版本、`compile_s`、`samples_ms`、`median_ms`；按 [benchmarks/common/timing.py:40–74](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/benchmarks/common/timing.py#L40-L74) 区分首次调用开销与预热后的测量。随后选一个形状加 `--trace beginner-trace`，检查 `segments_ms`、`kernel_ms` 是否成功输出；对应 [benchmarks/softmax_attention/attention_block.py:430–443](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/benchmarks/softmax_attention/attention_block.py#L430-L443)。
- **产出：** 两个形状的基线记录，以及对“当前花在 attention 还是投影/布局上”的一条有测量依据的观察。首轮不设必须达到的加速比；先建立可靠测量，再决定是否尝试 block/stage 调优。

## 7. 初步判断

我的建议是从“数学 reference → 小形状 API 验证 → dense kernel 的内存与流水线 → TPU 实测”推进。已有 PyTorch 模型实现经验可以直接用于核对 attention 公式、mask 和张量形状；主要要补的是显式存储层次、物理 layout、异步搬运与数值/性能验证。这个顺序对应项目已经存在的 reference、入口、Pallas scratch 和计时工具，而不是凭空设计学习工程：[tests/softmax_attention/util.py:181–273](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/tests/softmax_attention/util.py#L181-L273)、[flywheel_tpu/softmax_attention/prefill.py:62–125](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/prefill.py#L62-L125)、[flywheel_tpu/softmax_attention/kernels/fwd_dense.py:797–982](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/flywheel_tpu/softmax_attention/kernels/fwd_dense.py#L797-L982)、[benchmarks/common/timing.py:40–74](https://github.com/heavyball-research/flywheel-tpu/blob/2ba7d193fb4c4d46940d86927f85047c67665ea0/benchmarks/common/timing.py#L40-L74)。

现阶段可以提出的具体理解是：接口层负责保持 attention 语义，调度与 layout 决定 TPU 如何高效执行，测试与 benchmark 分别约束正确性和速度。尚未做硬件实验前，不把任何调参方向写成已经成立的性能结论。

