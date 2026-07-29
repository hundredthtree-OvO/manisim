# Matched Branch 实验路线

本文件只描述反事实分支与鲁棒动作选择研究，不替代 `BUILD.md` 和
`RESEARCH_ROADMAP.md` 中的交互控制、数据采集与长期功能路线。

## 1. 核心问题

研究假设：

> 在接触关键状态上，固定完整物理状态并真实执行多个局部动作片段，可以获得
> 比独立 rollout 更可归因的动作效果数据；这种 matched branch 数据后续可能
> 改善未知动力学下的局部动作选择。

当前阶段只验证数据基础，不训练 effect model、critic、Residual RL 或 VLA。

## 2. 术语边界

- `checkpoint`：仿真、控制器和项目运行状态的可恢复快照。
- `anchor`：触发 checkpoint 和动作分支的物理/任务事件。
- `branch`：从同一 checkpoint 执行的一个固定长度动作片段。
- `branch group`：同一 anchor 下的一组动作分支及真实执行结果。
- `matched-state intervention`：固定动力学内从同一状态执行不同动作。

本阶段不改变质量、摩擦或控制延迟，因此不宣称跨动力学分支来自完全相同的
物理状态。

## 3. Gate A：Checkpoint Fidelity

### 3.1 保存范围

- ManiSkill scene state：actor pose、线速度、角速度、articulation 状态；
- Panda controller state；
- `CommandExecutor` 与 `WorkspaceGuard` 内部状态；
- Task 和 scripted source 的阶段及累计状态；
- Python、NumPy、Torch RNG；
- 实验调用方补充的标量状态。

### 3.2 回归流程

```text
运行到 pre-lift
→ 保存 checkpoint
→ 执行固定 H 步抬升
→ 恢复 checkpoint
→ 重复执行相同 H 步动作 N 次
→ 比较 TCP、物体轨迹、接触力和最终状态
```

初始验收：

- 重复分支轨迹长度一致；
- 无 NaN、无状态恢复异常；
- TCP/物体最终位置的重复极差不高于 2 mm；
- 抓取结果一致；
- 接触力差异被记录，不预先假定严格相同。

若该门槛失败，停止 branch 学习路线并先修复恢复边界。

## 4. Gate B：单锚点固定动力学 Branch Collector

唯一锚点：

```text
pre_lift := Panda 已判断 target grasped，尚未开始抬升
```

第一组 intervention：

| branch | dwell | lift scale | XY offset |
| --- | ---: | ---: | ---: |
| base | 0 | 1.0 | 0 |
| wait-5 | 5 steps | 1.0 | 0 |
| slow | 0 | 0.5 | 0 |
| fast | 0 | 1.5 | 0 |
| x-plus | 0 | 1.0 | +3 mm X |
| x-minus | 0 | 1.0 | -3 mm X |
| hold-m050 | 0 | 1.0 | 0 |
| hold-m025 | 0 | 1.0 | 0 |
| hold-000 | 0 | 1.0 | 0 |
| hold-p025 | 0 | 1.0 | 0 |
| hold-p050 | 0 | 1.0 | 0 |
| hold-p100 | 0 | 1.0 | 0 |
| hold-p250 | 0 | 1.0 | 0 |

固定：

- 一个 cube；
- 一个物理参数配置；
- 固定 downward orientation；
- 固定 branch horizon；
- 所有分支复用同一 `CommandExecutor` 安全链。
- `hold-*` 将夹爪闭合目标从 `-1.0` 逐级放宽到
  `-0.5 / -0.25 / 0.0 / 0.025 / 0.05 / 0.10 / 0.25`，用于寻找“中心仍成功、
  边界先失败”的夹持裕量区间；正值微扫用于定位接近方块宽度处的释放阈值，
  不等同于直接执行完全打开。

每步结果：

- TCP、target pose/velocity；
- qpos、qvel；
- 左右指尖力、物体合力、非预期接触力；
- grasped；
- canonical target、safe target 和实际 Panda action。

每个 branch 汇总：

- 最终 TCP/物体位移；
- 最大与积分 grip/object/unintended force；
- 是否保持抓取；
- 最大横向滑移；
- checkpoint ID、anchor ID、intervention 参数。

## 5. 当前不做

- 多动力学 Action × Dynamics 网格；
- 自动 anchor discovery；
- 多任务 anchor DSL；
- 图像输入；
- effect model、pair/group loss；
- candidate reranking；
- Residual RL、在线 RL；
- sim-to-real。

## 6. 后续 Gate

只有 Gate A/B 通过后：

1. 检查 branch 间效果差异是否显著大于重复噪声；
2. 加入第二个物理事件任务，随后抽象共享 event detector；
3. 再引入 aligned multi-dynamics anchors；
4. 比较 independent regression、pairwise 和 groupwise；
5. 最后决定是否需要 reranking、offline residual 或 RL。

## 7. 首轮实测结果

2026-07-29，固定 cube、固定动力学、`pre_lift`：

- 相同抬升动作重复 3 次；
- 最终 TCP 位置极差：`0.0 m`；
- 最终物体位置极差：`0.0 m`；
- TCP 和物体的整段轨迹重复极差：`0.0 m`；
- 所有重复分支均保持抓取；
- Gate A 在当前 CPU PhysX、固定 seed 条件下通过。

6 个 intervention 均成功执行并产生完整 70 步记录。首轮出现一个可检测的
瞬态差异，但任务结果与滑移差异仍然很弱：

- 所有分支均保持抓取；
- 最大相对 XY 滑移约 `0.083～0.085 mm`；
- 最大 grip force 约 `28.02 N`；
- `wait_5` 在停留期间的物体合力峰值约 `0.684 N`，其余立即抬升分支约
  `0.003 N`；
- 除上述等待瞬态外，没有出现掉落、明显滑移或任务结果分化。

结论：

> Gate B 的恢复、执行与记录管线通过，也能分辨等待造成的接触瞬态；但标称
> 中心抓取距离失败边界太远，当前数据不足以支持“鲁棒动作选择”学习。下一轮
> 应改变 anchor 状态质量或扩大局部 intervention，使分支产生滑移/掉落等结果
> 分化；不应先增加模型复杂度。

可复现实验命令：

```bash
uv run mani-sim-branch-collect --config configs/demo0.yaml
```

结果写入：

```text
runs/experiments/<experiment-id>/
├── report.json
├── checkpoint_repeats.json
└── branch_group.json
```

## 8. Gate C：边界 Anchor 扫描

下一轮不改变动力学，也不增加学习模型。通过闭合夹爪前 TCP 相对物体中心的
偏移，改变 `pre_lift` anchor 的初始抓取质量：

- 中心抓取；
- X 轴 `±5 / ±10 / ±15 / ±20 / ±25 / ±30 mm`；
- Y 轴 `±5 / ±10 / ±15 / ±20 / ±25 / ±30 mm`。

粗扫描定位转折区后，X 轴 `26～29 mm`、Y 轴 `21～24 mm` 按 1 mm
自适应细化；内部稳定区域不做密集笛卡尔积扫描。

每个偏移从同一 episode-start checkpoint 独立恢复和构造。无法形成抓取的
偏移仍写入扫描报告，但不会伪装成合法 `pre_lift` anchor；成功形成抓取的
anchor 还必须通过 checkpoint 恢复后的抓取复验，才执行 Gate B 的同一组六个
intervention。报告分别保留 `formed_pre_lift` 和 `restorable_pre_lift`，避免
把边界瞬间的脆弱 grasp 判定当成可重复实验锚点。

验收重点：

1. 找到至少一个中心成功、较大偏移失败的方向；
2. 找到至少一个仍能形成抓取、但 branch 间出现掉落或明显滑移分化的边界
   anchor；
3. 若所有有效 anchor 仍完全稳定，下一轮才扫描夹爪闭合目标或闭合时机。

运行：

```bash
uv run mani-sim-branch-collect \
  --config configs/demo0.yaml \
  --anchor-sweep
```

定位边界后，只复测中心与四个边界 anchor：

```bash
uv run mani-sim-branch-collect \
  --config configs/demo0.yaml \
  --anchor-sweep \
  --boundary-only
```

输出：

```text
runs/experiments/<experiment-id>-anchor-sweep/
├── anchor_sweep_report.json
└── x<offset>_y<offset>_mm.json
```

## 9. Gate C 实测结果

固定动力学下的粗扫与 1 mm 边界细化得到：

- X 正向最后可恢复 anchor：`+26 mm`，`+27 mm` 无法形成抓取；
- X 负向最后可恢复 anchor：`-25 mm`，`-26 mm` 只能瞬时形成抓取，恢复
  checkpoint 后 grasp 判定失效；
- Y 正负向最后可恢复 anchor：`±22 mm`，`±23 mm` 无法形成抓取；
- 最靠边的可恢复 anchor 在原六个运动 intervention 下仍全部保持抓取；
- `hold_m050 → hold_p100` 形成连续夹持裕量曲线。X 边界最大滑移约从
  `0.29 mm` 增到 `0.38 mm`，中心约从 `0.07 mm` 增到 `0.14 mm`；
- `hold_p250` 对中心和四个边界 anchor 均释放，最终物体高度约 `0.02 m`，
  没有出现仅边界先二值失败的窄区间。

此外发现 PhysX 接触拓扑限制：释放分支使夹爪与物体分离后，仅恢复 scene、
controller 和项目状态不足以恢复接触缓存，后续 grasp 复验可能失败。因此
`hold_p250` 必须作为 branch group 的最后一条 intervention；不能把
checkpoint fidelity 从“接触保持型分支”外推到“接触断开后继续分支”。

阶段结论：

> Gate C 找到了可重复的几何抓取边界，也获得了显著高于重复噪声的连续滑移与
> 力响应差异；但当前干预没有产生有意义的边界专属二值成败。下一阶段的数据
> 标签应优先使用连续 effect（相对滑移、夹持力、物体力和末端偏差），而不是
> 只使用 success/drop。若仍要研究二值鲁棒选择，应改变动力学或物体几何，
> 不应继续手工堆叠更激进的同类夹爪阈值。

本轮主要结果：

```text
runs/experiments/20260729-032334-694495-anchor-sweep/
```

## 10. Gate D：Effect Dataset v1

Gate D 先固化数据契约，不训练模型。`mani-sim.effect.v1` 每条样本对应同一
anchor 下的一个 intervention：

- `sample_id`：实验、anchor、intervention 的唯一组合；
- `matched_group_id`：同一 checkpoint 的分支组；
- `anchor`：类型、抓取偏移、TCP 与物体初始位置；
- `intervention`：等待、抬升尺度、XY 偏移和夹爪目标；
- `labels`：相对滑移、力峰值与积分、最终 TCP/物体位移、grasp transition；
- `effect_delta_from_base`：相对同 anchor `base` 分支的效果差；
- `source`：原始 branch group 和 checkpoint 引用。

逐步轨迹不复制进 effect dataset，继续保留在原始 branch JSON 中，避免派生
数据重复膨胀。checkpoint repeats 用于生成独立 `noise_baseline`；不能用不同
anchor 的方差冒充恢复噪声。

构建：

```bash
uv run mani-sim-effect-dataset \
  runs/experiments/20260729-032334-694495-anchor-sweep \
  --output-dir runs/datasets/prelift-boundary-v1 \
  --fidelity-group \
    runs/experiments/20260729-030555-476003/checkpoint_repeats.json
```

输出：

```text
runs/datasets/prelift-boundary-v1/
├── manifest.json
└── effects.jsonl
```

首个数据集结果：

- `65` 条样本；
- `5` 个 matched groups，每组 `13` 个 intervention；
- sample ID 全部唯一；
- `60 held / 5 lost`；
- `5` 个 lost 均来自组末尾的 `hold_p250`；
- 最大相对 base 的滑移增量约 `0.908 mm`；
- 三次 checkpoint repeats 的已记录连续指标极差为 `0`。

当前只有五个 anchor，不划分 train/validation/test，也不据此报告泛化性能。
下一 Gate 应先增加独立 anchor 状态和重复 seed，再按完整
`matched_group_id` 划分，严禁把同一 checkpoint 的不同 branch 分到不同集合。

## 11. Gate E：随机场景与无泄漏划分

使用 `configs/manual_randomized.yaml` 和显式 `--seed`，让 seed 同时决定仿真
RNG 和方块绝对位置。每个 seed 采集中心及四个相对边界 anchor：

```bash
uv run mani-sim-branch-collect \
  --config configs/manual_randomized.yaml \
  --anchor-sweep \
  --boundary-only \
  --seed 0
```

Effect Dataset builder 支持合并多个实验目录。划分有两种严格层级：

- `matched_group`：同 checkpoint 的 branch 不跨集合；
- `experiment`：同 seed/绝对场景内的所有 anchor 都不跨集合。

研究绝对位置泛化时必须使用更严格的 `experiment`：

```bash
uv run mani-sim-effect-dataset \
  runs/experiments/20260729-033547-361925-anchor-sweep \
  runs/experiments/20260729-033614-792521-anchor-sweep \
  runs/experiments/20260729-033644-779238-anchor-sweep \
  --output-dir runs/datasets/prelift-randomized-seed012-v1 \
  --fidelity-group \
    runs/experiments/20260729-030555-476003/checkpoint_repeats.json \
  --split-ratios 0.7 0.15 0.15 \
  --split-seed 17 \
  --split-unit experiment
```

实测结果：

- 三个 seed 的方块 XY 分别约为
  `(0.469, -0.055) / (0.452, 0.108) / (0.417, -0.048) m`；
- seed 0 的相对 `Y=-22 mm` anchor 在新绝对位置下无法形成稳定抓取；
- 最终得到 `14/15` 个可恢复 matched groups；
- 合并后共 `182` 条样本，`168 held / 14 lost`；
- train/validation/test 按整个 experiment 分别包含 `1/1/1` 个 seed、
  `4/5/5` 个 matched groups和 `52/65/65` 条样本；
- matched-group 泄漏数和 experiment 泄漏数均为 `0`。

该数据只验证多场景采集与无泄漏划分管线。每个 split 只有一个 seed，不能用于
可信的模型比较。进入首个 effect baseline 前，建议至少采集 10 个独立场景
seed，并保证 validation/test 各不少于 2 个 seed；如果边界 anchor 随工作区
位置系统性漂移，则下一步应学习或标定位置条件化边界，而不是丢弃失败 anchor。

## 12. Gate F：首个 Effect Regression Baseline

补采 seed 3～9 后，共覆盖 seed 0～9：

- `46` 个可恢复 matched groups；
- `598` 条 branch effect 样本；
- 按整个 scene seed 做 `6/2/2` train/validation/test 划分；
- train/validation/test 分别包含 `27/10/9` 个 matched groups；
- 连续回归只使用 maintained-grasp 分支，确定性释放的 `hold_p250` 单独统计。

首轮不训练大模型。对照为：

1. `global_mean`：忽略 intervention 和 anchor；
2. `intervention_mean`：每个动作在训练集上的平均效果，是必须击败的强对照；
3. `action_ridge`：只看 intervention 参数；
4. `state_action_ridge`：anchor、action 及二者交互；
5. `state_knn`：同 intervention 内按 anchor 状态近邻预测。

连续目标为相对同 anchor base branch 的：

- XY 滑移差；
- grip/object force 峰值差；
- grip/object force impulse 差；
- 最终物体 XYZ 位移差。

训练：

```bash
uv run mani-sim-effect-baseline \
  runs/datasets/prelift-randomized-seed00-09-v1 \
  --output-dir runs/baselines/prelift-effect-ridge-v1
```

为避免一次场景划分偶然乐观，额外对 scene-level split seed 0～9 做十次复验，
并汇总到：

```text
runs/baselines/prelift-effect-ridge-cross-split-summary.json
```

结果：

- `state_action_ridge` 仅 `4/10` 次击败 intervention mean；
- ridge 平均 normalized RMSE gain 为 `-0.065`，即平均更差；
- `state_knn` 为 `6/10` 次胜出，平均 gain 为 `+0.035`；
- kNN 的滑移 RMSE 平均为 intervention mean 的 `0.888`，`8/10` 次胜出；
- kNN 的物体 X 位移 RMSE 比例为 `0.665`，`10/10` 次胜出；
- 物体力峰值与物体力积分比例分别为 `1.062/1.072`，没有稳定改善。

阶段性结论：

> 当前结果提供了“弱但非零”的状态条件化 effect 信号：局部几何位移和滑移
> 可以从相似 anchor 获得小幅、较稳定的预测收益。但核心接触力 effect 尚未
> 被当前状态表示可靠预测，线性 state-action 模型也没有稳定超过动作均值。
> 因此现阶段可以继续研究 matched-branch 数据，但不能宣称主要 hypothesis
> 已验证，更不能直接进入 Residual RL。

下一轮最值得验证的不是换更大的网络，而是补足 anchor 中缺失的接触状态：

- 左右指尖接触点与法向；
- 夹爪开度和两侧力不对称；
- 物体相对夹爪的完整位姿；
- pre-lift 前短窗口的速度与力历史。

明确的继续/停止判据：加入这些物理状态后，跨 scene 的 effect model 应在至少
`8/10` 个划分上击败 intervention mean，并让 object force/impulse RMSE
平均降低至少 `10%`；否则当前“基于 anchor 状态预测局部接触 effect”的路线
应降级为数据分析工具，而不作为主要学习贡献。
