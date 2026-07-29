# mani-sim 个人研究路线

## 1. 项目定位

本项目不是通用机器人仿真平台，而是一套围绕个人研究兴趣持续演进的轻量
实验脚手架。核心闭环是：

```text
鼠标示范 + 自动采集
-> 策略训练与执行
-> 鼠标介入修正
-> 利用示范、失败和介入数据继续学习
```

当前 `interactive-v1` 已达到阶段性人工采集目标：

- Panda 单臂和固定向下末端姿态；
- TOP XY、FRONT XZ、WRIST 多视图；
- 鼠标位置控制、键盘高度/深度和夹爪控制；
- 可达域投影、碰撞保护和 reset 稳定性；
- Pick-and-Place 任务阶段与成功判据；
- 分项指尖、物体和非预期接触力记录；
- 独立实时力曲线窗口；
- 正式 session/episode 记录。

下一阶段优先形成可重复的数据和学习闭环，不提前建设完整的通用任务、
训练或分布式平台。

## 2. 两种数据扩展方式

### 2.1 人工轨迹并行重放

人工只操作一次，记录 TCP 目标、夹爪命令和时序；随后在多个随机化环境中
并行重放同一操作意图。

```text
human episode
-> N 个质量/摩擦/控制误差不同的环境
-> 每个环境独立执行可达投影、安全保护和底层控制
-> N 条带共同 parent_episode_id 的派生轨迹
```

它用于动力学对照、force 分析、鲁棒性增强和 counterfactual replay，不应
被视为 N 条独立人工示范。训练和评估划分必须按 `parent_episode_id` 分组，
避免同源轨迹泄漏。

重放的是 canonical 操作意图：

```text
raw/safe TCP target
gripper command
control mode
```

不直接复制归一化 action，使每个环境能够根据自身状态重新计算控制动作。

### 2.2 自动 Pick-and-Place

自动策略完全不依赖鼠标，按照每个环境的实时状态独立推进：

```text
approach -> descend -> close -> lift -> transport -> lower -> open
```

同一个 batch 中，各环境可以处于不同任务阶段，并独立成功、失败和 reset。
它是真正的大规模自动采集入口，适合不同物体位置、质量、摩擦和视觉外观。

人工重放与自动策略是两个 `ActionSource`，但共用同一个 Parallel Collector：

```text
source = trajectory_replay
source = scripted_pick_place
```

建议先实现单环境自动策略，再批量化；随后加入人工轨迹并行重放。

## 3. RoboCasa 资产使用范围

近期不直接迁移完整 RoboCasa 厨房和长程任务，而是在当前 Panda 桌面任务中
复用 ManiSkill 已适配的 RoboCasa 资产。资产分为三类。

### 3.1 目标物

逐步将红色方块替换为经过筛选的可抓取物体：

- 杯子、瓶子、罐头、盒子和水果；
- 尺寸小于 Panda 夹爪最大开口；
- 适合当前垂直二指抓取；
- 碰撞模型、质量、惯量和几何原点合理；
- 第一批避免透明、强非凸、侧抓和柔性物体。

每个资产进入采集集前执行：

```text
load -> settle -> approach -> grasp -> lift -> place -> release
```

并记录尺寸、质量、类别、资产 ID 和失败原因。

### 3.2 放置目标

绿色无碰撞目标区逐步替换为更真实的 receptacle 或目标表面：

- 盘子、托盘、碗：具有物理几何和接触；
- 桌垫、布面：近期作为刚性薄垫或纯视觉区域；
- 指定台面区域：通过语义区域而不是绿色标记表达。

盘子或托盘的成功判据不只检查 XY：

```text
目标物已进入 receptacle 范围
目标物已释放
高度和速度进入稳定范围
必要时确认与 receptacle 接触
```

ManiSkill 3 当前不以柔性体任务为重点，因此“布”暂不模拟真实褶皱、拉伸和
软接触。先使用刚性薄 mesh 或无碰撞视觉标记；只有研究问题明确依赖柔性
动力学时，才单独评估其他后端或柔性方案。

### 3.3 干扰物

加入随机物体干扰是合理的，但按难度分层：

1. **视觉干扰**：无碰撞物体、纹理和颜色变化，用于视觉泛化；
2. **静态物理干扰**：不会移动的障碍，用于路径与碰撞保护；
3. **动态干扰物**：可推动、可碰撞的非目标物，用于杂乱场景和误抓分析。

所有干扰物必须与任务目标具有稳定 ID，记录：

```text
target_object_id
receptacle_id
distractor_ids
```

初期每个场景放置 1～3 个间距充足的干扰物，先避免目标被遮挡或初始碰撞。
自动策略稳定后再增加 clutter 密度。动态干扰物的接触力应与目标物夹持力、
地面/静态障碍力分开记录。

RoboCasa 完整厨房、柜体任务、Fetch/mobile manipulation 和长程语义任务属于
远期方向，不是当前资产替换工作的隐含范围。

## 4. 最小可扩展结构

只保留四个稳定概念：

```text
Scenario
    对象、receptacle、干扰物、初始状态和随机化

Task
    阶段、成功/失败、reward、UI 与记录字段

ActionSource
    mouse、scripted、trajectory replay、policy、intervention

Collector
    单环境交互或多环境并行
```

不构建一个覆盖所有任务的超级状态机。每个任务拥有自己的任务逻辑和自动
策略，但共享 observation、canonical action 和 episode schema：

```text
tasks/
├── pick_place.py
├── push.py
└── pull.py

policies/scripted/
├── pick_place.py
├── push.py
└── pull.py
```

多任务按三层推进：

1. 同任务换物体：Pick-and-Place + 多种 RoboCasa 目标物/receptacle；
2. 同动作空间换任务：Push、Reach、简单 Stack；
3. 需要新姿态的任务：Pull、Drawer、Handle，再开放 yaw 或完整 6D。

暂不进入完整厨房长程任务。

## 5. 数据契约

所有人工、自动、重放和介入 episode 至少记录：

```text
observation
├── top/front/wrist RGB
├── robot state
├── object/receptacle state
├── force
└── task instruction

action
├── canonical target
├── controller action
├── action_source
├── policy_action
├── human_action
└── executed_action

episode
├── task/asset/randomization
├── success/failure
├── intervention intervals
├── parent_episode_id
└── derived replay index
```

当前 JSONL 继续作为可读原始记录。确定具体训练方案后，再增加 ManiSkill
HDF5 或 LeRobot exporter，不同时维护多套主记录格式。

## 6. 学习与人工介入路线

先得到能够自动执行的策略，再研究介入和适应：

```text
人工示范 + 自动采集
-> 简单 BC sanity check 或 SmolVLA 微调
-> policy autonomous rollout
-> 鼠标立即接管
-> 记录 policy/human/executed action
-> episode 间更新 LoRA/adapter
```

人工接管时必须：

- 清空 VLA 尚未执行的 action chunk；
- 独立记录介入开始、结束和原因；
- 不覆盖 policy 原始 proposal；
- 保留介入前后的 force、任务阶段和失败趋势。

优先研究 episode 间的 test-time adaptation，而不是在控制循环内持续更新整个
VLA。RL 只在形成明确问题后加入，例如 demonstration 初始化、force-aware
reward、collision penalty 或 residual policy；不从零训练大规模 RL。

## 7. 推荐实施顺序

### Milestone A：自动采集最小闭环

已完成统一 `TaskSpaceCommand`、`ActionSource`、`CommandExecutor` 和
`RuntimeObservation` 插口；当前人工模式和单环境
`ScriptedPickPlaceSource` 已通过同一执行链运行，并共享 UI、force 与
session/episode 记录。固定人工模式用于调试，随机人工模式和自动策略已共享
同一位置分布与 seed 语义，可进行成对比较。

下一步：

1. 已完成位置随机化、批量 episode 上限和 session 统计最小闭环；
2. 将随机位置基线扩大到 50～100 条；
3. 为 episode 加入 top/front/wrist 图像；
4. 逐项验证尺寸、质量和摩擦随机化；
5. 将自动策略和任务状态改为 batch-aware；
6. 进行 `16 -> 32 -> 64` 环境吞吐与显存基准。

### Milestone B：真实资产与随机化

1. 下载并审计 ManiSkill 适配的 RoboCasa 资产；
2. 筛选 3～5 个适合垂直抓取的目标物；
3. 加入盘子/托盘和刚性布垫目标；
4. 加入 1～3 个视觉干扰物；
5. 再加入静态和动态物理干扰；
6. 比较 cube-only 与 mixed-object 成功率和 force。

### Milestone C：人工轨迹扩增

1. 记录可重放的 canonical target；
2. 实现并行 trajectory replay；
3. 增加质量、摩擦、位姿和控制误差随机化；
4. 记录同源关系、成功率和接触力差异；
5. 建立按 parent episode 分组的数据划分。

### Milestone D：策略与介入

1. 用简单 BC 验证图像/action 对齐，或直接选择 SmolVLA；
2. 完成自动 rollout 和标准 evaluation；
3. 加入 policy/human 控制权仲裁；
4. 采集介入与失败恢复数据；
5. 进行 episode 间 LoRA/adapter 更新；
6. 根据研究问题再选择 RL 或更强的 test-time training。

## 8. 暂不优先

- 完整 RoboCasa 厨房和 365 个任务迁移；
- 移动底盘和长程规划；
- 在线持续更新整个 VLA；
- 从零开始的大规模 RL；
- 真实柔性布料动力学；
- 为尚未选择的训练框架提前维护多套数据格式；
- 为所有未来任务构建通用抽象。

每一阶段都应产生可独立比较的结果，而不是只增加基础设施。
