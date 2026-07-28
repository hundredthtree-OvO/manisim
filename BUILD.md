# mani-sim 构建与开发说明

人工采集之后的自动采集、RoboCasa 资产、多任务、策略介入和学习路线见
[RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md)。

## 1. 项目目标

基于 ManiSkill 3 构建可复用的单臂研究脚手架，用于：

- 鼠标连续控制 Panda TCP；
- TOP XY、FRONT XZ 与腕部观察；
- 可达域投影、碰撞保护和接触监测；
- Pick-and-Place 等任务验证；
- 后续替换物体、任务、输入策略和数据记录方式。

当前重点是低延迟、连续、可解释的交互控制，不追求“鼠标与机械臂零延迟
锁定”。实际运动仍受控制周期、IK、PD、关节限位和 PhysX 动力学约束。

## 2. 环境与运行

要求 Linux 图形会话、NVIDIA/Vulkan 驱动和 `uv`。

```bash
uv sync --locked
uv run mani-sim --config configs/demo0.yaml
```

当前关键版本：

```text
ManiSkill 3.0.1
SAPIEN 3.0.3
PyTorch 2.13.0
```

重新标定固定姿态可达域：

```bash
uv run mani-sim-calibrate
```

验证：

```bash
uv run pytest -q
uv run pytest -q -s tests/integration
uv run mani-sim --config configs/demo0.yaml --max-steps 100
```

后两项分别需要 NVIDIA/Vulkan 和图形会话。

参考资料：

- [ManiSkill Installation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
- [Controllers / Action Spaces](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html)
- [Teleoperation](https://maniskill.readthedocs.io/en/latest/user_guide/data_collection/teleoperation.html)

## 3. 当前操作

```text
鼠标          当前 active view 的二维 TCP 控制
U / J         TOP 中控制 Z；FRONT 中控制 Y
Space         切换夹爪开合
1             TOP XY
2             FRONT XZ
3             WRIST，只观察
R             复位机器人、场景和任务
Q             退出
```

视图映射：

| 模式 | 鼠标水平 | 鼠标垂直 | U/J |
| --- | --- | --- | --- |
| TOP XY | world X | world Y | world Z |
| FRONT XZ | world X | world Z | world Y |
| WRIST | 不控制 | 不控制 | 不控制 |

TOP 与 FRONT 共享同一个世界 TCP 目标。切换视图不会改变目标；进入新的控制
视图后必须移动鼠标至少 3 px 才重新接管。

红点表示原始鼠标目标，绿点表示经过工作区、可达域和安全保护后的命令目标。
标记本身没有碰撞，可以进入目标物体；实际机械臂服从物理接触。

## 4. 当前场景与 Pick-and-Place

默认场景：

- Panda `panda_wristcam`；
- 4 cm 动态红色方块，初始中心 `(0.45, 0.00, 0.02) m`；
- 无碰撞绿色放置区，中心 `(0.30, 0.30)`；
- 可选静态障碍，默认关闭。

推荐手动流程：

1. 移动到方块正上方；
2. 降至方块中心 `z=0.02 m`；
3. Space 闭合；
4. 抬升并移动到绿色区域；
5. 降至地面；
6. Space 张开；
7. 保持张开并向上撤离。

任务状态：

```text
approaching
-> approached
-> grasped
-> lifted
-> transported
-> released
-> placed
```

成功放置要求方块曾被抬升、进入目标 XY 容差、夹爪解除抓持，并回落至地面
高度容差内。仅原地张开不立即算释放，因为手指可能仍与方块接触。

## 5. 初始化与 reset

初始化关节状态定义在：

```text
src/mani_sim/robot_setup.py
PANDA_SAFE_QPOS
```

当前保持固定向下姿态，实际初始 TCP 约为：

```text
(0.615010, 0.000000, 0.449907) m
```

reset policy：

```yaml
reset:
  policy: hold_tcp
  pointer_settle_steps: 2
  pointer_rearm_pixels: 3.0
```

首次启动、按 R 和自动 reset 都会：

- 将命令目标同步到实际 TCP；
- 将夹爪命令恢复为张开；
- 重置任务、安全状态和轨迹步长基准；
- 忽略窗口初始化阶段的鼠标漂移；
- 等待明确鼠标移动后再接管。

因此 reset 后不会自动追踪另一个高度或产生首帧目标跳跃。U/J 在鼠标尚未
接管时只沿当前 TCP 的对应轴运动。

## 6. 相机与 active viewport

按键直接切换主 viewport：

- `1`：主窗口 TOP，使用完整窗口控制 XY；
- `2`：主窗口 FRONT，使用完整窗口控制 XZ；
- 主窗口 TOP 时右侧预览 FRONT，主窗口 FRONT 时右侧预览 TOP；
- 右侧第二张图始终为 `hand_camera`；
- `3`：不接管鼠标，主窗口保持最后一个控制相机。

FRONT 相机为正前视：

```text
eye    = (0.45, -1.20, 0.35)
target = (0.45,  0.00, 0.35)
up     = world +Z
```

FRONT 主窗口鼠标通过 viewer 的真实 model/projection matrix 生成射线，
再与当前 `world Y = target_depth_y` 平面求交。实际矩阵验证：

```text
屏幕向右 -> world +X
屏幕向上 -> world +Z
```

TOP/FRONT 切换时同步主相机位姿和 FOV，但不修改世界 TCP。新视图仍需
3 px 鼠标移动才接管，右侧面板区域在两种模式下都屏蔽输入。

### 统一状态面板

右下 `RuntimeStatusPanel` 分为两类字段：

```text
全局：active view、轴映射、TCP、接触力/阈值、safety、recording
任务：task、phase、grasped、goal distance、success
```

全局字段由 runtime 提供；任务字段通过通用 `Task.ui_fields()` 注入，面板
不引用具体任务类。未来 Push/Pull 只需要返回自己的字段，不需要复制 UI。
状态面板区域与相机面板一样屏蔽鼠标命令。

## 7. 控制与可达保护

当前控制器：

```text
pd_ee_delta_pos
action = [dx, dy, dz, gripper]
```

TCP 三维位置可控，末端完整旋转姿态固定为向下。不是某一个 Panda 关节被
锁死；IK 会协调多个关节保持固定 TCP 姿态。

控制链：

```text
raw pointer target
-> workspace pre-clip
-> reachability projection
-> projected-step suppression
-> scene collision guard
-> workspace/stall guard
-> EE servo
-> normalized Panda action
```

必须先执行 workspace pre-clip。FRONT 射线可产生标定高度外的原始目标，
例如 `z=-0.191 m`；该值保留给红色标记，但控制目标会先裁剪至
`[0.02, 0.65] m`，不会传入可达地图导致异常。

### 可达地图

`calibrations/panda_fixed_orientation.json` 使用：

- 14 个高度层：`0.02, 0.05, 0.10, ..., 0.65 m`；
- 2.5 cm 全域粗网格；
- 边界附近 5 mm 自适应细化；
- 严格 IK、关节余量、FK 位置和姿态验证；
- 最大连通域过滤；
- 沿射线边界投影；
- 相邻高度层连续插值；
- 投影目标单步跳变限制。

新初始化姿态下，`z=0.02 m` 粗网格为 982 / 1125 点可达，覆盖率 87.3%。

### 在线安全

- 工作区 X/Y/Z 限制；
- 静态障碍膨胀 AABB 预保护；
- Panda 非基座链节与地面/障碍的接触力监测；
- 非预期接触超过 8 N 时冻结目标；
- IK 无进展时进入带滞回的 stall 状态。

目标物体不做几何排斥，否则无法抓取；方块、夹爪和地面的约束由 PhysX
处理。当前保护是任务空间安全层，不等价于整臂 swept-volume 运动规划。

## 8. 代码架构

```text
src/mani_sim/
├── app.py
├── action_sources/
│   ├── base.py
│   └── mouse.py
├── assets/
│   ├── object_spec.py
│   └── object_factory.py
├── environments/
│   └── scenario.py
├── tasks/
│   ├── base.py
│   └── pick_place.py
├── runtime/
│   ├── command_executor.py
│   ├── observation.py
│   ├── contact_forces.py
│   └── reset_manager.py
├── control/
├── input/
├── mapping/
├── visualization/
├── recording/
│   ├── episode_recorder.py
│   └── jsonl_recorder.py
├── reachability.py
├── calibration.py
└── robot_setup.py
```

职责边界：

```text
App                 运行时编排
ActionSource        根据 observation 产生 canonical task-space command
CommandExecutor     统一执行可达、碰撞、stall、servo 和 action 转换
RuntimeObservation  Task、UI、record 和未来策略共用的单帧状态
Scenario            实体注册、查询和复位
ObjectFactory       根据规格创建 actor
Task                阶段、成功判据和任务记录字段
Controller/Safety   命令跟踪与保护
EpisodeRecorder     运行时记录接口
```

现有 `task_scene.py` 和 `task_progress.py` 只保留兼容导出。后续新增物体不应
写入任务状态机；新增任务也不应直接读取鼠标或调用 `env.step()`。

当前人工模式已接入统一链路：

```text
MouseActionSource
-> TaskSpaceCommand(source=human)
-> CommandExecutor
-> env.step
-> RuntimeObservation
-> Task / UI / EpisodeRecorder
```

配置入口为：

```yaml
collection:
  source: mouse
```

当前只开放 `mouse`；下一阶段加入 `scripted_pick_place` 时，不修改安全和
控制执行链。每帧记录新增 `action_source`，旧字段保持兼容。

目标演进：

```text
configs/
├── runtime/
├── robots/
├── objects/
└── tasks/

tasks/
├── pick_place.py
├── push.py
└── pull.py
```

任务通过对象 ID 引用场景实体，使同一任务可替换方块、球体、瓶子或 mesh，
同一物体也能用于 Pick、Push、Pull。

## 9. 轨迹记录

每次运行创建一个不会覆盖旧数据的 session：

```text
runs/<YYYYMMDD-HHMMSS-ffffff>/
├── metadata.json
├── episodes.jsonl
└── episodes/
    ├── episode_000000.jsonl
    └── episode_000001.jsonl
```

`metadata.json` 保存 `mani-sim.session.v1` schema、session ID、UTC 创建时间、
完整 YAML 配置，以及 Python、ManiSkill、SAPIEN、PyTorch 版本。
`episodes.jsonl` 是 episode 索引，记录文件名、起止时间、步数、结束原因和最终
任务字段。启动时建立 episode 0；按 R、terminated 或 truncated 会封闭当前
episode 并开启新 episode。退出、关闭窗口和 `--max-steps` 也有明确结束原因。

每个控制步仍为 JSONL 一行，并新增：

- `schema_version`、`session_id`、`episode_id`；
- episode 内的 `episode_step` 和 session 内连续的 `global_step`。

原有字段保持兼容，主要包含：

- step、时间戳、active view；
- 鼠标像素和输入有效性；
- 原始目标、安全目标、实际 TCP；
- `target_height_m`、`target_depth_y_m`；
- action、夹爪命令、qpos、qvel；
- 可达投影、碰撞保护和接触状态；
- 方块位置、抓取状态、任务阶段和目标距离；
- 跟踪误差和安全目标步长。

当前暂不记录图像。现有平铺字段存在 step 前后采样差异，具体语义已写入
metadata；正式训练数据导出时再转换成
`observation_t -> command_t -> action_t -> observation_t+1 -> task_state_t+1`，
避免现在破坏已有分析脚本。

## 10. 接触力记录与实时曲线设计

ManiSkill 的 pairwise contact force 是 PhysX 接触冲量除以仿真时间步，单位
为 N。后续应记录原始三维向量和模长，并区分三类来源：

- **分项指尖力**：左指↔目标物、右指↔目标物分别记录。用于识别单边接触、
  夹持是否平衡、夹得过紧以及切向滑移；稳定抓取不能只看两指合力，因为
  左右力可能相互抵消。
- **物体接触合力**：目标物来自全部接触的合力。用于观察落地支撑、碰撞冲击、
  抬升、推拉载荷和放置后的稳定过程。它是接触合力，不含重力；静止于地面时
  接触力通常与重力平衡，不能把该值直接叫作动力学总力。
- **障碍/非预期力**：机器人各非基座 link 与地面、静态障碍的 pairwise
  force，分别记录并汇总 max/sum。用于碰撞保护、定位碰撞对象和评估冲击；
  必须与“夹爪接触任务物体”这种预期接触分离。

建议 UI 仍在同一 SAPIEN 窗口内实现：

```text
physics/control sampling (raw, every step)
-> ForceHistory fixed ring buffer (3–5 s)
-> display decimation (10–20 Hz)
-> NumPy rasterizer -> RGBA texture -> UIPicture
```

不建议每控制帧重绘 Matplotlib，也不建议为曲线增加第二个交互窗口。当前
SAPIEN UI 没有现成 plot widget，因此采用轻量栅格图最稳定：原始数据完整
落盘，UI 只显示降采样视图。指尖左右力放一张图，物体合力与非预期力分图；
当前 `ForceMonitorPanel` 已与 `RuntimeStatusPanel` 分离：运行状态恢复为紧凑
文本，独立 Force monitor 窗口显示最近 5 秒真正的 RGBA 彩色折线。蓝色为
双指有效夹持力，绿色为物体接触合力，红色为非预期接触，橙色为 8 N 阈值。
三条曲线共用 `0–40 N` 固定 Y 轴；实测抓取时左右指力约 28 N，因此不采用
原计划的 20 N 上限。

绘图由 NumPy 栅格化后上传 `RenderTexture2D`，贴到远离任务场景且无碰撞的
专用平面，再由 `force_chart_camera` 输出给 `UIPicture`。刷新不创建第二个
操作系统窗口，也不进入物理系统；TOP/FRONT/WRIST 相机的 far plane 看不到
该平面。平面位于局部 YZ、相机沿其 +X 法向正视，宽高比与 320×190 纹理
一致。上传前根据该平面的 UV 方向转置图像，最终横轴为时间（左旧右新），
纵轴为力（下低上高）；GPU 回归会分别检查图像左右半区，防止曲线区域再次
被裁掉。Force monitor 区域同样屏蔽鼠标控制。显示层不改写原始记录。

## 11. 当前验证基线

最新验证：

```text
普通测试：54 passed，4 GPU tests skipped
RTX 4070 SUPER：4 physical/render integration tests passed
GUI：100-step refactored manual-control smoke passed
```

关键结果：

- Pick-and-Place 最终 XY 误差约 3.38 mm；
- 抓取时左右指尖力约 28.03 / 28.00 N，物体接触合力约 0.64 N；
- 最终方块中心高度约 0.02 m；
- 低位地面非预期接触峰值 0 N；
- 静态障碍保护目标的障碍接触峰值 0 N；
- 初始化和 reset 首帧 action 为零；
- FRONT 上下越界输入不会进入标定范围外。

已知的 NumPy `__array_wrap__` 警告来自 ManiSkill 3.0.1 依赖路径，不影响
当前测试结果。

## 12. 后续路线

本阶段已完成单环境 `ScriptedPickPlaceSource`：

- `collection.source` 可选 `mouse` 或 `scripted_pick_place`；
- 人工与自动模式共用 observation、canonical command、安全执行链和 recorder；
- 自动模式保留 TOP/FRONT/WRIST、状态面板和 Force monitor 可视化；
- 状态与逐帧记录包含 `action_source` 和 `policy_phase`；
- 成功稳定后自动开启下一 episode，超时记录为 `policy_timeout`；
- 700 步真实 GUI 回归完成 2 条连续成功 episode，每条 290 步，最终
  方块到目标中心误差约 1.53 mm。

随后完成第一层 task randomization 与批量统计：

- 每条 episode 使用独立 seed 随机化 cube XY 和 goal XY；
- 随机范围限制在当前低位可达区域，并约束起点/目标最小间距；
- `--episodes N` 支持连续可视化采集 N 条后自动退出；
- session 自动生成 `summary.json`，统计成功率、结束原因、阶段耗时和三类力；
- `mani-sim-report runs/<session-id>` 可离线重算；
- 5 条随机位置真实 GUI/PhysX 回归为 `5/5` 成功，平均 279 步，非预期
  接触力峰值 `0 N`。

建议顺序：

1. 将随机位置批次扩大到 50～100 条，建立首个稳定基线；
2. 加入 top/front/wrist 图像记录；
3. 再逐项加入 cube 尺寸、质量和摩擦随机化，单变量评估失败边界；
4. 将已验证的自动策略和任务状态改为 batch-aware；
5. 再进入 RoboCasa 物体/receptacle 和人工轨迹并行重放；
6. 根据学习目标进入 SmolVLA、鼠标介入或 RL。
