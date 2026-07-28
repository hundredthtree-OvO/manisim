# mani-sim 构建与开发说明

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
├── assets/
│   ├── object_spec.py
│   └── object_factory.py
├── environments/
│   └── scenario.py
├── tasks/
│   ├── base.py
│   └── pick_place.py
├── runtime/
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
Scenario            实体注册、查询和复位
ObjectFactory       根据规格创建 actor
Task                阶段、成功判据和任务记录字段
Controller/Safety   命令跟踪与保护
EpisodeRecorder     运行时记录接口
```

现有 `task_scene.py` 和 `task_progress.py` 只保留兼容导出。后续新增物体不应
写入任务状态机；新增任务也不应直接读取鼠标或调用 `env.step()`。

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

当前默认写入：

```text
runs/demo0.jsonl
```

每个控制步一行，主要包含：

- step、时间戳、active view；
- 鼠标像素和输入有效性；
- 原始目标、安全目标、实际 TCP；
- `target_height_m`、`target_depth_y_m`；
- action、夹爪命令、qpos、qvel；
- 可达投影、碰撞保护和接触状态；
- 方块位置、抓取状态、任务阶段和目标距离；
- 跟踪误差和安全目标步长。

当前限制：

- 每次启动覆盖旧文件；
- R 不创建独立 episode；
- 没有 schema/version/config 元数据；
- 没有图像；
- 个别字段存在 step 前后采样时序差异。

正式数据采集前应升级为：

```text
runs/<session>/
├── metadata.json
└── episodes/
    ├── episode_000000.jsonl
    └── episode_000001.jsonl
```

每帧明确组织为
`observation_t -> command_t -> action_t -> observation_t+1 -> task_state_t+1`。

## 10. 当前验证基线

最新验证：

```text
普通测试：41 passed，3 GPU tests skipped
RTX 4070 SUPER：3 physical integration tests passed
GUI：100-step smoke passed
```

关键结果：

- Pick-and-Place 最终 XY 误差约 3.38 mm；
- 最终方块中心高度约 0.02 m；
- 低位地面非预期接触峰值 0 N；
- 静态障碍保护目标的障碍接触峰值 0 N；
- 初始化和 reset 首帧 action 为零；
- FRONT 上下越界输入不会进入标定范围外。

已知的 NumPy `__array_wrap__` 警告来自 ManiSkill 3.0.1 依赖路径，不影响
当前测试结果。

## 11. 后续路线

建议顺序：

1. 人工验证 FRONT 映射、深度方向和视图切换手感；
2. UI 显示 active view、任务阶段、抓取状态和接触阈值；
3. 将记录升级为正式 session/episode 格式；
4. 增加 yaw，保持 roll/pitch 固定；
5. 增加 Push 任务和任务专用姿态预设；
6. Pull/Drawer 阶段加入水平抓取姿态；
7. 复杂障碍增加整臂距离查询或运动规划；
8. 最后再进入完整 6D 位姿控制。

暂不优先实现力曲线。当前更有价值的是记录分项指尖/物体/障碍力并离线分析；
进入 Push、Pull 或力控实验后再考虑在 UI 中显示最近 3–5 秒曲线。
