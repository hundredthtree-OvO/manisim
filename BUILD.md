# mani-sim 构建与开发方案

## 1. 目标

在本仓库中搭建一个基于 ManiSkill 3 的、可继续替换和扩展策略的单臂仿真脚手架。首个里程碑（Demo 0）只实现：

- Franka Panda 固定安装在水平场景平面上；
- 固定相机视角；
- 鼠标位置映射为一个水平工作平面上的 TCP 目标位置；
- 机械臂通过 ManiSkill 的末端控制器连续跟随目标；
- TCP 姿态和工作高度固定；
- 目标不可达时稳定停在朝目标方向的可达边界附近；
- 可切换夹爪开合；
- 记录目标 TCP、实际 TCP、动作和饱和状态，供后续策略接口与回放复用。

Demo 0 是控制链路和研究基础设施验证，不把抓取、接触任务、强化学习或 sim-to-real 纳入首期范围。

## 2. 对现有方案的评估

### 2.1 结论

方案总体可行，且 `pd_ee_delta_pos` 适合作为第一版控制器。ManiSkill 的 Panda 末端位置控制器会将三维末端增量通过 IK 转换为关节目标，再由关节 PD 驱动；夹爪动作是一维量。官方 `interactive_panda` 也证明了鼠标、GUI、Panda 和轨迹执行可以组合，但它采用“拖动 gizmo，按键后进行运动规划”，不是连续鼠标伺服。

推荐保留原方案的以下部分：

- 采用物理跟随，不逐帧直接写关节状态；
- 用射线与工作平面求交，不直接线性缩放屏幕像素；
- 第一版固定 TCP 高度和姿态；
- 使用有界的 delta action 连续闭环；
- UI 的夹爪 toggle 转换为内部绝对状态；
- 把输入、映射、安全约束、控制和记录拆开。

### 2.2 必须修正或验证的部分

#### “立即跟随”应定义为低延迟追踪，而不是零延迟锁定

真实链路包括 GUI 事件、渲染、控制器、IK、PD 和物理步进。首期应追求稳定且可测量的低延迟响应，不承诺 TCP 与鼠标像素逐帧完全重合。直接设置关节位置虽能获得视觉上的刚性跟随，但会破坏动力学一致性，不适合作为策略研究的默认路径。

#### 不可达目标不能仅依赖 IK “自然卡住”

持续向不可达点发送 delta action 可能出现停滞、抖动、奇异位形附近跳动，或停在与期望方向不一致的位置。“无进展检测”可以作为 Demo 0 的保护机制，但不能被描述成严格的最大可达投影。

首版采用两层处理：

1. 对每步 TCP 位移限幅，避免鼠标跳变形成大动作；
2. 用距离、实际 TCP 进展和连续停滞计数形成滞回状态机，饱和后冻结或减小命令，目标重新进入可行方向后自动恢复。

沿射线做 IK 可行性二分投影放到后续里程碑。实施前还需确认 ManiSkill 当前版本是否暴露稳定的无副作用 IK 查询；若只能调用控制器内部接口，不应为了 Demo 0 依赖私有 API。

#### 鼠标输入与相机反投影是首要技术风险

官方 click-and-drag 示例依赖 SAPIEN viewer 的交互能力，但它不等同于提供稳定、公开的“每帧鼠标像素 + 相机矩阵”高层 API。应先做 viewer 技术探针，确认：

- 能否持续取得鼠标窗口坐标以及窗口尺寸；
- 能否区分鼠标位于视口内、拖动相机和操作 UI 的状态；
- 能否从当前相机获得可靠的 view/projection 或等价标定信息；
- HiDPI、窗口缩放和坐标原点是否影响像素到射线的计算；
- viewer 事件轮询能否与 `env.step()` 保持非阻塞。

若当前 SAPIEN viewer 只提供不稳定的私有入口，第一版应增加一个很薄的 `PointerSource` 适配层，将具体 GUI 实现隔离；不要让环境和控制器直接依赖窗口库。

#### 动作归一化必须以实际 action space 为准

ManiSkill 的多数 Panda PD 控制器使用 `[-1, 1]` 归一化动作。代码不能假定动作前三维就是以米为单位的 delta，也不能硬编码夹爪索引和值。启动时应读取并打印 controller/action-space 配置，并用小幅单轴动作验证方向、尺度和夹爪维度。

## 3. 技术基线

### 3.1 环境

- 平台：优先 Linux + NVIDIA GPU；
- Python：首版固定 Python 3.11；
- 环境与锁文件：`uv`；
- 仿真：ManiSkill 稳定版，不使用 nightly；
- 首版：单环境、CPU physics、GUI rendering；
- 依赖版本由 `uv.lock` 固定，不在文档中硬编码未经验证的 ManiSkill/Torch 版本组合。

当前开发机检查结果（2026-07-27，已在沙箱外复核）：

- `uv 0.11.20` 可用；
- `uv` 项目使用 CPython `3.11.15`；
- NVIDIA GeForce RTX 4070 SUPER、驱动 `580.126.09`、CUDA `13.0` 可用；
- 系统未安装 `vulkaninfo` 命令，但 ManiSkill/SAPIEN GUI 实际渲染初始化成功。

当前锁定的关键依赖为 ManiSkill `3.0.1`、SAPIEN `3.0.3` 和
PyTorch `2.13.0+cu130`。100 步 GUI 冒烟测试已经覆盖 Panda、
viewer、鼠标读取、相机矩阵、目标 marker、IK/PD、动作和记录链路。

### 3.2 初始化命令

首次实现阶段执行，而不是在本次文档阶段安装：

```bash
uv init --python 3.11
uv add mani-skill torch
uv lock
uv run python -m mani_skill.examples.demo_random_action
```

说明：

- PyTorch 的 CPU/CUDA 构建应按开发机驱动和官方 PyTorch 安装源选择；不要在未检查驱动前盲目固定 CUDA wheel；
- ManiSkill 官方说明，状态仿真不要求额外渲染依赖，但 GUI rendering 需要可工作的 Vulkan 驱动；
- 建议将资产目录放在仓库外，通过 `MS_ASSET_DIR` 配置，避免大体积资产进入版本控制；
- 可以设置 `MS_SKIP_ASSET_DOWNLOAD_PROMPT=1`，使缺失资产在自动化测试中直接失败而不是阻塞等待输入。

环境验收：

```bash
uv run python -c "import mani_skill, torch; print(mani_skill.__version__); print(torch.__version__)"
vulkaninfo --summary
uv run python -m mani_skill.examples.demo_random_action
uv run python -m mani_skill.examples.teleoperation.interactive_panda -e PickCube-v1
```

最后一条用于验证 viewer、Panda 和鼠标交互基础能力，不作为本项目的实现入口。

## 4. 建议的仓库结构

```text
mani-sim/
├── pyproject.toml
├── uv.lock
├── README.md
├── BUILD.md
├── src/
│   └── mani_sim/
│       ├── __init__.py
│       ├── app.py
│       ├── config.py
│       ├── envs/
│       │   └── mouse_reach_env.py
│       ├── input/
│       │   ├── base.py
│       │   └── sapien_pointer.py
│       ├── mapping/
│       │   └── screen_to_plane.py
│       ├── control/
│       │   ├── command.py
│       │   ├── ee_servo.py
│       │   └── workspace_guard.py
│       ├── visualization/
│       │   └── target_markers.py
│       └── recording/
│           └── trajectory_recorder.py
├── configs/
│   └── demo0.yaml
├── scripts/
│   └── run_demo0.py
└── tests/
    ├── test_screen_to_plane.py
    ├── test_ee_servo.py
    └── test_workspace_guard.py
```

边界约定：

- `envs` 只负责场景、Panda、相机和 ManiSkill 生命周期；
- `input` 输出屏幕坐标或无输入状态，不产生机器人动作；
- `mapping` 是纯数学模块，便于无 GUI 单测；
- `control` 接收统一命令并输出符合当前 action space 的动作；
- `recording` 只观察数据，不改变控制；
- `app.py` 负责调度，不承载射线、IK 或安全算法。

统一命令接口建议为：

```python
@dataclass(frozen=True)
class TaskSpaceCommand:
    target_position: np.ndarray
    target_orientation: np.ndarray
    gripper_position: float
    timestamp: float
    valid: bool = True
```

即使 Demo 0 固定姿态，也保留 `target_orientation`，以便后续将鼠标输入替换为策略或六自由度设备时不改变上层协议。

## 5. Demo 0 数据流

```text
viewer mouse event
        |
        v
PointerSource -- invalid/outside viewport --> hold last safe target
        |
        v
pixel_to_world_ray(camera calibration)
        |
        v
ray_plane_intersection(z = work_height)
        |
        v
workspace bounds + table clearance
        |
        v
TaskSpaceCommand
        |
        v
bounded delta servo + saturation hysteresis
        |
        v
ManiSkill action adapter (arm + gripper)
        |
        v
env.step() -> actual TCP -> recorder/overlay/next servo step
```

### 5.1 射线与平面求交

设世界坐标系射线为：

```text
r(t) = o + t d
```

工作平面为 `z = z_work`，则：

```text
t = (z_work - o_z) / d_z
```

只有在 `|d_z|` 大于阈值且 `t > 0` 时交点有效。无效、鼠标离开视口或 UI 捕获鼠标时保持最后一个安全目标，不向机器人发送突变目标。

必须用测试覆盖：

- 视口中心像素的预期射线；
- 四角像素方向；
- 平行射线；
- 相机位姿变换；
- HiDPI framebuffer 与 window size 不同的缩放；
- NDC 的 Y 轴翻转。

### 5.2 连续末端伺服

推荐初始控制模式：

```text
pd_ee_delta_pos
```

每个控制周期：

```text
error = target_position - actual_tcp_position
requested_delta = gain * error
metric_delta = clip_norm(requested_delta, max_delta_m)
action_delta = controller_adapter.normalize(metric_delta)
```

初始参数只作为调试起点：

- `gain`: `0.5`；
- `max_delta_m`: `0.01 m/control step`；
- `deadband_m`: `0.002 m`；
- 控制频率目标：`20–50 Hz`，以实际环境配置为准；
- 仿真频率高于或等于控制频率，具体比例显式记录在配置中。

最终值必须通过阶跃响应和快速鼠标扫动验证，不能仅凭视觉感受确定。

### 5.3 饱和状态机

记录：

- 当前目标距离；
- 本步 TCP 实际位移；
- 最近若干步朝目标方向的投影进展；
- 连续低进展步数；
- 当前是否饱和。

进入饱和需要同时满足“目标仍远”和“连续多步低进展”；退出饱和需要目标明显改变方向或距离重新缩小，形成滞回，避免边界处逐帧切换。

饱和时优先保持最后一个稳定目标或显著降低 delta，不累计更大的内部目标。UI 同时显示：

- 红色：原始鼠标目标；
- 绿色：当前送入控制器的安全目标；
- 文本或颜色：`tracking` / `saturated` / `input_invalid`。

## 6. 配置面

`configs/demo0.yaml` 至少包含：

```yaml
simulation:
  sim_backend: cpu
  num_envs: 1
  control_freq_hz: 30

robot:
  uid: panda
  control_mode: pd_ee_delta_pos
  gripper_open: 1.0
  gripper_closed: -1.0

camera:
  name: demo_view
  width: 1280
  height: 720
  pose: null  # 实施时填写经验证的显式位姿

workspace:
  work_height_m: 0.25
  table_clearance_m: 0.02
  xy_bounds_m:
    x: [0.20, 0.80]
    y: [-0.50, 0.50]

servo:
  gain: 0.5
  max_delta_m: 0.01
  deadband_m: 0.002
  progress_epsilon_m: 0.0005
  saturation_steps: 10

input:
  gripper_toggle_key: space
```

这些边界值不是 Panda 真实工作空间模型，只是防止明显危险输入的粗约束。实现时以机器人基座坐标系定义并明确转换到世界坐标系。

## 7. 分阶段实施

### 阶段 A：环境与 API 探针

交付：

- `pyproject.toml`、`uv.lock` 和最小 README；
- 可启动的官方随机动作示例；
- Vulkan/viewer 验证记录；
- 一个临时只读探针，打印实际 action space、TCP pose、相机参数和鼠标事件；
- 明确使用的公开 API；任何不得不使用的私有 API 单独封装并记录版本风险。

通过标准：

- 新环境可由 `uv sync --locked` 重建；
- GUI 窗口能稳定运行至少 5 分钟；
- 鼠标坐标持续更新且不阻塞仿真；
- 已验证 framebuffer/window 坐标关系。

### 阶段 B：纯数学映射与基础场景

交付：

- 固定 Panda、平面、工作相机；
- 屏幕射线与工作平面求交模块；
- 原始目标可视化标记；
- 对投影数学的无 GUI 单元测试。

通过标准：

- 已知相机姿态下的像素到平面结果满足数值误差要求；
- 调整窗口大小后映射仍正确；
- 鼠标离开视口不会产生跳变目标。

### 阶段 C：连续伺服与夹爪

交付：

- action adapter；
- 有界 delta servo；
- 空格键切换内部绝对夹爪目标；
- 目标与实际 TCP 叠加显示；
- 控制器单元测试和运行脚本。

通过标准：

- 可达区域内连续跟随，无 NaN、关节越界或明显持续振荡；
- 快速扫动鼠标时动作始终落在 action space 内；
- 停止移动鼠标后 TCP 收敛到 deadband；
- UI 事件和仿真循环不互相阻塞。

### 阶段 D：不可达保护与记录

交付：

- 饱和滞回状态机；
- 红/绿目标及状态显示；
- JSONL 或 NPZ 轨迹记录；
- 确定性回放所需的配置、seed 和版本元数据。

每步至少记录：

```text
timestamp, mouse_pixel, raw_target_world, safe_target_world,
actual_tcp_pose, action, gripper_target, saturation_state,
qpos, qvel
```

通过标准：

- 不可达目标下不会持续高频抖动；
- 从不可达区返回后可自动恢复跟随；
- 连续运行 10 分钟无 NaN 或失控；
- 记录数据长度、形状和时间戳单调性可自动验证。

### 阶段 E：研究脚手架稳定化

交付：

- `PointerSource` 与 `PolicySource` 使用同一 `TaskSpaceCommand`；
- headless smoke test；
- 配置校验、seed、日志和异常退出清理；
- CI 中不依赖 GPU 的数学和控制逻辑测试。

完成后再进入 Demo 1（滚轮高度）、Demo 2（PushCube 接触）和 Demo 3（抓取）。

## 8. 测试与验收指标

### 自动测试

- `screen_to_plane` 的确定性数值测试；
- delta 限幅、deadband 和 action normalization；
- 饱和状态进入/退出的滞回；
- 无效鼠标输入保持最后安全目标；
- 配置解析和单位检查；
- recorder schema、数组形状和时间戳。

### 手工验证

- 慢速圆周、快速水平扫动、窗口四角和鼠标离窗；
- 目标跨越粗工作区边界；
- 持续指向明显不可达位置；
- 在饱和状态下快速切回可达区；
- 夹爪多次切换；
- 窗口缩放以及相机交互冲突；
- 5 分钟常规运行和 10 分钟边界压力运行。

### 建议量化指标

- 控制循环实际频率及 P50/P95 周期；
- 鼠标事件到目标更新延迟；
- 可达目标的稳态 TCP 误差；
- 阶跃目标的上升和稳定时间；
- 边界状态下实际 TCP 速度峰值；
- action clipping 比例；
- dropped/invalid pointer event 数量。

首轮不要预设脱离硬件的严格毫秒阈值。阶段 A 测出基线后，在配置和测试说明中固定合理门槛。

## 9. 主要风险与决策

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 当前会话无可用 NVIDIA 驱动 | 无法验证 GUI/Vulkan | 在图形会话先跑 `vulkaninfo` 和官方 viewer 示例 |
| viewer 鼠标 API 不稳定 | 版本升级易破坏 | `PointerSource` 隔离；阶段 A 锁定并记录 API |
| 窗口坐标与 framebuffer 坐标不一致 | 射线方向错误 | 显式缩放、Y 翻转测试和角点标定 |
| normalized action 被误当作米 | 运动尺度错误 | action adapter + 启动探针 + 单轴校准 |
| 不可达点导致抖动 | 体验差且污染数据 | 每步限幅、进展检测、滞回和目标冻结 |
| 固定姿态限制实际工作空间 | 边界比预期小 | 作为 Demo 0 明确约束；后续再加入姿态控制 |
| GUI 与控制循环耦合 | 低帧率或输入延迟 | 记录循环时序；保持单线程最小实现，必要时再解耦输入采样 |
| 直接复用任务环境带入多余逻辑 | 脚手架边界混乱 | 构建最小自定义环境，仅借用官方 Panda agent/controller |

## 10. 首期明确不做

- 不逐帧写 `qpos` 实现几何锁定；
- 不对每次鼠标移动调用 motion planner；
- 不把工作空间近似成单一球体并宣称严格可达；
- 不在验证 API 前调用 ManiSkill/SAPIEN 私有 IK 或窗口内部对象；
- 不引入 ROS、额外 GUI 框架或新的运动规划依赖；
- 不在 Demo 0 中同时解决高度、姿态、碰撞抓取和策略训练；
- 不把渲染资产或大规模数据提交进 Git。

## 11. 开发启动顺序

下一次开始实现时严格按以下顺序：

1. 初始化 `uv` 项目并锁定环境；
2. 验证 Vulkan、官方随机动作和 `interactive_panda`；
3. 完成 viewer/action/camera 技术探针；
4. 根据探针结果冻结 Demo 0 的公开接口；
5. 先写 `screen_to_plane`、servo 和 saturation 的单元测试；
6. 构建最小场景并接入目标可视化；
7. 接入连续伺服；
8. 最后接入饱和保护、记录和压力验证。

## 12. 参考资料

- [ManiSkill Installation](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html)
- [Controllers / Action Spaces](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html)
- [Teleoperation](https://maniskill.readthedocs.io/en/latest/user_guide/data_collection/teleoperation.html)
- [interactive_panda.py](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/examples/teleoperation/interactive_panda.py)

## 13. Demo 0.5 可达域改进结果

### 13.1 已实现

2026-07-27 在 Demo 0 基础上完成：

- 新增 `mani-sim-calibrate`，使用 ManiSkill CPU Pinocchio IK 对固定 TCP
  姿态进行无副作用网格采样；
- 标定文件记录机器人、控制模式、初始关节状态、固定 TCP 姿态、网格间距
  以及各高度的可达点；
- Demo 启动时加载标定文件，将不可达鼠标目标投影到同高度可达域；
- 可达网格单元内部保留连续鼠标目标，仅在目标离开可达单元时投影，避免
  所有运动都出现 2.5 cm 量化跳变；
- 保留原有 stall 检测作为第二层保护，处理离散网格边界误差和实际 PD
  跟踪失败；
- 相机改为固定俯视视角，并将屏幕宽方向对齐工作区较长的 Y 方向。当前
  视野约覆盖 `x=[0.13, 0.77]`、`y=[-0.58, 0.58]`，不再产生原斜视
  相机在近平行射线处出现的数十米远目标。

运行：

```bash
uv run mani-sim-calibrate
uv run mani-sim --config configs/demo0.yaml
```

### 13.2 多高度标定结果

采样范围：

- `x=[0.15, 0.75] m`；
- `y=[-0.55, 0.55] m`；
- 网格间距 `0.025 m`；
- 每层 `1125` 个固定姿态 TCP 目标；
- Panda 使用 Demo 0 的确定性初始关节状态。

| TCP 高度 | 可达点 | 可达率 | 中心线最远 X |
| --- | ---: | ---: | ---: |
| `0.45 m` | 920 / 1125 | **81.8%** | `0.725 m` |
| `0.55 m` | 789 / 1125 | 70.1% | `0.650 m` |
| `0.65 m` | 588 / 1125 | 52.3% | `0.575 m` |

结论：

- 当前固定末端姿态下，`z=0.45 m` 的覆盖最好，因此不提高默认工作高度；
- 不可达区主要位于远端角落，真实边界不是矩形或单一半径；
- `z=0.45 m` 的中心线上 `x=0.75 m` 不可达，最近标定目标为
  `x=0.725 m`；
- 估计的屏幕远端角落 `(0.767, ±0.577, 0.45)` 会投影到约
  `(0.55, ±0.475, 0.45)`，投影距离约 `0.24 m`。

### 13.3 验证结果

- 15 项单元测试通过；
- 100 步启用可达域的 GUI 集成测试通过；
- 记录动作全部位于 `[-1, 1]`，无 NaN 或 Infinity；
- 当前鼠标样点映射为 `(0.324, -0.443, 0.45)`，位于可达单元内，
  因而保持连续目标、没有发生网格吸附；
- PyTorch CUDA 与 RTX 4070 SUPER 已在沙箱外验证可用。

### 13.4 已知限制

- 标定结果只对当前初始关节状态、固定 TCP 姿态和 Panda 基座位姿有效；
- “IK 返回成功”只表示运动学可解，尚未检查沿途自碰撞、桌面碰撞和关节
  余量；
- 最近离散点投影不保证鼠标方向上的最大可达点，也可能在非凸边界附近
  横向跳到另一支 IK 区域；
- 当前 2.5 cm 网格适合 Demo 体验，不适合高精度接触任务；
- 标定尚未缓存多组 IK seed，因此可能漏掉需要不同初始解的可达区域。

## 14. 下一步

下一阶段建议优先做“连续可达边界与稳定性”，暂不进入高度滚轮或抓取：

1. 将 2.5 cm 可达点构造成保守的二维占据栅格/连通区域，过滤孤立 IK 点；
2. 在原始鼠标目标与当前 TCP 的射线上二分搜索最后可达点，使不可达目标
   表现为朝鼠标方向伸展到边界，而不是欧氏最近点横向吸附；
3. 对边界 IK 解增加关节限位余量、FK 残差和解连续性检查；
4. 记录 `raw/projected/tcp` 三条轨迹，量化投影跳变、稳态误差和边界抖动；
5. 完成上述稳定性后，再实现滚轮控制 Z，并为每个高度层做插值或在线投影。

## 15. 连续边界与严格标定实施结果

### 15.1 关键根因修复

实施严格关节限位校验时发现，内置 `Empty-v1` 不会给 Panda 设置桌面任务
使用的初始关节状态。原 Demo 实际从 9 维全零 `qpos` 启动，其中
`panda_joint4=0` 已超出其合法上限 `-0.0698 rad`。这会造成：

- Pinocchio 以非法种子求出大量关节越界解；
- 离线 IK 将目标判断为可达，但物理关节驱动无法执行；
- TCP 在看似可达的绿色目标前停止，触发 stall；
- 旧版 81.8% 可达率混入了不可执行解。

现在启动、键盘复位、episode 自动复位和标定器统一调用
`initialize_panda()`，使用 ManiSkill `TableSceneBuilder` 的标准 Panda 姿态：

```text
[0, pi/8, 0, -5pi/8, 0, 3pi/4, pi/4, 0.04, 0.04]
```

200 步 GUI 集成测试中，实际关节距离最近限位仍有 `0.924 rad`，没有
关节越界。

### 15.2 严格可达性判定

每个采样目标现在必须依次满足：

1. Pinocchio IK 返回成功；
2. 7 个臂关节距离上下限至少 `0.02 rad`；
3. 将 IK 解带回 Pinocchio FK 后，位置残差不超过 `0.002 m`；
4. 固定姿态残差不超过 `0.02 rad`。

重新标定结果：

| TCP 高度 | 严格可达 | IK 失败 | 限位余量拒绝 | FK 拒绝 | 最大 FK 位置残差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0.45 m` | 920 | 205 | 0 | 0 | `0.0001003 m` |
| `0.55 m` | 789 | 336 | 0 | 0 | `0.0001002 m` |
| `0.65 m` | 588 | 537 | 0 | 0 | `0.0000999 m` |

修复初始姿态后，所有 IK 成功解均满足当前限位余量与 FK 阈值。相比旧地图，
`z=0.65 m` 少了 2 个不可执行点。

### 15.3 主连通域与沿射线投影

运行时会对每个高度层做 8 邻域连通分量分析，只保留最大连通区。当前三层
分别保留 `920/789/588` 点，没有孤立点被移除。

不可达目标不再直接吸附到欧氏最近点。算法现在：

1. 从当前 TCP 的 XY 位置朝鼠标目标均匀推进；
2. 找到进入主可达区后的第一个离开位置；
3. 在最后可达与最先不可达之间执行 12 次二分；
4. 返回射线上的边界目标；
5. 只有射线完全没有穿过可达区时才使用最近点兜底。

从 `(0.3, 0, 0.45)` 指向屏幕远端角落时：

```text
mouse  = (0.767, ±0.577, 0.45)
target = (0.613, ±0.387, 0.45)
method = ray_boundary
```

目标保持在鼠标方向上，不再横向吸附到 `(0.55, ±0.475)`。

### 15.4 验证

- 18 项单元测试通过；
- 覆盖主连通域过滤、射线方向保持和边界二分；
- 200 步 GUI 集成测试通过；
- 轨迹全程有限值，关节未越界；
- 固定鼠标目标下观测到最小 TCP 跟踪误差 `0.0015 m`；
- 记录新增 `projection_method`、`tracking_error_m` 和
  `safe_target_step_m`，可用于量化投影跳变与跟踪质量。

### 15.5 下一步

下一阶段建议做边界稳定性和更细精度，而不是立刻增加新自由度：

1. 将边界附近的 2.5 cm 网格自适应细化到 5 mm，内部区域保持粗网格；
2. 让射线起点使用“上一帧安全目标”与实际 TCP 的连续组合，减少快速鼠标
   运动时边界随 TCP 滞后漂移；
3. 用录制的 `safe_target_step_m` 检测并抑制非凸边界处的目标跳跃；
4. 加入实际物理步进后的可达回归测试，而不只依赖 IK/FK；
5. 稳定后再进入滚轮 Z 控制和跨高度层连续插值。

## 16. 自适应边界与物理回归实施结果

### 16.1 25 mm 全域 + 5 mm 边界细化

标定文件升级为 schema v2。每个高度层先执行原有 25 mm 全域严格采样，
然后找出“至少有一个 8 邻域不可达”的粗可达边界单元，仅在这些单元周围
追加 5 mm 网格采样。内部区域仍使用 O(1) 粗栅格查询，边界查询优先使用
细栅格结果。

| 高度 | 粗样本 | 粗可达 | 5 mm 边界样本 | 细样本可达 |
| --- | ---: | ---: | ---: | ---: |
| `0.45 m` | 1125 | 920 | 5505 | 4514 |
| `0.55 m` | 1125 | 789 | 5375 | 4397 |
| `0.65 m` | 1125 | 588 | 5385 | 4182 |

新版标定文件约 `1.20 MB`。三层主连通域仍分别保留
`920/789/588` 个粗点，没有发现需要删除的孤立粗 IK 点。

边界搜索步长由原来的 `6.25 mm`（25 mm 的四分之一）降低到
`2.5 mm`，随后继续执行 12 次二分，因此离散边界判断的主要误差现在来自
5 mm 占据单元，而不是射线采样间隔。

### 16.2 混合射线起点

射线不再完全从滞后的实际 TCP 出发。默认起点为：

```text
origin = 0.7 * previous_safe_target + 0.3 * actual_tcp
```

这样仍保留实际机械臂状态的反馈，但快速移动鼠标时，边界方向主要继承上一帧
已经确认的安全目标，减少 TCP 追踪滞后造成的边界漂移。权重可通过
`reachability.previous_safe_target_weight` 调整。

### 16.3 投影跳变抑制

当目标来自 `ray_boundary` 或 `nearest_fallback` 投影时，相邻安全目标
最大移动量默认限制为 `0.03 m`。限幅后的候选点必须仍在标定可达域内，
否则保持上一安全目标。

100 步 GUI 测试中：

- 96 步鼠标目标位于可达域内；
- 4 步触发 `ray_boundary`；
- 这 4 步均触发跳变抑制；
- 抑制后的每步安全目标位移均严格为 `0.03 m`；
- GUI、IK/PD、marker 和 JSONL 记录链路正常。

轨迹新增：

```text
projection_origin_world
projection_suppressed
projection_method
safe_target_step_m
tracking_error_m
```

### 16.4 实际物理步进回归

新增 GPU/Vulkan 集成测试：

```bash
uv run pytest -q -s tests/integration/test_physical_reachability.py
```

测试对 5 个代表性目标分别复位到标准 Panda 安全姿态，执行 240 个真实
`env.step()`，经过 IK、关节 PD 和 CPU PhysX 后检查最终 TCP，而不是只检查
离线 IK/FK。

| 目标 `(x, y, z) m` | 最终 TCP 误差 |
| --- | ---: |
| `(0.30, 0.00, 0.45)` | `0.902 mm` |
| `(0.55, 0.00, 0.45)` | `0.728 mm` |
| `(0.45, 0.35, 0.45)` | `0.567 mm` |
| `(0.45, -0.35, 0.45)` | `0.592 mm` |
| `(0.70, 0.00, 0.45)` | `0.639 mm` |

最大误差 `0.902 mm`，低于测试门槛 `15 mm`。测试同时会在没有外部
NVIDIA/Vulkan 设备的沙箱内自动跳过。

当前验证汇总：

- 20 项普通测试通过；
- 1 项 GPU 物理回归通过；
- 100 步 GUI 自适应地图测试通过；
- `uv lock --check` 和 Python bytecode 编译通过。

### 16.5 下一步：滚轮 Z 与跨层连续插值

边界稳定性阶段完成后，下一阶段可以进入 Demo 1：

1. 鼠标滚轮改变 TCP 高度，并设置每帧与总高度限幅；
2. 不直接跳到最近的 `0.45/0.55/0.65 m` 层，而是在相邻层之间插值可达
   边界；
3. 标定高度层加密到 5 cm 间隔，必要时在局部在线验证 IK；
4. 保持固定姿态，先验证三维位置跟踪和桌面安全距离；
5. 将物理回归扩展为跨高度轨迹，检查垂直运动、边界收缩和层切换连续性。

## 17. 高度控制与多机位界面决策

### 17.1 输入键位

高度控制采用：

```text
U：按住持续升高
J：按住持续降低
```

没有使用 W/S，因为 SAPIEN viewer 默认将其用于相机前后移动；没有使用
Q/E，因为 Q 已作为退出键。U/J 与现有 `Space/R/Q/1/2/3` 不冲突，
也沿用 ManiSkill 官方 Panda 交互示例中上/下移动的直观含义。

内部按速度积分，而不是按键事件产生固定跳变：

```text
z_next = clip(z + direction * 0.12 m/s * control_dt, 0.45, 0.65)
```

同时按下 U/J 时方向抵消。高度、按键速度和上下界均为 YAML 配置。

### 17.2 当前相机与控制模式

当前阶段仍只实现一种鼠标控制模式：

```text
1：TOP XY，默认且唯一接收鼠标控制
```

主 viewport 保持固定俯视相机，鼠标控制世界 X/Y，U/J 控制世界 Z。
前视 XZ 的输入映射暂不实现，避免在 Pick/Pull 场景和控制平面尚未确定前
引入第二套交互语义。

预留模式：

```text
2：FRONT XZ，当前只观察，后续鼠标控制 X/Z、按键控制深度
3：WRIST，始终只观察，不直接接管鼠标
```

### 17.3 单窗口多机位

机器人切换为 `panda_wristcam`，与普通 Panda 使用相同安全初始关节状态和
末端控制器。当前注册三台相机：

- 主 viewer 相机：固定俯视控制视图；
- `front_observer`：前方斜视固定相机；
- `hand_camera`：ManiSkill Panda wrist camera。

SAPIEN 3.0.3 没有现成的多 viewport 主渲染布局，但原生 ImGui 支持
`UIPicture`。因此实现为同一个 RenderWindow 内：

- 主场景占据主 viewport；
- 右侧辅助面板同时渲染 FRONT 和 WRIST；
- 不创建 OpenCV 或第二个 SAPIEN 窗口；
- 面板按 `2/3` 显示 ACTIVE 标签，但当前不接受鼠标目标；
- 鼠标位于辅助面板上时输入无效，保持上一安全目标。

ManiSkill 会将实际相机名加上 `scene-0_` 前缀，面板使用后缀匹配
`front_observer/hand_camera`。探针确认 viewer 中存在：

```text
scene-0_base_camera
scene-0_hand_camera
scene-0_render_camera
scene-0_front_observer
```

### 17.4 切换时目标不跳变

active view 状态独立于 `TaskSpaceCommand`：

- 从 1 切换到 2/3：立即停止接收鼠标，保持上一世界坐标安全目标；
- 从 2/3 切回 1：记录切换时鼠标像素；
- 鼠标移动不足 3 px：继续保持目标；
- 鼠标明确移动后：俯视 XY 重新接管。

因此按 `1/2/3` 本身不会修改 TCP 目标。该状态机已用纯单元测试覆盖。

### 17.5 跨高度连续可达投影

运行高度允许在 `0.45–0.65 m` 连续变化。目标 Z 位于两个标定层之间时：

1. 在上下两个高度层分别执行自适应边界射线投影；
2. 按 Z 的层间比例线性插值两个边界 XY；
3. 保持目标的连续 Z；
4. 继续应用 3 cm 投影跳变抑制和 stall 保护。

例如 `z=0.50 m` 会在 `0.45/0.55 m` 两层之间插值，不会突然吸附到其中
一层。单元测试已验证上下层边界不同时，中间高度边界位于两者之间。

### 17.6 验证结果

- 27 项普通测试通过；
- 1 项 GPU 物理回归通过；
- 50/100 步单窗口多相机 GUI 冒烟测试通过；
- viewer 已确认同时注册前视和腕部相机；
- `panda_wristcam` 重新标定后，三层粗/细可达统计与 Panda 保持一致；
- 物理回归扩展为 7 个目标，包括 `z=0.50/0.60 m` 两个插值高度。
- 高度端层使用 `1 mm` 容差，避免射线求交的浮点误差将
  `0.44999999 m` 误判为标定范围外。

七个目标最终 TCP 误差：

```text
0.888, 0.668, 0.627, 0.660, 0.818, 0.670, 0.758 mm
```

最大误差 `0.888 mm`。

### 17.7 front_xz 备用设计

后续实现 FRONT XZ 时保持同一个世界坐标 `TaskSpaceCommand`，仅替换输入
映射：

```text
mouse horizontal -> world X
mouse vertical   -> world Z
U/J or新深度键   -> world Y
```

实施前需要先确定 Pull/Drawer 场景的正方向、相机是否镜像 X、深度键语义
以及遮挡处理。TOP XY、FRONT XZ 和 WRIST 不应拥有各自独立的 TCP 目标。

### 17.8 下一步

下一步进入任务场景前的最后一轮交互验证：

1. 人工操作 U/J 和 1/2/3，确认按键手感、面板尺寸和鼠标屏蔽区域；
2. 记录一段同时包含 XY、Z 和视图切换的轨迹，检查目标连续性；
3. 根据体验调整 `vertical_speed_mps=0.12` 和辅助面板尺寸；
4. 验证后进入 PushCube，再进入 Pick-and-Place；
5. FRONT XZ 等到 Pull/Drawer 场景开始时实现。

## 18. 低高度与地面物体可行性探针

### 18.1 严格 IK/FK 标定

在不覆盖正式标定文件的情况下，临时采样
`z=0.05/0.10/0.15/0.20/0.30/0.45 m`，其余条件与 schema v2 正式
标定一致：

| TCP 高度 | 粗可达点 | 可达率 |
| --- | ---: | ---: |
| `0.05 m` | 967 / 1125 | 86.0% |
| `0.10 m` | 961 / 1125 | 85.4% |
| `0.15 m` | 961 / 1125 | 85.4% |
| `0.20 m` | 967 / 1125 | 86.0% |
| `0.30 m` | 1013 / 1125 | 90.0% |
| `0.45 m` | 920 / 1125 | 81.8% |

固定向下姿态在低位没有明显运动学覆盖问题；`0.15 m` 以下的可达率甚至
略高于当前默认 `0.45 m`。

### 18.2 实际物理步进与地面接触

在 `(x=0.4, y=0)` 对各高度分别执行 300 个真实 IK + PD + PhysX
控制步，并逐链统计与 ground 的接触力：

| TCP 目标高度 | 最终误差 | 非基座地面接触 |
| --- | ---: | --- |
| `0.05 m` | `0.705 mm` | 无 |
| `0.10 m` | `0.877 mm` | 无 |
| `0.15 m` | `0.862 mm` | 无 |
| `0.20 m` | `0.764 mm` | 无 |
| `0.30 m` | `0.930 mm` | 无 |
| `0.45 m` | `0.663 mm` | 无 |

补充下限探针：

| TCP 目标高度 | 结果 |
| --- | --- |
| `0.04 m` | 无地面接触 |
| `0.03 m` | 无地面接触 |
| `0.02 m` | 无地面接触 |
| `0.00 m` | 左右手指峰值约 `49.2/55.6 N`，TCP 被顶在 `z≈0.0119 m` |

### 18.3 结论

- 物体可以直接放在 `z=0` 的地面上；
- TCP 目标应该是物体的抓取中心，而不是地面。例如 4 cm 高物体的中心约为
  `z=0.02 m`；
- `z=0` 不应作为正常 TCP 下限，因为它会把手指压入地面；
- 当前单点实测支持将下限扩展到 `0.03 m`，保守默认建议先用 `0.05 m`；
- 正式扩大范围前仍需在整个 XY 边界做低位碰撞采样，而不能仅依赖中心点；
- 加入具体物体后，下限应为 `object_grasp_height` 或
  `surface_height + clearance`，而不是一个对所有任务固定的世界 Z。

下一步若批准扩大正式范围，建议重新标定
`z=0.05:0.05:0.65 m`，将 YAML 下限改为 `0.05 m`，并增加低位 XY
边界的物理碰撞回归。

## 19. 正式扩展至地面物体工作高度

### 19.1 配置与标定

正式连续高度范围已由 `0.45–0.65 m` 扩展为 `0.05–0.65 m`。
`mani-sim-calibrate` 默认按 5 cm 间隔生成 13 个高度层，每层保持
2.5 cm 粗网格与边界附近 5 mm 自适应细化：

| TCP 高度 | 粗可达点 | 可达率 | 边界细化点 |
| --- | ---: | ---: | ---: |
| `0.05 m` | 967 / 1125 | 86.0% | 5915 |
| `0.10 m` | 961 / 1125 | 85.4% | 5915 |
| `0.15 m` | 961 / 1125 | 85.4% | 5915 |
| `0.20 m` | 967 / 1125 | 86.0% | 5915 |
| `0.25 m` | 986 / 1125 | 87.6% | 5755 |
| `0.30 m` | 1013 / 1125 | 90.0% | 5375 |
| `0.35 m` | 1003 / 1125 | 89.2% | 5110 |
| `0.40 m` | 967 / 1125 | 86.0% | 5410 |
| `0.45 m` | 920 / 1125 | 81.8% | 5505 |
| `0.50 m` | 861 / 1125 | 76.5% | 5490 |
| `0.55 m` | 789 / 1125 | 70.1% | 5375 |
| `0.60 m` | 700 / 1125 | 62.2% | 5460 |
| `0.65 m` | 588 / 1125 | 52.3% | 5385 |

正式地图已写入 `calibrations/panda_fixed_orientation.json`。运行时在相邻
高度层间连续插值，因此 U/J 不会按 5 cm 离散跳层。

### 19.2 低位 XY 边界物理回归

GPU 集成测试加入 `z=0.05 m` 的中心、内侧、前伸和左右边界目标：

```text
(0.40,  0.00), (0.30,  0.00), (0.55,  0.00),
(0.40,  0.40), (0.40, -0.40)
```

这些点均先通过正式可达地图检查，再执行真实 IK、PD 与 PhysX 步进。连同
原有跨高度目标共 12 个目标，最终 TCP 误差为：

```text
0.888, 0.668, 0.627, 0.660, 0.818, 0.670,
0.758, 0.705, 0.816, 0.693, 0.704, 0.676 mm
```

最大误差为 `0.888 mm`；所有 `z=0.05 m` 目标运动过程中，Panda
非基座链节与地面的峰值接触力为 `0 N`。

### 19.3 验证与结论

- 27 项普通测试通过，1 项 GPU 测试在无外部设备的沙箱内按预期跳过；
- 外部 RTX 4070 SUPER 上的 GPU 物理回归通过；
- 100 步 GUI 冒烟通过，启动时正确加载全部 13 个高度层；
- 正式下限现为 `0.05 m`，足以控制地面上常见小物体的抓取中心；
- `z=0` 仍不是安全 TCP 命令，物体任务应叠加物体抓取高度和表面余量。

### 19.4 下一步

下一阶段进入第一个物体任务时，应把“地图可达”与“场景无碰撞”分开：

1. 加入地面立方体及尺寸、抓取中心和安全余量配置；
2. 为物体和桌面障碍增加在线碰撞/接近保护，不能只依赖空场景可达地图；
3. 录制包含 XY、Z、视图切换和夹爪状态的完整轨迹；
4. 先验证接近、闭合和抬升三个阶段，再构建完整 Pick-and-Place；
5. 保留 FRONT XZ 设计，等 Pull/Drawer 任务需要深度控制时再启用。
