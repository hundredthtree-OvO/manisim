# mani-sim

基于 ManiSkill 3 的鼠标连续控制 Panda 研究脚手架。默认场景包含地面上的
4 cm 动态目标立方体：鼠标决定 TCP 的 XY 目标，U/J 连续控制高度，
控制器通过 IK + PD 追踪；可达域、场景几何和接触力共同保护命令。

## 环境

要求 Linux 图形会话、可用的 NVIDIA/Vulkan 驱动以及 `uv`。

```bash
uv sync --locked
```

当前锁定的关键版本是 ManiSkill 3.0.1、SAPIEN 3.0.3、PyTorch 2.13.0。

## 运行

### 人工鼠标采集

固定场景用于熟悉控制和调试：

```bash
uv run mani-sim --config configs/demo0.yaml
```

与自动策略使用相同位置分布的正式人工采集：

```bash
uv run mani-sim --config configs/manual_randomized.yaml
```

随机人工模式会在首次启动时使用 `simulation.seed`，每次按 `R` 封闭当前
episode、将 seed 加一，并重新生成 Cube 和目标区。状态面板显示
`EPISODE SEED` 和 `SCENE: RANDOMIZED`；轨迹也记录 `episode_seed`、
`cube_initial_position` 和 `goal_position`。因此人工和自动记录中相同 seed
对应相同初始场景，可以直接配对比较。

操作：

- 移动鼠标：移动水平工作平面上的 TCP 目标；
- `U` / `J`：持续升高 / 降低 TCP，范围为 `0.02–0.65 m`；
- `Space`：切换夹爪开合；
- `1`：俯视 XY 主控制视图；
- `2`：前视 XZ 控制视图；
- `3`：高亮腕部观察视图，冻结鼠标目标；
- `R`：复位；
- `Q`：退出。

首次启动和按 `R` 后，控制目标会保持在 Panda 的实际 TCP，不会立即跳到
其他高度。当前初始化 TCP 高度约为 `0.45 m`，与默认工作高度一致。
窗口稳定 2 个控制步后，鼠标需要移动至少 3 像素才重新接管 XY；
在此之前按 U/J 只会沿当前 TCP 的 XY 垂直移动。复位同时将夹爪命令恢复为
张开状态。

按 `1/2` 会将主 viewport 切换为 TOP/FRONT，相应模式都使用完整主窗口
接收鼠标。按 `1` 时右侧预览 FRONT；按 `2` 时右侧自动改为预览 TOP；
第二张小窗始终是 WRIST。辅助预览不接收控制。切换后需要再次移动至少
3 像素才重新接管，因此切换本身不会改变世界 TCP 目标。WRIST 始终只观察。

FRONT 模式映射为：

```text
鼠标向右/左：世界 X 增大/减小
鼠标向上/下：世界 Z 增大/减小
U / J：世界 Y 增大/减小（远离/靠近前视相机）
```

TOP 与 FRONT 共用同一个世界 TCP 目标。FRONT 调整的高度会成为切回 TOP
后的控制平面高度；TOP 调整的 Y 也会成为 FRONT 的深度平面。

右下状态面板统一显示 active view、当前轴映射、TCP、非预期接触力/阈值、
安全状态和记录状态。任务区域由当前 Task 注入；Pick-and-Place 显示阶段、
抓取状态、目标距离和成功状态。状态面板不接收鼠标控制。

红点是原始鼠标目标，绿点是经过可达域投影和稳定保护后的控制目标。

### 自动 Pick-and-Place

```bash
uv run mani-sim --config configs/scripted_pick_place.yaml
```

自动模式仍会打开同一个可视化窗口。机械臂由确定性航点策略控制，TOP、FRONT、
WRIST 画面、目标标记、任务状态和实时力曲线都会继续刷新；`1/2/3` 仍可切换
观察视图，`R` 可重新开始当前 episode，`Q` 退出。鼠标、`U/J` 和 `Space`
不会覆盖自动策略。

自动配置默认启用窄范围、可复现的位置随机化：每条 episode 根据独立 seed
重新生成方块 XY 和目标区 XY，同时强制两者至少间隔 `0.18 m`。当前范围经过
固定向下夹爪的真实物理回归，适合先验证采集链路；它还不包含尺寸、质量、
摩擦、光照或障碍随机化。

[manual_randomized.yaml](configs/manual_randomized.yaml) 与
[scripted_pick_place.yaml](configs/scripted_pick_place.yaml) 使用完全相同的
位置范围和最小距离约束；两者只在 `collection.source` 和 episode 推进方式上
不同。修改实验分布时应同步修改并运行配置测试，避免人工/自动数据悄然失配。

状态面板的 `source` 显示 `scripted_pick_place`，`policy phase` 依次经过
`approach -> descend -> close -> lift -> transport -> lower -> open ->
retreat`。成功后默认保留 25 个稳定步，自动封闭当前 episode、复位场景并开始
下一条；单条超过 800 步会以 `policy_timeout` 结束。这两个值可在
[scripted_pick_place.yaml](configs/scripted_pick_place.yaml) 的
`collection.success_settle_steps` 和 `collection.max_episode_steps` 修改。

两种模式都使用相同的 `RuntimeObservation -> TaskSpaceCommand ->
CommandExecutor` 安全执行链，并写入正式 session：

```text
runs/<session-id>/
├── metadata.json
├── episodes.jsonl
└── episodes/episode_*.jsonl
```

每帧记录包含 `action_source` 和 `policy_phase`，episode 索引包含
`success`、`policy_timeout`、手动复位或退出等结束原因。

批量采集 20 条并在完成后自动生成统计：

```bash
uv run mani-sim --config configs/scripted_pick_place.yaml --episodes 20
```

结果写入当前 session 的 `summary.json`，包含成功率、结束原因、平均步数、
各策略阶段平均耗时，以及 grip/object/unintended 三类力的均值、峰值和
P95。已有 session 也可以重新统计：

```bash
uv run mani-sim-report runs/<session-id>
```

立方体默认位于 `(0.45, 0.0)`，边长 4 cm，小于 Panda 约 8 cm 的最大
夹爪开口。建议先移动到立方体正上方，再用 `J` 下降至抓取中心
`z=0.02 m`，按 `Space` 闭合夹爪，最后按 `U` 抬升。红色鼠标目标是
无碰撞的可视标记，可以进入目标立方体；绿色安全目标和实际机械臂运动仍
受可达域及 PhysX 接触约束。静态障碍默认关闭，可在 YAML 中开启。

绿色方形区域是默认放置目标，中心位于 `(0.30, 0.30)`。抓起后保持夹爪
闭合，将方块移动到绿色区域上方，再降至 `z=0.02 m`；按 `Space` 张开，
随后按 `U` 向上撤离。方块进入 4 cm XY 容差、夹爪释放且方块落稳到地面
高度后，任务阶段记为 `placed`。

配置入口为 [configs/demo0.yaml](configs/demo0.yaml)。完整设计、阶段划分和风险说明见 [BUILD.md](BUILD.md)。

## 重新标定可达域

当固定姿态、初始关节状态或工作区发生变化时，重新生成可达域：

```bash
uv run mani-sim-calibrate
```

默认包含抓取中心层 `z=0.02 m`，并对 `z=0.05–0.65 m` 按 5 cm 间隔
生成其余 13 个高度层，逐层进行
2.5 cm 全域 IK 采样，并在检测出的边界周围自适应追加 5 mm 采样，结果写入
`calibrations/panda_fixed_orientation.json`。运行 Demo 时，矩形边角等
不可达鼠标目标会沿混合射线起点投影到最后一个已标定可达边界；只有射线
未穿过主可达区时才退化为最近点投影。

## 验证

```bash
uv run pytest
uv run mani-sim --max-steps 100
uv run mani-sim --config configs/scripted_pick_place.yaml --max-steps 700
```

后两条需要图形会话；第三条还能覆盖自动抓取、放置、成功复位和可视化路径。

真实 Panda IK + PD 物理回归需要 NVIDIA/Vulkan 图形环境：

```bash
uv run pytest -q -s tests/integration
```
