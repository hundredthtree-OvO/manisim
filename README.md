# mani-sim

基于 ManiSkill 3 的鼠标连续控制 Panda 研究脚手架。Demo 0 在空场景水平平面上运行单机械臂：鼠标位置决定固定高度的 TCP 目标，控制器通过 IK + PD 连续追踪；不可达区域由进展检测和饱和滞回保护。

## 环境

要求 Linux 图形会话、可用的 NVIDIA/Vulkan 驱动以及 `uv`。

```bash
uv sync --locked
```

当前锁定的关键版本是 ManiSkill 3.0.1、SAPIEN 3.0.3、PyTorch 2.13.0。

## 运行

```bash
uv run mani-sim --config configs/demo0.yaml
```

操作：

- 移动鼠标：移动水平工作平面上的 TCP 目标；
- `U` / `J`：持续升高 / 降低 TCP，范围为 `0.05–0.65 m`；
- `Space`：切换夹爪开合；
- `1`：俯视 XY 主控制视图；
- `2`：高亮前视辅助视图，预留 XZ 控制，当前冻结鼠标目标；
- `3`：高亮腕部观察视图，冻结鼠标目标；
- `R`：复位；
- `Q`：退出。

前视和腕部图像显示在同一个 SAPIEN 窗口的辅助面板中，不创建额外控制
窗口。辅助面板区域不会产生鼠标目标；从 2/3 切回 1 后，需要再次移动鼠标
至少 3 像素才会重新接管，因此切换本身不会改变世界 TCP 目标。

红点是原始鼠标目标，绿点是经过可达域投影和稳定保护后的控制目标。默认轨迹写入 `runs/demo0.jsonl`。

配置入口为 [configs/demo0.yaml](configs/demo0.yaml)。完整设计、阶段划分和风险说明见 [BUILD.md](BUILD.md)。

## 重新标定可达域

当固定姿态、初始关节状态或工作区发生变化时，重新生成可达域：

```bash
uv run mani-sim-calibrate
```

默认对 `z=0.05–0.65 m` 按 5 cm 间隔生成 13 个高度层，逐层进行
2.5 cm 全域 IK 采样，并在检测出的边界周围自适应追加 5 mm 采样，结果写入
`calibrations/panda_fixed_orientation.json`。运行 Demo 时，矩形边角等
不可达鼠标目标会沿混合射线起点投影到最后一个已标定可达边界；只有射线
未穿过主可达区时才退化为最近点投影。

## 验证

```bash
uv run pytest
uv run mani-sim --max-steps 100
```

第二条需要图形会话，用于有限步 GUI 冒烟测试。

真实 Panda IK + PD 物理回归需要 NVIDIA/Vulkan 图形环境：

```bash
uv run pytest -q -s tests/integration/test_physical_reachability.py
```
