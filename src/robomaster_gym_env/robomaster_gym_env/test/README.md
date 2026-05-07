# 测试模块

本目录包含端到端通信测试相关文件，用于验证 Gym 环境与 Gazebo 仿真之间的双向 ROS2 通信。

---

## 文件说明

| 文件 | 描述 |
|------|------|
| `__init__.py` | 测试模块初始化 |
| `sim_data_publisher.py` | Gazebo 仿真数据发布器 (随机数据) |
| `test_node.py` | 端到端通信测试节点 |

---

## sim_data_publisher.py

以 ROS2 发布器的形式发送随机仿真数据，模拟 Gazebo 仿真节点的全部输出。

### 功能

发布的话题与 Gazebo 完全一致:

| 话题 | 消息类型 | 随机数据范围 |
|------|----------|-------------|
| `/{ns}/robot_base/odom` | Odometry | 位置 1~27m, 1~14m |
| `/{ns}/chassis_odometry_gt` | Odometry | 同上 |
| `/{ns}/robot_base/gimbal_state` | Gimbal | yaw ±π, pitch ±0.5 |
| `/referee_system/{name}/robot_status` | RobotStatus | HP 100~400, 弹药 0~200 |
| `/referee_system/pose_info` | TFMessage | 6 个机器人随机位置 |
| `/referee_system/game_status` | GameStatus | RUNNING 状态 |
| `/referee_system/outpost_status` | RobotStatus | HP 0~1500 |
| `/referee_system/base_status` | RobotStatus | HP 0~5000 |
| `/{ns}/livox/imu` | Imu | 随机加速度/角速度 |
| `/referee_system/{name}/enable_power` | Bool | True |
| `/referee_system/{name}/enable_control` | Bool | True |
| `/referee_system/attack_info` | String | 偶尔产生攻击信息 |

### 用法

```bash
# 直接运行
ros2 run robomaster_gym_env sim_data_publisher

# 指定参数
ros2 run robomaster_gym_env sim_data_publisher \
    --robot-name red_standard_robot1 \
    --namespace red_standard_robot1 \
    --team red \
    --rate 50.0 \
    --duration 30

# 查看帮助
ros2 run robomaster_gym_env sim_data_publisher --help
```

### 参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--robot-name` | red_standard_robot1 | 机器人名称 |
| `--namespace` | red_standard_robot1 | 机器人命名空间 |
| `--team` | red | 队伍 (red/blue) |
| `--rate` | 50.0 | 发布频率 (Hz) |
| `--duration` | 30.0 | 持续时间 (秒)，0 表示无限 |

---

## test_node.py

端到端通信测试节点，验证 Gym 环境与 Gazebo 仿真之间的双向通信。

### 测试内容

| 测试项 | 验证内容 |
|--------|----------|
| 测试1: 空间定义 | 观测空间/动作空间结构正确 |
| 测试2: Gazebo→Env | env 内部的 `sensor_data`/`state_data`/`referee_data` 是否收到数据 |
| 测试3: Env→Gazebo | `CommandMonitor` 节点监听 env 发出的控制命令，统计接收条数 |
| 测试4: 观测值合理性 | `all_robots` 形状 (10,4)、HP/弹药范围、`damage_per_step` 形状等 |

### 方向1: Gazebo -> Env

检查 env 是否收到以下数据:

- 里程计 (odom)
- 里程计真值 (chassis_odometry_gt)
- IMU 数据
- 机器人状态 (robot_status)
- 位姿信息 (pose_info)
- 攻击信息 (attack_info)
- 游戏状态 (game_status)
- 前哨站状态 (outpost_status)
- 基地状态 (base_status)
- 电源使能 (enable_power)
- 控制使能 (enable_control)

### 方向2: Env -> Gazebo

`CommandMonitor` 节点监听 env 发布的控制话题:

- `/{ns}/cmd_vel` - 底盘速度命令
- `/{ns}/robot_base/gimbal_cmd` - 云台控制命令
- `/{ns}/robot_base/shoot_cmd` - 射击命令
- `/{ns}/cmd_shoot` - 射击命令 (简化)

### 用法

```bash
# 确保 sim_data_publisher 或 Gazebo 已启动
ros2 run robomaster_gym_env gym_test_node
```

### 输出示例

```
============================================================
  RoboMaster 端到端通信测试
============================================================
  Robot: red_standard_robot1
  Namespace: red_standard_robot1
  控制频率: 10.0 Hz
============================================================

--- 测试1: 观测空间 & 动作空间 ---
观测空间 (Dict):
  all_robots: shape=(10, 4), dtype=float32
  own_hp: Discrete(401)
  ...
[PASS] 空间定义正常

--- 测试2: Gazebo -> Env (仿真数据接收) ---
  [OK] 里程计: x=14.23, y=7.56, z=0.00
  [OK] IMU: accel=(0.12, -0.34, 9.81)
  [OK] 机器人状态: HP=287/400, 弹药=156
  ...
[PASS] Gazebo -> Env 通信正常

--- 测试3: Env -> Gazebo (控制命令发送) ---
  Step   1: vel=(+1.23, -0.45), shoot=3, reward=0.0100
  ...
  方向2 命令接收统计:
    [OK] cmd_vel: 30 条
    [OK] cmd_shoot: 6 条
[PASS] Env -> Gazebo 通信正常

--- 测试4: 观测值合理性检查 ---
  [OK] all_robots: shape=(10, 4), 有效机器人=6
  [OK] own_hp: 287
  ...
[PASS] 观测值合理

============================================================
  测试汇总
============================================================
  方向1 (Gazebo->Env): 11/11 话题收到数据
  方向2 (Env->Gazebo): 4/4 命令话题有数据
  观测值合理性:       5/5 检查通过

  [ALL PASS] 端到端通信测试全部通过!
============================================================
```

---

## 一键启动测试

使用 `run_comm_test.sh` 脚本一键启动测试:

```bash
# 使用随机数据发布器 (无需 Gazebo)
bash run_comm_test.sh

# 使用真实 Gazebo 仿真
bash run_comm_test.sh --gazebo

# 其他选项
bash run_comm_test.sh --duration 60
bash run_comm_test.sh --robot blue_standard_robot1 --team blue
```

---

## 测试流程

### 方式1: 使用随机数据发布器

```bash
# 终端1
ros2 run robomaster_gym_env sim_data_publisher

# 终端2
ros2 run robomaster_gym_env gym_test_node
```

### 方式2: 使用真实 Gazebo 仿真

```bash
# 终端1
ros2 launch rmu_gazebo_simulator bringup_sim_headless.launch.py

# 终端2
ros2 run robomaster_gym_env gym_test_node
```

---

## 注意事项

1. **rmoss_interfaces**: 如果 `rmoss_interfaces` 不可用，部分话题将无法发布/订阅
2. **ROS2 环境**: 确保已 `source install/setup.bash`
3. **端口冲突**: 如果有多个 ROS2 节点运行，注意端口冲突
4. **QoS 配置**: 测试节点使用 `RELIABLE` QoS，与 Gym 环境保持一致
