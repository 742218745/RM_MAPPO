# RoboMaster Gazebo Gym Environment

基于 ROS2 Gazebo 仿真的 RoboMaster 强化学习训练环境。

---

## 目录

- [功能特性](#功能特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [多智能体环境](#多智能体环境)
- [观测空间](#观测空间)
- [动作空间](#动作空间)
- [ROS2 Topics](#ros2-topics)
- [奖励设计](#奖励设计)
- [课程学习训练指南](#课程学习训练指南)
- [配置系统](#配置系统)
- [与强化学习框架集成](#与强化学习框架集成)
- [端到端通信测试](#端到端通信测试)
- [架构设计](#架构设计)
- [常见问题](#常见问题)
- [注意事项](#注意事项)

---

## 功能特性

- ✅ 完整的 Gymnasium 接口实现 (支持新版 5 元组 step)
- ✅ 支持所有 ROS2 传感器数据 (相机、激光雷达、IMU 等)
- ✅ 支持底盘、云台、射击控制
- ✅ 集成裁判系统 (HP 管理、弹丸计数、比赛状态)
- ✅ 灵活的配置系统
- ✅ 支持多机器人环境
- ✅ 端到端通信测试工具

---

## 安装

### 依赖

```bash
# ROS2 依赖 (假设已安装 ROS2 Humble)
sudo apt update
sudo apt install -y \
    ros-humble-rclpy \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-nav-msgs \
    ros-humble-tf2-msgs \
    ros-humble-std-msgs \
    ros-humble-example-interfaces

# Python 依赖
pip3 install gymnasium numpy opencv-python transforms3d shapely
```

### 编译

```bash
cd ~/ros_ws
colcon build --packages-select robomaster_gym_env rmoss_interfaces
source install/setup.bash
```

---

## 快速开始

### 1. 启动 Gazebo 仿真

在**第一个终端**:

```bash
source ~/ros_ws/install/setup.bash
ros2 launch rmu_gazebo_simulator bringup_sim.launch.py
```

等待 Gazebo 完全启动，看到机器人模型加载完成。

### 2. 运行测试

在**第二个终端**:

```bash
source ~/ros_ws/install/setup.bash
ros2 run robomaster_gym_env gym_test_node
```

### 3. 基本使用代码

```python
import rclpy
import gymnasium
from robomaster_gym_env import RoboMasterGazeboEnv, GymEnvConfig

# 初始化 ROS2
rclpy.init()

# 创建配置
config = GymEnvConfig()
config.robot_name = "red_standard_robot1"
config.robot_namespace = "red_standard_robot1"
config.team = "red"

# 创建环境
env = RoboMasterGazeboEnv(config)

# 重置环境 (返回 2 元组)
observation, info = env.reset()

# 执行动作 (返回 5 元组)
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

env.close()
rclpy.shutdown()
```

### 4. 键盘控制示例

```bash
python3 ~/ros_ws/src/robomaster_gym_env/examples/keyboard_control.py
```

控制键位:
- **W/S**: 前进/后退
- **A/D**: 左移/右移
- **J/L**: 云台 yaw 左/右
- **I/K**: 云台 pitch 上/下
- **空格**: 射击
- **ESC**: 退出

---

## 多智能体环境

`RoboMasterMultiAgentEnv` 支持同时控制多个机器人（红方+蓝方），提供并行 step 接口。

### 基本使用

```python
from robomaster_gym_env import RoboMasterMultiAgentEnv

# 创建双智能体环境 (红蓝各一个步兵)
env = RoboMasterMultiAgentEnv(
    agents_config={
        'red_standard_robot1': {'team': 'red'},
        'blue_standard_robot1': {'team': 'blue'},
    }
)

# reset 返回所有智能体的观测
obs = env.reset()
# obs = {
#     'red_standard_robot1': { 'all_robots': ..., 'own_hp': ..., ... },
#     'blue_standard_robot1': { 'all_robots': ..., 'own_hp': ..., ... },
# }

# 并行 step: 所有智能体同时执行动作
actions = {
    name: env.action_spaces[name].sample()
    for name in env.agents
}
obs, rewards, terminateds, truncateds, infos = env.pstep(actions)
# obs:        Dict[agent_name, observation]
# rewards:    Dict[agent_name, float]
# terminateds: Dict[agent_name, bool]
# truncateds:  Dict[agent_name, bool]
# infos:      Dict[agent_name, dict]

env.close()
```

### 自定义智能体配置

```python
# 只控制红方 (单智能体退化为多智能体)
env = RoboMasterMultiAgentEnv(
    agents_config={
        'red_standard_robot1': {'team': 'red'},
    }
)

# 控制更多机器人
env = RoboMasterMultiAgentEnv(
    agents_config={
        'red_standard_robot1': {'team': 'red'},
        'red_standard_robot2': {'team': 'red'},
        'blue_standard_robot1': {'team': 'blue'},
    }
)

# 覆盖控制频率
env = RoboMasterMultiAgentEnv(
    agents_config={
        'red_standard_robot1': {'team': 'red', 'control_frequency': 20.0},
        'blue_standard_robot1': {'team': 'blue'},
    }
)
```

### 接口说明

| 方法 | 说明 |
|------|------|
| `reset()` | 重置仿真，返回 `Dict[agent_name, observation]` |
| `pstep(actions)` | 并行 step，所有智能体同时执行动作 |
| `close()` | 关闭所有 ROS2 接口 |
| `agents` | 智能体名称列表 |
| `action_spaces` | `Dict[agent_name, gymnasium.spaces.Dict]` |
| `observation_spaces` | `Dict[agent_name, gymnasium.spaces.Dict]` |

### 设计要点

- **命名空间隔离**: 每个智能体有独立的 `ROS2Interface`，发布/订阅各自命名空间下的话题
- **并行 step**: `pstep(actions_dict)` 所有智能体同时执行动作，返回各自的 obs/reward/terminated/truncated/info
- **共享仿真**: `reset()` 只调用一次 `reset_simulation()`，避免冲突
- **独立状态**: 每个智能体独立维护 hp、ammo、复活等状态
- **灵活配置**: `agents_config` 可指定任意数量的机器人，不限于红蓝各一个

### 与单智能体环境的关系

| 特性 | `RoboMasterGazeboEnv` | `RoboMasterMultiAgentEnv` |
|------|----------------------|--------------------------|
| 控制机器人数 | 1 | 任意 |
| step 接口 | `step(action)` | `pstep(actions_dict)` |
| 返回值 | 5 元组 | 5 个字典 |
| ROS2Interface | 1 个 | 每个智能体 1 个 |
| reset 仿真 | 每次调用 | 只调用一次 |

---

## 观测空间

环境提供以下观测数据:

| 观测名称 | 形状 | 类型 | 描述 |
|---------|------|------|------|
| `all_robots` | (10, 4) | float32 | 所有机器人位置 [id, team, x, y] |
| `own_hp` | () | int | 己方血量 |
| `own_ammo` | () | int | 己方弹药量 |
| `team_economy` | () | int | 我方经济 |
| `remaining_steps` | () | int | 剩余步数 |
| `judge_countdown_steps` | () | int | 判负步数 |
| `damage_per_step` | (1,) | float32 | 每步伤害能力 |
| `outpost_hp` | () | int | 前哨站血量 |
| `base_hp` | () | int | 基地血量 |
| `base_exposed` | () | int | 基地展开状态 (0/1) |
| `ammo_consumed_per_step` | () | int | 每步弹药消耗 |
| `revive_waiting_steps` | () | int | 复活等待步数 |

---

## 动作空间

环境接受以下动作:

| 动作名称 | 形状 | 类型 | 描述 |
|---------|------|------|------|
| `chassis_velocity` | (2,) | float32 | 底盘速度 [linear_x, linear_y]，范围 [-2, 2] m/s |
| `shoot` | () | Discrete(9) | 射击动作 (0=不射击, 1-6=射击机器人, 7=前哨站, 8=基地) |

**注意**: 云台锁定初始朝向 (yaw=0, pitch=0)，不作为动作空间的一部分，无论是否射击都不改变朝向。

---

## ROS2 Topics

### 控制命令 (发布)

| Topic | 消息类型 | 描述 |
|-------|---------|------|
| `/{robot_ns}/cmd_vel` | geometry_msgs/Twist | 底盘速度命令 |
| `/{robot_ns}/robot_base/gimbal_cmd` | rmoss_interfaces/GimbalCmd | 云台控制命令 |
| `/{robot_ns}/robot_base/shoot_cmd` | rmoss_interfaces/ShootCmd | 射击命令 |
| `/{robot_ns}/cmd_shoot` | example_interfaces/UInt8 | 射击命令 (简化) |

### 传感器数据 (订阅)

| Topic | 消息类型 | 描述 |
|-------|---------|------|
| `/{robot_ns}/robot_base/odom` | nav_msgs/Odometry | 里程计 |
| `/{robot_ns}/chassis_odometry_gt` | nav_msgs/Odometry | 底盘里程计真值 |
| `/{robot_ns}/robot_base/gimbal_state` | rmoss_interfaces/Gimbal | 云台状态 |
| `/{robot_ns}/livox/imu` | sensor_msgs/Imu | IMU 数据 (同时用于翻车检测) |

### 裁判系统 (订阅)

| Topic | 消息类型 | 描述 |
|-------|---------|------|
| `/referee_system/{robot_name}/robot_status` | rmoss_interfaces/RobotStatus | 机器人状态 |
| `/referee_system/{robot_name}/enable_power` | std_msgs/Bool | 电源使能 |
| `/referee_system/{robot_name}/enable_control` | std_msgs/Bool | 控制使能 |
| `/referee_system/pose_info` | tf2_msgs/TFMessage | 机器人位姿 |
| `/referee_system/attack_info` | std_msgs/String | 攻击信息 |
| `/referee_system/outpost_status` | rmoss_interfaces/RobotStatus | 前哨站状态 |
| `/referee_system/base_status` | rmoss_interfaces/RobotStatus | 基地状态 |
| `/referee_system/game_status` | rmoss_interfaces/GameStatus | 游戏状态 |

---

## 奖励设计

默认奖励配置:

```python
reward_config = {
    'hit_enemy': 50.0,           # 命中敌人奖励
    'be_hit': -50.0,             # 被击中惩罚
    'survive_per_step': 0.01,    # 存活奖励
    'ammo_usage': -0.1,          # 弹丸消耗惩罚
    'out_of_boundary': -100.0,   # 出界/死亡惩罚
    'tumble': -10.0,             # 翻车惩罚
    'near_enemy_bonus': 0.1,     # 在敌方 4m 范围内的奖励
    'distance_reward': 1.0,      # 距离渐变奖励权重
    'distance_shaping': 2.0,     # 距离缩减塑形奖励权重
    'max_field_distance': 30.0,  # 场地最大距离(归一化用)
}
```

### 奖励项详解

| # | 奖励项 | 配置键 | 公式 | 说明 |
|---|--------|--------|------|------|
| 1 | 存活奖励 | `survive_per_step` | `+0.01` (每步) | 鼓励存活 |
| 2 | 被击中惩罚 | `be_hit` | `hp_loss * be_hit / max_hp` | 按HP损失比例惩罚 |
| 3 | 弹药消耗惩罚 | `ammo_usage` | `ammo_used * (-0.1)` | 抑制浪费弹药 |
| 4 | 命中敌人奖励 | `hit_enemy` | `+50.0` (命中时) | 鼓励命中敌方 |
| 5 | 死亡惩罚 | `out_of_boundary` | `-100.0` | 死亡时大惩罚 |
| 6 | 贴近敌方奖励 | `near_enemy_bonus` | `count * 0.1` | 4m内每个敌方+0.1 |
| 7 | **距离渐变奖励** | `distance_reward` | `(1 - dist/max_dist) * weight` | 距离越近奖励越大，全程有梯度 |
| 8 | **距离缩减塑形奖励** | `distance_shaping` | `(last_dist - cur_dist) * weight` | 靠近为正，远离为负 |

### 距离奖励设计思路

传统的 `near_enemy_bonus` 是**二值阈值奖励**——只有进入4m范围才有信号，从远处导航时全程无梯度。新增的两个距离奖励解决了这个问题：

- **距离渐变奖励(#7)**：提供绝对位置信号。距离0m时奖励为 `distance_reward`，距离30m时奖励为0，中间线性插值。确保每一步都有梯度引导红方靠近蓝方。
- **距离缩减塑形奖励(#8)**：提供相对变化信号。每靠近1m奖励 `+distance_shaping`，远离1m惩罚 `-distance_shaping`。比绝对距离更稳定，直接反映"是否在靠近"。

---

## 课程学习训练指南

### 概述

课程学习（Curriculum Learning）通过**由易到难**逐步增加蓝方机器人的初始距离，帮助红方机器人循序渐进地学习导航能力。

每次 `env.reset()` 时，系统会根据当前训练阶段自动随机设置蓝方机器人的位置。

### 课程学习配置

```python
curriculum_config = {
    'enabled': True,          # 是否启用课程学习
    'stage': 1,               # 当前训练阶段 (1-4)
    'stage_ranges': {         # 各阶段蓝方距离范围 (相对于红方初始位置)
        1: (3.0, 6.0),       # 近距离: 3-6m
        2: (6.0, 12.0),      # 中距离: 6-12m
        3: (12.0, 20.0),     # 远距离: 12-20m
        4: (3.0, 25.0),      # 全场随机: 3-25m
    },
    'stage_episodes': {       # 各阶段训练episode数 (达到后自动升级)
        1: 200,
        2: 300,
        3: 500,
        4: -1,               # -1 表示不自动升级
    },
}
```

### 四个训练阶段

#### 阶段1：近距离导航（3-6m）

**目标**：让红方学会"靠近敌方→获得奖励"的基本关联。

蓝方在红方3-6m范围内随机出现，红方只需短距离移动即可获得距离奖励信号。

```python
config = GymEnvConfig()
# 默认 stage=1, 无需额外配置
```

**观察指标**：
- 红方平均到敌方距离是否在几个episode内快速下降
- 距离缩减塑形奖励的均值是否为正（说明在靠近）

**判断是否可以升级**：80%以上episode中红方能稳定靠近到4m以内。

#### 阶段2：中距离导航（6-12m）

**目标**：扩展导航距离，学会中距离追踪。

```python
config = GymEnvConfig()
config.curriculum_config['stage'] = 2
```

**调整建议**：
- 如果阶段1表现良好，直接进入阶段2
- 如果阶段1表现差，增加阶段1的 `stage_episodes` 到 500
- 可适当增大 `distance_shaping` 到 3.0，强化靠近信号

#### 阶段3：远距离导航（12-20m）

**目标**：远距离导航，接近实战初始距离。

```python
config = GymEnvConfig()
config.curriculum_config['stage'] = 3
```

**调整建议**：
- 增大 `distance_shaping` 到 3.0-5.0
- 如果红方经常被击杀，增大 `be_hit` 惩罚到 -100.0
- 可以开始关注 `hit_enemy` 奖励是否被触发

#### 阶段4：全场泛化（3-25m）

**目标**：在任意距离下都能导航，泛化到实战场景。

```python
config = GymEnvConfig()
config.curriculum_config['stage'] = 4
```

**调整建议**：
- 可以开始加入蓝方移动策略（随机游走/规则躲避）
- 逐步减小 `distance_reward` 权重，增大 `hit_enemy` 权重
- 从"导航训练"过渡到"战斗训练"

### 奖励权重调整原则

不同训练阶段应使用不同的奖励权重组合：

| 训练场景 | distance_reward | distance_shaping | be_hit | hit_enemy | 说明 |
|---------|----------------|------------------|--------|-----------|------|
| 纯导航训练 | 1.0 | 2.0 | -50.0 | 0 | 距离奖励主导，不关心战斗 |
| 导航+规避 | 1.0 | 2.0 | -100.0 | 50.0 | 引入被击中风险意识 |
| 战斗为主 | 0.5 | 1.0 | -100.0 | 100.0 | 命中奖励主导 |
| 实战微调 | 0.2 | 0.5 | -80.0 | 80.0 | 均衡各奖励项 |

**核心原则**：先让红方学会"靠近"（距离奖励主导），再逐步引入"规避"和"战斗"（被击中惩罚和命中奖励主导）。

### 自定义课程学习参数

```python
config = GymEnvConfig()

# 自定义各阶段距离范围
config.curriculum_config['stage_ranges'] = {
    1: (2.0, 5.0),    # 更近的起始距离
    2: (5.0, 10.0),
    3: (10.0, 18.0),
    4: (2.0, 28.0),   # 更大范围
}

# 自定义各阶段episode数
config.curriculum_config['stage_episodes'] = {
    1: 500,   # 更多episode确保学稳
    2: 500,
    3: 1000,
    4: -1,    # 不自动升级
}
```

### 禁用课程学习

如果不想使用课程学习，蓝方位置将保持 `gz_world.yaml` 中的固定位置 (25.6, 6.45)：

```python
config = GymEnvConfig()
config.curriculum_config['enabled'] = False
```

### 从检查点恢复训练

从之前保存的检查点恢复时，需要手动设置课程学习阶段：

```python
config = GymEnvConfig()
config.curriculum_config['stage'] = 3  # 从阶段3继续
# 加载检查点后继续训练
```

### 训练流程总览

```
阶段1 (近距离 3-6m, 200 episodes)
  │  目标: 学会靠近敌方
  │  奖励: distance_reward=1.0, distance_shaping=2.0
  │
  ▼ 自动升级
阶段2 (中距离 6-12m, 300 episodes)
  │  目标: 扩展导航距离
  │  奖励: 同上, 可增大 distance_shaping
  │
  ▼ 自动升级
阶段3 (远距离 12-20m, 500 episodes)
  │  目标: 远距离导航
  │  奖励: 增大 be_hit 惩罚, 启用 hit_enemy
  │
  ▼ 自动升级
阶段4 (全场 3-25m, 不自动升级)
  │  目标: 全场泛化 + 战斗
  │  奖励: 减小距离奖励, 增大战斗奖励
  │
  ▼
实战部署
```

---

## 配置系统

### 环境配置

```python
from robomaster_gym_env import GymEnvConfig

config = GymEnvConfig()
config.robot_name = "red_standard_robot1"
config.robot_namespace = "red_standard_robot1"
config.team = "red"
config.control_frequency = 50.0  # Hz
```

### 速度限制配置

```python
# 调整速度限制
config.chassis_velocity_limit['linear_x_max'] = 3.0  # m/s
config.chassis_velocity_limit['linear_y_max'] = 3.0  # m/s
```

---

## 与强化学习框架集成

### Stable Baselines3

```python
from stable_baselines3 import PPO
from robomaster_gym_env import RoboMasterGazeboEnv

env = RoboMasterGazeboEnv()
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_robomaster")
```

### RLlib

```python
from ray import tune
from ray.rllib.agents import ppo

config = ppo.DEFAULT_CONFIG.copy()
config["env"] = RoboMasterGazeboEnv
tune.run("PPO", config=config, stop={"training_iteration": 100})
```

---

## 端到端通信测试

测试 Gym 环境与 Gazebo 仿真之间的双向 ROS2 通信是否连贯:

- **方向1 (Gazebo -> Env)**: 仿真数据发布 -> env 订阅接收
- **方向2 (Env -> Gazebo)**: env 发布控制命令 -> Gazebo 接收

### 一键启动测试

```bash
# 方式1: 使用随机数据发布器 (无需启动 Gazebo, 快速验证)
bash run_comm_test.sh

# 方式2: 使用真实 Gazebo 仿真
bash run_comm_test.sh --gazebo

# 其他选项
bash run_comm_test.sh --duration 60
bash run_comm_test.sh --robot blue_standard_robot1 --team blue
```

### 手动分步测试

```bash
# 终端1: 启动随机数据发布器
ros2 run robomaster_gym_env sim_data_publisher

# 终端2: 启动端到端测试节点
ros2 run robomaster_gym_env gym_test_node
```

### 测试内容

| 测试项 | 说明 |
|--------|------|
| 测试1: 空间定义 | 验证观测空间和动作空间的结构 |
| 测试2: Gazebo->Env | 检查 env 是否收到里程计、IMU、裁判系统等数据 |
| 测试3: Env->Gazebo | 检查 env 发出的 cmd_vel、gimbal_cmd、shoot_cmd 是否到达 |
| 测试4: 观测值合理性 | 验证 all_robots 形状、HP/弹药范围等 |

---

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    强化学习算法 (用户代码)                          │
│                  (PPO, DQN, SAC, etc.)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              RoboMasterGazeboEnv (Gymnasium接口)                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  - observation_space: Dict[传感器数据]                    │  │
│  │  - action_space: Dict[控制命令]                           │  │
│  │  - step(action) -> (obs, reward, terminated, truncated, info) │
│  │  - reset() -> (observation, info)                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────┬───────────────────────────────────────────┬───────────┘
         │                                           │
         ▼                                           ▼
┌─────────────────────────────┐    ┌──────────────────────────────┐
│   ObservationSpace          │    │   ActionSpace                │
│  ┌───────────────────────┐ │    │  ┌────────────────────────┐ │
│  │ • all_robots (10,4)    │ │    │  │ • chassis_velocity (2) │ │
│  │ • own_hp (int)         │ │    │  │ • shoot (Discrete(9))  │ │
│  │ • own_ammo (int)       │ │    │  └────────────────────────┘ │
│  │ • team_economy (int)   │ │    └──────────────────────────────┘
│  │ • ...                  │ │
│  └───────────────────────┘ │
└─────────────────────────────┘
         │                                           │
         ▼                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ROS2Interface (ROS2通信层)                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  订阅器: odom, imu, robot_status, pose_info, ...          │  │
│  │  发布器: cmd_vel, gimbal_cmd, shoot_cmd                   │  │
│  │  服务: exchange_ammo, control_task                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ROS2 / Gazebo 仿真环境                        │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流

**观测数据流 (Gazebo → Gym)**:

```
Gazebo传感器 → ROS2 Topics → ROS2Interface (回调) → ObservationSpace → Gym观测
```

**控制数据流 (Gym → Gazebo)**:

```
Gym动作 → ActionSpace → ROS2Interface (发布) → ROS2 Topics → Gazebo控制器
```

### 类关系图

```
RoboMasterGazeboEnv (gymnasium.Env)
    │
    ├─── ROS2Interface
    │       ├── publishers: Dict[str, Publisher]
    │       ├── subscribers: Dict[str, Subscriber]
    │       ├── sensor_data: Dict
    │       ├── state_data: Dict
    │       └── referee_data: Dict
    │
    ├─── ObservationSpace
    │       ├── observation_space: gymnasium.spaces.Dict
    │       └── get_observation() -> Dict
    │
    ├─── ActionSpace
    │       ├── action_space: gymnasium.spaces.Dict
    │       └── parse_action() -> Dict
    │
    └─── RewardCalculator
            └── calculate_reward() -> float
```

### 目录结构

```
robomaster_gym_env/
├── robomaster_gym_env/
│   ├── __init__.py              # 包初始化
│   ├── config.py                # 配置定义
│   ├── robomaster_env.py        # 单智能体 Gymnasium 环境主类
│   ├── multi_agent_env.py       # 多智能体并行环境
│   ├── ros2_interface.py        # ROS2 通信接口
│   ├── observation_space.py     # 观测空间定义
│   ├── action_space.py          # 动作空间定义
│   ├── reward_calculator.py     # 奖励计算器
│   ├── interface_adapter.py     # 接口适配层
│   ├── data_processor.py        # 数据处理器
│   ├── unknown_state_handler.py # Unknown 状态处理
│   ├── env_renderer.py          # 环境渲染器
│   └── ros2_interface.py        # ROS2 通信接口 (含翻车检测)
├── test/
│   ├── __init__.py              # 测试模块初始化
│   ├── sim_data_publisher.py    # 仿真数据发布器 (测试用)
│   └── test_node.py             # 端到端通信测试节点
├── examples/
│   ├── random_agent.py          # 随机策略示例
│   ├── keyboard_control.py      # 键盘控制示例
│   ├── example.py               # 基本使用示例
│   └── test_action_publish.py   # 动作发布测试
├── launch/
│   └── gym_env.launch.py        # Launch 文件
└── README.md
```

---

## 常见问题

### Q1: 找不到 rmoss_interfaces

**问题**: `ImportError: cannot import name 'ChassisCmd' from 'rmoss_interfaces'`

**解决**:
```bash
cd ~/ros_ws
colcon build --packages-select rmoss_interfaces
source install/setup.bash
```

### Q2: Gazebo 无法启动

**解决**:
```bash
gazebo --version
sudo apt install ros-humble-gazebo-ros-pkgs
```

### Q3: 没有图像数据

**解决**:
```bash
ros2 topic echo /red_standard_robot1/front_industrial_camera/image --once
```

### Q4: 性能太慢

**解决**:
```python
config.control_frequency = 20.0  # 从 50Hz 降到 20Hz
```

### Q5: step() 返回值数量不对

**问题**: 旧版 gym 返回 4 元组，新版 gymnasium 返回 5 元组

**解决**: 使用 gymnasium API:
```python
obs, reward, terminated, truncated, info = env.step(action)
obs, info = env.reset()
```

---

## 注意事项

1. **启动 Gazebo 仿真**: 在使用 Gym 环境前，需要先启动 Gazebo 仿真:
   ```bash
   ros2 launch rmu_gazebo_simulator bringup_sim.launch.py
   ```

2. **ROS2 消息定义**: 确保已编译 `rmoss_interfaces` 包

3. **性能优化**: 
   - 调整控制频率以平衡性能和响应速度
   - 使用 headless 模式运行 Gazebo

4. **多机器人**: 使用 `RoboMasterMultiAgentEnv` 同时控制多个机器人，或创建多个 `RoboMasterGazeboEnv` 实例

5. **通信测试**: 修改通信接口后，建议运行 `bash run_comm_test.sh` 验证双向通信正常

6. **Gymnasium API**: 本项目使用新版 gymnasium API (5 元组 step, 2 元组 reset)

---

## License

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request!
