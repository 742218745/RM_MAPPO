# RoboMaster RL Training Workspace

基于 ROS2 Humble + Gazebo Fortress 的 RoboMaster 强化学习训练工作空间，为 RoboMaster 机甲大师赛开发 **MAPPO (Multi-Agent Proximal Policy Optimization)** 导航和战斗策略。

---

## 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [环境要求](#环境要求)
- [环境安装](#环境安装)
- [工作空间构建](#工作空间构建)
- [运行仿真与训练](#运行仿真与训练)
- [强化学习环境设计](#强化学习环境设计)
- [MAPPO算法与网络结构](#mappo算法与网络结构)
- [课程学习与特化训练](#课程学习与特化训练)
- [工作空间结构](#工作空间结构)
- [ROS2包说明](#ros2包说明)
- [Colcon常用指令](#colcon常用指令)
- [常见问题](#常见问题)

---

## 项目概述

本项目使用 MAPPO 强化学习算法，在 ROS2 + Gazebo 仿真环境中训练 RoboMaster 机器人实现端到端的自主导航与战斗决策。

### 核心特性

- **完整仿真环境**: 基于 RMUC 2026 赛季赛场地 (28m x 15m)，包含坡道、前哨站、基地等结构
- **Gymnasium 标准接口**: 环境封装为 Gymnasium Env，支持 step/reset/render
- **MAPPO 算法**: 多智能体近端策略优化，Actor-Critic 架构，PPO-Clip 更新
- **课程学习**: 4 阶段由易到难，从近距离导航逐步过渡到全场泛化
- **特化训练**: 针对特定路径 (起点→坡道→目标) 的分阶段引导训练
- **仿真加速**: real_time_factor = 4.5x，支持无 GUI 模式服务器训练
- **温和重置**: 只重置机器方位姿，不重启 Gazebo，大幅缩短重置时间

---

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│                  训练控制层                          │
│  (MAPPO训练循环、检查点管理、暂停控制)               │
├─────────────────────────────────────────────────────┤
│              算法层                                  │
│  (Actor网络、Critic网络、PPO更新器、经验缓冲区)      │
├─────────────────────────────────────────────────────┤
│              环境层                                  │
│  (Gymnasium Env、观察空间、动作空间、奖励计算)       │
├─────────────────────────────────────────────────────┤
│              通信层                                  │
│  (ROS2 Interface、话题订阅/发布、服务调用)           │
├─────────────────────────────────────────────────────┤
│              仿真层                                  │
│  (Gazebo物理引擎、机器人模型、场地模型、裁判系统)    │
└─────────────────────────────────────────────────────┘
```

---

## 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Ubuntu | 22.04 LTS | ROS2 Humble 要求 |
| ROS2 | Humble Hawksbill | 机器人操作系统 |
| Gazebo | Ignition Fortress | 仿真引擎 (非经典版 Gazebo) |
| Python | 3.10 | Ubuntu 22.04 默认 |
| C++ | 14 | Gazebo 插件构建标准 |
| DDS | FastRTPS | `rmw_fastrtps_cpp`，需禁用共享内存 |
| PyTorch | >= 1.13 | MAPPO 训练 (建议 CUDA 版本) |
| Gymnasium | >= 0.26 | 强化学习环境接口 (非旧版 gym) |

### Python 依赖

```
numpy
gymnasium
opencv-python
shapely
transforms3d
torch
xmacro
matplotlib
```

### ROS2 系统包依赖

```
ros-humble-ros-gz          # ROS2-Gazebo 桥接
ros-humble-rclcpp          # C++ 客户端库
ros-humble-rclpy           # Python 客户端库
ros-humble-geometry-msgs
ros-humble-sensor-msgs
ros-humble-nav-msgs
ros-humble-tf2-msgs
ros-humble-std-msgs
ros-humble-example-interfaces
ros-humble-image-transport
ros-humble-cv-bridge
ros-humble-camera-info-manager
ros-humble-xacro
ros-humble-nav2-common
ros-humble-robot-state-publisher
```

### Ignition (Gazebo) 系统包依赖

```
ignition-fortress          # 完整 Fortress 发行版
libignition-cmake2-dev
ignition-msgs8
ignition-transport11
ignition-gazebo6
```

---

## 环境安装

### 1. 安装 ROS2 Humble

```bash
# 设置 locale
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# 添加 ROS2 apt 源
sudo apt install -y software-properties-common
sudo add-apt-repository -y universe
sudo apt update && sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 安装 ROS2 Humble
sudo apt update
sudo apt install -y ros-humble-desktop

# 环境变量
source /opt/ros/humble/setup.bash
```

### 2. 安装 Ignition Fortress

```bash
sudo apt install -y wget lsb-release gnupg2

# 添加 Gazebo 源
sudo wget -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
  https://packages.osrfoundation.org/gpg.key
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
  http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# 安装 Fortress
sudo apt update
sudo apt install -y ignition-fortress
```

### 3. 安装 ROS2-Gazebo 桥接

```bash
sudo apt install -y ros-humble-ros-gz
```

### 4. 安装 Python 依赖

```bash
# xmacro (SDF 模型宏处理)
pip install xmacro

# 强化学习相关
pip install gymnasium numpy opencv-python shapely transforms3d matplotlib

# PyTorch (根据 CUDA 版本选择, 参考 https://pytorch.org)
pip install torch  # CPU 版本
# pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
# pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
```

### 5. 安装无 GUI 仿真依赖 (可选, 服务器训练需要)

```bash
sudo apt install -y xvfb
```

---

## 工作空间构建

### 1. 克隆仓库

```bash
git clone https://github.com/742218745/RM_MAPPO.git ~/ros_ws
cd ~/ros_ws
```

### 2. 安装系统依赖

```bash
# 确保 ROS2 环境已 source
source /opt/ros/humble/setup.bash

# rosdep 自动安装所有 package.xml 中声明的系统依赖
sudo rosdep init  # 首次使用
rosdep update
rosdep install -r --from-paths src --ignore-src --rosdistro humble -y
```

### 3. 构建

```bash
source /opt/ros/humble/setup.bash

# 完整构建 (推荐 Release 模式以提升 Gazebo 插件性能)
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=release

# Source 工作空间
source install/setup.bash
```

### 4. 增量构建 (开发时常用)

```bash
# 仅构建指定包及其依赖
colcon build --packages-up-to robomaster_gym_env

# 仅构建指定包 (不含依赖)
colcon build --packages-select rmoss_interfaces

# 构建多个包
colcon build --packages-select rmoss_interfaces robomaster_gym_env robomaster_mappo
```

---

## 运行仿真与训练

### 重要: DDS 环境变量

仿真和训练进程**必须**设置相同的 DDS 环境变量，否则 Gazebo 会崩溃：

```bash
export ROS_DISABLE_FASTRTPS_SHM=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

### 方式一: 使用脚本启动 (推荐)

**终端 1 - 启动 Gazebo 仿真 (有 GUI)**：

```bash
bash start_sim.sh
```

**终端 1 - 启动 Gazebo 仿真 (无 GUI, 服务器训练)**：

```bash
bash start_sim_headless.sh
```

**终端 2 - 启动 MAPPO 训练**：

```bash
bash run_train.sh
```

> 训练日志自动保存到 `train_log.txt`，检查点保存到 `checkpoints_nav_train/`。

### 方式二: 手动启动

**终端 1 - 仿真**：

```bash
source install/setup.bash
export ROS_DISABLE_FASTRTPS_SHM=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch rmu_gazebo_simulator bringup_sim.launch.py
```

**终端 2 - 训练**：

```bash
source install/setup.bash
export ROS_DISABLE_FASTRTPS_SHM=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
python3 run_train.py
```

### 训练参数

`run_train.py` 中的默认训练配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_episodes` | 3000 | 训练回合数 |
| `rollout_steps` | 2048 | 每回合步数 |
| `lr` | 3e-4 | 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_epsilon` | 0.2 | PPO 裁剪范围 |
| `ppo_epochs` | 8 | PPO 更新轮数 |
| `minibatch_size` | 64 | 小批量大小 |
| `entropy_coef` | 0.05 | 熵正则化系数 |
| `save_interval` | 100 | 检查点保存间隔 |
| `checkpoint_dir` | `checkpoints_nav_train/` | 检查点目录 |

训练会自动加载 `mappo_latest.pt` 继续训练。

### 测试训练好的模型

```bash
python3 test_model.py
```

### 实时监控机器人位置

```bash
python3 monitor_pos.py
```

### 测试机器人控制

```bash
source install/setup.bash

# 底盘控制
ros2 run rmoss_gz_base test_chassis_cmd.py \
  --ros-args -r __ns:=/red_standard_robot1/robot_base -p v:=0.3 -p w:=0.3

# 云台控制
ros2 run rmoss_gz_base test_gimbal_cmd.py \
  --ros-args -r __ns:=/red_standard_robot1/robot_base

# 射击控制
ros2 run rmoss_gz_base test_shoot_cmd.py \
  --ros-args -r __ns:=/red_standard_robot1/robot_base
```

---

## 强化学习环境设计

### 观察空间 (13维 Dict)

| 观测项 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `all_robots` | Box | (10, 4) | 所有机器人位置 [id, team, x, y] |
| `own_hp` | Discrete | 401 | 己方血量 (0-400) |
| `own_ammo` | Discrete | 301 | 己方弹药量 (0-300) |
| `team_economy` | Discrete | 401 | 队伍经济 |
| `remaining_steps` | Discrete | 2049 | 剩余步数 |
| `judge_countdown_steps` | Discrete | 2049 | 判负步数 |
| `damage_per_step` | Box | (1,) | 每步伤害能力 |
| `outpost_hp` | Discrete | 1501 | 前哨站血量 (0-1500) |
| `base_hp` | Discrete | 5001 | 基地血量 (0-5000) |
| `base_exposed` | Discrete | 2 | 基地展开状态 |
| `target_direction` | Box | (2,) | 目标相对方向 [dx, dy] |
| `ammo_consumed_per_step` | Discrete | 301 | 每步弹药消耗 |
| `revive_waiting_steps` | Discrete | 2049 | 复活等待步数 |

### 动作空间

| 动作项 | 类型 | 范围 | 说明 |
|--------|------|------|------|
| `chassis_velocity` | MultiDiscrete | [5, 5] | 底盘速度等级 {-2, -1, 0, 1, 2} m/s |
| `shoot` | Discrete | 9 | 0=不射击, 1-6=射击机器人, 7=前哨站, 8=基地 |

### 奖励函数

多层次奖励体系：

- **基础奖励**: 存活(+0.01/步)、被击(-50)、命中(+50)、弹药消耗(-0.1/发)、死亡(-20)、近敌(+0.1/敌)
- **距离塑形奖励**: 距离渐变 (越近越大) + 距离缩减塑形 (靠近为正, 远离为负)
- **特化模式奖励**: 分阶段距离塑形 + 爬坡奖惩 + 速度方向一致性 + 速度大小 + 时间惩罚 + 翻车/碰墙惩罚

### 终止条件

- 血量归零 (HP <= 0)
- 翻车 (IMU姿态角 > 45度, 连续5次确认)
- 出界 (距边界 < 1m)
- 步数达到上限 (2048步, truncated)

---

## MAPPO算法与网络结构

### Actor网络 (66,355 参数)

```
观测 Dict
  ├── all_robots (10,4) → RobotEncoder → robot_feat (64)
  ├── 13个标量 ────────→ StateEncoder → state_feat (64)
  └── concat → Fusion MLP → hidden (128)
                        ├── ChassisHeadX → Categorical(5)
                        ├── ChassisHeadY → Categorical(5)
                        └── ShootHead ──→ Categorical(9)
```

### Critic网络 (48,673 参数)

```
全局状态 → RobotEncoder + StateEncoder → Fusion MLP → ValueHead → V(s)
```

Actor 和 Critic 共享编码器结构，但不共享权重。

### PPO超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 学习率 | 3e-4 | Adam 优化器 |
| 折扣因子 γ | 0.99 | 重视长远奖励 |
| GAE λ | 0.95 | 偏差-方差平衡 |
| PPO裁剪 ε | 0.2 | 限制策略比率在 [0.8, 1.2] |
| PPO更新轮数 | 8 | 每批数据重复更新轮数 |
| 小批量大小 | 64 | PPO 更新时的小批量 |
| 熵正则化系数 | 0.05 | 鼓励探索 |
| 梯度裁剪范数 | 0.5 | 防止梯度爆炸 |

---

## 课程学习与特化训练

### 课程学习 (4阶段)

| 阶段 | 距离范围 | Episodes | 目标 |
|------|---------|----------|------|
| 1 | 3-6m | 300 | 学会靠近近距离目标 |
| 2 | 6-12m | 400 | 扩展到中距离导航 |
| 3 | 12-20m | 600 | 扩展到远距离导航 |
| 4 | 3-25m | 不自动升级 | 全场泛化 + 战斗 |

每个阶段通过虚拟蓝方位置控制任务难度，虚拟位置仅用于奖励计算，不实际移动 Gazebo 中的蓝方。

### 特化训练

针对特定路径进行专项训练：

```
起点(8.64, 3.65) → 坡道中间点(4.81, 2.47) → 目标(14.0, 7.5)
```

- **阶段1**: 引导去中间点，强化爬坡奖励 (权重3.0)
- **阶段2**: 引导去目标点，强化下坡惩罚 (权重-1.0，防止掉回坡下)
- 包含: 距离塑形奖励、爬坡/下坡奖惩、速度方向一致性、卡住检测与回退

---

## 工作空间结构

```
ros_ws/
├── src/                              # 源码
│   ├── rmoss_interfaces/             # 消息/服务定义
│   ├── rmoss_core/                   # 核心功能 (工具/通信/相机/弹道)
│   │   ├── rmoss_util/              #   公共工具
│   │   ├── rmoss_base/              #   SBC与MCU通信
│   │   ├── rmoss_cam/               #   相机ROS封装
│   │   ├── rmoss_projectile_motion/ #   弹道逆运动学求解
│   │   └── rmoss_core/              #   元包
│   ├── rmoss_gazebo/                 # Gazebo仿真
│   │   ├── rmoss_gz_plugins/        #   Gazebo插件 (麦轮/射击/灯条)
│   │   ├── rmoss_gz_base/           #   机器人基座接口
│   │   ├── rmoss_gz_cam/            #   相机接口
│   │   └── rmoss_gz_bridge/         #   Ignition-ROS桥接
│   ├── rmoss_gz_resources/           # Gazebo模型资源 (SDF)
│   ├── pb2025_robot_description/     # 2025机器人描述 (步兵/哨兵)
│   ├── sdformat_tools/               # SDF/URDF工具
│   ├── rmu_gazebo_simulator/         # RMU仿真环境集成 (launch/裁判系统)
│   ├── robomaster_gym_env/           # Gymnasium强化学习环境
│   │   ├── robomaster_env.py        #   核心环境类
│   │   ├── observation_space.py     #   观察空间定义
│   │   ├── action_space.py          #   动作空间定义
│   │   ├── reward_calculator.py     #   奖励计算器
│   │   ├── ros2_interface.py        #   ROS2通信管理器
│   │   ├── config.py                #   环境配置
│   │   └── env_renderer.py          #   2D俯视图渲染器
│   └── robomaster_mappo/             # MAPPO训练算法
│       ├── train.py                 #   训练主循环
│       ├── actor.py                 #   Actor网络
│       ├── critic.py                #   Critic网络
│       ├── rollout_buffer.py        #   经验回放缓冲区
│       └── obs_preprocessor.py      #   观测预处理
├── build/                            # 构建中间产物 (gitignore)
├── install/                          # 安装产物 (gitignore)
├── log/                              # 构建日志 (gitignore)
├── checkpoints_nav_train/            # 训练检查点 (gitignore)
├── run_train.py                      # 训练启动脚本 (Python)
├── run_train.sh                      # 训练启动脚本 (Shell)
├── start_sim.sh                      # 启动仿真 (有GUI)
├── start_sim_headless.sh             # 启动仿真 (无GUI)
├── test_model.py                     # 模型测试脚本
├── monitor_pos.py                    # 实时位置监控脚本
├── final_summary.py                  # 修改总结文档
├── thesis.md                         # 毕业论文
└── README.md
```

---

## ROS2包说明

| 包名 | 构建类型 | 说明 |
|------|---------|------|
| `rmoss_interfaces` | ament_cmake | RMOSS 消息/服务/动作定义 |
| `rmoss_util` | ament_cmake | 公共工具 (调试/图像/ROS 封装) |
| `rmoss_base` | ament_cmake | SBC 与 MCU 通信 |
| `rmoss_cam` | ament_cmake | 相机 ROS 封装 (USB/虚拟) |
| `rmoss_projectile_motion` | ament_cmake | 弹道逆运动学求解 |
| `rmoss_core` | ament_cmake | 元包 (聚合上述 4 包) |
| `rmoss_gz_plugins` | ament_cmake | Gazebo 插件 (麦轮/射击/灯条) |
| `rmoss_gz_base` | ament_cmake | Gazebo 机器人基座接口 |
| `rmoss_gz_cam` | ament_cmake | Gazebo 相机接口 |
| `rmoss_gz_bridge` | ament_cmake | Ignition-ROS 桥接 |
| `rmoss_gz_resources` | ament_cmake | Gazebo 模型资源 (SDF) |
| `pb2025_robot_description` | ament_cmake | 2025 机器人描述 (步兵/哨兵) |
| `sdformat_tools` | ament_python | SDF/URDF 工具 (xmacro4sdf 等) |
| `rmu_gazebo_simulator` | ament_cmake | RMU 仿真环境集成 (launch/裁判系统) |
| `robomaster_gym_env` | ament_python | Gymnasium 强化学习环境 |
| `robomaster_mappo` | ament_python | MAPPO 训练算法 |

### 构建顺序 (关键依赖链)

```
rmoss_interfaces → rmoss_core → rmoss_gazebo → rmoss_gz_resources
    → pb2025_robot_description → rmu_gazebo_simulator
    → robomaster_gym_env → robomaster_mappo
```

---

## 常见问题

### Gazebo 启动后崩溃 / RTPS_TRANSPORT_SHM 错误

确保仿真和训练进程都设置了相同的环境变量：

```bash
export ROS_DISABLE_FASTRTPS_SHM=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

### GPU 渲染问题 (libEGL warning / Segmentation fault)

取消 `start_sim.sh` 中以下行的注释：

```bash
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=3.3
```

### 找不到 rmoss_interfaces

```bash
colcon build --packages-select rmoss_interfaces
source install/setup.bash
```

### rosdep 找不到某些依赖

部分依赖 (如 `ignition-*`) 需要先添加 Gazebo apt 源后再运行 `rosdep install`。

### 残留进程导致端口占用

脚本会自动清理，手动清理：

```bash
pkill -9 -f "ign gazebo"
pkill -9 -f "ros_gz_bridge"
rm -rf /dev/shm/fastrtps_*
```

---

## License

Apache License 2.0 / MIT
