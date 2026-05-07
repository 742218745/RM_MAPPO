# RoboMaster RL Training Workspace

基于 ROS2 Humble + Gazebo Fortress 的 RoboMaster 强化学习训练工作空间，为 RoboMaster 机甲大师赛开发 MAPPO 导航和战斗策略。


## 目录

- [环境要求](#环境要求)
- [环境安装](#环境安装)
- [工作空间构建](#工作空间构建)
- [运行仿真与训练](#运行仿真与训练)
- [Colcon 常用指令](#colcon-常用指令)
- [工作空间结构](#工作空间结构)
- [ROS2 包说明](#ros2-包说明)
- [课程学习](#课程学习)
- [常见问题](#常见问题)

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

# vcstool2 (仓库导入工具)
sudo pip install vcstool2

# 强化学习相关
pip install gymnasium numpy opencv-python shapely transforms3d

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

### 1. 获取源码

如果已有完整工作空间，跳过此步。从零开始：

```bash
mkdir -p ~/ros_ws && cd ~/ros_ws

# 克隆主仓库
git clone https://github.com/SMBU-PolarBear-Robotics-Team/rmu_gazebo_simulator.git src/rmu_gazebo_simulator

# 导入依赖仓库
vcs import src < src/rmu_gazebo_simulator/dependencies.repos
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
source /home/xufurui/ros_ws/install/setup.bash
export ROS_DISABLE_FASTRTPS_SHM=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch rmu_gazebo_simulator bringup_sim.launch.py
```

**终端 2 - 训练**：

```bash
source /home/xufurui/ros_ws/install/setup.bash
export ROS_DISABLE_FASTRTPS_SHM=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
python3 run_train.py
```

### 训练参数

`run_train.py` 中的默认训练配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_episodes` | 1500 | 训练回合数 |
| `rollout_steps` | 2048 | 每回合步数 |
| `ppo_epochs` | 8 | PPO 更新轮数 |
| `minibatch_size` | 64 | 小批量大小 |
| `log_interval` | 10 | 日志打印间隔 |
| `save_interval` | 50 | 检查点保存间隔 |
| `checkpoint_dir` | `checkpoints_nav_train/` | 检查点目录 |

训练会自动加载 `mappo_latest.pt` 继续训练。

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

## Colcon 常用指令

### 构建

```bash
# 构建所有包
colcon build

# 构建 + 符号链接安装 (开发推荐, 避免每次改 Python 都 rebuild)
colcon build --symlink-install

# Release 模式构建 (提升 C++ 运行性能)
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=release

# 仅构建指定包
colcon build --packages-select <包名>

# 构建指定包及其依赖
colcon build --packages-up-to <包名>

# 构建自上次构建以来有变更的包
colcon build --packages-above <包名>

# 跳过指定包
colcon build --packages-skip <包名1> <包名2>

# 并行构建 (默认自动检测 CPU 核心数)
colcon build --parallel-workers 4
```

### 查看构建信息

```bash
# 列出所有已构建的包
colcon list

# 列出指定包的依赖
colcon list --packages-up-to <包名>

# 查看构建结果
colcon list --names-only
```

### 测试

```bash
# 运行所有包的测试
colcon test

# 运行指定包的测试
colcon test --packages-select <包名>

# 查看测试结果
colcon test-result --all
colcon test-result --verbose
```

### 清理

```bash
# 清理构建产物 (保留 install 和 log)
rm -rf build/

# 完全清理
rm -rf build/ install/ log/
```

---

## 工作空间结构

```
ros_ws/
├── src/                              # 源码
│   ├── rmoss_interfaces/             # 消息/服务定义
│   ├── rmoss_core/                   # 核心功能 (工具/通信/相机/弹道)
│   ├── rmoss_gazebo/                 # Gazebo 仿真 (插件/基座/相机/桥接)
│   ├── rmoss_gz_resources/           # Gazebo 模型资源 (SDF)
│   ├── pb2025_robot_description/     # 2025 机器人描述 (步兵/哨兵)
│   ├── sdformat_tools/               # SDF/URDF 工具
│   ├── rmu_gazebo_simulator/         # RMU 仿真环境集成
│   ├── robomaster_gym_env/           # Gymnasium 强化学习环境
│   └── robomaster_mappo/             # MAPPO 训练算法
├── build/                            # 构建中间产物
├── install/                          # 安装产物
├── log/                              # 构建日志
├── checkpoints_nav_train/            # 训练检查点
├── run_train.py                      # 训练启动脚本 (Python)
├── run_train.sh                      # 训练启动脚本 (Shell)
├── start_sim.sh                      # 启动仿真 (有 GUI)
├── start_sim_headless.sh             # 启动仿真 (无 GUI)
└── README.md
```

---

## ROS2 包说明

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

## 课程学习

Gym 环境内置 4 阶段课程学习，由易到难逐步增加敌方距离：

| 阶段 | 距离范围 | Episodes | 目标 |
|------|---------|----------|------|
| 1 | 3-6m | 200 | 学会靠近敌方 |
| 2 | 6-12m | 300 | 扩展导航距离 |
| 3 | 12-20m | 500 | 远距离导航 |
| 4 | 3-25m | 不自动升级 | 全场泛化 + 战斗 |

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
