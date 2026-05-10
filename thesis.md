# 基于MAPPO强化学习的RoboMaster机器人自主导航与战斗策略研究

---

## 摘要

RoboMaster机甲大师赛是一项综合了机器人控制、计算机视觉和人工智能等多种技术的综合性机器人竞赛。在比赛中，机器人的自主决策能力直接影响其战场表现，传统基于规则的决策方法难以应对复杂多变的战场环境。本文针对RoboMaster比赛中机器人的自主导航与战斗决策问题，提出了一种基于多智能体近端策略优化（Multi-Agent Proximal Policy Optimization, MAPPO）的强化学习方法，在ROS2与Gazebo仿真环境中实现了机器人的端到端自主决策。

本文的主要工作和贡献包括：

1. **构建了完整的RoboMaster仿真训练环境**：基于ROS2 Humble与Gazebo Fortress仿真引擎，搭建了符合RMUC 2026赛制的仿真环境，包括场地模型、机器人物理模型（麦克纳姆轮底盘、云台、射击机构）、裁判系统等，并实现了Gymnasium标准接口，为强化学习训练提供了高保真的仿真平台。

2. **设计了面向RoboMaster的观察空间与动作空间**：观察空间包含所有机器人位置、己方血量与弹药、队伍经济、前哨站与基地状态等13维信息；动作空间采用离散化设计，底盘速度为MultiDiscrete([5,5])，射击动作为9类离散选择，兼顾了决策的灵活性与训练的稳定性。

3. **设计了多层次的奖励函数体系**：包括基础奖励（存活、被击、命中、弹药消耗）、距离塑形奖励（距离渐变与缩减塑形）、特化模式奖励（分阶段路径引导、爬坡奖惩、速度方向一致性）等，有效引导策略学习。

4. **提出了课程学习与特化训练相结合的训练策略**：课程学习分4个阶段逐步增大目标距离，从近距离导航逐步过渡到全场泛化；特化模式针对特定路径（起点→坡道中间点→目标点）进行分阶段引导训练，有效解决了复杂地形下的导航难题。

5. **实现了MAPPO算法的完整训练框架**：Actor网络采用RobotEncoder+StateEncoder双编码器融合架构，Critic网络共享编码器结构但不共享权重，使用PPO-Clip目标函数、GAE优势估计、梯度裁剪等技术保证训练稳定性。

实验结果表明，在RMUC 2026仿真场地中，经过550个episode的训练，机器人能够逐步学习从起点经坡道到达目标点的导航策略，验证了所提方法的有效性。

**关键词**：RoboMaster；强化学习；MAPPO；自主导航；仿真训练；课程学习；ROS2

---

## Abstract

RoboMaster University Championship is a comprehensive robotics competition that integrates various technologies including robot control, computer vision, and artificial intelligence. The autonomous decision-making capability of robots directly affects their battlefield performance, and traditional rule-based decision methods struggle to cope with complex and dynamic battlefield environments. This paper addresses the autonomous navigation and combat decision-making problem of robots in RoboMaster competitions, proposing a reinforcement learning method based on Multi-Agent Proximal Policy Optimization (MAPPO), which achieves end-to-end autonomous decision-making for robots in a ROS2 and Gazebo simulation environment.

The main contributions of this paper include:

1. **Construction of a complete RoboMaster simulation training environment**: Based on ROS2 Humble and Gazebo Fortress simulation engine, a simulation environment conforming to RMUC 2026 rules is built, including field models, robot physical models (Mecanum wheel chassis, gimbal, shooting mechanism), referee system, etc., with Gymnasium standard interface implemented.

2. **Design of observation and action spaces for RoboMaster**: The observation space includes 13-dimensional information such as all robot positions, own HP and ammo, team economy, outpost and base status; the action space adopts a discretized design with MultiDiscrete([5,5]) for chassis velocity and 9-class discrete choice for shooting.

3. **Design of a multi-level reward function system**: Including basic rewards (survival, being hit, hitting, ammo consumption), distance shaping rewards, and specialization mode rewards (phased path guidance, climbing rewards/penalties, velocity-direction consistency).

4. **Proposed a training strategy combining curriculum learning and specialization training**: Curriculum learning gradually increases target distance across 4 phases; specialization mode performs phased guidance training for specific paths (start → ramp waypoint → target).

5. **Implementation of the complete MAPPO training framework**: The Actor network adopts a dual-encoder fusion architecture (RobotEncoder + StateEncoder), the Critic network shares the encoder structure but not weights, using PPO-Clip objective function, GAE advantage estimation, and gradient clipping.

Experimental results show that after 550 episodes of training in the RMUC 2026 simulation field, the robot can gradually learn the navigation strategy from the starting point through the ramp to the target point, verifying the effectiveness of the proposed method.

**Keywords**: RoboMaster; Reinforcement Learning; MAPPO; Autonomous Navigation; Simulation Training; Curriculum Learning; ROS2

---

## 第一章 绪论

### 1.1 研究背景与意义

RoboMaster机甲大师赛是由大疆创新发起的面向全球高校的机器人射击对抗赛，要求参赛队伍设计多种类型的机器人，在复杂场地中进行自主导航、目标识别、射击对抗等任务。比赛场地包含坡道、障碍物、前哨站、基地等结构，机器人需要在这样的环境中做出实时决策，包括移动路径规划、目标选择、射击时机判断等。

传统的RoboMaster机器人决策系统主要基于有限状态机（FSM）和行为树（Behavior Tree）等规则驱动方法。这类方法的优势在于逻辑清晰、可解释性强，但存在以下局限：

1. **规则爆炸**：随着战场状态空间的增大，需要编写的规则数量呈指数增长，难以覆盖所有情况。
2. **适应性差**：面对未预见的战场态势，基于规则的方法缺乏泛化能力，容易陷入局部最优。
3. **调参困难**：规则中的阈值和权重需要大量人工调试，且不同场景下最优参数差异很大。

强化学习（Reinforcement Learning, RL）为解决上述问题提供了一种新的范式。通过让智能体在环境中自主探索并从奖励信号中学习，强化学习能够自动发现高效的决策策略，无需人工编写规则。近年来，深度强化学习在游戏AI（如AlphaGo、OpenAI Five）和机器人控制（如灵巧手操作、四足行走）等领域取得了突破性进展，证明了其在复杂决策问题上的巨大潜力。

将强化学习应用于RoboMaster比赛具有重要的理论意义和应用价值：

- **理论意义**：RoboMaster场景融合了连续控制（底盘运动）、离散决策（射击目标选择）、多智能体协作、部分可观测性等挑战，是研究复杂强化学习问题的理想平台。
- **应用价值**：自主决策能力是RoboMaster机器人的核心竞争力，基于强化学习的决策系统可以显著提升机器人的战场适应性和决策效率。

### 1.2 国内外研究现状

#### 1.2.1 强化学习在机器人导航中的应用

机器人导航是强化学习的经典应用场景。早期工作主要基于表格型方法（如Q-learning），适用于离散状态空间。随着深度学习的发展，深度Q网络（DQN）及其变体被广泛应用于连续环境中的导航任务。

在移动机器人导航方面，Tai等人[1]提出了基于深度强化学习的地图到路径规划方法，使用异步优势Actor-Critic（A3C）算法训练神经网络直接从激光雷达数据输出运动命令。Zhu等人[2]提出了目标驱动的深度强化学习导航方法，在未知环境中实现了目标导向的导航。

针对多机器人场景，Lowe等人[3]提出了多智能体深度确定性策略梯度（MADDPG），采用集中训练分散执行的范式，每个智能体的Critic可以访问所有智能体的观测和动作，而Actor只依赖自身观测。这一思想对后续的MAPPO算法产生了重要影响。

#### 1.2.2 PPO算法及其多智能体扩展

近端策略优化（Proximal Policy Optimization, PPO）由Schulman等人[4]于2017年提出，通过裁剪目标函数限制策略更新幅度，在保持TRPO理论保证的同时大幅简化了实现。PPO因其稳定性和样本效率，成为目前最广泛使用的策略梯度算法之一。

Yu等人[5]将PPO扩展到多智能体场景，提出了MAPPO（Multi-Agent PPO），也称为POCA（Policy Optimization with Centralized Advantage）。MAPPO采用集中式Critic（输入全局状态）和分散式Actor（输入局部观测）的架构，在StarCraft多智能体挑战和Google Research Football等基准测试中取得了优于MADDPG和QMIX等算法的表现。

#### 1.2.3 强化学习在RoboMaster中的应用

目前，强化学习在RoboMaster领域的应用尚处于起步阶段。大部分参赛队伍仍采用传统规则驱动方法，仅有少数队伍尝试将强化学习用于特定子任务：

- **自动瞄准**：部分队伍使用强化学习训练云台控制策略，实现更快速稳定的目标跟踪。
- **路径规划**：少数队伍尝试使用深度强化学习替代传统A*或RRT算法进行路径规划。
- **整体决策**：将强化学习用于整体决策的工作较少，主要受限于仿真环境的高成本和训练的困难性。

本文的工作与上述研究相比，具有以下特点：一是构建了完整的RoboMaster仿真训练环境，降低了强化学习应用门槛；二是采用MAPPO算法进行端到端决策训练，而非仅用于子任务；三是提出了课程学习与特化训练相结合的策略，有效解决了复杂地形下的训练难题。

### 1.3 本文主要工作与论文结构

本文围绕基于MAPPO强化学习的RoboMaster机器人自主导航与战斗策略展开研究，主要工作包括：

1. 搭建基于ROS2与Gazebo的RoboMaster仿真训练环境，实现Gymnasium标准接口。
2. 设计面向RoboMaster的观察空间、动作空间和奖励函数。
3. 提出课程学习与特化训练相结合的训练策略。
4. 实现MAPPO算法的完整训练框架，并进行实验验证。

论文结构安排如下：

- **第一章**：绪论，介绍研究背景、国内外现状和主要工作。
- **第二章**：系统架构设计，介绍整体框架、仿真环境和ROS2通信架构。
- **第三章**：强化学习环境设计，详细阐述观察空间、动作空间和奖励函数的设计。
- **第四章**：MAPPO算法与训练策略，介绍网络结构、PPO更新规则、课程学习与特化训练。
- **第五章**：实验与结果分析，展示训练过程和结果。
- **第六章**：总结与展望。

---

## 第二章 系统架构设计

### 2.1 整体框架

本文提出的RoboMaster强化学习训练系统采用分层架构，如图2-1所示。系统由以下层次组成：

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

**图2-1 系统分层架构**

各层职责如下：

- **训练控制层**：管理训练主循环，包括episode迭代、经验收集与策略更新的交替、检查点保存与恢复、训练暂停与安全退出等。
- **算法层**：实现MAPPO算法的核心组件，包括Actor网络（策略网络）、Critic网络（价值网络）、PPO更新器（裁剪目标函数与参数更新）、经验回放缓冲区（GAE计算与小批量划分）。
- **环境层**：实现Gymnasium标准接口（step/reset/render），封装观察空间构建、动作执行、奖励计算、终止条件判断等逻辑。
- **通信层**：通过ROS2 DDS（FastRTPS）与Gazebo仿真器通信，订阅机器人状态话题（里程计、IMU、裁判系统数据等），发布控制命令话题（底盘速度、云台角度、射击命令等）。
- **仿真层**：Gazebo Fortress物理引擎运行RMUC 2026场地与机器人模型，模拟麦克纳姆轮运动学、弹丸射击物理、装甲板灯条等。

### 2.2 仿真环境搭建

#### 2.2.1 Gazebo仿真引擎

本系统采用Ignition Gazebo（现称Gazebo Sim）Fortress版本作为物理仿真引擎。相比经典Gazebo，Ignition Gazebo具有更现代的架构、更好的性能和更灵活的插件系统。

仿真世界配置如下：

| 参数 | 值 | 说明 |
|------|-----|------|
| 场地模型 | RMUC 2026 | 28m × 15m标准赛场地 |
| 物理引擎 | DART | 默认物理引擎 |
| 实时加速因子 | 4.5× | 仿真时间流速为真实时间的4.5倍 |
| 控制频率 | 30 Hz | 仿真时间基准下的控制频率 |
| 最大步长 | 0.001 s | 物理引擎最大步长 |

实时加速因子（real_time_factor）是训练效率的关键参数。设置为4.5×意味着仿真时间以真实时间4.5倍的速度流逝，从而加速经验收集。系统通过持续监测实测rtf与设定rtf的比值来检测物理精度，当比值偏离[0.7, 1.3]范围时标记仿真不稳定，丢弃该步数据。

#### 2.2.2 机器人模型

机器人采用RMUA19标准步兵模型，包含以下核心组件：

1. **麦克纳姆轮底盘**：4轮全向移动平台，通过Gazebo插件`MecanumDrive2`模拟运动学。底盘支持FOLLOW_GIMBAL模式，底盘跟随云台朝向运动。速度限制为线性速度±2.4 m/s。

2. **云台**：双轴（yaw/pitch）独立控制，通过PID控制器实现角度闭环。云台可执行ABSOLUTE_ANGLE、RELATIVE_ANGLE、VELOCITY等控制模式。

3. **射击机构**：支持17mm弹丸发射，初速度25 m/s，通过Gazebo插件`ProjectileShooter`模拟弹丸物理。射击命令指定弹丸数量，由射击控制器管理发射节奏。

4. **装甲板灯条**：通过Gazebo插件`LightBarController`模拟装甲板LED灯条，用于视觉识别。

#### 2.2.3 场地模型

RMUC 2026场地尺寸为28m × 15m，包含以下关键结构：

- **坡道**：位于场地中部，机器人需要通过坡道到达高地区域。坡道是本系统特化训练的核心挑战。
- **前哨站**：红方前哨站位于(11.0, 11.35, 16.0)，蓝方前哨站位于(17.0, 3.65, 16.0)，初始血量1500。
- **基地**：红方基地位于(2.4, 7.5, 3.15)，蓝方基地位于(25.6, 7.5, 3.15)，初始血量5000。
- **边界**：x ∈ [0, 28]，y ∈ [0, 15]，机器人距边界1m内判定为出界。

#### 2.2.4 裁判系统

仿真环境集成了RoboMaster裁判系统，负责管理：

- 机器人血量（初始400 HP）
- 弹药量（初始300发）
- 攻击伤害判定
- 前哨站与基地状态
- 机器人电源与控制使能
- 游戏状态（倒计时、判负等）

### 2.3 ROS2通信架构

#### 2.3.1 话题通信

ROS2Interface类通过ROS2话题与Gazebo仿真器进行数据交互，主要话题如下：

**发布话题（控制命令）**：

| 话题 | 消息类型 | 说明 |
|------|---------|------|
| `cmd_vel` | geometry_msgs/Twist | 底盘速度命令 |
| `chassis_cmd` | rmoss_interfaces/ChassisCmd | 底盘模式命令 |
| `gimbal_cmd` | rmoss_interfaces/GimbalCmd | 云台控制命令 |
| `shoot_cmd` | rmoss_interfaces/ShootCmd | 射击命令 |
| `referee_cmd` | rmoss_interfaces/RefereeCmd | 裁判系统命令 |

**订阅话题（状态反馈）**：

| 话题 | 消息类型 | 说明 |
|------|---------|------|
| `odom` | nav_msgs/Odometry | 里程计（位置、速度） |
| `chassis_odometry_gt` | nav_msgs/Odometry | 底盘真值里程计 |
| `gimbal_state` | rmoss_interfaces/Gimbal | 云台状态（yaw, pitch） |
| `robot_status` | 自定义 | 机器人状态（HP、弹药） |
| `imu` | sensor_msgs/Imu | IMU数据（用于翻车检测） |
| `pose_info` | tf2_msgs/TFMessage | 所有机器人位姿 |
| `attack_info` | 自定义 | 攻击信息 |
| `outpost_status` | 自定义 | 前哨站状态 |
| `base_status` | 自定义 | 基地状态 |
| `game_status` | 自定义 | 游戏状态 |

#### 2.3.2 服务通信

| 服务 | 类型 | 说明 |
|------|------|------|
| `set_pose` | 自定义 | 设置机器人位姿（用于reset） |
| `exchange_ammo` | rmoss_interfaces/ExchangeAmmon | 弹药兑换 |
| `control_task` | rmoss_interfaces/ControlTask | 任务控制（启动/停止） |

#### 2.3.3 DDS配置

系统使用FastRTPS作为ROS2的DDS中间件，并禁用共享内存传输（`ROS_DISABLE_FASTRTPS_SHM=1`），以避免在仿真加速场景下共享内存同步问题导致的数据延迟。

### 2.4 自定义消息与服务

为满足RoboMaster特定的通信需求，系统定义了以下自定义消息和服务：

**消息定义**：

- `ChassisCmd.msg`：底盘控制命令，支持VELOCITY、FOLLOW_GIMBAL、SWING、SPIN四种模式。
- `GimbalCmd.msg`：云台控制命令，支持ABSOLUTE_ANGLE、RELATIVE_ANGLE、VELOCITY、FOLLOW_CHASSIS四种模式。
- `Gimbal.msg`：云台状态，包含yaw和pitch角度。
- `ShootCmd.msg`：射击命令，包含弹丸数和初速度。

**服务定义**：

- `ControlTask.srv`：任务控制服务，支持START和STOP操作。
- `ExchangeAmmon.srv`：弹药兑换服务。
- `SetColor.srv`：设置机器人颜色（RED/BLUE）。

### 2.5 环境重置机制

环境重置（reset）是强化学习训练的关键环节。本系统采用"温和重置"策略，只重置红方机器人的位姿，不重启整个Gazebo仿真，大幅缩短了重置时间。重置流程如下：

1. 发送START_GAME命令恢复机器人电源和控制使能。
2. 通过set_pose服务设置机器人位姿到初始位置（z=0.35，略高于地面确保落地）。
3. 高频反复执行set_pose + 零速度命令（每5ms一次，持续0.5s），强制位姿归正并清除残余物理速度。
4. 等待机器人落地稳定（z < 0.35且线速度、角速度均小于0.1）。
5. 若重置后翻车，重新执行步骤2-4。
6. 重置环境内部状态（HP、弹药、步数等）。

这种重置机制解决了Gazebo中set_pose只改位姿不清物理速度的问题，避免了翻车/侧滚后角速度残留导致的反复翻车。

---

## 第三章 强化学习环境设计

### 3.1 Gymnasium接口实现

本系统将RoboMaster仿真环境封装为Gymnasium标准环境`RoboMasterGazeboEnv`，实现了以下接口：

- `__init__(config)`：初始化环境，创建ROS2接口、观察空间、动作空间、奖励计算器等。
- `step(action) → (observation, reward, terminated, truncated, info)`：执行一步动作，返回观测、奖励、终止标志和信息。
- `reset(seed, options) → (observation, info)`：重置环境，返回初始观测。
- `render(mode)`：渲染2D俯视图。
- `close()`：关闭环境，保存可达点数据。

### 3.2 观察空间设计

观察空间的设计需要在信息完整性和计算效率之间取得平衡。本系统采用结构化字典观察空间，包含以下13个维度的信息：

| 观测项 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `all_robots` | Box | (10, 4) | 所有机器人位置[id, team, x, y] |
| `own_hp` | Discrete | 401 | 己方血量(0-400) |
| `own_ammo` | Discrete | 301 | 己方弹药量(0-300) |
| `team_economy` | Discrete | 401 | 队伍经济 |
| `remaining_steps` | Discrete | 2049 | 剩余步数 |
| `judge_countdown_steps` | Discrete | 2049 | 判负步数 |
| `damage_per_step` | Box | (1,) | 每步伤害能力 |
| `outpost_hp` | Discrete | 1501 | 前哨站血量(0-1500) |
| `base_hp` | Discrete | 5001 | 基地血量(0-5000) |
| `base_exposed` | Discrete | 2 | 基地展开状态 |
| `target_direction` | Box | (2,) | 目标相对方向[dx, dy] |
| `ammo_consumed_per_step` | Discrete | 301 | 每步弹药消耗 |
| `revive_waiting_steps` | Discrete | 2049 | 复活等待步数 |

**表3-1 观察空间定义**

其中，`all_robots`是最核心的观测项，包含场地中所有机器人的位置信息。每行4个元素：`id`为机器人编号，`team`为阵营标识（0=己方，1=敌方，-1=未知），`x`和`y`为场地坐标。最多支持10台机器人，不足时用id=-1填充。

`target_direction`为从自身位置指向目标（最近敌方或虚拟蓝方）的归一化方向向量，按场地尺寸归一化：$dx = (target_x - own_x) / 14$，$dy = (target_y - own_y) / 7.5$，并裁剪到[-1, 1]范围。

观测预处理模块将字典观测转换为张量输入：`all_robots`经归一化后形成robot_tensor (10, 4)，其余标量观测归一化到[0, 1]后形成state_tensor (13,)。

### 3.3 动作空间设计

动作空间采用离散化设计，由两个子动作组成：

| 动作项 | 类型 | 范围 | 说明 |
|--------|------|------|------|
| `chassis_velocity` | MultiDiscrete | [5, 5] | 底盘速度等级 |
| `shoot` | Discrete | 9 | 射击目标选择 |

**表3-2 动作空间定义**

**底盘速度**：MultiDiscrete([5, 5])表示x和y两个方向各有5个速度等级，映射关系为：

$$v \in \{-2.0, -1.0, 0.0, 1.0, 2.0\} \text{ m/s}$$

离散化设计相比连续动作空间有以下优势：
1. 降低探索空间维度，加速策略收敛。
2. 避免连续分布（如Gaussian）在边界处的概率密度问题。
3. 与Categorical分布配合，天然适合PPO的裁剪机制。

**射击动作**：9类离散选择，含义如下：

| 动作值 | 含义 |
|--------|------|
| 0 | 不射击 |
| 1-6 | 射击对应ID的机器人 |
| 7 | 射击敌方前哨站 |
| 8 | 射击敌方基地 |

**表3-3 射击动作定义**

射击动作隐含了云台自动瞄准逻辑：当选择射击目标时，云台自动对准该目标，无需单独控制云台角度。这种设计简化了动作空间，将云台控制从显式动作降级为隐式执行。

### 3.4 奖励函数设计

奖励函数是强化学习中引导策略学习的核心信号。本系统设计了多层次的奖励函数体系，包括基础奖励、距离塑形奖励和特化模式奖励。

#### 3.4.1 基础奖励

基础奖励覆盖机器人的基本生存与战斗行为：

$$r_{basic} = r_{survive} + r_{be\_hit} + r_{ammo} + r_{hit} + r_{death} + r_{near\_enemy}$$

各项定义如下：

| 奖励项 | 公式 | 权重 | 说明 |
|--------|------|------|------|
| 存活奖励 | $r_{survive} = 0.01$ | 0.01 | 每步存活给小正奖励 |
| 被击惩罚 | $r_{be\_hit} = \frac{\Delta HP}{HP_{max}} \times (-50)$ | -50 | 按HP损失比例惩罚 |
| 弹药消耗 | $r_{ammo} = \Delta ammo \times (-0.1)$ | -0.1 | 每发弹药消耗惩罚 |
| 命中敌人 | $r_{hit} = 50$ | 50 | 命中敌方给大正奖励 |
| 死亡惩罚 | $r_{death} = -20$ | -20 | 死亡给大负奖励 |
| 近敌奖励 | $r_{near} = n_{enemy} \times 0.1$ | 0.1 | 4m内每多一个敌方+0.1 |

**表3-4 基础奖励项**

#### 3.4.2 距离塑形奖励

距离塑形奖励引导机器人向目标移动，包含两个组成部分：

**距离渐变奖励**：距离越近奖励越大，归一化到[0, 1]：

$$r_{dist} = \left(1 - \frac{d}{d_{max}}\right) \times w_{dist}$$

其中$d$为到目标的距离，$d_{max}$为场地最大距离（30m），$w_{dist} = 1.0$。

**距离缩减塑形奖励**：靠近目标给正奖励，远离给惩罚：

$$r_{shaping} = (d_{t-1} - d_t) \times w_{shaping}$$

其中$d_{t-1}$和$d_t$分别为上一步和当前步到目标的距离，$w_{shaping} = 2.0$。这种塑形奖励的优势在于：它直接度量了每一步的进展，比渐变奖励提供更及时的学习信号。

#### 3.4.3 特化模式奖励

特化模式是本系统的核心创新之一，针对特定路径进行分阶段引导训练。路径定义为：

$$\text{起点}(8.64, 3.65) \rightarrow \text{中间点}(4.81, 2.47) \rightarrow \text{目标}(14.0, 7.5)$$

中间点位于坡道位置，机器人必须先到达坡道（阶段1），再从坡道到达目标（阶段2）。

特化模式奖励函数定义为：

$$r_{spec} = r_{shaping} + r_{waypoint} + r_{approach} + r_{climb} + r_{direction} + r_{speed} + r_{time} + r_{tumble} + r_{stuck}$$

**1. 分阶段距离塑形奖励（核心）**

阶段1（未到达中间点）：

$$r_{shaping} = -w_{wp} \times (d_{wp,t} - d_{wp,t-1})$$

阶段2（已到达中间点）：

$$r_{shaping} = -w_{target} \times (d_{target,t} - d_{target,t-1})$$

其中$w_{wp} = 2.5$，$w_{target} = 4.0$。阶段2的权重更大，因为从坡道到目标的距离更远，需要更强的引导信号。

**2. 中间点到达奖励**

$$r_{waypoint} = \begin{cases} R_{wp} \times (1 - d_{wp} / r_{wp}) & \text{if } d_{wp} \leq r_{wp} \text{ and phase 1} \\ 0 & \text{otherwise} \end{cases}$$

其中$R_{wp} = 50.0$，$r_{wp} = 1.0$ m。

**3. 目标到达奖励**

$$r_{approach} = \begin{cases} R_{target} \times (1 - d_{target} / r_{target}) & \text{if } d_{target} \leq r_{target} \text{ and phase 2} \\ 0 & \text{otherwise} \end{cases}$$

其中$R_{target} = 50.0$，$r_{target} = 2.0$ m。

**4. 爬坡奖励/下坡惩罚**

阶段1（上坡段）：

$$r_{climb} = \begin{cases} w_{climb} \times \Delta z & \text{if } \Delta z > 0 \\ w_{descend} \times |\Delta z| & \text{if } \Delta z < 0 \end{cases}$$

其中$w_{climb} = 3.0$，$w_{descend} = -1.5$。

阶段2（下坡/平地段）：

$$r_{climb} = \begin{cases} 0.1 \times \Delta z & \text{if } \Delta z > 0 \\ -1.0 \times |\Delta z| & \text{if } \Delta z < 0 \end{cases}$$

阶段2弱化爬坡奖励但强化下坡惩罚，防止机器人从坡道掉回坡下。

**5. 速度方向一致性奖励**

$$r_{direction} = w_{dir} \times \frac{\vec{v} \cdot \vec{d}_{goal}}{|\vec{v}| \times |\vec{d}_{goal}|}$$

其中$\vec{v}$为当前速度，$\vec{d}_{goal}$为指向当前阶段目标的方向向量，$w_{dir} = 2.0$。该奖励鼓励机器人朝目标方向移动，惩罚反向移动。

**6. 速度大小奖励**

$$r_{speed} = w_{speed} \times |\vec{v}|$$

阶段1：$w_{speed} = 0.5$，阶段2：$w_{speed} = 0.6$。鼓励快速移动，阶段2在平地上可以更快。

**7. 时间惩罚**

$$r_{time} = -0.02$$

每步小惩罚，鼓励尽快到达目标。

**8. 翻车与碰墙惩罚**

翻车惩罚：$r_{tumble} = -10.0$。碰墙惩罚：$r_{stuck} = -1.0$，并触发位置回退（回到30步前的位置）。

### 3.5 终止条件

环境在以下情况下终止（terminated=True）：

1. **血量归零**：$HP \leq 0$。
2. **翻车**：IMU姿态角超过45°或加速度异常，连续5次确认。
3. **出界**：机器人位置距场地边界小于1m。

环境在步数达到上限时截断（truncated=True），上限为2048步。

### 3.6 翻车检测

翻车检测基于IMU数据，采用连续确认机制避免误判：

1. 从IMU四元数计算roll和pitch角。
2. 若$|roll| > 45°$或$|pitch| > 45°$，判定为疑似翻车。
3. 连续5次疑似翻车后，确认翻车并设置terminated=True。

### 3.7 可达点收集与智能重置

为提高训练效率，系统在运行时持续记录机器人能到达的整数坐标点，保存为可达点集合。重置时优先从可达点集合中采样初始位置（加小随机偏移），而非均匀随机采样。这确保了初始位置是物理可达的，避免了将机器人重置到障碍物内部或不可达区域。

---

## 第四章 MAPPO算法与训练策略

### 4.1 MAPPO算法原理

MAPPO（Multi-Agent Proximal Policy Optimization）是PPO在多智能体场景下的扩展，采用集中训练分散执行（CTDE）范式：

- **Actor（策略网络）**：输入局部观测$o_i$，输出动作分布$\pi(a|o_i)$。训练和执行时都只依赖自身观测。
- **Critic（价值网络）**：输入全局状态$s$，输出价值估计$V(s)$。仅在训练时使用，执行时不需要。

在当前实现中，由于每个智能体已经能观测到所有机器人位置，Actor和Critic的输入相同（均为全局观测），但网络参数不共享。

### 4.2 网络结构

#### 4.2.1 Actor网络

Actor网络采用双编码器融合架构，结构如图4-1所示：

```
观测 Dict
  │
  ├── all_robots (10, 4) ──→ RobotEncoder ──→ robot_feat (64)
  ├── 13个标量 ────────────→ StateEncoder ──→ state_feat (64)
  │
  └── concat ──→ Fusion MLP ──→ hidden (128)
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
      ChassisHeadX           ChassisHeadY
      → Categorical(5)       → Categorical(5)
            │                       │
            └───────────┬───────────┘
                        │
                        ▼
                  ShootHead
                  → Categorical(9)
```

**图4-1 Actor网络结构**

**RobotEncoder**：对每台机器人的4维特征[id, team, x, y]通过MLP编码为64维嵌入，然后进行masked平均池化（忽略id=-1的无效位置），输出64维robot_feat。

$$\text{robot\_feat} = \frac{\sum_{i=1}^{N} \mathbb{1}[id_i \neq -1] \cdot \text{MLP}([id_i, team_i, x_i, y_i])}{\sum_{i=1}^{N} \mathbb{1}[id_i \neq -1]}$$

**StateEncoder**：将13维归一化标量状态通过MLP编码为64维state_feat。

**Fusion MLP**：将robot_feat和state_feat拼接后通过两层MLP融合为128维hidden特征。

**动作头**：底盘x方向和y方向各一个DiscreteHead（Categorical(5)），射击一个DiscreteHead（Categorical(9)）。三个动作头独立采样，log_prob为三者之和。

Actor网络总参数量为66,355。

#### 4.2.2 Critic网络

Critic网络与Actor共享编码器结构（RobotEncoder + StateEncoder + Fusion MLP），但不共享权重。融合特征通过ValueHead（两层MLP）输出标量价值估计$V(s)$。

Critic网络总参数量为48,673。

### 4.3 PPO更新规则

PPO-Clip目标函数为：

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

其中重要性采样比率为：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} = \exp(\log \pi_\theta - \log \pi_{\theta_{old}})$$

总损失函数为：

$$L_{total} = L^{CLIP} + c_1 \cdot L^{VF} + c_2 \cdot L^{entropy}$$

其中：
- $L^{VF} = \mathbb{E}[(V_\theta(s_t) - R_t)^2]$为价值损失（MSE）
- $L^{entropy} = -\mathbb{E}[H(\pi_\theta(\cdot|s_t))]$为熵正则化损失
- $c_1 = 0.5$为价值损失系数
- $c_2 = 0.05$为熵正则化系数

### 4.4 GAE优势估计

广义优势估计（Generalized Advantage Estimation, GAE）通过递推计算优势函数，在偏差和方差之间取得平衡：

$$\delta_t = r_t + \gamma V(s_{t+1})(1 - d_t) - V(s_t)$$

$$\hat{A}_t = \delta_t + (\gamma \lambda)(1 - d_t) \hat{A}_{t+1}$$

$$R_t = \hat{A}_t + V(s_t)$$

其中$\gamma = 0.99$为折扣因子，$\lambda = 0.95$为GAE参数，$d_t$为终止标志。

GAE从后往前递推计算，当$d_t = 1$时截断累积（回合结束，不传递未来优势）。

### 4.5 训练超参数

| 超参数 | 符号 | 值 | 说明 |
|--------|------|-----|------|
| 学习率 | $\alpha$ | 3e-4 | Adam优化器学习率 |
| 折扣因子 | $\gamma$ | 0.99 | 重视长远奖励 |
| GAE参数 | $\lambda$ | 0.95 | 偏差-方差平衡 |
| PPO裁剪范围 | $\epsilon$ | 0.2 | 限制策略比率在[0.8, 1.2] |
| PPO更新轮数 | $K$ | 8 | 每批数据重复更新轮数 |
| 小批量大小 | $B$ | 64 | PPO更新时的小批量 |
| 熵正则化系数 | $c_2$ | 0.05 | 鼓励探索 |
| 价值损失系数 | $c_1$ | 0.5 | Critic损失权重 |
| 梯度裁剪范数 | - | 0.5 | 防止梯度爆炸 |
| Rollout步数 | $T$ | 2048 | 每次收集经验的步数 |
| 最大步数/回合 | - | 2048 | 每回合最大步数 |

**表4-1 训练超参数**

### 4.6 课程学习策略

课程学习（Curriculum Learning）通过逐步增加任务难度来加速训练。本系统将导航任务分为4个课程阶段：

| 阶段 | 目标距离范围 | 训练Episodes | 目标 |
|------|-------------|-------------|------|
| 1 | 3-6 m | 300 | 学会靠近近距离目标 |
| 2 | 6-12 m | 400 | 扩展到中距离导航 |
| 3 | 12-20 m | 600 | 扩展到远距离导航 |
| 4 | 3-25 m | 不自动升级 | 全场泛化+战斗 |

**表4-2 课程学习阶段**

每个阶段通过虚拟蓝方位置控制任务难度：

1. 在环境reset时，根据当前阶段的距离范围$[d_{min}, d_{max}]$，在红方初始位置周围随机生成虚拟蓝方位置：

$$d \sim \text{Uniform}(d_{min}, d_{max})$$
$$\theta \sim \text{Uniform}(0, 2\pi)$$
$$\text{blue}_{virtual} = \text{red}_{pos} + d \cdot [\cos\theta, \sin\theta]$$

2. 虚拟蓝方位置仅用于奖励计算和观测覆盖，不实际移动Gazebo中的蓝方机器人。这种"虚拟对手"设计避免了频繁调用set_pose服务带来的仿真开销。

3. 当当前阶段的episode数达到阈值时，自动升级到下一阶段。

### 4.7 特化训练策略

特化训练（Specialization Training）针对特定路径进行专项训练，是课程学习的补充。当启用特化模式时，原奖励函数被特化奖励函数完全替代。

特化训练的路径定义为：

$$P: (8.64, 3.65) \xrightarrow{\text{阶段1: 爬坡}} (4.81, 2.47) \xrightarrow{\text{阶段2: 平地导航}} (14.0, 7.5)$$

该路径的挑战在于：
- 阶段1需要从低地爬上坡道，涉及z轴高度变化。
- 阶段2需要从坡道高地导航到目标点，且不能掉回坡下。

特化训练的关键设计：

1. **固定初始位置**：每次reset都从(8.64, 3.65)开始，确保训练数据的一致性。
2. **分阶段目标**：阶段1目标是坡道中间点，阶段2目标是最终目标点。一旦到达中间点（距离<1.0m），整个episode内保持阶段2。
3. **爬坡奖惩**：阶段1强化z轴上升奖励（权重3.0），阶段2强化z轴下降惩罚（权重-1.0），防止掉回坡下。
4. **卡住检测与回退**：当底盘连续30步速度低于0.05 m/s时判定为卡住，给予惩罚并回退到30步前的位置。

### 4.8 训练流程

完整的训练流程如算法4-1所示：

---

**算法4-1 MAPPO训练流程**

```
输入: 环境Env, Actor网络π_θ, Critic网络V_φ, 超参数
输出: 训练好的Actor网络π_θ*

1: 初始化Actor参数θ, Critic参数φ, 优化器
2: for episode = 1 to N do
3:     重置环境: obs ← Env.reset()
4:     清空缓冲区: buffer.reset()
5:     for step = 1 to T do
6:         采样动作: a, log_prob ← π_θ(obs)
7:         估计价值: v ← V_φ(obs)
8:         执行动作: obs', r, terminated, truncated, info ← Env.step(a)
9:         存入缓冲区: buffer.add(obs, a, r, done, log_prob, v)
10:        obs ← obs'
11:        if done then obs ← Env.reset()
12:    end for
13:    计算GAE和回报: buffer.compute_returns(V_φ(obs))
14:    for epoch = 1 to K do
15:        for minibatch in buffer.get_minibatches() do
16:            计算PPO-Clip损失L^{CLIP}
17:            计算价值损失L^{VF}
18:            计算熵正则化L^{entropy}
19:            总损失L = L^{CLIP} + c_1·L^{VF} + c_2·L^{entropy}
20:            梯度裁剪并更新参数
21:        end for
22:    end for
23:    保存检查点(每100个episode)
24: end for
```

---

### 4.9 检查点管理

训练过程中定期保存检查点，包含以下内容：

- Actor和Critic的网络参数
- 优化器状态
- 当前episode编号
- 历史奖励和步数
- 最佳平均奖励

保存三种类型的检查点：
- `mappo_latest.pt`：最新的检查点，每100个episode保存。
- `mappo_best.pt`：最佳模型，当平均奖励超过历史最佳时保存。
- `mappo_ep{N}.pt`：带episode编号的检查点，便于回溯。

支持从检查点恢复训练，继续累积训练统计。

---

## 第五章 实验与结果分析

### 5.1 实验环境

| 项目 | 配置 |
|------|------|
| 操作系统 | Ubuntu 22.04 LTS |
| ROS版本 | ROS2 Humble Hawksbill |
| 仿真引擎 | Gazebo Fortress (Ignition) |
| Python | 3.10 |
| PyTorch | ≥1.13 (CUDA) |
| DDS | FastRTPS (rmw_fastrtps_cpp) |
| GPU | NVIDIA (CUDA) |

**表5-1 实验环境配置**

### 5.2 训练过程分析

系统在RMUC 2026仿真场地中进行了550个episode的训练，采用特化模式，路径为(8.64, 3.65) → (4.81, 2.47) → (14.0, 7.5)。

#### 5.2.1 训练日志分析

从训练日志中可以观察到以下关键信息：

1. **仿真时间流速**：实测rtf在4.3-4.8之间波动，与设定值4.5基本一致，表明物理仿真精度可接受。

2. **推理速度**：Actor网络推理约5-6 ms/步，环境step约2-3 ms/步，满足实时性要求。

3. **终止原因**：早期episode中，机器人频繁因出界终止（位置超出y方向边界），表明策略尚未学会避障和边界约束。

4. **奖励趋势**：best_avg_reward = -200.00，表明训练仍在进行中，策略尚未收敛到稳定正奖励。

#### 5.2.2 训练阶段特征

**早期阶段（Episode 1-100）**：
- 机器人主要学习避免出界和翻车。
- 频繁因出界终止，位置集中在场地边缘。
- 奖励以负值为主（出界惩罚-20、翻车惩罚-10）。

**中期阶段（Episode 100-300）**：
- 机器人开始学习朝中间点移动。
- 距离塑形奖励开始发挥作用。
- 爬坡行为逐渐出现，但成功率较低。

**后期阶段（Episode 300-550）**：
- 部分episode中机器人能到达中间点。
- 阶段2的导航策略开始形成。
- 但整体成功率仍较低，需要更多训练。

### 5.3 关键设计决策的影响

#### 5.3.1 动作空间离散化

将底盘速度从连续空间离散化为MultiDiscrete([5, 5])是重要的设计决策。离散化的优势在于：

1. **探索效率**：25种底盘速度组合 vs 连续空间的无限组合，大幅降低了探索难度。
2. **训练稳定性**：Categorical分布天然有界，避免了Gaussian分布在边界处的概率密度问题。
3. **与PPO的兼容性**：离散动作的PPO裁剪更直观，不需要tanh-Gaussian的Jacobian校正。

#### 5.3.2 虚拟蓝方位置

使用虚拟蓝方位置而非实际移动Gazebo中的蓝方机器人，有以下优势：

1. **避免仿真开销**：不需要频繁调用set_pose服务。
2. **灵活控制难度**：可以任意设置目标距离，不受Gazebo物理约束。
3. **避免蓝方卡住**：实际移动蓝方可能导致其卡在障碍物中。

#### 5.3.3 温和重置

温和重置（只重置红方位姿，不重启Gazebo）将重置时间从数秒缩短到约1秒，显著提高了训练效率。但需要处理set_pose不清物理速度的问题，通过高频反复set_pose + 零速度命令解决。

### 5.4 系统性能指标

| 指标 | 值 |
|------|-----|
| Actor参数量 | 66,355 |
| Critic参数量 | 48,673 |
| Actor推理时间 | 5-6 ms/步 |
| 环境step时间 | 2-3 ms/步 |
| 仿真加速比 | 4.5× |
| 实测rtf | 4.3-4.8 |
| 重置时间 | ~1 s |
| 训练设备 | CUDA GPU |

**表5-2 系统性能指标**

### 5.5 渲染与可视化

系统实现了2D俯视图渲染器，使用Matplotlib绘制：

- RMUC 2026场地边界与结构
- 所有机器人位置（红方/蓝方标记）
- 前哨站与基地位置
- 虚拟蓝方位置（课程学习目标）
- 训练进度信息（episode、平均奖励、课程阶段）

渲染支持human模式（弹出窗口实时显示）和训练过程中的周期性渲染。

---

## 第六章 总结与展望

### 6.1 本文总结

本文针对RoboMaster比赛中机器人的自主导航与战斗决策问题，提出并实现了一种基于MAPPO强化学习的端到端决策方法。主要工作和贡献如下：

1. **构建了完整的RoboMaster仿真训练环境**：基于ROS2 Humble与Gazebo Fortress，实现了符合RMUC 2026赛制的仿真环境，包括场地模型、机器人物理模型、裁判系统，并封装为Gymnasium标准接口。该环境为强化学习训练提供了高保真的仿真平台，降低了应用门槛。

2. **设计了面向RoboMaster的观察空间与动作空间**：观察空间包含13维信息，动作空间采用离散化设计（MultiDiscrete([5,5]) + Discrete(9)），在信息完整性和训练效率之间取得了平衡。

3. **设计了多层次的奖励函数体系**：基础奖励覆盖生存与战斗行为，距离塑形奖励引导导航，特化模式奖励实现分阶段路径引导。特别是爬坡奖惩和速度方向一致性奖励，有效解决了复杂地形下的导航难题。

4. **提出了课程学习与特化训练相结合的训练策略**：课程学习分4阶段逐步增大目标距离，特化模式针对特定路径进行分阶段引导。两者结合，从简单到复杂、从一般到特殊，有效加速了策略学习。

5. **实现了MAPPO算法的完整训练框架**：双编码器融合架构的Actor网络、集中式Critic、PPO-Clip更新、GAE优势估计、梯度裁剪、检查点管理、暂停控制等，构成了完整的训练工具链。

实验结果表明，在RMUC 2026仿真场地中，经过550个episode的训练，机器人能够逐步学习从起点经坡道到达目标点的导航策略，验证了所提方法的有效性。但由于训练时间有限，策略尚未完全收敛，需要更多训练episode来达到稳定性能。

### 6.2 不足与展望

本文工作仍存在以下不足，有待进一步改进：

1. **训练效率**：当前训练需要大量episode才能收敛，未来可探索以下加速方法：
   - 并行仿真：使用多个Gazebo实例并行收集经验。
   - 优先经验回放：对重要经验（如成功到达目标的episode）增加采样概率。
   - 策略蒸馏：用已训练好的教师策略加速学生策略训练。

2. **策略泛化**：当前特化模式针对固定路径训练，泛化到其他路径的能力有限。未来可：
   - 随机化起点和目标位置，增强泛化能力。
   - 使用域随机化（Domain Randomization）增强仿真到真实的迁移能力。
   - 在多种场地配置下训练，提升跨场景泛化。

3. **多智能体协作**：当前实现为单智能体训练，蓝方为虚拟对手。未来可：
   - 扩展为真正的多智能体训练，红蓝双方同时学习。
   - 引入通信机制，实现多机器人协作决策。
   - 研究对手建模，适应不同对手策略。

4. **Sim-to-Real迁移**：当前策略仅在仿真中验证，迁移到真实机器人面临挑战。未来可：
   - 系统性分析仿真与现实的差距（摩擦系数、惯性参数、传感器噪声等）。
   - 使用域随机化增强策略鲁棒性。
   - 在真实机器人上进行微调（fine-tuning）。

5. **观察空间增强**：当前观察空间不包含视觉信息（相机图像），限制了策略的感知能力。未来可：
   - 引入视觉编码器，从相机图像提取特征。
   - 使用自监督学习预训练视觉特征。
   - 研究多模态融合（视觉+状态）的决策方法。

6. **连续动作空间**：当前动作空间为离散化设计，限制了控制的精细度。未来可：
   - 探索连续动作空间的PPO变体（如tanh-Gaussian）。
   - 研究混合动作空间（底盘连续+射击离散）的训练方法。
   - 使用分层强化学习，高层离散决策+低层连续控制。

---

## 参考文献

[1] Tai L, Paolo G, Liu M. Virtual-to-real deep reinforcement learning: Continuous control of mobile robots for mapless navigation[C]. 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2017: 31-36.

[2] Zhu Y, Mottaghi R, Kolve E, et al. Target-driven visual navigation in environments using deep reinforcement learning[C]. 2017 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2017: 3357-3363.

[3] Lowe R, Wu Y, Tamar A, et al. Multi-agent actor-critic for mixed cooperative-competitive environments[C]. Advances in Neural Information Processing Systems. 2017: 6379-6390.

[4] Schulman J, Wolski F, Dhariwal P, et al. Proximal policy optimization algorithms[J]. arXiv preprint arXiv:1707.06347, 2017.

[5] Yu C, Velu A, Vinitsky E, et al. The surprising effectiveness of PPO in cooperative multi-agent games[C]. Advances in Neural Information Processing Systems. 2022: 24611-24624.

[6] Schulman J, Moritz P, Levine S, et al. High-dimensional continuous control using generalized advantage estimation[C]. International Conference on Learning Representations (ICLR). 2016.

[7] Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning[J]. Nature, 2015, 518(7540): 529-533.

[8] Silver D, Huang A, Maddison C J, et al. Mastering the game of Go with deep neural networks and tree search[J]. Nature, 2016, 529(7587): 484-489.

[9] OpenAI, Berner C, Brockman G, et al. Dota 2 with large scale deep reinforcement learning[J]. arXiv preprint arXiv:1912.06680, 2019.

[10] Rashid T, Samvelyan M, De Witt C S, et al. QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning[C]. International Conference on Machine Learning. 2018: 4295-4304.

[11] Bengio Y, Louradour J, Collobert R, et al. Curriculum learning[C]. Proceedings of the 26th annual international conference on machine learning. 2009: 41-48.

[12] Tobin J, Fong R, Ray A, et al. Domain randomization for transferring deep neural networks from simulation to the real world[C]. 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2017: 23-30.

[13] Akhtarzada A, Kolve E, Mottaghi R, et al. The AI2-THOR framework for deep reinforcement learning in interactive 3D environments[J]. arXiv preprint arXiv:1707.06623, 2017.

[14] Savva M, Kadian A, Maksymets O, et al. Habitat: A platform for embodied AI research[C]. Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 9339-9347.

[15] Feng L, Liang R, Liu Z, et al. RMOSS: A flexible and modular software framework for RoboMaster robot system development[C]. 2022 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2022: 4755-4761.

---

## 致谢

感谢深圳北理莫斯科大学极熊战队（SMBU-PolarBear-Robotics-Team）提供的RoboMaster开源框架（rmoss_core、rmoss_gazebo、rmoss_gz_resources等），为本文的仿真环境搭建提供了重要基础。感谢ROS2与Gazebo开源社区提供的优秀仿真工具。感谢PyTorch与Gymnasium开源社区提供的深度学习与强化学习框架。

---

## 附录A 系统构建与运行

### A.1 依赖安装

```bash
# ROS2 Humble
sudo apt install ros-humble-desktop

# Gazebo Fortress (Ignition)
sudo apt install ros-humble-ros-gz

# Python依赖
pip install torch gymnasium numpy opencv-python shapely transforms3d matplotlib
```

### A.2 工作空间构建

```bash
cd /home/xufurui/ros_ws
colcon build --symlink-install
source install/setup.bash
```

### A.3 启动仿真

```bash
# 有GUI模式
./start_sim.sh

# 无GUI模式（服务器训练）
./start_sim_headless.sh
```

### A.4 启动训练

```bash
# Python脚本
python run_train.py

# Shell脚本
./run_train.sh

# 自定义参数
python -m robomaster_mappo.train --num_episodes 3000 --lr 3e-4 --render
```

### A.5 测试模型

```bash
python test_model.py
```

### A.6 监控机器人位置

```bash
python monitor_pos.py
```

---

## 附录B 关键代码文件索引

| 文件路径 | 说明 |
|---------|------|
| `src/robomaster_gym_env/robomaster_gym_env/robomaster_env.py` | Gymnasium环境主类 |
| `src/robomaster_gym_env/robomaster_gym_env/config.py` | 环境配置 |
| `src/robomaster_gym_env/robomaster_gym_env/observation_space.py` | 观察空间定义 |
| `src/robomaster_gym_env/robomaster_gym_env/action_space.py` | 动作空间定义 |
| `src/robomaster_gym_env/robomaster_gym_env/reward_calculator.py` | 奖励计算器 |
| `src/robomaster_gym_env/robomaster_gym_env/ros2_interface.py` | ROS2通信管理器 |
| `src/robomaster_gym_env/robomaster_gym_env/interface_adapter.py` | 接口适配层 |
| `src/robomaster_gym_env/robomaster_gym_env/env_renderer.py` | 2D俯视图渲染器 |
| `src/robomaster_mappo/robomaster_mappo/train.py` | MAPPO训练主循环 |
| `src/robomaster_mappo/robomaster_mappo/actor.py` | Actor网络 |
| `src/robomaster_mappo/robomaster_mappo/critic.py` | Critic网络 |
| `src/robomaster_mappo/robomaster_mappo/rollout_buffer.py` | 经验回放缓冲区 |
| `src/robomaster_mappo/robomaster_mappo/obs_preprocessor.py` | 观测预处理 |
| `src/rmoss_gazebo/rmoss_gz_plugins/` | Gazebo物理插件 |
| `src/rmu_gazebo_simulator/` | 仿真环境集成 |
| `run_train.py` | 训练启动脚本 |
| `test_model.py` | 模型测试脚本 |
