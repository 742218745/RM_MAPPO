"""
Gym环境配置文件
定义所有ROS2节点、Topics、消息类型等配置信息
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class GymEnvConfig:
    """Gym环境配置"""
    # 机器人配置
    robot_name: str = "red_standard_robot1"
    robot_namespace: str = "red_standard_robot1"  # ✅ 修复:不要以斜杠开头
    team: str = "red"

    # 仿真配置
    use_sim_time: bool = True
    timeout_steps: int = 10000
    control_frequency: float = 30.0  # Hz (仿真时间基准)

    # 动作空间配置
    # 云台锁定初始朝向, 不作为动作空间
    action_config: Dict[str, bool] = field(default_factory=lambda: {
        'chassis_velocity': True,
        'shoot': True,
    })

    # 奖励配置
    reward_config: Dict[str, float] = field(default_factory=lambda: {
        'hit_enemy': 50.0,
        'be_hit': -50.0,
        # 存活奖励
        'survive_per_step': 0.01,
        # 弹药惩罚
        'ammo_usage': -0.1,
        'out_of_boundary': -20.0,  # 出界惩罚 100→20, 避免压过到达奖励
        # 翻车惩罚 (翻车时 terminated=True, 奖励覆盖为此值)
        'tumble': -10.0,
        # 碰墙惩罚 (底盘卡住时每步惩罚)
        'stuck_penalty': -1.0,
        # 在敌方4m范围内的奖励
        'near_enemy_bonus': 0.1,
        # 距离渐变奖励: (1 - distance/max_distance) * weight, 距离越近奖励越大
        'distance_reward': 1.0,
        # 距离缩减塑形奖励: (last_dist - cur_dist) * weight, 靠近为正远离为负
        'distance_shaping': 2.0,
        # 场地最大距离(用于距离渐变奖励归一化)
        'max_field_distance': 30.0,
    })

    # 课程学习配置
    curriculum_config: Dict[str, any] = field(default_factory=lambda: {
        # 是否启用课程学习
        'enabled': True,
        # 当前训练阶段: 1=近距离, 2=中距离, 3=远距离, 4=全场随机
        'stage': 1,
        # 各阶段蓝方距离范围(相对于红方初始位置的最小/最大距离, 单位m)
        'stage_ranges': {
            1: (3.0, 6.0),    # 近距离: 3-6m
            2: (6.0, 12.0),   # 中距离: 6-12m
            3: (12.0, 20.0),  # 远距离: 12-20m
            4: (3.0, 25.0),   # 全场随机: 3-25m
        },
        # 各阶段训练episode数(达到后自动升级阶段)
        'stage_episodes': {
            1: 300,
            2: 400,
            3: 600,
            4: -1,  # -1表示不自动升级
        },
        # 使用虚拟蓝方位置(不移动Gazebo中的蓝方, 仅在奖励计算中使用)
        'use_virtual_blue': True,
        # 虚拟蓝方位置是否覆盖观测空间中的蓝方位置
        # True: 观测中蓝方位置=虚拟位置(策略网络看到虚拟目标)
        # False: 观测中蓝方位置=真实Gazebo位置(策略网络看到真实位置)
        'virtual_blue_override_obs': True,
    })

    # 虚拟蓝方位置(由课程学习在reset时生成, 不影响Gazebo仿真)
    virtual_blue_x: float = 9.4   # 默认与gz_world.yaml中蓝方初始位置一致
    virtual_blue_y: float = 9.5

    # 特化模式配置 (用于专项训练, 启用后原奖励不生效)
    # 分阶段引导: 起点 → 中间点(坡) → 目标点
    specialize_config: Dict[str, any] = field(default_factory=lambda: {
        'enabled': True,
        # 固定初始位置 (不随机落点)
        'start_x': 8.64,
        'start_y': 3.65,
        # 固定目标位置 (地图中心)
        'target_x': 14.0,
        'target_y': 7.5,
        # 中间点坐标 (必须经过的坡道点)
        'waypoint_x': 4.81,
        'waypoint_y': 2.47,
        'waypoint_arrive_radius': 1.0,    # 到达中间点的判定半径(m)
        'waypoint_arrive_reward': 50.0,   # 到达中间点的奖励 10.0→50.0
        'waypoint_shaping_weight': 2.5,   # 阶段1: 距离塑形权重(靠近中间点) 3.0→2.5
        # 目标到达奖励
        'approach_radius': 2.0,           # 到达目标的判定半径(m)
        'approach_reward': 50.0,          # 到达目标的奖励 10.0→50.0
        'target_shaping_weight': 4.0,     # 阶段2: 距离塑形权重(靠近目标) 2.5→4.0
        # 爬坡奖励 (阶段1: 上坡段, 强化)
        'climb_reward': 3.0,              # 阶段1: z轴上升奖励 1.0→3.0
        'descend_penalty': -1.5,          # 阶段1: z轴下降惩罚 -0.5→-1.5
        # 爬坡奖励 (阶段2: 下坡/平地段, 弱化)
        'climb_reward_phase2': 0.1,       # 阶段2: z轴上升奖励(弱)
        'descend_penalty_phase2': -1.0,   # 阶段2: z轴下降惩罚(强, 防止掉回坡下)
        # 速度方向与目标方向一致性
        'direction_reward': 2.0,          # 速度朝目标方向投影的奖励 1.0→2.0
        # 速度大小奖励 (鼓励快速移动)
        'speed_reward': 0.5,              # 阶段1: 速度大小奖励 0.2→0.5
        'speed_reward_phase2': 0.6,       # 阶段2: 速度大小奖励(平地可更快) 0.3→0.6
        # 反向速度惩罚
        'reverse_penalty': -1.0,          # 速度与目标方向反向时的惩罚 -0.5→-1.0
        # 时间惩罚 (每步)
        'time_penalty': -0.02,
        # 卡住时回退步数
        'stuck_rollback_steps': 30,
    })

    # 控制限制
    chassis_velocity_limit: Dict[str, float] = field(default_factory=lambda: {
        'linear_x_max': 2.4,    # m/s (每步0.08m @ 30Hz)
        'linear_y_max': 2.4,    # m/s
    })

    shoot_config: Dict[str, float] = field(default_factory=lambda: {
        'projectile_velocity': 25.0,  # m/s
        'max_projectile_num': 5,
        'cooldown_time': 0.1,   # s
    })

    # 场地固定目标坐标 (单位: m)
    # 红方前哨站/基地 (己方为红时)
    red_outpost_position: Tuple[float, float, float] = (11.0, 11.35, 16.0)
    red_base_position: Tuple[float, float, float] = (2.4, 7.5, 3.15)
    # 蓝方前哨站/基地 (己方为蓝时)
    blue_outpost_position: Tuple[float, float, float] = (17.0, 3.65, 16.0)
    blue_base_position: Tuple[float, float, float] = (25.6, 7.5, 3.15)


# 默认配置实例
DEFAULT_GYM_CONFIG = GymEnvConfig()
