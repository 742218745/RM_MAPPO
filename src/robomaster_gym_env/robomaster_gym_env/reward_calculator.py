"""
奖励计算器
基于ROS2数据计算强化学习奖励
"""

from typing import Dict, Any
import numpy as np


class RewardCalculator:
    """奖励计算器
    
    从ROS2接口获取数据并计算奖励,不再维护状态
    """

    def __init__(self, reward_config: Dict[str, float]):
        """
        初始化奖励计算器

        Args:
            reward_config: 奖励配置
        """
        self.reward_config = reward_config

        # 上一步的状态(用于计算变化)
        self.last_hp = 400
        self.last_ammo = 300
        self.last_nearest_enemy_distance = None

    def calculate_reward(
        self,
        current_hp: int,
        current_ammo: int,
        attack_info: str = "",
        is_alive: bool = True,
        near_enemy_count: int = 0,
        nearest_enemy_distance: float = float('inf'),
        max_field_distance: float = 30.0,
        virtual_blue_distance: float = None,
        direction_alignment: float = None,
    ) -> float:
        """
        计算当前步的奖励

        Args:
            current_hp: 当前血量
            current_ammo: 当前弹药数
            attack_info: 攻击信息
            is_alive: 是否存活
            near_enemy_count: 4m范围内的敌方机器人数量
            nearest_enemy_distance: 最近敌方机器人的距离(米)
            max_field_distance: 场地最大距离(用于归一化)
            virtual_blue_distance: 虚拟蓝方距离(米), 若提供则替代nearest_enemy_distance用于距离奖励
            direction_alignment: 速度在目标方向上的投影 [-1,1], 正值=朝目标走, 负值=背离

        Returns:
            float: 奖励值
        """
        reward = 0.0

        # 1. 存活奖励
        if is_alive:
            reward += self.reward_config.get('survive_per_step', 0)

        # 2. 被击中惩罚
        hp_loss = self.last_hp - current_hp
        if hp_loss > 0:
            max_hp = 400  # 默认最大血量
            reward += hp_loss * self.reward_config.get('be_hit', 0) / max_hp

        # 3. 弹丸消耗惩罚
        ammo_used = self.last_ammo - current_ammo
        if ammo_used > 0:
            reward += ammo_used * self.reward_config.get('ammo_usage', -0.1)

        # 4. 命中敌人奖励 (从攻击信息解析)
        if attack_info and 'hit' in attack_info.lower():
            reward += self.reward_config.get('hit_enemy', 100.0)

        # 5. 死亡惩罚
        if not is_alive:
            reward += self.reward_config.get('out_of_boundary', -100.0)

        # 6. 在敌方4m范围内的奖励
        if near_enemy_count > 0:
            reward += near_enemy_count * self.reward_config.get('near_enemy_bonus', 0.1)

        # 决定用于距离奖励的距离值: 优先使用虚拟蓝方距离
        effective_distance = nearest_enemy_distance
        if virtual_blue_distance is not None:
            effective_distance = virtual_blue_distance

        # 7. 距离渐变奖励: 距离越近奖励越大, 归一化到[0, 1]再乘以权重
        distance_reward_weight = self.reward_config.get('distance_reward', 0.0)
        if distance_reward_weight != 0 and effective_distance < float('inf'):
            # 归一化: 距离0时奖励为1, 距离max_field_distance时奖励为0
            normalized_distance = min(effective_distance / max_field_distance, 1.0)
            distance_reward = (1.0 - normalized_distance) * distance_reward_weight
            reward += distance_reward

        # 8. 距离缩减塑形奖励: 靠近敌方为正, 远离为负
        distance_shaping_weight = self.reward_config.get('distance_shaping', 0.0)
        if distance_shaping_weight != 0 and self.last_nearest_enemy_distance is not None \
                and effective_distance < float('inf'):
            # delta > 0 表示靠近了(上一步距离 - 当前距离)
            delta = self.last_nearest_enemy_distance - effective_distance
            reward += delta * distance_shaping_weight

        # 更新上一步状态
        self.last_hp = current_hp
        self.last_ammo = current_ammo
        if effective_distance < float('inf'):
            self.last_nearest_enemy_distance = effective_distance

        return reward

    def reset(self):
        """重置奖励计算器"""
        self.last_hp = 400
        self.last_ammo = 300
        self.last_nearest_enemy_distance = None
