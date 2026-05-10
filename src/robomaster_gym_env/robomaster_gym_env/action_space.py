"""
动作空间定义
定义Gym环境的动作空间,包括底盘控制、射击等

云台锁定初始朝向, 不作为动作空间的一部分
"""

import numpy as np
from typing import Dict, Any, Tuple
from gymnasium import spaces


class ActionSpace:
    """动作空间管理类"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化动作空间

        Args:
            config: 动作配置字典,包含控制限制等
        """
        self.config = config
        
        # 离散速度等级映射表
        self.velocity_levels = [-2.0, -1.0, 0.0, 1.0, 2.0]
        
        self.action_space = self._build_action_space()

    def _build_action_space(self) -> spaces.Dict:
        """构建动作空间 - 使用离散速度空间"""
        action_spaces = {}

        # 1. 底盘速度控制 - 改为离散空间
        if self.config.get('chassis_velocity', False):
            # 离散速度等级: [-2, -1, 0, 1, 2] m/s
            # MultiDiscrete([5, 5]) 表示两个维度,每个维度0-4
            action_spaces['chassis_velocity'] = spaces.MultiDiscrete([5, 5])

        # 2. 射击控制
        if self.config.get('shoot', False):
            # 离散动作: 0=不射击, 1~6=射击机器人, 7=射击前哨站, 8=射击基地
            action_spaces['shoot'] = spaces.Discrete(9)

        # 3. 金币复活控制
        if self.config.get('revive_with_coins', False):
            # bool类型: 0=不复活, 1=花费金币复活
            action_spaces['revive_with_coins'] = spaces.Discrete(2)

        # 4. 远程弹药兑换
        if self.config.get('remote_ammo_exchange', False):
            # bool类型: 0=不兑换, 1=远程兑换弹药(150金币100发)
            action_spaces['remote_ammo_exchange'] = spaces.Discrete(2)

        # 5. 非远程弹药兑换
        if self.config.get('local_ammo_exchange', False):
            # bool类型: 0=不兑换, 1=非远程兑换弹药(10金币10发)
            action_spaces['local_ammo_exchange'] = spaces.Discrete(2)

        return spaces.Dict(action_spaces)

    def parse_action(self, action: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        解析动作,转换为控制命令

        Args:
            action: 动作字典

        Returns:
            控制命令字典
        """
        commands = {}

        # 1. 底盘速度 - 从离散索引映射到实际速度
        if 'chassis_velocity' in action:
            vel_idx = action['chassis_velocity']  # [idx_x, idx_y], 每个在0-4之间
            linear_x = self.velocity_levels[int(vel_idx[0])]
            linear_y = self.velocity_levels[int(vel_idx[1])]
            
            commands['chassis'] = {
                'linear_x': linear_x,
                'linear_y': linear_y,
                'angular_z': 0.0
            }

        # 2. 射击
        if 'shoot' in action:
            shoot_action = int(action['shoot'])
            if shoot_action == 0:
                commands['shoot'] = {'fire': False}
            elif 1 <= shoot_action <= 6:
                # 射击机器人 (1-6)
                commands['shoot'] = {'fire': True, 'target_type': 'robot', 'target_id': shoot_action}
            elif shoot_action == 7:
                # 射击前哨站
                commands['shoot'] = {'fire': True, 'target_type': 'outpost'}
            elif shoot_action == 8:
                # 射击基地
                commands['shoot'] = {'fire': True, 'target_type': 'base'}

        # 4. 金币复活
        if 'revive_with_coins' in action:
            commands['revive_with_coins'] = bool(action['revive_with_coins'])

        # 5. 远程弹药兑换
        if 'remote_ammo_exchange' in action:
            commands['remote_ammo_exchange'] = bool(action['remote_ammo_exchange'])

        # 6. 非远程弹药兑换
        if 'local_ammo_exchange' in action:
            commands['local_ammo_exchange'] = bool(action['local_ammo_exchange'])

        return commands

    def get_empty_action(self) -> Dict[str, np.ndarray]:
        """获取空动作(用于初始化)"""
        action = {}

        for key, space in self.action_space.spaces.items():
            if isinstance(space, spaces.Box):
                action[key] = np.zeros(space.shape, dtype=space.dtype)
            elif isinstance(space, spaces.Discrete):
                action[key] = 0

        return action

    def sample_random_action(self) -> Dict[str, np.ndarray]:
        """采样随机动作"""
        action = {}

        for key, space in self.action_space.spaces.items():
            action[key] = space.sample()

        return action

    def clip_action(self, action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        裁剪动作到合法范围

        Args:
            action: 原始动作

        Returns:
            裁剪后的动作
        """
        clipped_action = {}

        for key, value in action.items():
            if key in self.action_space.spaces:
                space = self.action_space.spaces[key]
                if isinstance(space, spaces.Box):
                    clipped_action[key] = np.clip(value, space.low, space.high)
                else:
                    clipped_action[key] = value

        return clipped_action

