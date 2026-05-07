"""
数据处理层

该模块负责处理和转换接口数据，包括：
1. 机器人位置数据处理
2. 游戏状态数据处理
3. 数据有效性验证

# 注:该文件怎么全是unknown:当 ROS2 数据还没到达时，所有方法都会返回 unknown 值，这是正常的容错设计。
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from .unknown_state_handler import UnknownStateHandler


class DataProcessor:
    """数据处理器

    负责处理和转换接口数据为观察空间所需的格式。
    """

    def __init__(self, max_robots: int = 10):
        """初始化数据处理器

        Args:
            max_robots: 最大机器人数量
        """
        self.max_robots = max_robots

    def process_robot_positions(
        self,
        robot_poses: Optional[List[Dict[str, Any]]],
        own_robot_id: int,
        own_team: str = 'red'
    ) -> np.ndarray:
        """处理所有机器人位置数据

        Args:
            robot_poses: 机器人位置列表，每个元素包含：
                - robot_id: 机器人ID
                - team: 队伍 ('red' or 'blue')
                - x: x坐标
                - y: y坐标
            own_robot_id: 自己的机器人ID
            own_team: 自身队伍颜色 ('red' or 'blue')，默认红色

        Returns:
            np.ndarray: 所有机器人位置数组 [10, 4] (id, team, x, y)
                - id: 机器人ID, -1表示unknown
                - team: 队伍关系, 0=己方(ally), 1=敌方(enemy), -1=unknown
                - x, y: 坐标, nan表示unknown
        """
        # 初始化结果数组，全部标记为unknown
        all_robots = UnknownStateHandler.create_unknown_robot_array(self.max_robots)

        # 如果数据缺失，直接返回unknown数组
        if robot_poses is None:
            return all_robots

        try:
            # 按顺序填充有效数据 (己方优先, 然后敌方)
            # 先排序: 己方在前, 敌方在后
            sorted_poses = sorted(robot_poses, key=lambda p: (
                0 if p.get('team', 'unknown') == own_team else
                1 if p.get('team', 'unknown') in ('red', 'blue') else 2
            ))

            idx = 0
            for pose in sorted_poses:
                if idx >= self.max_robots:
                    break

                robot_id = pose.get('robot_id', UnknownStateHandler.UNKNOWN_INT)
                team_str = pose.get('team', 'unknown')
                x = pose.get('x', UnknownStateHandler.UNKNOWN_FLOAT)
                y = pose.get('y', UnknownStateHandler.UNKNOWN_FLOAT)

                # 根据自身颜色判断己方/敌方
                # 0=己方(ally), 1=敌方(enemy), -1=unknown
                if team_str == own_team:
                    team = 0  # 己方
                elif team_str in ('red', 'blue'):
                    team = 1  # 敌方
                else:
                    team = -1  # unknown

                all_robots[idx] = [idx, team, x, y]
                idx += 1

            return all_robots

        except Exception as e:
            # 处理失败，返回unknown数组
            return UnknownStateHandler.create_unknown_robot_array(self.max_robots)

    def process_game_state(
        self,
        game_state: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """处理游戏状态数据

        Args:
            game_state: 游戏状态字典，包含：
                - own_hp: 己方血量
                - own_ammo: 己方弹药量
                - team_economy: 我方经济
                - judge_countdown: 判负时间

        Returns:
            Dict[str, Any]: 处理后的游戏状态
        """
        # 如果数据缺失，返回unknown状态
        if game_state is None:
            return {
                'own_hp': UnknownStateHandler.UNKNOWN_INT,
                'own_ammo': UnknownStateHandler.UNKNOWN_INT,
                'team_economy': UnknownStateHandler.UNKNOWN_INT,
                'judge_countdown': UnknownStateHandler.UNKNOWN_INT
            }

        try:
            # 提取并验证数据
            own_hp = game_state.get('own_hp', UnknownStateHandler.UNKNOWN_INT)
            own_ammo = game_state.get('own_ammo', UnknownStateHandler.UNKNOWN_INT)
            team_economy = game_state.get('team_economy', UnknownStateHandler.UNKNOWN_INT)
            judge_countdown = game_state.get('judge_countdown', UnknownStateHandler.UNKNOWN_INT)

            # 验证数据有效性
            own_hp = self._validate_int(own_hp, min_val=0)
            own_ammo = self._validate_int(own_ammo, min_val=0)
            team_economy = self._validate_int(team_economy, min_val=0)
            judge_countdown = self._validate_int(judge_countdown, min_val=0)

            return {
                'own_hp': own_hp,
                'own_ammo': own_ammo,
                'team_economy': team_economy,
                'judge_countdown': judge_countdown
            }

        except Exception as e:
            # 处理失败，返回unknown状态
            return {
                'own_hp': UnknownStateHandler.UNKNOWN_INT,
                'own_ammo': UnknownStateHandler.UNKNOWN_INT,
                'team_economy': UnknownStateHandler.UNKNOWN_INT,
                'judge_countdown': UnknownStateHandler.UNKNOWN_INT
            }

    def process_remaining_steps(
        self,
        current_step: int,
        max_steps: int
    ) -> int:
        """处理剩余步数

        Args:
            current_step: 当前步数
            max_steps: 最大步数

        Returns:
            int: 剩余步数
        """
        try:
            remaining = max_steps - current_step
            if remaining < 0:
                return 0
            return remaining
        except Exception:
            return UnknownStateHandler.UNKNOWN_INT

    def validate_data(self, data: Any, data_type: str) -> bool:
        """验证数据有效性

        Args:
            data: 待验证的数据
            data_type: 数据类型

        Returns:
            bool: True表示有效, False表示无效
        """
        if data is None:
            return False

        if UnknownStateHandler.is_unknown(data):
            return False

        # 根据数据类型进行验证
        if data_type == 'robot_array':
            if not isinstance(data, np.ndarray):
                return False
            if data.shape != (self.max_robots, 4):
                return False
            return True

        elif data_type == 'int':
            if not isinstance(data, (int, np.integer)):
                return False
            return data >= 0

        elif data_type == 'float':
            if not isinstance(data, (float, np.floating)):
                return False
            return not np.isnan(data) and data >= 0

        return True

    # ==================== 辅助方法 ====================

    def _validate_int(
        self,
        value: int,
        min_val: int = 0,
        max_val: int = None
    ) -> int:
        """验证整数值

        Args:
            value: 待验证的值
            min_val: 最小值
            max_val: 最大值

        Returns:
            int: 验证后的值, 无效时返回UNKNOWN_INT
        """
        if value == UnknownStateHandler.UNKNOWN_INT:
            return value

        if not isinstance(value, (int, np.integer)):
            return UnknownStateHandler.UNKNOWN_INT

        if value < min_val:
            return UnknownStateHandler.UNKNOWN_INT

        if max_val is not None and value > max_val:
            return UnknownStateHandler.UNKNOWN_INT

        return int(value)

    def _validate_float(
        self,
        value: float,
        min_val: float = 0.0,
        max_val: float = None
    ) -> float:
        """验证浮点数值

        Args:
            value: 待验证的值
            min_val: 最小值
            max_val: 最大值

        Returns:
            float: 验证后的值, 无效时返回UNKNOWN_FLOAT
        """
        if np.isnan(value):
            return value

        if not isinstance(value, (float, np.floating)):
            return UnknownStateHandler.UNKNOWN_FLOAT

        if value < min_val:
            return UnknownStateHandler.UNKNOWN_FLOAT

        if max_val is not None and value > max_val:
            return UnknownStateHandler.UNKNOWN_FLOAT

        return float(value)
