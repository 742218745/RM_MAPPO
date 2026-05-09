"""
接口适配层

该模块负责适配ROS2接口, 提供统一的数据获取接口,
并处理数据缺失和异常情况。
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from .unknown_state_handler import UnknownStateHandler


@dataclass
class InterfaceData:
    """接口数据容器

    用于存储从各个接口获取的原始数据。
    """
    # 机器人位置信息
    pose_info: Optional[Any] = None

    # 裁判系统数据
    robot_status: Optional[Dict[str, Any]] = None
    game_status: Optional[Dict[str, Any]] = None

    # 环境状态
    current_step: int = 0
    max_steps: int = 2048


class InterfaceAdapter:
    """接口适配器

    封装ROS2接口, 提供统一的数据获取方法,
    处理数据缺失和异常情况。
    """

    def __init__(
        self,
        ros2_interface,
        own_robot_id: int = 0,
        own_team: str = 'red'
    ):
        """初始化接口适配器

        Args:
            ros2_interface: ROS2接口对象
            own_robot_id: 自己的机器人ID
            own_team: 自己的队伍颜色 ('red' or 'blue')，默认红色
        """
        self.ros2_interface = ros2_interface
        self.own_robot_id = own_robot_id
        self.own_team = own_team

    def get_all_robot_poses(self) -> Optional[List[Dict[str, Any]]]:
        """获取所有机器人的位置信息

        Returns:
            Optional[List[Dict]]: 机器人位置列表，每个元素包含：
                - robot_id: 机器人ID
                - team: 队伍 ('red' or 'blue')
                - x: x坐标
                - y: y坐标
            如果数据缺失返回None
        """
        try:
            # 从ROS2接口获取pose_info
            referee_data = self.ros2_interface.get_referee_data()
            pose_info = referee_data.get('pose_info')

            if pose_info is None:
                return None

            # 解析位置信息
            # Gazebo -allow_renaming 会产生带_0后缀的模型 (实际在场地上的)
            # 优先使用_0后缀的, 忽略无后缀的(可能是异常位置)
            robot_poses = []
            seen_teams = set()
            # 先收集_0后缀的
            for transform in pose_info:
                child_frame_id = transform.child_frame_id
                if 'standard_robot' not in child_frame_id:
                    continue
                if not child_frame_id.endswith('_0'):
                    continue
                # 提取队伍
                team = self._extract_team(child_frame_id)
                if team in seen_teams:
                    continue
                seen_teams.add(team)
                robot_id = self._extract_robot_id(child_frame_id)
                x = transform.transform.translation.x
                y = transform.transform.translation.y
                robot_poses.append({
                    'robot_id': robot_id,
                    'team': team,
                    'x': x,
                    'y': y
                })
            # 如果_0后缀的没找到, 回退到无后缀的
            if not robot_poses:
                for transform in pose_info:
                    child_frame_id = transform.child_frame_id
                    if 'standard_robot' not in child_frame_id:
                        continue
                    if child_frame_id.endswith('_0'):
                        continue
                    team = self._extract_team(child_frame_id)
                    if team in seen_teams:
                        continue
                    seen_teams.add(team)
                    robot_id = self._extract_robot_id(child_frame_id)
                    x = transform.transform.translation.x
                    y = transform.transform.translation.y
                    robot_poses.append({
                        'robot_id': robot_id,
                        'team': team,
                        'x': x,
                        'y': y
                    })

            return robot_poses

        except Exception as e:
            # 数据获取失败，返回None
            return None

    def get_game_state(self) -> Optional[Dict[str, Any]]:
        """获取游戏状态信息

        Returns:
            Optional[Dict]: 游戏状态字典，包含：
                - own_hp: 己方血量
                - own_ammo: 己方弹药量
                - team_economy: 我方经济
                - judge_countdown: 判负时间
            如果数据缺失返回None
        """
        try:
            # 从裁判系统获取数据
            robot_status = self.ros2_interface.get_robot_status()

            if robot_status is None:
                return None

            # 提取状态信息
            game_state = {
                'own_hp': robot_status.get('remain_hp', 400),  # 默认400
                'own_ammo': robot_status.get('projectile_num', 300),  # 默认300
                'team_economy': 0,  # 默认0
                'judge_countdown': 2100  # 默认2100
            }

            # 尝试获取game_status（如果可用）
            referee_data = self.ros2_interface.get_referee_data()
            if 'game_status' in referee_data:
                game_status = referee_data['game_status']
                game_state['team_economy'] = game_status.get('team_economy', 0)
                game_state['judge_countdown'] = game_status.get('judge_countdown', 2100)

            return game_state

        except Exception as e:
            # 数据获取失败，返回None
            return None

    def get_own_team(self) -> str:
        """获取自身队伍颜色

        优先从ROS2仿真接口读取，如果不可用则使用初始化时的默认值。

        Returns:
            str: 自身队伍颜色 ('red' or 'blue')
        """
        try:
            # 尝试从ROS2接口获取自身颜色
            if self.ros2_interface is not None:
                team_from_sim = self.ros2_interface.get_own_team()
                if team_from_sim in ('red', 'blue'):
                    # 更新缓存的own_team
                    self.own_team = team_from_sim
                    return team_from_sim
        except Exception:
            pass

        # 回退到默认值
        return self.own_team

    def get_map_info(self) -> Optional[Dict[str, Any]]:
        """获取地图信息

        Returns:
            Optional[Dict]: 地图信息字典
        """
        # 地图信息从Gazebo仿真获取，这里返回None
        return None

    def get_outpost_and_base_state(self) -> Tuple[int, int, bool]:
        """获取前哨站和基地状态

        Returns:
            Tuple[int, int, bool]: (前哨站血量, 基地血量, 基地展开状态)
        """
        try:
            # 从ROS2接口获取数据
            referee_data = self.ros2_interface.get_referee_data()

            # 获取前哨站血量
            outpost_hp = UnknownStateHandler.UNKNOWN_INT
            if 'outpost_status' in referee_data:
                outpost_hp = referee_data['outpost_status'].get('remain_hp', 1500)  # 默认1500

            # 获取基地血量和展开状态
            base_hp = UnknownStateHandler.UNKNOWN_INT
            base_exposed = False
            if 'base_status' in referee_data:
                base_status = referee_data['base_status']
                base_hp = base_status.get('remain_hp', 5000)  # 默认5000
                base_exposed = base_status.get('base_exposed', False)

            return outpost_hp, base_hp, base_exposed

        except Exception as e:
            # 数据获取失败，返回默认值
            return 1500, 5000, False

    def handle_data_missing(self, data_key: str) -> Any:
        """处理数据缺失情况

        Args:
            data_key: 数据键名

        Returns:
            对应类型的unknown值
        """
        # 根据数据键名返回对应的unknown值
        key_type_mapping = {
            'all_robots': 'robot',
            'movable_range': 'range',
            'own_hp': 'int',
            'own_ammo': 'int',
            'team_economy': 'int',
            'remaining_steps': 'int',
            'judge_countdown': 'int',
            'damage_per_step': 'float'
        }

        data_type = key_type_mapping.get(data_key, 'int')
        return UnknownStateHandler.get_unknown_value(data_type)

    # ==================== 辅助方法 ====================

    def _extract_robot_id(self, frame_id: str) -> int:
        """从frame_id中提取机器人ID

        Args:
            frame_id: frame ID字符串，例如 'red_standard_robot1'

        Returns:
            int: 机器人ID
        """
        try:
            # 假设格式为 'red_standard_robot1' 或 'blue_standard_robot3'
            # 提取最后的数字
            import re
            match = re.search(r'robot(\d+)', frame_id)
            if match:
                return int(match.group(1))
            return UnknownStateHandler.UNKNOWN_INT
        except Exception:
            return UnknownStateHandler.UNKNOWN_INT

    def _extract_team(self, frame_id: str) -> str:
        """从frame_id中提取队伍信息

        Args:
            frame_id: frame ID字符串

        Returns:
            str: 'red' or 'blue'
        """
        if 'red' in frame_id.lower():
            return 'red'
        elif 'blue' in frame_id.lower():
            return 'blue'
        else:
            return 'unknown'
