"""
观察空间定义模块

该模块定义了简化后的观察空间，包含核心决策信息：
1. 所有机器人位置（包含自己）
2. 己方血量
3. 己方弹药量
4. 我方经济
5. 剩余步数
6. 判负步数 (1秒=5步)
7. 每步伤害能力
8. 前哨站血量
9. 基地血量
10. 基地展开状态

注意：已删除激光雷达距离数据
"""

import gymnasium
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .unknown_state_handler import UnknownStateHandler
from .interface_adapter import InterfaceAdapter
from .data_processor import DataProcessor


@dataclass
class ObservationConfig:
    """观察空间配置

    Attributes:
        max_robots: 最大机器人数量
        max_hp: 最大血量 (默认400)
        max_ammo: 最大弹药量 (默认300)
        max_economy: 最大经济值 (默认400)
        max_steps: 最大步数 (默认2048)
        max_countdown_steps: 最大判负步数 (默认10500, 1秒=5步)
        max_outpost_hp: 前哨站最大血量 (默认1500)
        max_base_hp: 基地最大血量 (默认5000)
        damage_per_step: 每步伤害 (默认10)
        enable_damage_per_step: 是否启用每步伤害
        ammo_per_step: 每步攻击消耗弹药量 (默认1)
        own_team: 自身队伍颜色 ('red' or 'blue')，默认红色
    """
    max_robots: int = 10
    max_hp: int = 400
    max_ammo: int = 300
    max_economy: int = 400
    max_steps: int = 2048
    max_countdown_steps: int = 2048
    max_outpost_hp: int = 1500
    max_base_hp: int = 5000
    damage_per_step: int = 10
    enable_damage_per_step: bool = True
    ammo_per_step: int = 1
    own_team: str = 'red'


class ObservationSpace:
    """观察空间类

    提供简化的观察空间定义和观察数据构建功能。
    """

    def __init__(
        self,
        config: ObservationConfig = None,
        interface_adapter: InterfaceAdapter = None
    ):
        """初始化观察空间

        Args:
            config: 观察空间配置
            interface_adapter: 接口适配器
        """
        self.config = config or ObservationConfig()
        self.interface_adapter = interface_adapter

        # 初始化数据处理器
        self.data_processor = DataProcessor(max_robots=self.config.max_robots)

        # 构建观察空间
        self.observation_space = self.build_observation_space()

    def build_observation_space(self) -> spaces.Dict:
        """构建Gym观察空间定义

        Returns:
            spaces.Dict: Gym观察空间
        """
        obs_space = spaces.Dict({
            # 所有机器人位置: [10, 4] (id, team, x, y)
            # 包含自己和其他所有机器人，共10台
            # team: 0=己方(ally), 1=敌方(enemy), -1=unknown
            # unknown时: id=-1表示该位置无效
            'all_robots': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.config.max_robots, 4),
                dtype=np.float32
            ),

            # 己方血量
            # unknown时: -1
            'own_hp': spaces.Discrete(self.config.max_hp + 1),

            # 己方弹药量
            # unknown时: -1
            'own_ammo': spaces.Discrete(self.config.max_ammo + 1),

            # 我方经济
            # unknown时: -1
            'team_economy': spaces.Discrete(self.config.max_economy + 1),

            # 剩余步数
            # unknown时: -1
            'remaining_steps': spaces.Discrete(self.config.max_steps + 1),

            # 判负步数 (1秒=5步)
            # unknown时: -1
            'judge_countdown_steps': spaces.Discrete(self.config.max_countdown_steps + 1),

            # 每步伤害能力
            # unknown时: nan
            'damage_per_step': spaces.Box(
                low=0.0,
                high=np.inf,
                shape=(1,),
                dtype=np.float32
            ),

            # 前哨站血量
            # unknown时: -1
            'outpost_hp': spaces.Discrete(self.config.max_outpost_hp + 1),

            # 基地血量
            # unknown时: -1
            'base_hp': spaces.Discrete(self.config.max_base_hp + 1),

            # 基地展开状态
            # True: 已展开, False: 未展开
            'base_exposed': spaces.Discrete(2),

            # 目标相对方向: [dx, dy] (归一化到[-1,1])
            # 从自身位置指向最近敌方/虚拟蓝方的方向向量
            # unknown时: [0, 0]
            'target_direction': spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32
            ),

            # 每步攻击消耗弹药量
            # unknown时: -1
            'ammo_consumed_per_step': spaces.Discrete(self.config.max_ammo + 1),

            # 复活等待步数
            # unknown时: -1
            'revive_waiting_steps': spaces.Discrete(self.config.max_steps + 1)
        })

        return obs_space

    def get_empty_observation(self) -> Dict[str, np.ndarray]:
        """获取空观察字典(所有项为unknown状态)

        Returns:
            Dict[str, np.ndarray]: 所有项为unknown的观察字典
        """
        return {
            'all_robots': UnknownStateHandler.create_unknown_robot_array(self.config.max_robots),
            'own_hp': UnknownStateHandler.UNKNOWN_INT,
            'own_ammo': UnknownStateHandler.UNKNOWN_INT,
            'team_economy': UnknownStateHandler.UNKNOWN_INT,
            'remaining_steps': UnknownStateHandler.UNKNOWN_INT,
            'judge_countdown_steps': UnknownStateHandler.UNKNOWN_INT,
            'damage_per_step': np.array([UnknownStateHandler.UNKNOWN_FLOAT], dtype=np.float32),
            'outpost_hp': UnknownStateHandler.UNKNOWN_INT,
            'base_hp': UnknownStateHandler.UNKNOWN_INT,
            'base_exposed': 0,
            'ammo_consumed_per_step': UnknownStateHandler.UNKNOWN_INT,
            'revive_waiting_steps': UnknownStateHandler.UNKNOWN_INT,
            'target_direction': np.array([0.0, 0.0], dtype=np.float32),
        }

    def is_valid(self, observation: Dict[str, np.ndarray]) -> bool:
        """验证观察数据的有效性

        Args:
            observation: 观察数据字典

        Returns:
            bool: True表示有效, False表示无效
        """
        # 检查所有必需的键是否存在
        required_keys = [
            'all_robots', 'own_hp', 'own_ammo',
            'team_economy', 'remaining_steps', 'judge_countdown_steps', 'damage_per_step',
            'outpost_hp', 'base_hp', 'base_exposed', 'ammo_consumed_per_step',
            'revive_waiting_steps', 'target_direction'
        ]

        for key in required_keys:
            if key not in observation:
                return False

        # 检查数组形状
        if observation['all_robots'].shape != (self.config.max_robots, 4):
            return False

        if observation['damage_per_step'].shape != (1,):
            return False

        return True

    def get_observation(
        self,
        all_robots: np.ndarray,
        own_hp: int,
        own_ammo: int,
        team_economy: int,
        remaining_steps: int,
        judge_countdown_steps: int,
        damage_per_step: float,
        outpost_hp: int,
        base_hp: int,
        base_exposed: bool,
        ammo_consumed_per_step: int,
        revive_waiting_steps: int,
        target_direction: np.ndarray = None,
    ) -> Dict[str, Any]:
        """构建观察字典

        Args:
            all_robots: 所有机器人位置 [10, 4]
            own_hp: 己方血量
            own_ammo: 己方弹药量
            team_economy: 我方经济
            remaining_steps: 剩余步数
            judge_countdown_steps: 判负步数 (1秒=5步)
            damage_per_step: 每步伤害
            outpost_hp: 前哨站血量
            base_hp: 基地血量
            base_exposed: 基地展开状态
            ammo_consumed_per_step: 每步攻击消耗弹药量
            revive_waiting_steps: 复活等待步数
            target_direction: 目标相对方向 [dx, dy] 归一化到[-1,1]

        Returns:
            Dict[str, Any]: 观察字典
        """
        if target_direction is None:
            target_direction = np.array([0.0, 0.0], dtype=np.float32)

        observation = {
            'all_robots': all_robots,
            'own_hp': own_hp,
            'own_ammo': own_ammo,
            'team_economy': team_economy,
            'remaining_steps': remaining_steps,
            'judge_countdown_steps': judge_countdown_steps,
            'damage_per_step': np.array([damage_per_step], dtype=np.float32),
            'outpost_hp': outpost_hp,
            'base_hp': base_hp,
            'base_exposed': 1 if base_exposed else 0,
            'ammo_consumed_per_step': ammo_consumed_per_step,
            'revive_waiting_steps': revive_waiting_steps,
            'target_direction': target_direction,
        }

        return observation

    def __repr__(self) -> str:
        """字符串表示"""
        return f"ObservationSpaceV2(config={self.config})"

    def get_observation_from_interface(
        self,
        current_step: int,
        max_steps: int,
        damage_per_step: float = None,
        ammo_consumed_per_step: int = None,
        own_robot_id: int = 0,
        # 环境状态信息（优先使用，不依赖ROS2接口）
        env_own_hp: int = None,
        env_own_ammo: int = None,
        env_remaining_steps: int = None,
        env_judge_countdown_steps: int = None,
        env_outpost_hp: int = None,
        env_base_hp: int = None,
        env_revive_waiting_steps: int = None,
        # 虚拟蓝方位置覆盖
        virtual_blue_pos: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """从接口获取观察数据（主方法）

        Args:
            current_step: 当前步数
            max_steps: 最大步数
            damage_per_step: 每步伤害（外部输入）
            ammo_consumed_per_step: 每步攻击消耗弹药量（外部输入）
            own_robot_id: 自己的机器人ID
            env_own_hp: 环境维护的血量（优先使用）
            env_own_ammo: 环境维护的弹药量（优先使用）
            env_remaining_steps: 环境维护的剩余步数（优先使用）
            env_judge_countdown_steps: 环境维护的判负步数（优先使用）
            env_outpost_hp: 环境维护的前哨站血量（优先使用）
            env_base_hp: 环境维护的基地血量（优先使用）
            env_revive_waiting_steps: 环境维护的复活等待步数（优先使用）
            virtual_blue_pos: 虚拟蓝方位置 (x, y), 若提供则覆盖all_robots中敌方位置

        Returns:
            Dict[str, Any]: 观察字典
        """
        # 1. 获取所有机器人位置
        all_robots = self._get_all_robots(own_robot_id)

        # 1.5 虚拟蓝方位置覆盖: 将敌方机器人的位置替换为虚拟蓝方位置
        if virtual_blue_pos is not None:
            vx, vy = virtual_blue_pos
            for i in range(all_robots.shape[0]):
                # team_code == 1 表示敌方
                if int(all_robots[i, 1]) == 1:
                    all_robots[i, 2] = vx  # x
                    all_robots[i, 3] = vy  # y

        # 2. 获取游戏状态（用于获取经济等仍需从ROS2获取的信息）
        game_state = self._get_game_state()

        # 3. 计算剩余步数（优先使用环境维护的值）
        if env_remaining_steps is not None:
            remaining_steps = env_remaining_steps
        else:
            remaining_steps = self.data_processor.process_remaining_steps(
                current_step, max_steps
            )

        # 4. 处理每步伤害
        damage = self._process_damage_per_step(damage_per_step)

        # 5. 获取前哨站和基地状态（优先使用环境维护的值）
        if env_outpost_hp is not None and env_base_hp is not None:
            outpost_hp = env_outpost_hp
            base_hp = env_base_hp
            # 仍需从接口获取基地展开状态
            _, _, base_exposed = self._get_outpost_and_base_state()
        elif env_outpost_hp is not None:
            outpost_hp = env_outpost_hp
            # 仍需从接口获取基地状态
            _, base_hp, base_exposed = self._get_outpost_and_base_state()
        elif env_base_hp is not None:
            base_hp = env_base_hp
            # 仍需从接口获取前哨站状态
            outpost_hp, _, base_exposed = self._get_outpost_and_base_state()
        else:
            outpost_hp, base_hp, base_exposed = self._get_outpost_and_base_state()

        # 6. 处理每步弹药消耗
        ammo_consumed = self._process_ammo_consumed_per_step(ammo_consumed_per_step)

        # 7. 处理血量（优先使用环境维护的值）
        own_hp = env_own_hp if env_own_hp is not None else game_state['own_hp']

        # 8. 处理弹药量（优先使用环境维护的值）
        own_ammo = env_own_ammo if env_own_ammo is not None else game_state['own_ammo']

        # 9. 处理判负步数（优先使用环境维护的值）
        if env_judge_countdown_steps is not None:
            judge_countdown_steps = env_judge_countdown_steps
        else:
            # 将判负时间(秒)转换为步数 (1秒=5步)
            judge_countdown_seconds = game_state['judge_countdown']
            judge_countdown_steps = judge_countdown_seconds * 5 if judge_countdown_seconds > 0 else 0

        # 10. 计算目标相对方向
        target_direction = self._compute_target_direction(
            all_robots, virtual_blue_pos
        )

        # 11. 构建观察字典
        observation = self.get_observation(
            all_robots=all_robots,
            own_hp=own_hp,
            own_ammo=own_ammo,
            team_economy=game_state['team_economy'],
            remaining_steps=remaining_steps,
            judge_countdown_steps=judge_countdown_steps,
            damage_per_step=damage,
            outpost_hp=outpost_hp,
            base_hp=base_hp,
            base_exposed=base_exposed,
            ammo_consumed_per_step=ammo_consumed,
            revive_waiting_steps=env_revive_waiting_steps if env_revive_waiting_steps is not None else 0,
            target_direction=target_direction,
        )

        return observation

    def _get_all_robots(self, own_robot_id: int) -> np.ndarray:
        """获取所有机器人位置

        Args:
            own_robot_id: 自己的机器人ID

        Returns:
            np.ndarray: 所有机器人位置数组 [10, 4]
                team字段: 0=己方(ally), 1=敌方(enemy), -1=unknown
        """
        if self.interface_adapter is None:
            return UnknownStateHandler.create_unknown_robot_array(self.config.max_robots)

        # 从接口动态获取自身队伍颜色
        own_team = self.interface_adapter.get_own_team()

        # 从接口获取位置数据
        robot_poses = self.interface_adapter.get_all_robot_poses()

        # 处理数据（传入own_team用于己方/敌方判断）
        all_robots = self.data_processor.process_robot_positions(
            robot_poses, own_robot_id, own_team=own_team
        )

        return all_robots

    def _get_game_state(self) -> Dict[str, Any]:
        """获取游戏状态

        Returns:
            Dict[str, Any]: 游戏状态字典
        """
        if self.interface_adapter is None:
            return {
                'own_hp': UnknownStateHandler.UNKNOWN_INT,
                'own_ammo': UnknownStateHandler.UNKNOWN_INT,
                'team_economy': UnknownStateHandler.UNKNOWN_INT,
                'judge_countdown': UnknownStateHandler.UNKNOWN_INT
            }

        # 从接口获取游戏状态
        game_state = self.interface_adapter.get_game_state()

        # 处理数据
        processed_state = self.data_processor.process_game_state(game_state)

        return processed_state

    def _process_damage_per_step(self, damage_per_step: float) -> float:
        """处理每步伤害

        Args:
            damage_per_step: 每步伤害值（外部输入）

        Returns:
            float: 处理后的伤害值
        """
        if not self.config.enable_damage_per_step:
            return UnknownStateHandler.UNKNOWN_FLOAT

        # 检查输入有效性
        if damage_per_step is None or np.isnan(damage_per_step):
            return UnknownStateHandler.UNKNOWN_FLOAT

        # 验证范围
        if damage_per_step < 0:
            return UnknownStateHandler.UNKNOWN_FLOAT

        return float(damage_per_step)

    def _get_outpost_and_base_state(self) -> Tuple[int, int, bool]:
        """获取前哨站和基地状态

        Returns:
            Tuple[int, int, bool]: (前哨站血量, 基地血量, 基地展开状态)
        """
        if self.interface_adapter is None:
            return (
                UnknownStateHandler.UNKNOWN_INT,
                UnknownStateHandler.UNKNOWN_INT,
                False
            )

        # 从接口获取前哨站和基地状态
        outpost_hp, base_hp, base_exposed = self.interface_adapter.get_outpost_and_base_state()

        return outpost_hp, base_hp, base_exposed

    def _process_ammo_consumed_per_step(self, ammo_consumed_per_step: int) -> int:
        """处理每步弹药消耗

        Args:
            ammo_consumed_per_step: 每步弹药消耗值（外部输入）

        Returns:
            int: 处理后的弹药消耗值
        """
        # 检查输入有效性
        if ammo_consumed_per_step is None:
            # 如果没有外部输入,使用配置的默认值
            return self.config.ammo_per_step

        # 验证范围
        if ammo_consumed_per_step < 0:
            return UnknownStateHandler.UNKNOWN_INT

        # 确保不超过最大弹药量
        if ammo_consumed_per_step > self.config.max_ammo:
            return self.config.max_ammo

        return int(ammo_consumed_per_step)

    def _compute_target_direction(
        self,
        all_robots: np.ndarray,
        virtual_blue_pos: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """计算从自身指向目标的方向向量 (归一化到[-1,1])

        优先使用虚拟蓝方位置, 否则找 all_robots 中最近的敌方。
        方向向量按场地尺寸归一化: dx/14, dy/7.5

        Args:
            all_robots: (10, 4) [id, team, x, y]
            virtual_blue_pos: 虚拟蓝方位置 (x, y)

        Returns:
            np.ndarray: (2,) [dx_norm, dy_norm]
        """
        # 找自身位置 (team=0 的第一个)
        own_x, own_y = None, None
        for i in range(all_robots.shape[0]):
            if int(all_robots[i, 1]) == 0:  # ally
                own_x = all_robots[i, 2]
                own_y = all_robots[i, 3]
                break

        if own_x is None:
            return np.array([0.0, 0.0], dtype=np.float32)

        # 确定目标位置
        target_x, target_y = None, None

        if virtual_blue_pos is not None:
            target_x, target_y = virtual_blue_pos
        else:
            # 找最近的敌方 (team=1)
            min_dist = float('inf')
            for i in range(all_robots.shape[0]):
                if int(all_robots[i, 1]) == 1:  # enemy
                    ex, ey = all_robots[i, 2], all_robots[i, 3]
                    dist = (ex - own_x) ** 2 + (ey - own_y) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        target_x, target_y = ex, ey

        if target_x is None:
            return np.array([0.0, 0.0], dtype=np.float32)

        # 计算方向并归一化
        dx = (target_x - own_x) / 14.0   # 场地半宽
        dy = (target_y - own_y) / 7.5    # 场地半高

        # clip 到 [-1, 1]
        dx = float(np.clip(dx, -1.0, 1.0))
        dy = float(np.clip(dy, -1.0, 1.0))

        return np.array([dx, dy], dtype=np.float32)
