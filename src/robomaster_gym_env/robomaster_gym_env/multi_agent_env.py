"""
RoboMaster 多智能体并行环境

支持同时控制多个机器人（红方+蓝方），提供并行 step 接口:
  - pstep(actions): 所有智能体同时执行动作
  - reset(): 重置仿真，返回所有智能体的初始观测

用法:
    from robomaster_gym_env import RoboMasterMultiAgentEnv

    env = RoboMasterMultiAgentEnv(
        agents_config={
            'red_standard_robot1': {'team': 'red'},
            'blue_standard_robot1': {'team': 'blue'},
        }
    )
    obs = env.reset()
    actions = {name: env.action_spaces[name].sample() for name in env.agents}
    obs, rewards, terminateds, truncateds, infos = env.pstep(actions)
"""

import gymnasium
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple, List

from .config import GymEnvConfig, DEFAULT_GYM_CONFIG
from .ros2_interface import ROS2Interface
from .observation_space import ObservationSpace, ObservationConfig
from .action_space import ActionSpace
from .reward_calculator import RewardCalculator
from .interface_adapter import InterfaceAdapter


class RoboMasterMultiAgentEnv(gymnasium.Env):
    """
    RoboMaster 多智能体并行环境

    每个智能体拥有独立的:
      - ROS2Interface (命名空间隔离)
      - InterfaceAdapter
      - ObservationSpace / ActionSpace
      - 环境状态 (hp, ammo, ...)

    共享:
      - Gazebo 仿真 (只 reset 一次)
      - 裁判系统数据 (全局话题)
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        agents_config: Optional[Dict[str, Dict[str, Any]]] = None,
        base_config: Optional[GymEnvConfig] = None,
    ):
        """
        Args:
            agents_config: 每个智能体的配置, key 为机器人名称, value 为覆盖字段
                默认: 红蓝各一个步兵
                {
                    'red_standard_robot1': {'team': 'red'},
                    'blue_standard_robot1': {'team': 'blue'},
                }
            base_config: 基础配置, 各智能体继承此配置再覆盖
        """
        super().__init__()

        if agents_config is None:
            agents_config = {
                'red_standard_robot1': {'team': 'red'},
                'blue_standard_robot1': {'team': 'blue'},
            }

        self.base_config = base_config if base_config is not None else DEFAULT_GYM_CONFIG
        self.agents: List[str] = list(agents_config.keys())
        self.agent_configs: Dict[str, GymEnvConfig] = {}

        # 为每个智能体创建配置和接口
        self.ros2_interfaces: Dict[str, ROS2Interface] = {}
        self.interface_adapters: Dict[str, InterfaceAdapter] = {}
        self.obs_managers: Dict[str, ObservationSpace] = {}
        self.action_managers: Dict[str, ActionSpace] = {}
        self.reward_calculators: Dict[str, RewardCalculator] = {}

        # 公开的动作/观测空间 (Dict: agent_name -> space)
        self.action_spaces: Dict[str, gymnasium.spaces.Dict] = {}
        self.observation_spaces: Dict[str, gymnasium.spaces.Dict] = {}

        for agent_name, overrides in agents_config.items():
            # 构建该智能体的配置
            config = GymEnvConfig(
                robot_name=agent_name,
                robot_namespace=agent_name,
                team=overrides.get('team', 'red'),
                use_sim_time=self.base_config.use_sim_time,
                control_frequency=overrides.get(
                    'control_frequency', self.base_config.control_frequency
                ),
            )
            self.agent_configs[agent_name] = config

            # ROS2 接口 (命名空间隔离)
            ros2_if = ROS2Interface(
                robot_name=config.robot_name,
                robot_namespace=config.robot_namespace,
                team=config.team,
                use_sim_time=config.use_sim_time,
            )
            self.ros2_interfaces[agent_name] = ros2_if

            # 接口适配器
            adapter = InterfaceAdapter(
                ros2_interface=ros2_if,
                own_robot_id=0,
                own_team=config.team,
            )
            self.interface_adapters[agent_name] = adapter

            # 观测空间
            obs_config = ObservationConfig()
            obs_config.own_team = config.team
            obs_mgr = ObservationSpace(config=obs_config, interface_adapter=adapter)
            self.obs_managers[agent_name] = obs_mgr
            self.observation_spaces[agent_name] = obs_mgr.observation_space

            # 动作空间
            act_mgr = ActionSpace(
                {**config.action_config, **config.chassis_velocity_limit}
            )
            self.action_managers[agent_name] = act_mgr
            self.action_spaces[agent_name] = act_mgr.action_space

            # 奖励计算器
            self.reward_calculators[agent_name] = RewardCalculator(config.reward_config)

        # 兼容 gymnasium.Env 的单一 action_space / observation_space
        # 使用第一个智能体的空间 (方便 gymnasium 兼容检查)
        first_agent = self.agents[0]
        self.action_space = self.action_spaces[first_agent]
        self.observation_space = self.observation_spaces[first_agent]

        # 环境状态
        self.current_step = 0
        self.max_steps = 2100
        self.control_period = 1.0 / self.base_config.control_frequency
        self.last_control_time = 0.0

        # 每个智能体的独立状态
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        for agent_name in self.agents:
            self.agent_states[agent_name] = self._init_agent_state()

        # 伤害计算参数
        self.max_attack_distance = 7.0
        self.damage_reduction_distance = 5.0
        self.damage_fluctuation_rate = 0.25
        self.damage_per_step = 10.0

        # 启动所有 ROS2 spinning
        for ros2_if in self.ros2_interfaces.values():
            ros2_if.start_spinning()

        agent_info = ', '.join(
            f'{name}({self.agent_configs[name].team})' for name in self.agents
        )
        print(f'RoboMasterMultiAgentEnv initialized: [{agent_info}]')

    def _init_agent_state(self) -> Dict[str, Any]:
        """初始化单个智能体的状态"""
        return {
            'own_hp': 400,
            'own_ammo': 300,
            'remaining_steps': self.max_steps,
            'judge_countdown_steps': 0,
            'outpost_hp': 1500,
            'base_hp': 5000,
            'robot_level': 1,
            'coin_revive_count': 0,
            'revive_waiting_steps': 0,
            'is_dead': False,
        }

    # ==================== 核心接口 ====================

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        重置环境, 返回所有智能体的初始观测

        Returns:
            Dict[agent_name, observation]
        """
        super().reset(seed=seed)

        # 只用第一个智能体的接口 reset 仿真 (避免重复 reset)
        first_ros2 = self.ros2_interfaces[self.agents[0]]
        first_ros2.reset_simulation()

        self.current_step = 0

        # 重置所有智能体状态
        for agent_name in self.agents:
            self.agent_states[agent_name] = self._init_agent_state()
            self.reward_calculators[agent_name].reset()

        time.sleep(0.1)

        return self._get_all_observations()

    def pstep(
        self,
        actions: Dict[str, Dict[str, np.ndarray]],
    ) -> Tuple[
        Dict[str, Dict[str, np.ndarray]],  # observations
        Dict[str, float],                   # rewards
        Dict[str, bool],                    # terminateds
        Dict[str, bool],                    # truncateds
        Dict[str, Dict[str, Any]],          # infos
    ]:
        """
        并行 step: 所有智能体同时执行动作

        Args:
            actions: {agent_name: action_dict}

        Returns:
            observations, rewards, terminateds, truncateds, infos
        """
        # 1. 执行所有智能体的动作
        for agent_name, action in actions.items():
            if agent_name in self.ros2_interfaces:
                self._execute_agent_action(agent_name, action)

        # 2. 等待控制周期
        self._wait_for_control_period()

        # 3. 更新步数
        self.current_step += 1

        # 4. 更新所有智能体状态
        for agent_name in self.agents:
            action = actions.get(agent_name, {})
            self._update_agent_state(agent_name, action)

        # 5. 获取观测
        observations = self._get_all_observations()

        # 6. 计算奖励
        rewards = {}
        for agent_name in self.agents:
            rewards[agent_name] = self._calculate_agent_reward(agent_name)

        # 7. 检查终止
        terminateds = {}
        truncateds = {}
        for agent_name in self.agents:
            terminateds[agent_name] = self._check_agent_terminated(agent_name)
            truncateds[agent_name] = self.current_step >= self.max_steps

        # 8. 额外信息
        infos = {
            agent_name: {
                'current_step': self.current_step,
                'max_steps': self.max_steps,
            }
            for agent_name in self.agents
        }

        return observations, rewards, terminateds, truncateds, infos

    def close(self):
        """关闭环境"""
        for ros2_if in self.ros2_interfaces.values():
            ros2_if.stop_spinning()
            ros2_if.destroy()

    # ==================== 内部方法 ====================

    def _execute_agent_action(self, agent_name: str, action: Dict[str, np.ndarray]):
        """执行单个智能体的动作"""
        ros2_if = self.ros2_interfaces[agent_name]

        # 底盘控制
        if 'chassis_velocity' in action:
            ros2_if.send_chassis_velocity(
                linear_x=action['chassis_velocity'][0],
                linear_y=action['chassis_velocity'][1],
                angular_z=0.0,
            )

        # 云台锁定初始朝向
        ros2_if.send_gimbal_angle(yaw=0.0, pitch=0.0)

        # 射击控制
        if 'shoot' in action:
            shoot_mode = int(action['shoot'])
            ros2_if.send_shoot_command(projectile_num=shoot_mode)

    def _get_all_observations(self) -> Dict[str, Dict[str, np.ndarray]]:
        """获取所有智能体的观测"""
        observations = {}
        for agent_name in self.agents:
            state = self.agent_states[agent_name]
            observations[agent_name] = self.obs_managers[agent_name].get_observation_from_interface(
                current_step=self.current_step,
                max_steps=self.max_steps,
                damage_per_step=self.damage_per_step,
                ammo_consumed_per_step=1,
                own_robot_id=0,
                env_own_hp=state['own_hp'],
                env_own_ammo=state['own_ammo'],
                env_remaining_steps=state['remaining_steps'],
                env_judge_countdown_steps=state['judge_countdown_steps'],
                env_outpost_hp=state['outpost_hp'],
                env_base_hp=state['base_hp'],
                env_revive_waiting_steps=state['revive_waiting_steps'],
            )
        return observations

    def _calculate_agent_reward(self, agent_name: str) -> float:
        """计算单个智能体的奖励"""
        ros2_if = self.ros2_interfaces[agent_name]
        config = self.agent_configs[agent_name]
        state = self.agent_states[agent_name]

        robot_status = ros2_if.get_robot_status()
        referee_data = ros2_if.get_referee_data()

        current_hp = 400
        current_ammo = 300
        if robot_status is not None:
            current_hp = robot_status.get('remain_hp', 400)
            total_ammo = robot_status.get('total_projectiles', 300)
            used_ammo = robot_status.get('used_projectiles', 0)
            current_ammo = total_ammo - used_ammo

        attack_info = referee_data.get('attack_info', '')
        is_alive = current_hp > 0

        near_enemy_count = self._count_nearby_enemies(agent_name, distance_threshold=4.0)

        nearest_enemy_distance = self._get_nearest_enemy_distance(agent_name)
        max_field_distance = config.reward_config.get('max_field_distance', 30.0)

        return self.reward_calculators[agent_name].calculate_reward(
            current_hp=current_hp,
            current_ammo=current_ammo,
            attack_info=attack_info,
            is_alive=is_alive,
            near_enemy_count=near_enemy_count,
            nearest_enemy_distance=nearest_enemy_distance,
            max_field_distance=max_field_distance,
        )

    def _count_nearby_enemies(self, agent_name: str, distance_threshold: float = 4.0) -> int:
        """计算指定智能体附近的敌方数量"""
        try:
            all_robots = self.interface_adapters[agent_name].get_all_robot_poses()
            if all_robots is None:
                return 0

            own_position = self.ros2_interfaces[agent_name].get_robot_position()
            if own_position is None:
                return 0

            own_x, own_y, own_z = own_position
            own_team = self.agent_configs[agent_name].team

            count = 0
            for robot in all_robots:
                robot_team = robot.get('team', 'unknown')
                robot_x = robot.get('x', 0)
                robot_y = robot.get('y', 0)
                if robot_team != own_team and robot_team != 'unknown':
                    distance = np.sqrt((robot_x - own_x) ** 2 + (robot_y - own_y) ** 2)
                    if distance <= distance_threshold:
                        count += 1
            return count
        except Exception:
            return 0

    def _get_nearest_enemy_distance(self, agent_name: str) -> float:
        """获取指定智能体到最近敌方的距离"""
        try:
            all_robots = self.interface_adapters[agent_name].get_all_robot_poses()
            if all_robots is None:
                return float('inf')

            own_position = self.ros2_interfaces[agent_name].get_robot_position()
            if own_position is None:
                return float('inf')

            own_x, own_y, own_z = own_position
            own_team = self.agent_configs[agent_name].team

            min_distance = float('inf')
            for robot in all_robots:
                robot_team = robot.get('team', 'unknown')
                robot_x = robot.get('x', 0)
                robot_y = robot.get('y', 0)
                if robot_team != own_team and robot_team != 'unknown':
                    distance = np.sqrt((robot_x - own_x) ** 2 + (robot_y - own_y) ** 2)
                    if distance < min_distance:
                        min_distance = distance
            return min_distance
        except Exception:
            return float('inf')

    def _check_agent_terminated(self, agent_name: str) -> bool:
        """检查单个智能体是否终止"""
        ros2_if = self.ros2_interfaces[agent_name]
        robot_status = ros2_if.get_robot_status()
        if robot_status is not None:
            if robot_status.get('remain_hp', 0) <= 0:
                return True
        if ros2_if.is_tumbled():
            return True
        return False

    def _update_agent_state(self, agent_name: str, action: Dict[str, np.ndarray]):
        """更新单个智能体的环境状态"""
        state = self.agent_states[agent_name]
        ros2_if = self.ros2_interfaces[agent_name]
        config = self.agent_configs[agent_name]

        # 1. 剩余步数
        state['remaining_steps'] = self.max_steps - self.current_step

        # 2. 复活等待递减
        if state['revive_waiting_steps'] > 0:
            state['revive_waiting_steps'] -= 1
            if state['revive_waiting_steps'] == 0 and state['is_dead']:
                state['is_dead'] = False
                state['own_hp'] = 400
                state['own_ammo'] = 300

        # 3. 射击处理
        if 'shoot' in action and not state['is_dead']:
            shoot_mode = int(action['shoot'])
            if shoot_mode > 0:
                state['own_ammo'] = max(0, state['own_ammo'] - 1)

                if 1 <= shoot_mode <= 6:
                    target_distance = self._get_target_distance(agent_name, shoot_mode)
                    actual_damage = self._calculate_damage_with_distance(
                        self.damage_per_step, target_distance
                    )
                elif shoot_mode == 7:
                    outpost_distance = self._get_outpost_distance(agent_name)
                    if outpost_distance <= 5.0:
                        actual_damage = self._calculate_damage_with_fluctuation(self.damage_per_step)
                        state['outpost_hp'] = max(0, state['outpost_hp'] - int(actual_damage))
                elif shoot_mode == 8:
                    base_distance = self._get_base_distance(agent_name)
                    if base_distance <= 5.0 and self._can_damage_base(agent_name):
                        actual_damage = self._calculate_damage_with_fluctuation(self.damage_per_step)
                        state['base_hp'] = max(0, state['base_hp'] - int(actual_damage))

        # 4. 死亡检查
        if state['own_hp'] <= 0 and not state['is_dead']:
            state['is_dead'] = True
            state['revive_waiting_steps'] = int(
                self.current_step / 50 + 50 + 100 * state['coin_revive_count']
            )

        # 5. 判负步数
        referee_data = ros2_if.get_referee_data()
        if referee_data is not None:
            judge_countdown_seconds = referee_data.get('judge_countdown', 0)
            state['judge_countdown_steps'] = (
                judge_countdown_seconds * 5 if judge_countdown_seconds > 0 else 0
            )

    def _get_target_distance(self, agent_name: str, target_id: int) -> float:
        """获取目标距离"""
        try:
            all_robots = self.interface_adapters[agent_name].get_all_robot_poses()
            if all_robots is None:
                return float('inf')
            own_position = self.ros2_interfaces[agent_name].get_robot_position()
            if own_position is None:
                return float('inf')
            own_x, own_y, own_z = own_position
            for robot in all_robots:
                if robot.get('id', -1) == target_id:
                    robot_x = robot.get('x', 0)
                    robot_y = robot.get('y', 0)
                    return np.sqrt((robot_x - own_x) ** 2 + (robot_y - own_y) ** 2)
            return float('inf')
        except Exception:
            return float('inf')

    def _get_outpost_distance(self, agent_name: str) -> float:
        """获取敌方前哨站距离"""
        try:
            own_position = self.ros2_interfaces[agent_name].get_robot_position()
            if own_position is None:
                return float('inf')
            own_x, own_y, own_z = own_position
            config = self.agent_configs[agent_name]
            if config.team == 'red':
                target = config.blue_outpost_position
            else:
                target = config.red_outpost_position
            if target[0] is None:
                return float('inf')
            return np.sqrt(
                (target[0] - own_x) ** 2 + (target[1] - own_y) ** 2 + (target[2] - own_z) ** 2
            )
        except Exception:
            return float('inf')

    def _get_base_distance(self, agent_name: str) -> float:
        """获取敌方基地距离"""
        try:
            own_position = self.ros2_interfaces[agent_name].get_robot_position()
            if own_position is None:
                return float('inf')
            own_x, own_y, own_z = own_position
            config = self.agent_configs[agent_name]
            if config.team == 'red':
                target = config.blue_base_position
            else:
                target = config.red_base_position
            if target[0] is None:
                return float('inf')
            return np.sqrt(
                (target[0] - own_x) ** 2 + (target[1] - own_y) ** 2 + (target[2] - own_z) ** 2
            )
        except Exception:
            return float('inf')

    def _can_damage_base(self, agent_name: str) -> bool:
        """判断是否可以伤害敌方基地"""
        try:
            outpost_hp, base_hp, base_exposed = \
                self.interface_adapters[agent_name].get_outpost_and_base_state()
            return base_exposed and outpost_hp == 0
        except Exception:
            return False

    def _calculate_damage_with_distance(self, base_damage: float, distance: float) -> float:
        """基于距离的伤害计算"""
        if distance > self.max_attack_distance:
            return 0.0
        if distance > self.damage_reduction_distance:
            base_damage *= 0.5
        fluctuation = np.random.uniform(
            -self.damage_fluctuation_rate, self.damage_fluctuation_rate
        )
        return max(0.0, base_damage * (1.0 + fluctuation))

    def _calculate_damage_with_fluctuation(self, base_damage: float) -> float:
        """带波动的伤害计算"""
        fluctuation = np.random.uniform(
            -self.damage_fluctuation_rate, self.damage_fluctuation_rate
        )
        return max(0.0, base_damage * (1.0 + fluctuation))

    def _wait_for_control_period(self):
        """等待控制周期"""
        current_time = time.time()
        elapsed = current_time - self.last_control_time
        if elapsed < self.control_period:
            time.sleep(self.control_period - elapsed)
        self.last_control_time = time.time()
