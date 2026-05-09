"""
RoboMaster Gazebo Gym环境主类
观察空间
"""

import gymnasium
import numpy as np
from typing import Dict, Any, Optional, Tuple
import time
import math
import os

from .config import GymEnvConfig, DEFAULT_GYM_CONFIG
from .ros2_interface import ROS2Interface
from .observation_space import ObservationSpace, ObservationConfig
from .action_space import ActionSpace
from .reward_calculator import RewardCalculator
from .interface_adapter import InterfaceAdapter
from .env_renderer import EnvRenderer


class RoboMasterGazeboEnv(gymnasium.Env):
    """
    RoboMaster Gazebo仿真Gym环境

    使用简化的观察空间，包含核心决策信息：
    1. 所有机器人位置（包含自己）
    2. 己方血量
    3. 己方弹药量
    4. 我方经济
    5. 剩余步数
    6. 判负时间
    7. 每步伤害能力
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
        self,
        config: Optional[GymEnvConfig] = None,
        obs_config: Optional[ObservationConfig] = None
    ):
        """
        初始化环境

        Args:
            config: 环境配置
            obs_config: 观察空间配置
        """
        super(RoboMasterGazeboEnv, self).__init__()

        # 配置
        self.config = config if config is not None else DEFAULT_GYM_CONFIG

        # ROS2接口
        self.ros2_interface = ROS2Interface(
            robot_name=self.config.robot_name,
            robot_namespace=self.config.robot_namespace,
            team=self.config.team,
            use_sim_time=self.config.use_sim_time
        )

        # 奖励计算器
        self.reward_calculator = RewardCalculator(
            self.config.reward_config
        )

        # 创建接口适配器
        self.interface_adapter = InterfaceAdapter(
            ros2_interface=self.ros2_interface,
            own_robot_id=0,  # TODO: 从配置获取
            own_team=self.config.team  # 自身队伍颜色，默认红色
        )

        # 创建观察空间
        obs_config = obs_config or ObservationConfig()
        obs_config.own_team = self.config.team  # 将自身颜色传入观察空间配置
        self.observation_space_manager = ObservationSpace(
            config=obs_config,
            interface_adapter=self.interface_adapter
        )

        self.observation_space = self.observation_space_manager.observation_space

        # 动作空间 (云台锁定初始朝向, 不作为动作)
        self.action_space_manager = ActionSpace(
            {**self.config.action_config, **self.config.chassis_velocity_limit}
        )
        self.action_space = self.action_space_manager.action_space

        # 环境状态
        self.current_step = 0
        self.max_steps = 2048  # 最大步数
        self.control_period = 1.0 / self.config.control_frequency
        self.last_control_time = 0.0
        
        # real_time_factor: 仿真加速倍数
        # 需要与 Gazebo 世界文件中的 real_time_factor 一致
        # 服务器 (CPU 30%): 5.0
        self.real_time_factor = 5.0

        # 每步伤害（固定为10）
        self.damage_per_step = 10.0

        # 每步弹药消耗（固定为1）
        self.ammo_consumed_per_step = 1

        # 环境维护的状态信息（不依赖ROS2接口）
        self.own_hp = 400  # 初始血量
        self.own_ammo = 300  # 初始弹药量
        self.remaining_steps = self.max_steps  # 剩余步数
        self.judge_countdown_steps = 0  # 判负步数
        self.outpost_hp = 1500  # 前哨站初始血量
        self.base_hp = 5000  # 基地初始血量

        # 伤害计算相关参数
        self.max_attack_distance = 7.0  # 最大攻击距离(米)
        self.damage_reduction_distance = 5.0  # 伤害衰减距离阈值(米)
        self.damage_fluctuation_rate = 0.25  # 伤害波动范围(25%)

        # 复活相关状态变量
        self.robot_level = 1  # 机器人等级(默认1级)
        self.coin_revive_count = 0  # 累计金币复活次数
        self.revive_waiting_steps = 0  # 复活等待步数
        self.is_dead = False  # 是否死亡

        # 弹药兑换相关参数
        self.remote_ammo_cost = 150  # 远程兑换弹药金币消耗
        self.remote_ammo_gain = 100  # 远程兑换弹药获得数量
        self.local_ammo_cost = 10  # 非远程兑换弹药金币消耗
        self.local_ammo_gain = 10  # 非远程兑换弹药获得数量

        # 渲染相关
        self.viewer = None
        self.render_mode = None
        self._last_reward = 0.0  # 最近一步的奖励, 供 render() 使用
        self._train_progress = None  # 训练进度信息, 供 render() 使用

        # 课程学习状态
        self._curriculum_episode_count = 0  # 当前阶段已训练的episode数
        self._curriculum_stage = self.config.curriculum_config.get('stage', 1)

        # 物理精度检测: 对比仿真时间与真实时间的流逝速率
        self._wall_time_start = time.time()
        self._sim_time_start = None  # 首次step时初始化
        self._real_time_factor_measured = 0.0  # 实测时间流速

        # 虚拟蓝方位置(课程学习生成, 不移动Gazebo中的蓝方)
        self._virtual_blue_x = self.config.virtual_blue_x
        self._virtual_blue_y = self.config.virtual_blue_y

        # 可达点收集: 运行时记录机器人能到达的整数坐标
        self._reachable_points_file = os.path.join(os.path.dirname(__file__), 'reachable_points.npy')
        self._reachable_points = set()  # set of (int_x, int_y)
        self._load_reachable_points()

        # 启动ROS2
        self.ros2_interface.start_spinning()

        print(f"RoboMasterGazeboEnv initialized")
        print(f"  Robot: {self.config.robot_name}")
        print(f"  Namespace: {self.config.robot_namespace}")
        print(f"  Observation space: {self.observation_space}")
        print(f"  Action space: {self.action_space}")

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作

        Args:
            action: 动作字典

        Returns:
            observation: 观察数据
            reward: 奖励值
            terminated: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        # 处理 ROS2 回调 (在主线程中非阻塞处理，确保数据最新)
        self.ros2_interface.spin_once()

        # 执行动作
        t0 = time.time()
        self._execute_action(action)
        t1 = time.time()

        # 等待控制周期
        self._wait_for_control_period()
        t2 = time.time()

        # 物理精度检测: 测量实际时间流速
        self._measure_real_time_factor()
        t3 = time.time()

        # 更新步数
        self.current_step += 1

        # 记录位置历史 (用于卡住时回退)
        own_pos = self.ros2_interface.get_robot_position()
        if own_pos is not None:
            self._position_history.append((own_pos[0], own_pos[1], own_pos[2]))
            # 只保留最近50步
            if len(self._position_history) > 50:
                self._position_history = self._position_history[-50:]

        # 收集可达点: 记录机器人当前整数坐标
        self._collect_reachable_point()

        # 更新环境状态信息
        self._update_env_state(action)

        # 获取观察
        observation = self._get_observation()

        # 计算奖励
        reward = self._calculate_reward()
        self._last_reward = reward  # 保存最近奖励, 供 render() 使用

        # 检查终止条件
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_steps

        # 翻车时覆盖奖励为翻车惩罚
        if terminated and self.ros2_interface.is_tumbled():
            reward = self.config.reward_config.get('tumble', -10.0)
            self._last_reward = reward

        # 出边界时覆盖奖励为出边界惩罚
        if terminated:
            own_x, own_y = self._get_own_position_safe()
            if own_x is not None:
                margin = 1.0  # 不能离边界 1m 内
                if own_x < margin or own_x > 28 - margin or own_y < margin or own_y > 15 - margin:
                    reward = self.config.reward_config.get('out_of_boundary', -100.0)
                    self._last_reward = reward

        # 检测仿真数据是否异常 (rtf 过低或过高)
        # rtf 正常范围: [set_rtf * 0.8, set_rtf * 1.2]
        # 异常时标记 info['sim_unstable'] = True, 训练循环应跳过该步
        sim_unstable = False
        measured_rtf = self._real_time_factor_measured
        set_rtf = self.real_time_factor
        if measured_rtf > 0 and set_rtf > 0:
            rtf_ratio = measured_rtf / set_rtf
            if rtf_ratio < 0.8 or rtf_ratio > 1.2:
                sim_unstable = True

        # 额外信息
        info = {
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'curriculum_stage': self._curriculum_stage,
            'virtual_blue_pos': (self._virtual_blue_x, self._virtual_blue_y),
            'sim_unstable': sim_unstable,
        }

        # 步内计时诊断 (每100步打印一次)
        if self.current_step % 100 == 0:
            t_end = time.time()
            total = (t_end - t0) * 1000
            action_ms = (t1 - t0) * 1000
            wait_ms = (t2 - t1) * 1000
            measure_ms = (t3 - t2) * 1000
            other_ms = total - action_ms - wait_ms - measure_ms
            print(f"  [STEP TIMING] total={total:.1f}ms | action={action_ms:.1f} | wait={wait_ms:.1f} | measure={measure_ms:.1f} | other={other_ms:.1f}")

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        重置环境

        Args:
            seed: 随机种子
            options: 重置选项

        Returns:
            observation: 初始观察
            info: 额外信息
        """
        super().reset(seed=seed)

        # 温和重置: 只重置红方机器人位姿, 不重置整个仿真
        # 先确认Gazebo中模型已加载, 避免set_pose创建额外模型
        pose_info = self.ros2_interface.referee_data.get('pose_info')
        model_exists = False
        if pose_info is not None:
            model_exists = any(
                t.child_frame_id == self.config.robot_name or
                t.child_frame_id == f"{self.config.robot_name}_0"
                for t in pose_info
            )

        # 出界/翻车/死亡后, 裁判系统可能已断电(enable_power=False)
        # 发送 START_GAME 命令恢复所有机器人的电源和控制, 否则 set_pose 后车无法移动
        try:
            from rmoss_interfaces.msg import RefereeCmd
            if 'referee_cmd' in self.ros2_interface.publishers:
                start_msg = RefereeCmd()
                start_msg.cmd = RefereeCmd.START_GAME
                start_msg.robot_name = ''
                self.ros2_interface.publishers['referee_cmd'].publish(start_msg)
                time.sleep(0.3)
        except (ImportError, KeyError):
            pass

        # 即使 pose_info 中找不到模型(出界后可能丢失), 也尝试 set_pose
        # Gazebo 的 set_pose service 按模型名查找, 只要模型还在世界就能设
        if True:
            # 先发送零速度命令，清除机器人的惯性速度
            # 否则 set_pose 只改位置不改速度，重置后机器人会带着旧速度乱飞
            self.ros2_interface.send_chassis_velocity(0.0, 0.0, 0.0)
            # 等待速度归零 (最多0.5秒)
            for _ in range(25):
                self.ros2_interface.spin_once()
                time.sleep(0.02)
            
            # 红方初始位置: 特化模式固定, 常规模式随机
            spec_cfg = self.config.specialize_config
            if spec_cfg.get('enabled', False):
                reset_x = spec_cfg.get('start_x', 8.64)
                reset_y = spec_cfg.get('start_y', 3.65)
            else:
                reset_x, reset_y = self._sample_reset_position()
            
            # 重置位置，z 稍微抬高确保落地
            self.ros2_interface.set_robot_pose(reset_x, reset_y, 0.35, 0.0)
            
            # 等待机器人落地并稳定 (缩短等待时间)
            time.sleep(0.2)
            
            # 检查机器人是否稳定（速度接近零且未翻车）
            stable_wait = 0
            max_stable_wait = 20  # 最多等待 1 秒
            while stable_wait < max_stable_wait:
                # 处理回调获取最新数据
                self.ros2_interface.spin_once()
                
                # 如果翻车了，重新重置位置
                if self.ros2_interface.is_tumbled():
                    print("[WARN] 重置后翻车，重新重置位置")
                    self.ros2_interface.send_chassis_velocity(0.0, 0.0, 0.0)
                    for _ in range(25):
                        self.ros2_interface.spin_once()
                        time.sleep(0.02)
                    # 翻车重试时仍用特化位置, 不破坏训练假设
                    if spec_cfg.get('enabled', False):
                        reset_x = spec_cfg.get('start_x', 8.64)
                        reset_y = spec_cfg.get('start_y', 3.65)
                    else:
                        reset_x, reset_y = self._sample_reset_position()
                    self.ros2_interface.set_robot_pose(reset_x, reset_y, 0.35, 0.0)
                    time.sleep(0.2)
                    stable_wait = 0
                    continue
                
                # 获取机器人速度（从 odom 数据）
                odom_data = self.ros2_interface.state_data.get('odom')
                if odom_data is not None and 'twist' in odom_data:
                    twist = odom_data['twist']
                    linear = twist.get('linear', {})
                    angular = twist.get('angular', {})
                    # 检查线速度和角速度是否足够小
                    linear_vel = (linear.get('x', 0)**2 + linear.get('y', 0)**2 + linear.get('z', 0)**2)**0.5
                    angular_vel = (angular.get('x', 0)**2 + angular.get('y', 0)**2 + angular.get('z', 0)**2)**0.5
                    if linear_vel < 0.1 and angular_vel < 0.1:
                        break
                time.sleep(0.05)
                stable_wait += 1
            
            if stable_wait >= max_stable_wait:
                print("[WARN] 机器人未能在规定时间内稳定，继续训练")
        else:
            print(f"[WARN] Gazebo中未找到模型{self.config.robot_name}, 跳过位姿重置")

        # 重置环境: 只重置内部状态, 不发任何Gazebo命令
        # 使用虚拟蓝方位置进行课程学习(不移动Gazebo中的蓝方)
        self._generate_virtual_blue_position()

        # 输出目标位置
        print(f"[Episode] 目标位置: ({self._virtual_blue_x:.2f}, {self._virtual_blue_y:.2f})")

        # 重置步数
        self.current_step = 0

        # 重置环境状态信息
        self.own_hp = 400  # 重置血量
        self.own_ammo = 300  # 重置弹药量
        self.remaining_steps = self.max_steps  # 重置剩余步数
        self.judge_countdown_steps = 0  # 重置判负步数
        self.outpost_hp = 1500  # 重置前哨站血量
        self.base_hp = 5000  # 重置基地血量

        # 重置复活相关状态
        self.robot_level = 1  # 重置等级
        self.coin_revive_count = 0  # 重置金币复活次数
        self.revive_waiting_steps = 0  # 重置复活等待步数
        self.is_dead = False  # 重置死亡状态
        self._speed_history = []  # 重置卡住检测历史
        self._spec_last_dist = None  # 重置特化模式距离历史
        self._spec_last_z = None  # 重置特化模式高度历史
        self._spec_waypoint_reached = False  # 重置中间点到达标记
        self._position_history = []  # 重置位置历史
        self._vel_dir_history = []  # 重置速度方向历史

        # 重置奖励计算器
        self.reward_calculator.reset()

        # 等待数据稳定
        time.sleep(0.1)

        # 获取初始观察
        observation = self._get_observation()

        info = {
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'curriculum_stage': self._curriculum_stage,
            'virtual_blue_pos': (self._virtual_blue_x, self._virtual_blue_y),
        }

        # 课程学习: 累计episode数并检查是否升级阶段
        self._curriculum_episode_count += 1
        self._check_curriculum_stage_upgrade()

        return observation, info

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """渲染环境

        绘制 2D 俯视图, 显示场地、机器人、前哨站/基地位置,
        以及右侧面板列出观察空间各项参数。

        Args:
            mode: 渲染模式
                - 'human': 弹出窗口实时显示 (默认)
                - 'rgb_array': 返回 RGB 图像数组 (暂未实现, 返回 None)

        Returns:
            mode='human' 时返回 None
            mode='rgb_array' 时返回 (H, W, 3) 的 uint8 数组 (暂未实现)
        """
        # 获取当前观察 (用于渲染)
        obs = self._get_observation()

        # 懒初始化渲染器 (第一次调用 render 时才创建)
        if self.viewer is None:
            self.viewer = EnvRenderer()

        # 调用渲染器绘制
        self.viewer.render(
            obs=obs,
            env_config=self.config,
            current_step=self.current_step,
            max_steps=self.max_steps,
            reward=getattr(self, '_last_reward', 0.0),
            team=self.config.team,
            virtual_blue_pos=(self._virtual_blue_x, self._virtual_blue_y),
            train_progress=getattr(self, '_train_progress', None),
        )

        return None

    def close(self):
        """关闭环境"""
        # 保存可达点
        self._save_reachable_points()
        if len(self._reachable_points) > 0:
            print(f"[INFO] 保存 {len(self._reachable_points)} 个可达点")

        if self.ros2_interface is not None:
            self.ros2_interface.stop_spinning()
            self.ros2_interface.destroy()

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # ==================== 内部方法 ====================

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """获取观察数据"""
        # 判断是否用虚拟蓝方位置覆盖观测中的敌方位置
        virtual_blue_pos = None
        curriculum = self.config.curriculum_config
        if curriculum.get('enabled', False) \
                and curriculum.get('use_virtual_blue', True) \
                and curriculum.get('virtual_blue_override_obs', True):
            virtual_blue_pos = (self._virtual_blue_x, self._virtual_blue_y)

        return self.observation_space_manager.get_observation_from_interface(
            current_step=self.current_step,
            max_steps=self.max_steps,
            damage_per_step=self.damage_per_step,
            ammo_consumed_per_step=self.ammo_consumed_per_step,
            own_robot_id=0,  # TODO: 从配置获取
            # 传递环境维护的状态信息
            env_own_hp=self.own_hp,
            env_own_ammo=self.own_ammo,
            env_remaining_steps=self.remaining_steps,
            env_judge_countdown_steps=self.judge_countdown_steps,
            env_outpost_hp=self.outpost_hp,
            env_base_hp=self.base_hp,
            env_revive_waiting_steps=self.revive_waiting_steps,
            virtual_blue_pos=virtual_blue_pos,
        )

    def _execute_action(self, action: Dict[str, np.ndarray]):
        """执行动作

        云台不控制, 保持不动。
        底盘用 FOLLOW_GIMBAL 模式, 平缓跟随云台。
        """
        # 底盘: FOLLOW_GIMBAL 模式
        if 'chassis_velocity' in action:
            self.ros2_interface.send_chassis_follow_gimbal(
                linear_x=action['chassis_velocity'][0],
                linear_y=action['chassis_velocity'][1],
            )

        # 射击控制
        if 'shoot' in action:
            shoot_mode = int(action['shoot'])
            self.ros2_interface.send_shoot_command(projectile_num=shoot_mode)

    def _check_stuck(self, threshold: float = 0.05, window: int = 30) -> bool:
        """检测底盘是否卡住

        当最近 window 步中, 底盘实际速度持续低于 threshold, 判定为卡住。

        Args:
            threshold: 速度阈值 (m/s), 低于此值视为静止
            window: 检测窗口 (步数)

        Returns:
            bool: True 表示卡住
        """
        odom = self.ros2_interface.get_odom()
        if odom is None:
            return False

        actual_speed = np.sqrt(
            odom.get('linear_x', 0)**2 + odom.get('linear_y', 0)**2
        )

        # 维护速度历史
        if not hasattr(self, '_speed_history'):
            self._speed_history = []
        self._speed_history.append(actual_speed)
        if len(self._speed_history) > window:
            self._speed_history = self._speed_history[-window:]

        # 窗口内速度都低于阈值 → 卡住
        if len(self._speed_history) >= window:
            return all(s < threshold for s in self._speed_history)
        return False

    def _calculate_specialize_reward(self, spec_cfg: Dict) -> float:
        """特化模式奖励计算 - 分阶段引导

        路径: 起点 → 中间点(坡) → 目标点
        阶段1(未到达中间点): 引导去中间点, 强化爬坡, 不惩罚远离目标
        阶段2(已到达中间点): 引导去目标点, 不惩罚远离中间点

        规则:
        - 分阶段距离塑形奖励: 靠近当前阶段目标给正奖励, 远离给惩罚
        - 中间点到达奖励: 首次进入waypoint半径内给大奖励
        - 目标到达奖励: 进入目标半径内给大奖励
        - 爬坡奖励: 阶段1(上坡段)强化, 阶段2弱化
        - 速度方向与目标方向一致性: 速度朝当前阶段目标方向给奖励
        - 速度大小奖励: 鼓励快速移动
        - 时间惩罚: 每步小惩罚
        - 翻车/碰墙惩罚
        """
        reward = 0.0

        # 获取当前位置
        own_position = self.ros2_interface.get_robot_position()
        if own_position is None:
            return 0.0
        own_x, own_y, own_z = own_position

        target_x = spec_cfg.get('target_x', 14.0)
        target_y = spec_cfg.get('target_y', 7.5)
        waypoint_x = spec_cfg.get('waypoint_x', 4.81)
        waypoint_y = spec_cfg.get('waypoint_y', 2.47)

        # 计算到各关键点的距离
        dist_to_target = np.sqrt((own_x - target_x)**2 + (own_y - target_y)**2)
        dist_to_waypoint = np.sqrt((own_x - waypoint_x)**2 + (own_y - waypoint_y)**2)

        # ---- 阶段判断 ----
        waypoint_arrive_radius = spec_cfg.get('waypoint_arrive_radius', 1.5)
        # 一旦到达中间点, 整个episode内保持阶段2
        if dist_to_waypoint <= waypoint_arrive_radius:
            self._spec_waypoint_reached = True

        is_phase1 = not self._spec_waypoint_reached  # 阶段1: 去中间点

        # ---- 1. 分阶段距离塑形奖励 (核心) ----
        # 阶段1: 靠近中间点给正奖励, 远离中间点给惩罚
        # 阶段2: 靠近目标点给正奖励, 远离目标点给惩罚
        if is_phase1:
            shaping_weight = spec_cfg.get('waypoint_shaping_weight', 3.0)
            if self._spec_last_dist is None:
                self._spec_last_dist = dist_to_waypoint
            dist_delta = dist_to_waypoint - self._spec_last_dist  # 正值=远离中间点
            reward -= shaping_weight * dist_delta  # 靠近中间点→正奖励, 远离→惩罚
            self._spec_last_dist = dist_to_waypoint
        else:
            shaping_weight = spec_cfg.get('target_shaping_weight', 3.0)
            if self._spec_last_dist is None:
                self._spec_last_dist = dist_to_target
            dist_delta = dist_to_target - self._spec_last_dist  # 正值=远离目标
            reward -= shaping_weight * dist_delta  # 靠近目标→正奖励, 远离→惩罚
            self._spec_last_dist = dist_to_target

        # ---- 2. 中间点到达奖励 ----
        waypoint_arrive_reward = spec_cfg.get('waypoint_arrive_reward', 10.0)
        if is_phase1 and dist_to_waypoint <= waypoint_arrive_radius:
            # 在中间点附近给额外奖励, 越近越大
            proximity = 1.0 - dist_to_waypoint / waypoint_arrive_radius
            reward += waypoint_arrive_reward * proximity

        # ---- 3. 目标到达奖励 ----
        approach_radius = spec_cfg.get('approach_radius', 2.0)
        approach_reward = spec_cfg.get('approach_reward', 10.0)
        if not is_phase1 and dist_to_target <= approach_radius:
            proximity = 1.0 - dist_to_target / approach_radius
            reward += approach_reward * proximity

        # ---- 4. 爬坡奖励/下坡惩罚 ----
        # 阶段1(上坡段): 强化爬坡奖励, 鼓励往上走
        # 阶段2(下坡/平地段): 弱化, 只惩罚下坡(防止掉回去)
        climb_reward = spec_cfg.get('climb_reward', 1.0)
        descend_penalty = spec_cfg.get('descend_penalty', -0.5)
        climb_reward_phase2 = spec_cfg.get('climb_reward_phase2', 0.1)
        descend_penalty_phase2 = spec_cfg.get('descend_penalty_phase2', -1.0)
        if self._spec_last_z is None:
            self._spec_last_z = own_z
        z_delta = own_z - self._spec_last_z  # 正值=上升
        if is_phase1:
            if z_delta > 0:
                reward += climb_reward * z_delta
            elif z_delta < 0:
                reward += descend_penalty * abs(z_delta)
        else:
            if z_delta > 0:
                reward += climb_reward_phase2 * z_delta
            elif z_delta < 0:
                reward += descend_penalty_phase2 * abs(z_delta)
        self._spec_last_z = own_z

        # ---- 5. 速度方向与目标方向一致性 ----
        # 速度朝当前阶段目标方向投影为正时给奖励
        odom = self.ros2_interface.get_odom()
        if odom is not None:
            vx = odom.get('linear_x', 0.0)
            vy = odom.get('linear_y', 0.0)
            speed = np.sqrt(vx**2 + vy**2)

            # 当前阶段目标方向
            if is_phase1:
                goal_dx = waypoint_x - own_x
                goal_dy = waypoint_y - own_y
            else:
                goal_dx = target_x - own_x
                goal_dy = target_y - own_y
            goal_dist = np.sqrt(goal_dx**2 + goal_dy**2)

            if speed > 0.05 and goal_dist > 0.1:
                # 速度在目标方向上的投影 (归一化)
                vel_dot_goal = (vx * goal_dx + vy * goal_dy) / (speed * goal_dist)
                # vel_dot_goal ∈ [-1, 1], 1=完全朝目标, -1=完全反向
                direction_reward = spec_cfg.get('direction_reward', 1.0)
                reward += direction_reward * vel_dot_goal

            # 速度大小奖励: 鼓励快速移动 (阶段1上坡时更重要)
            speed_reward = spec_cfg.get('speed_reward', 0.2)
            speed_reward_phase2 = spec_cfg.get('speed_reward_phase2', 0.3)
            if is_phase1:
                reward += speed_reward * speed
            else:
                reward += speed_reward_phase2 * speed

            # 反向速度惩罚 (速度与目标方向夹角>90°)
            if speed > 0.05 and goal_dist > 0.1:
                vel_dot_goal = (vx * goal_dx + vy * goal_dy) / (speed * goal_dist)
                reverse_penalty = spec_cfg.get('reverse_penalty', -0.5)
                if vel_dot_goal < 0:
                    reward += reverse_penalty * abs(vel_dot_goal)

        # ---- 6. 时间惩罚 ----
        time_penalty = spec_cfg.get('time_penalty', -0.02)
        reward += time_penalty

        # ---- 7. 翻车惩罚 ----
        if self.ros2_interface.is_tumbled():
            reward += self.config.reward_config.get('tumble', -10.0)

        # ---- 8. 碰墙惩罚 + 回退 ----
        if self._check_stuck():
            reward += self.config.reward_config.get('stuck_penalty', -1.0)
            rollback_steps = spec_cfg.get('stuck_rollback_steps', 30)
            if len(self._position_history) >= rollback_steps:
                old_x, old_y, old_z = self._position_history[-rollback_steps]
                # 先清零速度, 否则 set_pose 后带着旧速度漂移
                self.ros2_interface.send_chassis_velocity(0.0, 0.0, 0.0)
                for _ in range(5):
                    self.ros2_interface.spin_once()
                    time.sleep(0.01)
                self.ros2_interface.set_robot_pose(old_x, old_y, 0.28, 0.0)
                self._spec_last_dist = None
                self._spec_last_z = None

        return reward

    def _get_enemy_outpost_position(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """获取敌方前哨站位置

        根据自身队伍颜色返回敌方前哨站的固定坐标。

        Returns:
            Tuple[x, y, z]: 敌方前哨站坐标，单位m
        """
        own_team = self.config.team
        if own_team == 'red':
            # 红方的敌方是蓝方
            return self.config.blue_outpost_position
        else:
            # 蓝方的敌方是红方
            return self.config.red_outpost_position

    def _get_enemy_base_position(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """获取敌方基地位置

        根据自身队伍颜色返回敌方基地的固定坐标。

        Returns:
            Tuple[x, y, z]: 敌方基地坐标，单位m
        """
        own_team = self.config.team
        if own_team == 'red':
            return self.config.blue_base_position
        else:
            return self.config.red_base_position

    def _wait_for_control_period(self):
        """等待控制周期
        
        control_period 是仿真时间基准，需要根据 real_time_factor 转换为真实时间。
        例如: control_period=0.033s (仿真时间), real_time_factor=10.0
        则真实时间等待 = 0.033 / 10.0 = 0.0033s
        
        等待期间持续处理 ROS2 回调，确保数据不积压。
        """
        # 将仿真时间转换为真实时间
        real_control_period = self.control_period / self.real_time_factor
        
        current_time = time.time()
        elapsed = current_time - self.last_control_time
        remaining = real_control_period - elapsed
        
        if remaining > 0:
            # 在等待期间处理 ROS2 回调，避免数据积压
            deadline = current_time + remaining
            while True:
                self.ros2_interface.spin_once()
                left = deadline - time.time()
                if left <= 0:
                    break
                time.sleep(min(0.001, left))
        
        self.last_control_time = time.time()

    def _measure_real_time_factor(self):
        """测量实际仿真时间流速，检测物理精度
        
        如果实测流速远低于设定值，说明物理计算跟不上，精度在下降。
        通过对比仿真时间流逝与真实时间流逝来计算实测 real_time_factor。
        """
        try:
            wall_time = time.time()
            
            # 通过 ROS2 clock 获取仿真时间
            # use_sim_time=True 时, get_clock() 返回的是仿真时钟
            clock = self.ros2_interface.node.get_clock()
            sim_time_msg = clock.now()
            sim_time_sec = sim_time_msg.nanoseconds / 1e9
            
            if self._sim_time_start is None or self._sim_time_start == 0:
                self._sim_time_start = sim_time_sec
                self._wall_time_start = wall_time
                return
            
            sim_elapsed = sim_time_sec - self._sim_time_start
            wall_elapsed = wall_time - self._wall_time_start
            
            if wall_elapsed > 2.0:  # 至少2秒真实时间后才计算
                if wall_elapsed > 0 and sim_elapsed > 0:
                    self._real_time_factor_measured = sim_elapsed / wall_elapsed
                # 重置基准，保持滑动窗口
                self._sim_time_start = sim_time_sec
                self._wall_time_start = wall_time
        except Exception:
            pass

    def _load_reachable_points(self):
        """从文件加载已保存的可达点"""
        try:
            if os.path.exists(self._reachable_points_file):
                arr = np.load(self._reachable_points_file)
                self._reachable_points = set(map(tuple, arr.astype(int).tolist()))
                print(f"[INFO] 加载 {len(self._reachable_points)} 个可达点")
        except Exception as e:
            print(f"[WARN] 加载可达点失败: {e}")
            self._reachable_points = set()

    def _save_reachable_points(self):
        """保存可达点到文件"""
        try:
            if self._reachable_points:
                arr = np.array(list(self._reachable_points), dtype=int)
                np.save(self._reachable_points_file, arr)
        except Exception as e:
            print(f"[WARN] 保存可达点失败: {e}")

    def _collect_reachable_point(self):
        """收集当前机器人位置的整数坐标作为可达点"""
        try:
            pos = self.ros2_interface.get_robot_position()
            if pos is not None:
                ix, iy = int(round(pos[0])), int(round(pos[1]))
                if (ix, iy) not in self._reachable_points:
                    self._reachable_points.add((ix, iy))
                    # 每100个新点保存一次
                    if len(self._reachable_points) % 100 == 0:
                        self._save_reachable_points()
        except Exception:
            pass

    def _sample_reset_position(self):
        """采样 reset 位置: 优先从可达点选取, 否则回退到均匀随机"""
        if len(self._reachable_points) >= 10:
            ix, iy = list(self._reachable_points)[np.random.randint(len(self._reachable_points))]
            # 整数坐标上加小随机偏移, 避免每次都精确在同一位置
            reset_x = ix + np.random.uniform(-0.3, 0.3)
            reset_y = iy + np.random.uniform(-0.3, 0.3)
            return reset_x, reset_y
        else:
            # 可达点不足, 回退到均匀随机
            reset_x = np.random.uniform(2.0, 12.0)
            reset_y = np.random.uniform(2.0, 13.0)
            return reset_x, reset_y

    def _calculate_reward(self) -> float:
        """计算奖励"""
        # 特化模式: 使用专用奖励逻辑, 原奖励不生效
        spec_cfg = self.config.specialize_config
        if spec_cfg.get('enabled', False):
            return self._calculate_specialize_reward(spec_cfg)

        # 从ROS2获取当前状态
        robot_status = self.ros2_interface.get_robot_status()
        referee_data = self.ros2_interface.get_referee_data()

        # 获取当前HP和弹药
        current_hp = 400  # 默认值
        current_ammo = 300  # 默认值
        if robot_status is not None:
            current_hp = robot_status.get('remain_hp', 400)
            total_ammo = robot_status.get('total_projectiles', 300)
            used_ammo = robot_status.get('used_projectiles', 0)
            current_ammo = total_ammo - used_ammo

        # 获取攻击信息
        attack_info = referee_data.get('attack_info', '')

        # 判断是否存活
        is_alive = current_hp > 0

        # 计算4m范围内的敌方机器人数量
        near_enemy_count = self._count_nearby_enemies(distance_threshold=4.0)

        # 计算最近敌方距离(用于距离渐变奖励和塑形奖励)
        nearest_enemy_distance = self._get_nearest_enemy_distance()

        # 场地最大距离(对角线约30m)
        max_field_distance = self.config.reward_config.get('max_field_distance', 30.0)

        # 计算虚拟蓝方距离(红方当前位置到虚拟蓝方位置的距离)
        virtual_blue_distance = None
        direction_alignment = None  # 方向跟随: 速度在目标方向上的投影
        curriculum = self.config.curriculum_config
        if curriculum.get('enabled', False) and curriculum.get('use_virtual_blue', True):
            own_position = self.ros2_interface.get_robot_position()
            if own_position is not None:
                own_x, own_y, own_z = own_position
                virtual_blue_distance = np.sqrt(
                    (self._virtual_blue_x - own_x) ** 2 +
                    (self._virtual_blue_y - own_y) ** 2
                )

                # 计算方向跟随: 当前速度在目标方向上的投影
                # 目标方向 = (蓝方 - 红方) / 距离
                if virtual_blue_distance > 0.1:
                    target_dx = (self._virtual_blue_x - own_x) / virtual_blue_distance
                    target_dy = (self._virtual_blue_y - own_y) / virtual_blue_distance
                    # 从 odom 获取当前速度
                    odom_data = self.ros2_interface.state_data.get('odom')
                    if odom_data is not None and 'twist' in odom_data:
                        twist = odom_data['twist']
                        linear = twist.get('linear', {})
                        vx = linear.get('x', 0.0)
                        vy = linear.get('y', 0.0)
                        # 速度在目标方向上的投影 (归一化到 [-1, 1])
                        speed = np.sqrt(vx**2 + vy**2)
                        if speed > 0.01:
                            direction_alignment = (vx * target_dx + vy * target_dy) / speed

        # 计算奖励
        reward = self.reward_calculator.calculate_reward(
            current_hp=current_hp,
            current_ammo=current_ammo,
            attack_info=attack_info,
            is_alive=is_alive,
            near_enemy_count=near_enemy_count,
            nearest_enemy_distance=nearest_enemy_distance,
            max_field_distance=max_field_distance,
            virtual_blue_distance=virtual_blue_distance,
            direction_alignment=direction_alignment,
        )

        # 碰墙惩罚: 底盘卡住时给惩罚, 让网络学会避墙
        if self._check_stuck():
            reward += self.config.reward_config.get('stuck_penalty', -1.0)

        return reward

    def _count_nearby_enemies(self, distance_threshold: float = 4.0) -> int:
        """计算指定距离范围内的敌方机器人数量

        Args:
            distance_threshold: 距离阈值(米)

        Returns:
            int: 范围内的敌方机器人数量
        """
        try:
            # 获取所有机器人位置
            all_robots = self.interface_adapter.get_all_robot_poses()
            if all_robots is None:
                return 0

            # 获取自己的位置
            own_position = self.ros2_interface.get_robot_position()
            if own_position is None:
                return 0

            own_x, own_y, own_z = own_position
            own_team = self.config.team

            # 统计范围内的敌方数量
            count = 0
            for robot in all_robots:
                robot_team = robot.get('team', 'unknown')
                robot_x = robot.get('x', 0)
                robot_y = robot.get('y', 0)

                # 只统计敌方机器人
                if robot_team != own_team and robot_team != 'unknown':
                    # 计算距离
                    distance = np.sqrt((robot_x - own_x)**2 + (robot_y - own_y)**2)
                    if distance <= distance_threshold:
                        count += 1

            return count

        except Exception:
            return 0

    def _get_own_position(self):
        """从pose_info获取自身机器人的实际位置

        Returns:
            tuple: (x, y) 或 (None, None) 如果数据不可用
        """
        try:
            referee_data = self.ros2_interface.get_referee_data()
            pose_info = referee_data.get('pose_info')
            if pose_info is None:
                return None, None
            own_name = self.config.robot_name  # e.g. 'red_standard_robot1'
            # 优先找_0后缀的 (Gazebo -allow_renaming产生的实际模型)
            own_name_0 = f"{own_name}_0"
            for t in pose_info:
                if t.child_frame_id == own_name_0:
                    return t.transform.translation.x, t.transform.translation.y
            # 回退到无后缀
            for t in pose_info:
                if t.child_frame_id == own_name:
                    return t.transform.translation.x, t.transform.translation.y
            # 都找不到, 返回None
            return None, None
        except Exception:
            return None, None

    def _get_own_position_safe(self, max_retries=3, retry_interval=0.005):
        """安全获取自身位置, 带重试机制

        set_robot_pose后Gazebo可能短暂不发布pose_info,
        重试几次确保能拿到数据。

        Returns:
            tuple: (x, y) 或 (None, None) 如果重试后仍不可用
        """
        for i in range(max_retries):
            x, y = self._get_own_position()
            if x is not None:
                return x, y
            time.sleep(retry_interval)
        return None, None

    def _generate_virtual_blue_position(self):
        """根据课程学习阶段生成虚拟蓝方位置

        在红方初始位置周围, 按当前阶段距离范围随机生成蓝方位置,
        存储在 self._virtual_blue_x/y 中, 仅用于奖励计算,
        不实际移动 Gazebo 中的蓝方机器人。
        """
        # 特化模式下使用固定目标位置
        spec_cfg = self.config.specialize_config
        if spec_cfg.get('enabled', False):
            self._virtual_blue_x = spec_cfg.get('target_x', 14.0)
            self._virtual_blue_y = spec_cfg.get('target_y', 7.5)
            return

        curriculum = self.config.curriculum_config
        if not curriculum.get('enabled', False) or not curriculum.get('use_virtual_blue', True):
            # 课程学习未启用, 使用默认蓝方位置
            self._virtual_blue_x = self.config.virtual_blue_x
            self._virtual_blue_y = self.config.virtual_blue_y
            return

        # 获取红方实际位置 (从pose_info读取)
        red_x, red_y = self._get_own_position_safe()
        if red_x is None:
            # 回退到场地中心
            red_x, red_y = 7.0, 7.5

        # 获取当前阶段的距离范围
        stage = self._curriculum_stage
        stage_ranges = curriculum.get('stage_ranges', {})
        dist_range = stage_ranges.get(stage, (3.0, 25.0))
        min_dist, max_dist = dist_range

        # 随机生成距离和角度
        distance = np.random.uniform(min_dist, max_dist)
        angle = np.random.uniform(0, 2 * np.pi)

        blue_x = red_x + distance * np.cos(angle)
        blue_y = red_y + distance * np.sin(angle)

        # 场地边界约束 (rmuc场地: x∈[0,28], y∈[0,15])
        self._virtual_blue_x = float(np.clip(blue_x, 1.0, 27.0))
        self._virtual_blue_y = float(np.clip(blue_y, 1.0, 14.0))

    def _apply_curriculum_blue_position(self):
        """根据课程学习阶段随机设置蓝方机器人位置

        在红方初始位置周围, 按当前阶段距离范围随机生成蓝方位置,
        并通过 set_robot_pose_by_name 设置蓝方位姿。
        """
        curriculum = self.config.curriculum_config
        if not curriculum.get('enabled', False):
            return

        # 获取红方实际位置 (从pose_info读取)
        red_x, red_y = self._get_own_position_safe()
        if red_x is None:
            # 回退到场地中心
            red_x, red_y = 7.0, 7.5

        # 获取当前阶段的距离范围
        stage = self._curriculum_stage
        stage_ranges = curriculum.get('stage_ranges', {})
        dist_range = stage_ranges.get(stage, (3.0, 25.0))
        min_dist, max_dist = dist_range

        # 随机生成距离和角度
        distance = np.random.uniform(min_dist, max_dist)
        angle = np.random.uniform(0, 2 * np.pi)

        blue_x = red_x + distance * np.cos(angle)
        blue_y = red_y + distance * np.sin(angle)

        # 场地边界约束 (rmuc场地: x∈[0,28], y∈[0,15])
        blue_x = np.clip(blue_x, 1.0, 27.0)
        blue_y = np.clip(blue_y, 1.0, 14.0)

        # 设置蓝方位姿 (z=0.28与gz_world.yaml一致, yaw朝向红方)
        yaw_to_red = np.arctan2(red_y - blue_y, red_x - blue_x)
        # z=4.0: 从空中自由下落, 避免卡在障碍物里
        self.ros2_interface.set_robot_pose_by_name(
            'blue_standard_robot1', blue_x, blue_y, 4.0, yaw_to_red
        )

    def _check_curriculum_stage_upgrade(self):
        """检查是否应升级课程学习阶段

        当当前阶段的episode数达到阈值时, 自动升级到下一阶段。
        """
        curriculum = self.config.curriculum_config
        if not curriculum.get('enabled', False):
            return

        stage = self._curriculum_stage
        stage_episodes = curriculum.get('stage_episodes', {})
        max_episodes = stage_episodes.get(stage, -1)

        # -1 表示不自动升级
        if max_episodes == -1:
            return

        if self._curriculum_episode_count >= max_episodes:
            next_stage = stage + 1
            max_stage = max(list(curriculum.get('stage_ranges', {}).keys()) + [4])
            if next_stage <= max_stage:
                self._curriculum_stage = next_stage
                self._curriculum_episode_count = 0
                print(f"[Curriculum] 升级到阶段 {next_stage}: "
                      f"蓝方距离范围 {curriculum['stage_ranges'].get(next_stage, 'N/A')}m")

    def _check_terminated(self) -> bool:
        """检查是否终止"""
        # 检查HP是否为0
        robot_status = self.ros2_interface.get_robot_status()
        if robot_status is not None:
            remain_hp = robot_status.get('remain_hp', 0)
            if remain_hp <= 0:
                return True

        # 检查是否翻车
        if self.ros2_interface.is_tumbled():
            return True

        # 检查是否出边界 (rmuc场地: x∈[0,28], y∈[0,15])
        # 不能离边界 1m 内
        own_x, own_y = self._get_own_position_safe()
        if own_x is not None:
            margin = 1.0
            if own_x < margin or own_x > 28 - margin or own_y < margin or own_y > 15 - margin:
                return True

        return False

    def _update_env_state(self, action: Dict[str, np.ndarray]):
        """更新环境状态信息

        Args:
            action: 当前步的动作
        """
        # 1. 更新剩余步数
        self.remaining_steps = self.max_steps - self.current_step

        # 2. 处理复活等待步数递减
        if self.revive_waiting_steps > 0:
            self.revive_waiting_steps -= 1
            # 复活完成
            if self.revive_waiting_steps == 0 and self.is_dead:
                self.is_dead = False
                self.own_hp = 400  # 复活后恢复满血
                self.own_ammo = 300  # 复活后恢复弹药

        # 3. 处理射击动作
        if 'shoot' in action and not self.is_dead:
            shoot_mode = int(action['shoot'])
            # shoot_mode: 0=不射击, 1-6=射击机器人, 7=射击前哨站, 8=射击基地
            if shoot_mode > 0:
                # 消耗弹药
                self.own_ammo = max(0, self.own_ammo - 1)

                # 根据目标类型计算伤害
                if 1 <= shoot_mode <= 6:
                    # 射击机器人
                    target_distance = self._get_target_distance(shoot_mode)
                    actual_damage = self._calculate_damage_with_distance(
                        base_damage=self.damage_per_step,
                        distance=target_distance
                    )

                elif shoot_mode == 7:
                    # 射击前哨站(须在5m内)
                    outpost_distance = self._get_outpost_distance()
                    if outpost_distance <= 5.0:
                        actual_damage = self._calculate_damage_with_fluctuation(self.damage_per_step)
                        # 更新前哨站血量
                        self.outpost_hp = max(0, self.outpost_hp - int(actual_damage))

                elif shoot_mode == 8:
                    # 射击基地(须在5m内, 且基地已展开且前哨站血量为0)
                    base_distance = self._get_base_distance()
                    if base_distance <= 5.0 and self._can_damage_base():
                        actual_damage = self._calculate_damage_with_fluctuation(self.damage_per_step)
                        # 更新基地血量
                        self.base_hp = max(0, self.base_hp - int(actual_damage))

        # 4. 处理金币复活
        if 'revive_with_coins' in action and action['revive_with_coins'] and self.is_dead:
            self._handle_coin_revive()

        # 5. 处理远程弹药兑换
        if 'remote_ammo_exchange' in action and action['remote_ammo_exchange'] and not self.is_dead:
            self._handle_remote_ammo_exchange()

        # 6. 处理非远程弹药兑换
        if 'local_ammo_exchange' in action and action['local_ammo_exchange'] and not self.is_dead:
            self._handle_local_ammo_exchange()

        # 7. 检查死亡状态
        if self.own_hp <= 0 and not self.is_dead:
            self.is_dead = True
            # 计算时间复活等待步数
            self.revive_waiting_steps = int(self.current_step / 50 + 50 + 100 * self.coin_revive_count)

        # 8. 更新判负步数（从ROS2接口获取）
        referee_data = self.ros2_interface.get_referee_data()
        if referee_data is not None:
            judge_countdown_seconds = referee_data.get('judge_countdown', 0)
            # 将秒转换为步数 (1秒=5步)
            self.judge_countdown_steps = judge_countdown_seconds * 5 if judge_countdown_seconds > 0 else 0

    def _calculate_damage_with_distance(
        self,
        base_damage: float,
        distance: float
    ) -> float:
        """计算基于距离的伤害（包含衰减和随机波动）

        Args:
            base_damage: 基础伤害值
            distance: 攻击距离(米)

        Returns:
            float: 实际伤害值
        """
        # 1. 距离检查：超过7m不造成伤害
        if distance > self.max_attack_distance:
            return 0.0

        # 2. 距离衰减：超过5m伤害减少50%
        if distance > self.damage_reduction_distance:
            base_damage *= 0.5

        # 3. 随机波动：在25%范围内随机波动
        # 生成[-0.25, 0.25]范围内的随机数
        fluctuation = np.random.uniform(-self.damage_fluctuation_rate, self.damage_fluctuation_rate)
        actual_damage = base_damage * (1.0 + fluctuation)

        # 确保伤害不为负
        return max(0.0, actual_damage)

    def _get_nearest_enemy_distance(self) -> float:
        """获取最近敌方机器人的距离

        Returns:
            float: 最近敌方距离(米)，如果没有敌方则返回无穷大
        """
        try:
            # 获取所有机器人位置
            all_robots = self.interface_adapter.get_all_robot_poses()
            if all_robots is None:
                return float('inf')

            # 获取自己的位置
            own_position = self.ros2_interface.get_robot_position()
            if own_position is None:
                return float('inf')

            own_x, own_y, own_z = own_position
            own_team = self.config.team

            # 查找最近的敌方机器人
            min_distance = float('inf')
            for robot in all_robots:
                robot_team = robot.get('team', 'unknown')
                robot_x = robot.get('x', 0)
                robot_y = robot.get('y', 0)

                # 只统计敌方机器人
                if robot_team != own_team and robot_team != 'unknown':
                    # 计算距离
                    distance = np.sqrt((robot_x - own_x)**2 + (robot_y - own_y)**2)
                    if distance < min_distance:
                        min_distance = distance

            return min_distance

        except Exception:
            return float('inf')

    def _calculate_damage_with_fluctuation(self, base_damage: float) -> float:
        """计算带随机波动的伤害

        Args:
            base_damage: 基础伤害值

        Returns:
            float: 实际伤害值
        """
        # 随机波动：在25%范围内随机波动
        fluctuation = np.random.uniform(-self.damage_fluctuation_rate, self.damage_fluctuation_rate)
        actual_damage = base_damage * (1.0 + fluctuation)
        return max(0.0, actual_damage)

    def _get_target_distance(self, target_id: int) -> float:
        """获取指定目标机器人的距离

        Args:
            target_id: 目标机器人ID (1-6)

        Returns:
            float: 目标距离(米)
        """
        try:
            # 获取所有机器人位置
            all_robots = self.interface_adapter.get_all_robot_poses()
            if all_robots is None:
                return float('inf')

            # 获取自己的位置
            own_position = self.ros2_interface.get_robot_position()
            if own_position is None:
                return float('inf')

            own_x, own_y, own_z = own_position

            # 查找指定ID的机器人
            for robot in all_robots:
                robot_id = robot.get('id', -1)
                if robot_id == target_id:
                    robot_x = robot.get('x', 0)
                    robot_y = robot.get('y', 0)
                    distance = np.sqrt((robot_x - own_x)**2 + (robot_y - own_y)**2)
                    return distance

            return float('inf')

        except Exception:
            return float('inf')

    def _get_outpost_distance(self) -> float:
        """获取敌方前哨站的3D距离

        Returns:
            float: 前哨站距离(米)
        """
        try:
            own_position = self.ros2_interface.get_robot_position()
            if own_position is None:
                return float('inf')

            own_x, own_y, own_z = own_position
            target_x, target_y, target_z = self._get_enemy_outpost_position()

            if target_x is None:
                return float('inf')

            distance = np.sqrt(
                (target_x - own_x)**2 +
                (target_y - own_y)**2 +
                (target_z - own_z)**2
            )
            return distance

        except Exception:
            return float('inf')

    def _get_base_distance(self) -> float:
        """获取敌方基地的3D距离

        Returns:
            float: 基地距离(米)
        """
        try:
            own_position = self.ros2_interface.get_robot_position()
            if own_position is None:
                return float('inf')

            own_x, own_y, own_z = own_position
            target_x, target_y, target_z = self._get_enemy_base_position()

            if target_x is None:
                return float('inf')

            distance = np.sqrt(
                (target_x - own_x)**2 +
                (target_y - own_y)**2 +
                (target_z - own_z)**2
            )
            return distance

        except Exception:
            return float('inf')

    def _can_damage_base(self) -> bool:
        """判断是否可以对敌方基地造成伤害

        条件: 基地已展开(base_exposed=True) 且 前哨站血量为0

        Returns:
            bool: True表示可以伤害基地
        """
        # 获取基地展开状态和前哨站血量
        try:
            outpost_hp, base_hp, base_exposed = \
                self.interface_adapter.get_outpost_and_base_state()

            # 基地必须已展开，且前哨站必须已被摧毁
            if base_exposed and outpost_hp == 0:
                return True
            return False
        except Exception:
            return False

    def _handle_coin_revive(self):
        """处理金币复活逻辑"""
        # 计算金币复活费用: 当前步数/300*80 + 机器人等级*20
        revive_cost = int(self.current_step / 300 * 80 + self.robot_level * 20)

        # 从ROS2接口获取当前金币
        game_state = self.interface_adapter.get_game_state()
        if game_state is not None:
            current_coins = game_state.get('team_economy', 0)

            # 检查金币是否足够
            if current_coins >= revive_cost:
                # 扣除金币并立即复活
                # TODO: 通过ROS2接口扣除金币
                self.coin_revive_count += 1
                self.is_dead = False
                self.own_hp = 400  # 复活后恢复满血
                self.own_ammo = 300  # 复活后恢复弹药
                self.revive_waiting_steps = 0

    def _handle_remote_ammo_exchange(self):
        """处理远程弹药兑换逻辑"""
        # 从ROS2接口获取当前金币
        game_state = self.interface_adapter.get_game_state()
        if game_state is not None:
            current_coins = game_state.get('team_economy', 0)

            # 检查金币是否足够 (150金币)
            if current_coins >= self.remote_ammo_cost:
                # 扣除金币并增加弹药
                # TODO: 通过ROS2接口扣除金币
                self.own_ammo = min(300, self.own_ammo + self.remote_ammo_gain)

    def _handle_local_ammo_exchange(self):
        """处理非远程弹药兑换逻辑"""
        # 暂时不产生效果
        # TODO: 未来实现补给区/增益点/前哨站增益点的弹药兑换功能
        pass
