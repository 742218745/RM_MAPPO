#!/usr/bin/env python3
"""
端到端通信测试节点

测试 Gym 环境与 Gazebo 仿真之间的双向 ROS2 通信是否连贯:

  方向1 (Gazebo -> Env): sim_data_publisher 发布随机仿真数据 -> env 订阅接收
  方向2 (Env -> Gazebo): env 发布控制命令 (cmd_vel, gimbal_cmd, shoot_cmd) -> Gazebo 接收

测试流程:
  1. 创建 env, 验证观测/动作空间
  2. 等待 sim_data_publisher 的数据到达 env (方向1验证)
  3. 执行随机动作, env 通过 ROS2 发布器发送控制命令 (方向2验证)
  4. 监听 env 发出的话题, 确认命令已发出
  5. 汇总通信连贯性报告

使用方法:
  终端1: ros2 run robomaster_gym_env sim_data_publisher
  终端2: ros2 run robomaster_gym_env gym_test_node
  或一键启动: bash run_comm_test.sh
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import time
from typing import Dict, Any, Optional

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from example_interfaces.msg import UInt8
from std_msgs.msg import Bool, String

try:
    from rmoss_interfaces.msg import GimbalCmd, ShootCmd
    RMOSS_AVAILABLE = True
except ImportError:
    RMOSS_AVAILABLE = False

from robomaster_gym_env import RoboMasterGazeboEnv, GymEnvConfig


class CommandMonitor(Node):
    """监听 env 发出的控制命令, 验证方向2 (Env -> Gazebo) 通信"""

    def __init__(self, robot_namespace: str = "red_standard_robot1"):
        super().__init__('command_monitor')

        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 记录收到的命令
        self.received = {
            'cmd_vel': 0,
            'gimbal_cmd': 0,
            'shoot_cmd': 0,
            'cmd_shoot': 0,
        }
        self.last_cmd_vel = None
        self.last_gimbal_cmd = None
        self.last_shoot_cmd = None

        # 订阅 env 发布的控制话题
        self.sub_cmd_vel = self.create_subscription(
            Twist, f'{robot_namespace}/cmd_vel',
            self._cmd_vel_cb, self.qos_profile)

        if RMOSS_AVAILABLE:
            self.sub_gimbal_cmd = self.create_subscription(
                GimbalCmd, f'{robot_namespace}/robot_base/gimbal_cmd',
                self._gimbal_cmd_cb, self.qos_profile)
            self.sub_shoot_cmd = self.create_subscription(
                ShootCmd, f'{robot_namespace}/robot_base/shoot_cmd',
                self._shoot_cmd_cb, self.qos_profile)

        self.sub_cmd_shoot = self.create_subscription(
            UInt8, f'{robot_namespace}/cmd_shoot',
            self._cmd_shoot_cb, self.qos_profile)

    def _cmd_vel_cb(self, msg: Twist):
        self.received['cmd_vel'] += 1
        self.last_cmd_vel = {
            'linear_x': msg.linear.x,
            'linear_y': msg.linear.y,
            'angular_z': msg.angular.z,
        }

    def _gimbal_cmd_cb(self, msg):
        self.received['gimbal_cmd'] += 1
        self.last_gimbal_cmd = {
            'yaw': msg.position.yaw,
            'pitch': msg.position.pitch,
        }

    def _shoot_cmd_cb(self, msg):
        self.received['shoot_cmd'] += 1
        self.last_shoot_cmd = {
            'projectile_num': msg.projectile_num,
        }

    def _cmd_shoot_cb(self, msg: UInt8):
        self.received['cmd_shoot'] += 1

    def get_summary(self) -> Dict[str, Any]:
        return {
            'received_counts': self.received.copy(),
            'last_cmd_vel': self.last_cmd_vel,
            'last_gimbal_cmd': self.last_gimbal_cmd,
            'last_shoot_cmd': self.last_shoot_cmd,
        }


def test_spaces(env):
    """测试1: 观测空间和动作空间"""
    print("\n" + "=" * 60)
    print("  测试1: 观测空间 & 动作空间")
    print("=" * 60)

    obs_space = env.observation_space
    print(f"\n观测空间 ({type(obs_space).__name__}):")
    for key, space in obs_space.spaces.items():
        if hasattr(space, 'shape'):
            print(f"  {key}: shape={space.shape}, dtype={space.dtype}")
        else:
            print(f"  {key}: {space}")

    act_space = env.action_space
    print(f"\n动作空间 ({type(act_space).__name__}):")
    for key, space in act_space.spaces.items():
        if hasattr(space, 'shape'):
            print(f"  {key}: shape={space.shape}, dtype={space.dtype}")
        else:
            print(f"  {key}: {space}")

    print("\n[PASS] 空间定义正常")


def test_gazebo_to_env(env, wait_seconds: float = 3.0):
    """测试2: Gazebo -> Env 方向通信 (env 能否收到仿真数据)"""
    print("\n" + "=" * 60)
    print("  测试2: Gazebo -> Env (仿真数据接收)")
    print("=" * 60)

    print(f"\n等待 {wait_seconds}s 让仿真数据到达 env...")
    time.sleep(wait_seconds)

    # 检查 env 内部数据
    ros2_if = env.ros2_interface
    sensor_data = ros2_if.sensor_data
    state_data = ros2_if.state_data
    referee_data = ros2_if.referee_data

    results = {}

    # 检查里程计
    has_odom = 'odom' in state_data
    results['odom'] = has_odom
    if has_odom:
        pos = state_data['odom']['pose']['position']
        print(f"  [OK] 里程计: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
    else:
        print(f"  [MISS] 里程计: 未收到数据")

    # 检查里程计真值
    has_gt = 'chassis_odometry_gt' in state_data
    results['chassis_odom_gt'] = has_gt
    if has_gt:
        pos = state_data['chassis_odometry_gt']['pose']['position']
        print(f"  [OK] 里程计真值: x={pos[0]:.2f}, y={pos[1]:.2f}")
    else:
        print(f"  [MISS] 里程计真值: 未收到数据")

    # 检查 IMU
    has_imu = 'imu' in sensor_data
    results['imu'] = has_imu
    if has_imu:
        acc = sensor_data['imu']['linear_acceleration']
        print(f"  [OK] IMU: accel=({acc[0]:.2f}, {acc[1]:.2f}, {acc[2]:.2f})")
    else:
        print(f"  [MISS] IMU: 未收到数据")

    # 检查机器人状态
    has_robot_status = 'robot_status' in referee_data
    results['robot_status'] = has_robot_status
    if has_robot_status:
        rs = referee_data['robot_status']
        print(f"  [OK] 机器人状态: HP={rs['remain_hp']}/{rs['max_hp']}, "
              f"弹药={rs['total_projectiles']-rs['used_projectiles']}")
    else:
        print(f"  [MISS] 机器人状态: 未收到数据")

    # 检查位姿信息
    has_pose = 'pose_info' in referee_data
    results['pose_info'] = has_pose
    if has_pose:
        n_robots = len(referee_data['pose_info'])
        print(f"  [OK] 位姿信息: {n_robots} 个机器人")
    else:
        print(f"  [MISS] 位姿信息: 未收到数据")

    # 检查攻击信息
    has_attack = 'attack_info' in referee_data
    results['attack_info'] = has_attack
    if has_attack:
        print(f"  [OK] 攻击信息: {referee_data['attack_info'][:50]}...")
    else:
        print(f"  [MISS] 攻击信息: 未收到数据")

    # 检查游戏状态
    has_game = 'game_status' in referee_data
    results['game_status'] = has_game
    if has_game:
        gs = referee_data['game_status']
        print(f"  [OK] 游戏状态: status={gs['game_status']}, "
              f"remaining_time={gs['remaining_time']}s")
    else:
        print(f"  [MISS] 游戏状态: 未收到数据")

    # 检查前哨站/基地
    has_outpost = 'outpost_status' in referee_data
    has_base = 'base_status' in referee_data
    results['outpost_status'] = has_outpost
    results['base_status'] = has_base
    if has_outpost:
        print(f"  [OK] 前哨站: HP={referee_data['outpost_status']['remain_hp']}")
    else:
        print(f"  [MISS] 前哨站: 未收到数据")
    if has_base:
        print(f"  [OK] 基地: HP={referee_data['base_status']['remain_hp']}")
    else:
        print(f"  [MISS] 基地: 未收到数据")

    # 检查使能
    has_power = 'enable_power' in referee_data
    has_control = 'enable_control' in referee_data
    results['enable_power'] = has_power
    results['enable_control'] = has_control
    if has_power:
        print(f"  [OK] 电源使能: {referee_data['enable_power']}")
    else:
        print(f"  [MISS] 电源使能: 未收到数据")
    if has_control:
        print(f"  [OK] 控制使能: {referee_data['enable_control']}")
    else:
        print(f"  [MISS] 控制使能: 未收到数据")

    # 汇总
    received = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  方向1 汇总: {received}/{total} 话题收到数据")

    if received == total:
        print("  [PASS] Gazebo -> Env 通信正常")
    elif received > 0:
        print("  [WARN] 部分话题未收到数据 (可能 sim_data_publisher 未启动或 rmoss_interfaces 不可用)")
    else:
        print("  [FAIL] 未收到任何数据, 请确认 sim_data_publisher 已启动")

    return results


def test_env_to_gazebo(env, monitor: CommandMonitor, max_steps: int = 30):
    """测试3: Env -> Gazebo 方向通信 (env 发出的控制命令能否被监听到)"""
    print("\n" + "=" * 60)
    print("  测试3: Env -> Gazebo (控制命令发送)")
    print("=" * 60)

    print(f"\n执行 {max_steps} 步随机动作, env 将通过 ROS2 发布器发送控制命令...")

    obs, info = env.reset(seed=42)
    total_reward = 0.0
    shoot_count = 0

    for step in range(max_steps):
        action = env.action_space.sample()

        # 确保至少有一些射击动作
        if step % 5 == 0:
            action['shoot'] = np.random.randint(1, 9)
            shoot_count += 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # 让 monitor 有机会接收回调
        rclpy.spin_once(monitor, timeout_sec=0.001)

        if step % 10 == 0:
            chassis = action.get('chassis_velocity', np.array([0, 0]))
            shoot = action.get('shoot', 0)
            print(f"  Step {step+1:3d}: vel=({chassis[0]:+.2f}, {chassis[1]:+.2f}), "
                  f"shoot={shoot}, reward={reward:.4f}")

        if terminated or truncated:
            print(f"  Episode 在第 {step+1} 步结束")
            break

    # 等待最后的命令到达
    time.sleep(0.5)
    rclpy.spin_once(monitor, timeout_sec=0.1)

    # 汇总
    summary = monitor.get_summary()
    print(f"\n  方向2 命令接收统计:")
    for topic, count in summary['received_counts'].items():
        status = "[OK]" if count > 0 else "[MISS]"
        print(f"    {status} {topic}: {count} 条")

    if summary['last_cmd_vel']:
        v = summary['last_cmd_vel']
        print(f"\n  最后一条 cmd_vel: linear_x={v['linear_x']:.3f}, "
              f"linear_y={v['linear_y']:.3f}, angular_z={v['angular_z']:.3f}")

    if summary['last_gimbal_cmd']:
        g = summary['last_gimbal_cmd']
        print(f"  最后一条 gimbal_cmd: yaw={g['yaw']:.3f}, pitch={g['pitch']:.3f}")

    if summary['last_shoot_cmd']:
        s = summary['last_shoot_cmd']
        print(f"  最后一条 shoot_cmd: projectile_num={s['projectile_num']}")

    # 判定
    cmd_vel_ok = summary['received_counts']['cmd_vel'] > 0
    cmd_shoot_ok = summary['received_counts']['cmd_shoot'] > 0

    if cmd_vel_ok and cmd_shoot_ok:
        print("\n  [PASS] Env -> Gazebo 通信正常 (cmd_vel + cmd_shoot 已发出)")
    elif cmd_vel_ok:
        print("\n  [WARN] cmd_vel 正常, 但 cmd_shoot 未发出 (可能射击动作未触发)")
    else:
        print("\n  [FAIL] 未检测到控制命令, 请检查 env 的发布器")

    print(f"\n  总奖励: {total_reward:.4f}, 射击次数: {shoot_count}")

    return summary


def test_observation_values(env):
    """测试4: 验证 env 输出的观测值合理性"""
    print("\n" + "=" * 60)
    print("  测试4: 观测值合理性检查")
    print("=" * 60)

    obs, info = env.reset(seed=99)

    checks = {}

    # all_robots 应该是 (10, 4) 的数组
    all_robots = obs.get('all_robots')
    if all_robots is not None:
        shape_ok = all_robots.shape == (10, 4)
        checks['all_robots_shape'] = shape_ok
        if shape_ok:
            valid_robots = np.sum(all_robots[:, 0] != -1)
            print(f"  [OK] all_robots: shape={all_robots.shape}, 有效机器人={valid_robots}")
        else:
            print(f"  [FAIL] all_robots: shape={all_robots.shape}, 期望 (10, 4)")
    else:
        checks['all_robots_shape'] = False
        print(f"  [FAIL] all_robots: 缺失")

    # own_hp 应该在合理范围
    own_hp = obs.get('own_hp')
    hp_ok = isinstance(own_hp, (int, np.integer)) and -1 <= own_hp <= 400
    checks['own_hp'] = hp_ok
    print(f"  {'[OK]' if hp_ok else '[FAIL]'} own_hp: {own_hp}")

    # own_ammo
    own_ammo = obs.get('own_ammo')
    ammo_ok = isinstance(own_ammo, (int, np.integer)) and -1 <= own_ammo <= 300
    checks['own_ammo'] = ammo_ok
    print(f"  {'[OK]' if ammo_ok else '[FAIL]'} own_ammo: {own_ammo}")

    # remaining_steps
    remaining = obs.get('remaining_steps')
    remaining_ok = isinstance(remaining, (int, np.integer)) and remaining >= 0
    checks['remaining_steps'] = remaining_ok
    print(f"  {'[OK]' if remaining_ok else '[FAIL]'} remaining_steps: {remaining}")

    # damage_per_step
    damage = obs.get('damage_per_step')
    if isinstance(damage, np.ndarray):
        damage_ok = damage.shape == (1,)
    else:
        damage_ok = False
    checks['damage_per_step'] = damage_ok
    print(f"  {'[OK]' if damage_ok else '[FAIL]'} damage_per_step: {damage}")

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    print(f"\n  观测值检查: {passed}/{total} 通过")

    if passed == total:
        print("  [PASS] 观测值合理")
    else:
        print("  [WARN] 部分观测值异常")

    return checks


def main():
    """主函数"""
    rclpy.init()

    robot_name = "red_standard_robot1"
    robot_namespace = "red_standard_robot1"

    try:
        # 创建配置
        config = GymEnvConfig()
        config.robot_name = robot_name
        config.robot_namespace = robot_namespace
        config.team = "red"
        config.control_frequency = 10.0  # 10Hz for testing

        # 创建环境
        env = RoboMasterGazeboEnv(config)

        # 创建命令监听器
        monitor = CommandMonitor(robot_namespace=robot_namespace)

        print("\n" + "=" * 60)
        print("  RoboMaster 端到端通信测试")
        print("=" * 60)
        print(f"  Robot: {robot_name}")
        print(f"  Namespace: {robot_namespace}")
        print(f"  控制频率: {config.control_frequency} Hz")
        print("=" * 60)

        # 测试1: 空间定义
        test_spaces(env)

        # 测试2: Gazebo -> Env
        dir1_results = test_gazebo_to_env(env, wait_seconds=3.0)

        # 测试3: Env -> Gazebo
        dir2_results = test_env_to_gazebo(env, monitor, max_steps=30)

        # 测试4: 观测值合理性
        obs_results = test_observation_values(env)

        # 最终汇总
        print("\n" + "=" * 60)
        print("  测试汇总")
        print("=" * 60)

        dir1_ok = sum(1 for v in dir1_results.values() if v)
        dir1_total = len(dir1_results)
        dir2_ok = sum(1 for c in dir2_results['received_counts'].values() if c > 0)
        dir2_total = len(dir2_results['received_counts'])
        obs_ok = sum(1 for v in obs_results.values() if v)
        obs_total = len(obs_results)

        print(f"\n  方向1 (Gazebo->Env): {dir1_ok}/{dir1_total} 话题收到数据")
        print(f"  方向2 (Env->Gazebo): {dir2_ok}/{dir2_total} 命令话题有数据")
        print(f"  观测值合理性:       {obs_ok}/{obs_total} 检查通过")

        all_pass = (dir1_ok == dir1_total and dir2_ok >= 2 and obs_ok == obs_total)
        if all_pass:
            print("\n  [ALL PASS] 端到端通信测试全部通过!")
        else:
            print("\n  [PARTIAL] 部分测试未通过, 请检查上方详细输出")

        print("=" * 60)

        # 关闭
        env.close()
        monitor.destroy_node()

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
