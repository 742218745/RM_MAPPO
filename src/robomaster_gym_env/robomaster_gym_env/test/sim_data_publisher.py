#!/usr/bin/env python3
"""
Gazebo 仿真数据发布器

以 ROS2 发布器的形式发送随机仿真数据, 让 Gym 环境接收,
用于测试 env 的订阅和数据解析是否正常工作。

发布的话题与 Gazebo 仿真节点发布的话题完全一致:
  - /{ns}/robot_base/odom          (Odometry)
  - /{ns}/chassis_odometry_gt      (Odometry)
  - /{ns}/robot_base/gimbal_state  (Gimbal)
  - /referee_system/{name}/robot_status (RobotStatus)
  - /referee_system/{name}/enable_power (Bool)
  - /referee_system/{name}/enable_control (Bool)
  - /referee_system/attack_info    (String)
  - /referee_system/pose_info      (TFMessage)
  - /referee_system/outpost_status (RobotStatus)
  - /referee_system/base_status    (RobotStatus)
  - /referee_system/game_status    (GameStatus)
  - /{ns}/livox/imu                (Imu)

使用方法:
  1. 不需要启动 Gazebo, 直接运行此节点
  2. 在另一个终端运行 test_node.py 验证 env 能否正确接收数据
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import time
import argparse

# ROS2 标准消息
from geometry_msgs.msg import Twist, TransformStamped, Pose, Accel
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from example_interfaces.msg import UInt8
from tf2_msgs.msg import TFMessage

# 自定义消息
try:
    from rmoss_interfaces.msg import (
        ChassisCmd, GimbalCmd, ShootCmd, Gimbal,
        RefereeCmd, RfidStatusArray,
        RobotStatus, GameStatus
    )
    RMOSS_AVAILABLE = True
except ImportError:
    RMOSS_AVAILABLE = False


class SimDataPublisher(Node):
    """仿真数据发布器 - 发送随机数据模拟 Gazebo 仿真输出"""

    def __init__(
        self,
        robot_name: str = "red_standard_robot1",
        robot_namespace: str = "red_standard_robot1",
        team: str = "red",
        publish_rate: float = 50.0
    ):
        super().__init__('sim_data_publisher')

        self.robot_name = robot_name
        self.robot_namespace = robot_namespace
        self.team = team
        self.publish_rate = publish_rate

        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self._pubs = {}
        self._init_publishers()

        self.step_count = 0
        self.get_logger().info(f'SimDataPublisher initialized')
        self.get_logger().info(f'  Robot: {robot_name}, NS: {robot_namespace}, Team: {team}')
        self.get_logger().info(f'  Rate: {publish_rate} Hz')

    def _init_publishers(self):
        """初始化所有发布器, 话题名与 Gazebo 仿真完全一致"""
        ns = self.robot_namespace
        name = self.robot_name

        # 里程计
        self._pubs['odom'] = self.create_publisher(
            Odometry, f'{ns}/robot_base/odom', self.qos_profile)
        self._pubs['chassis_odom_gt'] = self.create_publisher(
            Odometry, f'{ns}/chassis_odometry_gt', self.qos_profile)

        # 云台状态
        if RMOSS_AVAILABLE:
            self._pubs['gimbal_state'] = self.create_publisher(
                Gimbal, f'{ns}/robot_base/gimbal_state', self.qos_profile)

        # 裁判系统
        if RMOSS_AVAILABLE:
            self._pubs['robot_status'] = self.create_publisher(
                RobotStatus, f'/referee_system/{name}/robot_status', self.qos_profile)
            self._pubs['outpost_status'] = self.create_publisher(
                RobotStatus, '/referee_system/outpost_status', self.qos_profile)
            self._pubs['base_status'] = self.create_publisher(
                RobotStatus, '/referee_system/base_status', self.qos_profile)
            self._pubs['game_status'] = self.create_publisher(
                GameStatus, '/referee_system/game_status', self.qos_profile)

        self._pubs['enable_power'] = self.create_publisher(
            Bool, f'/referee_system/{name}/enable_power', self.qos_profile)
        self._pubs['enable_control'] = self.create_publisher(
            Bool, f'/referee_system/{name}/enable_control', self.qos_profile)
        self._pubs['attack_info'] = self.create_publisher(
            String, '/referee_system/attack_info', self.qos_profile)
        self._pubs['pose_info'] = self.create_publisher(
            TFMessage, '/referee_system/pose_info', self.qos_profile)

        # IMU
        self._pubs['imu'] = self.create_publisher(
            Imu, f'{ns}/livox/imu', self.qos_profile)

    def publish_all(self):
        """发布所有随机数据"""
        now = self.get_clock().now().to_msg()
        self.step_count += 1

        # 1. 里程计 (随机位置在场地范围内: ~28m x 15m)
        x = np.random.uniform(1, 27)
        y = np.random.uniform(1, 14)
        z = 0.0
        yaw = np.random.uniform(-np.pi, np.pi)
        qz = np.sin(yaw / 2)
        qw = np.cos(yaw / 2)

        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"
        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = z
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw
        odom_msg.twist.twist.linear.x = np.random.uniform(-2, 2)
        odom_msg.twist.twist.linear.y = np.random.uniform(-2, 2)
        odom_msg.twist.twist.angular.z = np.random.uniform(-1, 1)
        self._pubs['odom'].publish(odom_msg)
        self._pubs['chassis_odom_gt'].publish(odom_msg)

        # 2. 云台状态
        if RMOSS_AVAILABLE and 'gimbal_state' in self._pubs:
            gimbal_msg = Gimbal()
            gimbal_msg.yaw = np.random.uniform(-3.14, 3.14)
            gimbal_msg.pitch = np.random.uniform(-0.5, 0.5)
            self._pubs['gimbal_state'].publish(gimbal_msg)

        # 3. 机器人状态
        if RMOSS_AVAILABLE and 'robot_status' in self._pubs:
            robot_id = 1 if self.team == 'red' else 4
            status_msg = RobotStatus()
            status_msg.id = robot_id
            status_msg.level = 3  # 步兵
            status_msg.name = self.robot_name
            status_msg.remain_hp = np.random.randint(100, 400)
            status_msg.max_hp = 400
            status_msg.total_projectiles = 300
            status_msg.used_projectiles = np.random.randint(0, 200)
            status_msg.hit_projectiles = np.random.randint(0, 50)
            status_msg.gt_tf.translation.x = x
            status_msg.gt_tf.translation.y = y
            status_msg.gt_tf.translation.z = z
            status_msg.gt_tf.rotation.x = 0.0
            status_msg.gt_tf.rotation.y = 0.0
            status_msg.gt_tf.rotation.z = qz
            status_msg.gt_tf.rotation.w = qw
            self._pubs['robot_status'].publish(status_msg)

        # 4. 前哨站状态
        if RMOSS_AVAILABLE and 'outpost_status' in self._pubs:
            outpost_msg = RobotStatus()
            outpost_msg.id = 0
            outpost_msg.level = 0
            outpost_msg.name = f"{self.team}_outpost"
            outpost_msg.remain_hp = np.random.randint(0, 1500)
            outpost_msg.max_hp = 1500
            self._pubs['outpost_status'].publish(outpost_msg)

        # 5. 基地状态
        if RMOSS_AVAILABLE and 'base_status' in self._pubs:
            base_msg = RobotStatus()
            base_msg.id = 0
            base_msg.level = 0
            base_msg.name = f"{self.team}_base"
            base_msg.remain_hp = np.random.randint(0, 5000)
            base_msg.max_hp = 5000
            self._pubs['base_status'].publish(base_msg)

        # 6. 游戏状态
        if RMOSS_AVAILABLE and 'game_status' in self._pubs:
            game_msg = GameStatus()
            game_msg.status = GameStatus.RUNNING
            game_msg.total_time = 210
            game_msg.current_time = np.random.randint(0, 210)
            self._pubs['game_status'].publish(game_msg)

        # 7. 使能信号
        power_msg = Bool()
        power_msg.data = True
        self._pubs['enable_power'].publish(power_msg)
        self._pubs['enable_control'].publish(power_msg)

        # 8. 攻击信息 (偶尔产生)
        attack_msg = String()
        if np.random.random() < 0.1:
            target_team = "blue" if self.team == "red" else "red"
            target_id = np.random.randint(1, 4)
            attack_msg.data = f"{self.robot_name}:shooter:{target_team}_standard_robot{target_id}:armor_front:target_collision"
        self._pubs['attack_info'].publish(attack_msg)

        # 9. 位姿信息 (所有机器人的 TF)
        pose_msg = TFMessage()
        for i in range(1, 4):
            for t in ['red', 'blue']:
                tf = TransformStamped()
                tf.header.stamp = now
                tf.header.frame_id = "world"
                tf.child_frame_id = f"{t}_standard_robot{i}"
                tf.transform.translation.x = np.random.uniform(1, 27)
                tf.transform.translation.y = np.random.uniform(1, 14)
                tf.transform.translation.z = 0.0
                tf.transform.rotation.x = 0.0
                tf.transform.rotation.y = 0.0
                tf.transform.rotation.z = np.sin(np.random.uniform(-np.pi, np.pi) / 2)
                tf.transform.rotation.w = np.cos(np.random.uniform(-np.pi, np.pi) / 2)
                pose_msg.transforms.append(tf)
        self._pubs['pose_info'].publish(pose_msg)

        # 10. IMU
        imu_msg = Imu()
        imu_msg.header.stamp = now
        imu_msg.header.frame_id = "imu_link"
        imu_msg.orientation.x = 0.0
        imu_msg.orientation.y = 0.0
        imu_msg.orientation.z = qz
        imu_msg.orientation.w = qw
        imu_msg.angular_velocity.x = np.random.uniform(-0.5, 0.5)
        imu_msg.angular_velocity.y = np.random.uniform(-0.5, 0.5)
        imu_msg.angular_velocity.z = np.random.uniform(-1, 1)
        imu_msg.linear_acceleration.x = np.random.uniform(-2, 2)
        imu_msg.linear_acceleration.y = np.random.uniform(-2, 2)
        imu_msg.linear_acceleration.z = 9.81
        self._pubs['imu'].publish(imu_msg)


def main():
    parser = argparse.ArgumentParser(description='Gazebo 仿真数据发布器 (随机数据)')
    parser.add_argument('--robot-name', type=str, default='red_standard_robot1')
    parser.add_argument('--namespace', type=str, default='red_standard_robot1')
    parser.add_argument('--team', type=str, default='red', choices=['red', 'blue'])
    parser.add_argument('--rate', type=float, default=50.0, help='发布频率 Hz')
    parser.add_argument('--duration', type=float, default=30.0, help='持续时间 秒 (0=无限)')
    args = parser.parse_args()

    rclpy.init()
    publisher = SimDataPublisher(
        robot_name=args.robot_name,
        robot_namespace=args.namespace,
        team=args.team,
        publish_rate=args.rate
    )

    print("\n" + "=" * 60)
    print("  Gazebo 仿真数据发布器 (随机数据)")
    print("=" * 60)
    print(f"  Robot: {args.robot_name}")
    print(f"  Namespace: {args.namespace}")
    print(f"  Team: {args.team}")
    print(f"  Rate: {args.rate} Hz")
    print(f"  Duration: {'infinite' if args.duration == 0 else f'{args.duration}s'}")
    print("=" * 60 + "\n")

    period = 1.0 / args.rate
    start_time = time.time()

    try:
        while True:
            if args.duration > 0 and (time.time() - start_time) > args.duration:
                break

            publisher.publish_all()
            rclpy.spin_once(publisher, timeout_sec=0.001)

            if publisher.step_count % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  [SimPublisher] step={publisher.step_count}, elapsed={elapsed:.1f}s")

            time.sleep(period)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        print(f"\n共发布 {publisher.step_count} 步数据")
        publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
