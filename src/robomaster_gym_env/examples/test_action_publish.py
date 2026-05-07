#!/usr/bin/env python3
"""
测试动作发布脚本
直接给动作空间赋值, 通过ROS2话题发布到Gazebo仿真

使用方法:
1. 先启动仿真: ros2 launch rmu_gazebo_simulator bringup_sim.launch.py
2. 运行此脚本: python3 test_action_publish.py
3. 在另一个终端监控话题: ros2 topic echo /red_standard_robot1/cmd_vel
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import time
import argparse


class ActionPublisher(Node):
    """动作发布节点"""

    def __init__(self, robot_namespace: str = "/red_standard_robot1", robot_name: str = "red_standard_robot1"):
        super().__init__('action_publisher')

        self.robot_namespace = robot_namespace
        self.robot_name = robot_name

        # QoS配置 (与Gym环境保持一致)
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 创建底盘速度发布器
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f'{robot_namespace}/cmd_vel',
            self.qos_profile
        )

        # 创建电源使能发布器
        self.enable_power_pub = self.create_publisher(
            Bool,
            f'/referee_system/{robot_name}/enable_power',
            self.qos_profile
        )

        # 创建控制使能发布器
        self.enable_control_pub = self.create_publisher(
            Bool,
            f'/referee_system/{robot_name}/enable_control',
            self.qos_profile
        )

        self.get_logger().info(f'Action Publisher initialized')
        self.get_logger().info(f'  Namespace: {robot_namespace}')
        self.get_logger().info(f'  Topic: {robot_namespace}/cmd_vel')
        self.get_logger().info(f'  Enable Power: /referee_system/{robot_name}/enable_power')
        self.get_logger().info(f'  Enable Control: /referee_system/{robot_name}/enable_control')

    def enable_power_and_control(self):
        """使能电源和控制"""
        # 使能电源
        msg_power = Bool()
        msg_power.data = True
        self.enable_power_pub.publish(msg_power)
        self.get_logger().info('✅ 电源已使能')

        # 使能控制
        msg_control = Bool()
        msg_control.data = True
        self.enable_control_pub.publish(msg_control)
        self.get_logger().info('✅ 控制已使能')

        # 等待使能生效
        time.sleep(0.5)

    def send_chassis_velocity(self, linear_x: float, linear_y: float, angular_z: float = 0.0):
        """
        发送底盘速度命令

        Args:
            linear_x: 前进速度 (m/s)
            linear_y: 横向速度 (m/s)
            angular_z: 旋转角速度 (rad/s), Gym动作空间中固定为0
        """
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.linear.y = float(linear_y)
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(angular_z)

        self.cmd_vel_pub.publish(msg)
        self.get_logger().info(
            f'Published: linear_x={linear_x:.2f}, linear_y={linear_y:.2f}, angular_z={angular_z:.2f}'
        )


def main():
    parser = argparse.ArgumentParser(description='测试动作发布到Gazebo仿真')
    parser.add_argument('--namespace', type=str, default='/red_standard_robot1',
                        help='机器人命名空间 (默认: /red_standard_robot1)')
    parser.add_argument('--robot-name', type=str, default='red_standard_robot1',
                        help='机器人名称 (默认: red_standard_robot1)')
    parser.add_argument('--linear-x', type=float, default=5.0,
                        help='前进速度 m/s (默认: 5.0)')
    parser.add_argument('--linear-y', type=float, default=5.0,
                        help='横向速度 m/s (默认: 5.0)')
    parser.add_argument('--angular-z', type=float, default=0.0,
                        help='旋转角速度 rad/s (默认: 0.0)')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='持续时间 秒 (默认: 5.0)')
    parser.add_argument('--rate', type=float, default=50.0,
                        help='发布频率 Hz (默认: 50.0)')
    parser.add_argument('--no-enable', action='store_true',
                        help='不自动使能电源和控制')

    args = parser.parse_args()

    # 初始化ROS2
    rclpy.init()

    # 创建发布节点
    publisher = ActionPublisher(robot_namespace=args.namespace, robot_name=args.robot_name)

    print("\n" + "="*60)
    print("动作发布测试")
    print("="*60)
    print(f"机器人名称: {args.robot_name}")
    print(f"机器人命名空间: {args.namespace}")
    print(f"底盘速度命令:")
    print(f"  linear_x (前进): {args.linear_x} m/s")
    print(f"  linear_y (横向): {args.linear_y} m/s")
    print(f"  angular_z (旋转): {args.angular_z} rad/s")
    print(f"持续时间: {args.duration} 秒")
    print(f"发布频率: {args.rate} Hz")
    print("="*60)

    # 使能电源和控制
    if not args.no_enable:
        print("\n【步骤1】使能电源和控制...")
        publisher.enable_power_and_control()
        print("✅ 使能完成\n")
    else:
        print("\n⚠️  跳过自动使能 (使用 --no-enable 参数)\n")

    print("【步骤2】开始发布速度命令...\n")

    # 计算发布次数
    period = 1.0 / args.rate
    num_publishes = int(args.duration * args.rate)

    try:
        for i in range(num_publishes):
            # 发送速度命令
            publisher.send_chassis_velocity(
                linear_x=args.linear_x,
                linear_y=args.linear_y,
                angular_z=args.angular_z
            )

            # 处理ROS2回调
            rclpy.spin_once(publisher, timeout_sec=0.001)

            # 等待下一个周期
            time.sleep(period)

        print(f"\n完成! 共发布 {num_publishes} 次命令")

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        # 清理
        publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
