#!/usr/bin/env python3
"""
实时监控机器人位置脚本

用法:
  source install/setup.bash
  python3 monitor_pos.py

  # 指定刷新频率
  python3 monitor_pos.py --freq 10

  # 只看红方
  python3 monitor_pos.py --robot red_standard_robot1
"""

import argparse
import time
import sys
import math

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


class PositionMonitor(Node):
    def __init__(self):
        super().__init__('position_monitor')
        self.robot_poses = {}  # name -> (x, y, z, yaw)
        self.sub = self.create_subscription(
            TFMessage,
            '/referee_system/pose_info',
            self.pose_callback,
            10,
        )

    def pose_callback(self, msg: TFMessage):
        for t in msg.transforms:
            name = t.child_frame_id
            x = t.transform.translation.x
            y = t.transform.translation.y
            z = t.transform.translation.z
            # 从四元数提取 yaw
            q = t.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            self.robot_poses[name] = (x, y, z, yaw)


def main():
    parser = argparse.ArgumentParser(description='实时监控机器人位置')
    parser.add_argument('--freq', type=float, default=5.0,
                        help='刷新频率 Hz (默认: 5)')
    parser.add_argument('--robot', type=str, default=None,
                        help='只看指定机器人 (如 red_standard_robot1)')
    args = parser.parse_args()

    rclpy.init()
    node = PositionMonitor()

    print(f"位置监控已启动 (刷新 {args.freq}Hz, Ctrl+C 退出)")
    print()

    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.1)

            # 清屏
            print("\033[2J\033[H", end='')

            poses = node.robot_poses
            if not poses:
                print("等待数据... (确保仿真已启动)")
            else:
                # 按名称排序
                names = sorted(poses.keys())
                if args.robot:
                    names = [n for n in names if args.robot in n]

                # 表头
                print(f"{'机器人':<30} {'X':>7} {'Y':>7} {'Z':>6} {'Yaw°':>7}")
                print("-" * 62)

                for name in names:
                    x, y, z, yaw = poses[name]
                    yaw_deg = math.degrees(yaw)
                    print(f"{name:<30} {x:7.2f} {y:7.2f} {z:6.2f} {yaw_deg:7.1f}")

                # 如果有红蓝双方，计算距离
                red_names = [n for n in poses if 'red' in n.lower()]
                blue_names = [n for n in poses if 'blue' in n.lower()]
                if red_names and blue_names:
                    print()
                    print("--- 距离 ---")
                    for rn in red_names:
                        rx, ry, _, _ = poses[rn]
                        for bn in blue_names:
                            bx, by, _, _ = poses[bn]
                            dist = math.sqrt((rx - bx)**2 + (ry - by)**2)
                            print(f"  {rn} <-> {bn}: {dist:.2f}m")

            print()
            print(f"[{time.strftime('%H:%M:%S')}] 刷新 {args.freq}Hz | Ctrl+C 退出")

            time.sleep(1.0 / args.freq)

    except KeyboardInterrupt:
        print("\n退出")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
