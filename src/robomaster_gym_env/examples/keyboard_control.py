#!/usr/bin/env python3
"""
示例: 使用键盘控制机器人
"""

import gymnasium
import numpy as np
from robomaster_gym_env import RoboMasterGazeboEnv, GymEnvConfig
import threading
import time


class KeyboardController:
    """键盘控制器"""

    def __init__(self):
        self.linear_x = 0.0
        self.linear_y = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.shoot = 0
        self.running = True

    def get_action(self):
        """获取当前动作"""
        return {
            'chassis_velocity': np.array([self.linear_x, self.linear_y], dtype=np.float32),
            'gimbal_angle': np.array([self.yaw, self.pitch], dtype=np.float32),
            'shoot': self.shoot
        }

    def update(self):
        """更新键盘输入"""
        import termios
        import tty
        import sys

        # 保存终端设置
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setraw(sys.stdin.fileno())

            while self.running:
                ch = sys.stdin.read(1)

                # 底盘控制: WASD (angular_z 固定为0，不作为动作)
                if ch == 'w':
                    self.linear_x = min(self.linear_x + 0.2, 2.0)
                elif ch == 's':
                    self.linear_x = max(self.linear_x - 0.2, -2.0)
                elif ch == 'a':
                    self.linear_y = max(self.linear_y - 0.2, -2.0)
                elif ch == 'd':
                    self.linear_y = min(self.linear_y + 0.2, 2.0)

                # 云台控制: 方向键
                elif ch == 'j':
                    self.yaw = max(self.yaw - 0.1, -3.14)
                elif ch == 'l':
                    self.yaw = min(self.yaw + 0.1, 3.14)
                elif ch == 'i':
                    self.pitch = min(self.pitch + 0.05, 0.5)
                elif ch == 'k':
                    self.pitch = max(self.pitch - 0.05, -0.5)

                # 射击: 空格
                elif ch == ' ':
                    self.shoot = 1
                else:
                    self.shoot = 0

                # 退出: ESC
                if ch == '\x1b':
                    self.running = False
                    break

        finally:
            # 恢复终端设置
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def keyboard_control_test():
    """键盘控制测试"""
    # 创建配置
    config = GymEnvConfig()
    config.robot_name = "red_standard_robot1"
    config.robot_namespace = "/red_standard_robot1"
    config.team = "red"
    config.timeout_steps = 10000
    config.control_frequency = 50.0

    # 创建环境
    env = RoboMasterGazeboEnv(config)

    # 创建键盘控制器
    controller = KeyboardController()

    # 启动键盘输入线程
    keyboard_thread = threading.Thread(target=controller.update, daemon=True)
    keyboard_thread.start()

    print("\n" + "="*50)
    print("Keyboard Control Mode")
    print("="*50)
    print("Chassis Control:")
    print("  W/S: Forward/Backward")
    print("  A/D: Left/Right")
    print("\nGimbal Control:")
    print("  J/L: Yaw Left/Right")
    print("  I/K: Pitch Up/Down")
    print("\nShoot:")
    print("  Space: Fire")
    print("\nESC: Exit")
    print("="*50 + "\n")

    # 运行环境
    obs = env.reset()

    try:
        while controller.running:
            # 获取动作
            action = controller.get_action()

            # 执行动作
            obs, reward, done, info = env.step(action)

            # 打印状态
            if env.current_step % 10 == 0:
                print(f"\rStep {env.current_step}: "
                      f"HP={info['referee_system']['remain_hp']}, "
                      f"Ammo={info['referee_system']['ammo_remaining']}, "
                      f"Reward={info['referee_system']['episode_reward']:.2f}",
                      end='', flush=True)

            # 渲染
            env.render(mode='human')

            if done:
                print("\nEpisode finished!")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        controller.running = False
        env.close()


if __name__ == '__main__':
    keyboard_control_test()
