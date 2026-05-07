#!/usr/bin/env python3
"""
示例: 使用随机策略测试环境
"""

import gymnasium
import numpy as np
from robomaster_gym_env import RoboMasterGazeboEnv, GymEnvConfig


def random_agent_test():
    """使用随机策略测试环境"""
    # 创建配置
    config = GymEnvConfig()
    config.robot_name = "red_standard_robot1"
    config.robot_namespace = "/red_standard_robot1"
    config.team = "red"
    config.timeout_steps = 500
    config.control_frequency = 20.0

    # 创建环境
    env = RoboMasterGazeboEnv(config)

    print("Starting random agent test...")

    # 运行一个episode
    obs = env.reset()
    total_reward = 0

    for step in range(config.timeout_steps):
        # 随机动作
        action = env.action_space.sample()

        # 执行动作
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # 每10步打印一次
        if step % 10 == 0:
            print(f"Step {step}: reward={reward:.3f}, total_reward={total_reward:.3f}")
            print(f"  HP: {info['referee_system']['remain_hp']}/{info['referee_system']['max_hp']}")
            print(f"  Ammo: {info['referee_system']['ammo_remaining']}")

        # 渲染
        env.render(mode='human')

        if done:
            print(f"Episode finished at step {step}")
            break

    print(f"Total reward: {total_reward:.3f}")
    env.close()


if __name__ == '__main__':
    random_agent_test()
