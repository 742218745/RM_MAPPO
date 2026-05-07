"""
使用观察空间的示例代码

演示如何使用RoboMasterGazeboEnv环境。
"""

import numpy as np
from robomaster_gym_env import (
    RoboMasterGazeboEnv,
    ObservationConfig,
    MapConfigLoader,
    DEFAULT_GYM_CONFIG
)


def example_basic_usage():
    """基本使用示例"""
    print("=" * 50)
    print("基本使用示例")
    print("=" * 50)

    # 创建环境（使用默认配置）
    env = RoboMasterGazeboEnv(
        config=DEFAULT_GYM_CONFIG
    )

    # 重置环境
    obs, info = env.reset()
    print(f"\n初始观察:")
    print(f"  all_robots shape: {obs['all_robots'].shape}")
    print(f"  movable_range shape: {obs['movable_range'].shape}")
    print(f"  own_hp: {obs['own_hp']}")
    print(f"  own_ammo: {obs['own_ammo']}")
    print(f"  team_economy: {obs['team_economy']}")
    print(f"  remaining_steps: {obs['remaining_steps']}")
    print(f"  judge_countdown: {obs['judge_countdown']}")
    print(f"  damage_per_step: {obs['damage_per_step']}")

    # 执行随机动作
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {i+1}:")
        print(f"  reward: {reward}")
        print(f"  own_hp: {obs['own_hp']}")
        print(f"  remaining_steps: {obs['remaining_steps']}")

        if terminated or truncated:
            print(f"\nEpisode finished!")
            break

    # 关闭环境
    env.close()


def example_with_map_config():
    """使用场地配置的示例"""
    print("\n" + "=" * 50)
    print("使用场地配置示例")
    print("=" * 50)

    # 加载场地配置
    map_config_file = "config/map_config.yaml"
    try:
        map_config = MapConfigLoader.load_from_yaml(map_config_file)
        print(f"\n场地配置加载成功:")
        print(f"  地图名称: {map_config.map_name}")
        print(f"  地图版本: {map_config.map_version}")
        print(f"  边界: {map_config.get_boundary_tuple()}")
        print(f"  障碍物数量: {len(map_config.obstacles)}")
    except Exception as e:
        print(f"\n场地配置加载失败: {e}")
        print("使用默认配置")
        map_config = MapConfigLoader.create_default_config()

    # 创建环境
    env = RoboMasterGazeboEnv(
        config=DEFAULT_GYM_CONFIG,
        map_config_file=map_config_file
    )

    # 重置环境
    obs, info = env.reset()

    print(f"\n初始观察:")
    print(f"  movable_range shape: {obs['movable_range'].shape}")
    if obs['movable_range'].size > 0:
        print(f"  可移动范围点数: {len(obs['movable_range'])}")

    env.close()


def example_with_custom_obs_config():
    """使用自定义观察配置的示例"""
    print("\n" + "=" * 50)
    print("自定义观察配置示例")
    print("=" * 50)

    # 创建自定义观察配置
    obs_config = ObservationConfig(
        max_robots=10,
        max_range_points=500,
        max_hp=2000,
        max_ammo=500,
        max_economy=10000,
        max_steps=5000,
        max_countdown=10000,
        enable_movable_range=True,
        enable_damage_per_step=True
    )

    print(f"\n观察配置:")
    print(f"  max_robots: {obs_config.max_robots}")
    print(f"  max_range_points: {obs_config.max_range_points}")
    print(f"  enable_movable_range: {obs_config.enable_movable_range}")
    print(f"  enable_damage_per_step: {obs_config.enable_damage_per_step}")

    # 创建环境
    env = RoboMasterGazeboEnv(
        config=DEFAULT_GYM_CONFIG,
        obs_config=obs_config
    )

    # 重置环境
    obs, info = env.reset()

    print(f"\n观察空间:")
    print(f"  {env.observation_space}")

    env.close()


def example_set_damage_per_step():
    """设置每步伤害的示例"""
    print("\n" + "=" * 50)
    print("设置每步伤害示例")
    print("=" * 50)

    # 创建环境
    env = RoboMasterGazeboEnv(
        config=DEFAULT_GYM_CONFIG
    )

    # 设置每步伤害（外部输入）
    damage_value = 10.5
    env.set_damage_per_step(damage_value)
    print(f"\n设置每步伤害: {damage_value}")

    # 重置环境
    obs, info = env.reset()

    print(f"\n初始观察:")
    print(f"  damage_per_step: {obs['damage_per_step']}")

    # 执行一步
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\n执行一步后:")
    print(f"  damage_per_step: {obs['damage_per_step']}")

    env.close()


if __name__ == "__main__":
    # 运行所有示例
    try:
        example_basic_usage()
    except Exception as e:
        print(f"基本使用示例失败: {e}")

    try:
        example_with_map_config()
    except Exception as e:
        print(f"场地配置示例失败: {e}")

    try:
        example_with_custom_obs_config()
    except Exception as e:
        print(f"自定义配置示例失败: {e}")

    try:
        example_set_damage_per_step()
    except Exception as e:
        print(f"设置伤害示例失败: {e}")

    print("\n" + "=" * 50)
    print("所有示例运行完成!")
    print("=" * 50)
