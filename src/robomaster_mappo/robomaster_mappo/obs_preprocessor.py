"""
观察空间预处理模块

把 Gym 环境返回的 Dict 观测转成网络能吃的张量:
  - all_robots (10, 4) -> 归一化后的 (10, 4)
  - 11个标量特征 -> 归一化后的 (11,) 向量

归一化方式: 除以各自的最大值, 映射到 [0, 1] 附近

注意: all_robots 中 team 字段编码已改为:
  0=己方(ally), 1=敌方(enemy), -1=unknown
  (原来是 0=red, 1=blue)
"""

import torch
import numpy as np
from typing import Dict


# ========== 各观测项的最大值 (用于归一化) ==========
# 和 observation_space.py 中 ObservationConfig 的默认值保持一致
OBS_MAX = {
    'own_hp': 400,
    'own_ammo': 300,
    'team_economy': 400,
    'remaining_steps': 2100,
    'judge_countdown_steps': 2100,
    'damage_per_step': 10.0,
    'outpost_hp': 1500,
    'base_hp': 5000,
    'base_exposed': 1,          # 只有 0/1
    'ammo_consumed_per_step': 300,
    'revive_waiting_steps': 2100,
}

# 场地尺寸 (RoboMaster 场地约 28m x 15m), 用于归一化坐标
FIELD_HALF_X = 14.0   # x 方向半场
FIELD_HALF_Y = 7.5    # y 方向半场


def preprocess_obs(obs: Dict[str, np.ndarray], device: str = 'cpu'):
    """
    将 Dict 观测预处理为两个张量

    Args:
        obs: Gym 环境返回的观测字典, 包含:
            - all_robots: (10, 4) [id, team, x, y]
                team: 0=己方(ally), 1=敌方(enemy), -1=unknown
            - own_hp, own_ammo, team_economy, remaining_steps,
              judge_countdown_steps, damage_per_step, outpost_hp,
              base_hp, base_exposed, ammo_consumed_per_step,
              revive_waiting_steps
        device: 'cpu' 或 'cuda'

    Returns:
        robot_tensor: (10, 4) 归一化后的机器人信息
        state_tensor: (11,)  归一化后的标量状态
    """
    # ----- 1. 处理 all_robots -----
    all_robots = obs['all_robots'].copy()  # (10, 4)

    # 将NaN替换为0 (pose_info不可见时all_robots为NaN)
    all_robots = np.nan_to_num(all_robots, nan=0.0)

    # 归一化坐标: x / 14, y / 7.5, 映射到 [-1, 1] 附近
    # id 和 team 保持原值 (离散标识, 不归一化)
    all_robots[:, 2] = all_robots[:, 2] / FIELD_HALF_X   # x
    all_robots[:, 3] = all_robots[:, 3] / FIELD_HALF_Y   # y

    robot_tensor = torch.FloatTensor(all_robots).to(device)  # (10, 4)

    # ----- 2. 处理标量特征 -----
    # 按固定顺序拼接, 和 OBS_MAX 的 key 顺序一致
    scalar_values = [
        float(obs['own_hp']) / OBS_MAX['own_hp'],
        float(obs['own_ammo']) / OBS_MAX['own_ammo'],
        float(obs['team_economy']) / OBS_MAX['team_economy'],
        float(obs['remaining_steps']) / OBS_MAX['remaining_steps'],
        float(obs['judge_countdown_steps']) / OBS_MAX['judge_countdown_steps'],
        float(obs['damage_per_step'].squeeze()) / OBS_MAX['damage_per_step'],
        float(obs['outpost_hp']) / OBS_MAX['outpost_hp'],
        float(obs['base_hp']) / OBS_MAX['base_hp'],
        float(obs['base_exposed']) / OBS_MAX['base_exposed'],
        float(obs['ammo_consumed_per_step']) / OBS_MAX['ammo_consumed_per_step'],
        float(obs['revive_waiting_steps']) / OBS_MAX['revive_waiting_steps'],
    ]

    state_tensor = torch.FloatTensor(scalar_values).to(device)  # (11,)

    return robot_tensor, state_tensor


def preprocess_obs_batch(obs_list, device: str = 'cpu'):
    """
    批量预处理观测 (用于训练时一次处理多条经验)

    Args:
        obs_list: 观测字典列表, 长度为 batch_size
        device: 'cpu' 或 'cuda'

    Returns:
        robot_batch: (batch, 10, 4)
        state_batch: (batch, 11)
    """
    robots, states = [], []
    for obs in obs_list:
        r, s = preprocess_obs(obs, device='cpu')
        robots.append(r)
        states.append(s)

    robot_batch = torch.stack(robots).to(device)   # (batch, 10, 4)
    state_batch = torch.stack(states).to(device)   # (batch, 11)

    return robot_batch, state_batch
