"""
MAPPO Actor 网络

动作空间:
  - chassis_velocity: (2,) 连续, 底盘速度 [linear_x, linear_y] (angular_z 固定为0)
  - shoot: 9类离散, 0=不射击, 1-6=射击机器人, 7=射击前哨站, 8=射击基地

云台角度不作为动作输出, 而是根据 shoot 目标自动计算:
  - shoot=0 (不射击) → 云台保持当前朝向
  - shoot=1~6 (射击机器人) → 云台对准该机器人
  - shoot=7 (射击前哨站)   → 云台对准前哨站
  - shoot=8 (射击基地)     → 云台对准基地

结构:

    观测 Dict
      │
      ├── all_robots (10, 4) ──→ RobotEncoder ──→ robot_feat (64)
      ├── 11个标量 ────────────→ StateEncoder ──→ state_feat (64)
      │
      └── concat ──→ Fusion MLP ──→ hidden (128)
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
          ChassisHead              ShootHead
          → tanh-Gaussian          → Categorical
          → (2,) 连续              → 9类 离散
                                        │
                                        ▼
                                 gimbal_auto_aim()
                                 → (2,) 云台角度 [yaw, pitch]
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

from .obs_preprocessor import preprocess_obs, preprocess_obs_batch


# ============================================================
#  编码器
# ============================================================

class RobotEncoder(nn.Module):
    """
    机器人位置编码器

    输入: all_robots (batch, 10, 4), 每行 [id, team, x, y]
          id=-1 表示该位置无效 (padding)

    做法: MLP 编码每台机器人 → masked 平均池化 → 输出
    """
    def __init__(self, input_dim=4, embed_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, robots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            robots: (batch, 10, 4)
        Returns:
            (batch, embed_dim)
        """
        features = self.mlp(robots)  # (batch, 10, embed_dim)
        mask = (robots[:, :, 0] != -1).unsqueeze(-1).float()  # (batch, 10, 1)
        summed = (features * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        return summed / count


class StateEncoder(nn.Module):
    """
    标量状态编码器

    输入: 11维归一化标量特征
    输出: embed_dim 维向量
    """
    def __init__(self, input_dim=13, embed_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.mlp(state)


# ============================================================
#  动作头
# ============================================================

class ContinuousHead(nn.Module):
    """
    连续动作头: tanh-Gaussian 分布

    保证输出严格在 [low, high] 范围内, 并做 Jacobian 校正
    """
    def __init__(self, hidden_dim: int, action_dim: int,
                 low: list, high: list):
        super().__init__()
        self.action_dim = action_dim

        self.mean_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.register_buffer('low', torch.FloatTensor(low))
        self.register_buffer('high', torch.FloatTensor(high))

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_raw = self.mean_net(hidden)
        mean = self._rescale(mean_raw)
        std = self.log_std.exp().unsqueeze(0).expand_as(mean)
        return mean, std

    def sample(self, mean: torch.Tensor, std: torch.Tensor,
               deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if deterministic:
            return mean, torch.zeros(mean.shape[0], device=mean.device)

        noise = torch.randn_like(mean)
        mean_raw = self._rescale_inv(mean)
        z_raw = mean_raw + std * noise
        action = self._rescale(z_raw)
        log_prob = self._compute_log_prob(z_raw, mean_raw, std)
        return action, log_prob

    def evaluate(self, action: torch.Tensor,
                 mean: torch.Tensor, std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_raw = self._rescale_inv(action)
        mean_raw = self._rescale_inv(mean)
        log_prob = self._compute_log_prob(z_raw, mean_raw, std)
        entropy = (0.5 + 0.5 * np.log(2 * np.pi) + self.log_std).sum()
        return log_prob, entropy.expand_as(log_prob)

    def _rescale(self, raw: torch.Tensor) -> torch.Tensor:
        tanh_val = torch.tanh(raw)
        return self.low + (self.high - self.low) * (tanh_val + 1.0) / 2.0

    def _rescale_inv(self, scaled: torch.Tensor) -> torch.Tensor:
        normalized = (2.0 * (scaled - self.low) / (self.high - self.low) - 1.0)
        normalized = normalized.clamp(-0.999, 0.999)
        return 0.5 * torch.log((1 + normalized) / (1 - normalized))

    def _compute_log_prob(self, z_raw, mean_raw, std):
        log_prob_gauss = -0.5 * (((z_raw - mean_raw) / std).pow(2)
                                  + np.log(2 * np.pi) + 2 * self.log_std)
        log_prob_gauss = log_prob_gauss.sum(dim=-1)
        tanh_val = torch.tanh(z_raw)
        log_det = torch.log(1 - tanh_val.pow(2) + 1e-6).sum(dim=-1)
        log_scale = torch.log((self.high - self.low) / 2.0 + 1e-6).sum()
        return log_prob_gauss - log_det - log_scale


class DiscreteHead(nn.Module):
    """
    离散动作头: Categorical 分布
    """
    def __init__(self, hidden_dim: int, n_categories: int):
        super().__init__()
        self.logit_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_categories),
        )
        self.n_categories = n_categories

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.logit_net(hidden)

    def sample(self, logits: torch.Tensor,
               deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate(self, action: torch.Tensor,
                 logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy()


# ============================================================
#  Actor 主网络
# ============================================================

class MAPPOActor(nn.Module):
    """
    MAPPO Actor 网络 

    动作空间:
        chassis_velocity: (2,) 连续, 范围 [-2,2], [-2,2] (angular_z 固定为0)
        shoot: 9类离散, 0=不射击, 1-6=射击机器人, 7=前哨站, 8=基地
    """
    def __init__(
        self,
        robot_embed_dim: int = 64,
        state_embed_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # ===== 编码器 =====
        self.robot_encoder = RobotEncoder(input_dim=4, embed_dim=robot_embed_dim)
        self.state_encoder = StateEncoder(input_dim=13, embed_dim=state_embed_dim)

        # ===== 融合层 =====
        fusion_input = robot_embed_dim + state_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ===== 动作头 (只有两个) =====
        # 底盘速度: [linear_x, linear_y] (angular_z 固定为0)
        self.chassis_head = ContinuousHead(
            hidden_dim=hidden_dim, action_dim=2,
            low=[-2.0, -2.0],
            high=[2.0, 2.0],
        )

        # 射击: 0=不射击, 1-6=射击机器人, 7=前哨站, 8=基地
        self.shoot_head = DiscreteHead(
            hidden_dim=hidden_dim, n_categories=9,
        )

    def forward(self, robot_tensor: torch.Tensor, state_tensor: torch.Tensor):
        """
        前向传播

        Args:
            robot_tensor: (batch, 10, 4)
            state_tensor: (batch, 11)

        Returns:
            chassis_mean, chassis_std: 底盘速度分布参数
            shoot_logits:              射击 logits
            hidden:                    融合特征
        """
        robot_feat = self.robot_encoder(robot_tensor)
        state_feat = self.state_encoder(state_tensor)
        hidden = self.fusion(torch.cat([robot_feat, state_feat], dim=-1))

        chassis_mean, chassis_std = self.chassis_head(hidden)
        shoot_logits = self.shoot_head(hidden)

        return chassis_mean, chassis_std, shoot_logits, hidden

    def get_action(self, obs: dict, deterministic: bool = False, device: str = 'cpu'):
        """
        从环境观测采样动作

        只输出 chassis_velocity 和 shoot, 云台角度由 Gym 环境内部
        根据 shoot 目标自动计算并发给 ROS2, 不需要 Actor 输出

        Args:
            obs: Gym 环境返回的观测字典
            deterministic: True 则取均值/argmax
            device: 'cpu' 或 'cuda'

        Returns:
            action:   动作字典, 包含 chassis_velocity 和 shoot
            log_prob: 标量
        """
        # 预处理
        robot_tensor, state_tensor = preprocess_obs(obs, device=device)
        robot_tensor = robot_tensor.unsqueeze(0)
        state_tensor = state_tensor.unsqueeze(0)

        # 前向
        chassis_mean, chassis_std, shoot_logits, _ = \
            self.forward(robot_tensor, state_tensor)

        # 采样
        chassis_action, chassis_logp = self.chassis_head.sample(
            chassis_mean, chassis_std, deterministic)
        shoot_action, shoot_logp = self.shoot_head.sample(
            shoot_logits, deterministic)

        # 组装动作字典 (不含 gimbal_angle, 环境内部自动计算)
        action = {
            'chassis_velocity': chassis_action.squeeze(0).detach().cpu().numpy(),
            'shoot': shoot_action.squeeze(0).detach().cpu().item(),
        }

        log_prob = (chassis_logp + shoot_logp).squeeze(0).detach().cpu().item()

        return action, log_prob

    def evaluate_actions(self, robot_batch, state_batch,
                         chassis_actions, shoot_actions):
        """
        评估已有动作的 log_prob 和 entropy (PPO 更新时用)

        Args:
            robot_batch:     (batch, 10, 4)
            state_batch:     (batch, 11)
            chassis_actions: (batch, 2)
            shoot_actions:   (batch,) 整数

        Returns:
            log_prob: (batch,)
            entropy:  (batch,)
        """
        chassis_mean, chassis_std, shoot_logits, _ = \
            self.forward(robot_batch, state_batch)

        chassis_logp, chassis_ent = self.chassis_head.evaluate(
            chassis_actions, chassis_mean, chassis_std)
        shoot_logp, shoot_ent = self.shoot_head.evaluate(
            shoot_actions, shoot_logits)

        return chassis_logp + shoot_logp, chassis_ent + shoot_ent
