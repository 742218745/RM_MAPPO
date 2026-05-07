"""
MAPPO Critic 网络

Critic 接收全局状态 s, 输出价值估计 V(s).

MAPPO 中:
  - Actor 输入: 局部观测 o_i → 动作分布 π(a|o_i)
  - Critic 输入: 全局状态 s → V(s)

当前环境中每个智能体已经能看到所有机器人位置,
所以 Actor 和 Critic 的输入相同 (都是全局观测).

结构:

    全局状态 s (同 Actor 输入)
      │
      ├── all_robots (10, 4) ──→ RobotEncoder ──→ robot_feat (64)
      ├── 11个标量 ────────────→ StateEncoder ──→ state_feat (64)
      │
      └── concat ──→ Fusion MLP ──→ hidden (128)
                            │
                            ▼
                      ValueHead
                      → 标量 V(s)
"""

import torch
import torch.nn as nn
from typing import Dict

from .actor import RobotEncoder, StateEncoder
from .obs_preprocessor import preprocess_obs, preprocess_obs_batch


class MAPPOCritic(nn.Module):
    """
    MAPPO Critic 网络

    输入全局状态, 输出价值估计 V(s).
    复用 Actor 的 RobotEncoder 和 StateEncoder 结构.
    """
    def __init__(
        self,
        robot_embed_dim: int = 64,
        state_embed_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # ===== 编码器 (与 Actor 共享结构, 但不共享权重) =====
        self.robot_encoder = RobotEncoder(input_dim=4, embed_dim=robot_embed_dim)
        self.state_encoder = StateEncoder(input_dim=11, embed_dim=state_embed_dim)

        # ===== 融合层 =====
        fusion_input = robot_embed_dim + state_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ===== 价值头 =====
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, robot_tensor: torch.Tensor, state_tensor: torch.Tensor):
        """
        前向传播

        Args:
            robot_tensor: (batch, 10, 4)
            state_tensor: (batch, 11)

        Returns:
            value: (batch,) 价值估计
        """
        robot_feat = self.robot_encoder(robot_tensor)
        state_feat = self.state_encoder(state_tensor)
        hidden = self.fusion(torch.cat([robot_feat, state_feat], dim=-1))
        value = self.value_head(hidden).squeeze(-1)  # (batch,)
        return value

    def get_value(self, obs: dict, device: str = 'cpu') -> float:
        """
        从环境观测计算价值估计 (单条经验, 推理用)

        Args:
            obs: Gym 环境返回的观测字典
            device: 'cpu' 或 'cuda'

        Returns:
            float: V(s)
        """
        robot_tensor, state_tensor = preprocess_obs(obs, device=device)
        robot_tensor = robot_tensor.unsqueeze(0)
        state_tensor = state_tensor.unsqueeze(0)

        value = self.forward(robot_tensor, state_tensor)
        return value.squeeze(0).detach().cpu().item()

    def get_value_batch(self, robot_batch: torch.Tensor,
                        state_batch: torch.Tensor) -> torch.Tensor:
        """
        批量计算价值估计 (训练用)

        Args:
            robot_batch: (batch, 10, 4)
            state_batch: (batch, 11)

        Returns:
            value: (batch,)
        """
        return self.forward(robot_batch, state_batch)
