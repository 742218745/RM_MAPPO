"""
PPO 经验回放缓冲区 (Rollout Buffer)

存储一个 rollout 周期内收集的所有经验, 用于 PPO 更新。

PPO 的训练流程:
  1. 用当前策略在环境中交互, 收集一批经验 (一个 rollout)
  2. 用这批经验计算 GAE (广义优势估计) 和 returns (回报)
  3. 用 PPO-Clip 目标函数更新 Actor 和 Critic
  4. 清空 buffer, 回到步骤 1

Buffer 中每条经验包含:
  - obs:       观测字典 (原始 Gym 观测)
  - action:    动作字典 {chassis_velocity: ndarray, shoot: int}
  - reward:    标量奖励
  - done:      是否终止 (terminated or truncated)
  - log_prob:  策略在该状态输出该动作的对数概率 log π(a|s)
  - value:     Critic 对该状态的价值估计 V(s)

计算 GAE 时还需要:
  - next_value: 最后一步之后 Critic 对下一个状态的估计 (用于 bootstrap)
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional


class RolloutBuffer:
    """PPO 经验回放缓冲区

    存储一个 rollout 的所有经验, 并提供计算 GAE 和转换为
    PyTorch Tensor 的方法。

    使用流程:
        buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
        # 收集经验
        for step in range(num_steps):
            buffer.add(obs, action, reward, done, log_prob, value)
        # 计算优势函数和回报
        buffer.compute_returns(next_value=critic.get_value(last_obs))
        # 取出数据用于训练
        for batch in buffer.get_minibatches(batch_size=64):
            ppo_update(batch)
        # 清空, 开始下一个 rollout
        buffer.reset()

    Attributes:
        gamma: 折扣因子, 控制未来奖励的衰减速度 (0~1)
            - 越接近1: 越重视长远奖励
            - 越接近0: 越重视即时奖励
            - 常用值: 0.99 或 0.995
        gae_lambda: GAE 的 lambda 参数, 控制偏差-方差权衡 (0~1)
            - lambda=0: 低方差高偏差 (只看一步 TD error)
            - lambda=1: 高方差低偏差 (看完整蒙特卡洛回报)
            - 常用值: 0.95
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """初始化缓冲区

        Args:
            gamma: 折扣因子
            gae_lambda: GAE lambda 参数
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # 经验存储列表
        self.observations: List[Dict[str, Any]] = []   # 观测列表
        self.actions: List[Dict[str, Any]] = []         # 动作列表
        self.rewards: List[float] = []                  # 奖励列表
        self.dones: List[bool] = []                     # 终止标志列表
        self.log_probs: List[float] = []                # log π(a|s) 列表
        self.values: List[float] = []                   # V(s) 列表

        # 计算后的优势函数和回报 (在 compute_returns 后填充)
        self.advantages: Optional[np.ndarray] = None    # 优势函数 A(s,a)
        self.returns: Optional[np.ndarray] = None       # 回报 R(s) = A(s,a) + V(s)

    def add(
        self,
        obs: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        done: bool,
        log_prob: float,
        value: float
    ):
        """向缓冲区添加一条经验

        Args:
            obs: 观测字典 (Gym 环境返回的原始观测)
            action: 动作字典 {chassis_velocity: ndarray, shoot: int}
            reward: 标量奖励
            done: 是否终止 (terminated or truncated)
            log_prob: 策略在该状态输出该动作的对数概率
            value: Critic 对该状态的价值估计
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns(self, next_value: float = 0.0):
        """计算 GAE (广义优势估计) 和回报

        GAE (Generalized Advantage Estimation) 是 PPO 中计算优势函数的方法。
        优势函数 A(s,a) 表示"在状态 s 执行动作 a 比平均水平好多少"。

        计算过程:
          1. 计算每一步的 TD error: δ_t = r_t + γ * V(s_{t+1}) * (1-done) - V(s_t)
          2. 递推计算 GAE: A_t = δ_t + (γ * λ) * (1-done) * A_{t+1}
          3. 计算回报: R_t = A_t + V(s_t)

        Args:
            next_value: 最后一步之后 Critic 对下一个状态的估计
                用于 bootstrap (当 rollout 没有真正结束时,
                用 Critic 估计来近似未来的价值)
        """
        num_steps = len(self.rewards)
        if num_steps == 0:
            return

        # 初始化优势函数数组
        self.advantages = np.zeros(num_steps, dtype=np.float32)
        self.returns = np.zeros(num_steps, dtype=np.float32)

        # ---- 从后往前递推计算 GAE ----
        # gae: 累积的优势估计值
        gae = 0.0

        for t in reversed(range(num_steps)):
            # 计算下一步的价值
            if t == num_steps - 1:
                # 最后一步: 使用 next_value (bootstrap)
                next_val = next_value
            else:
                # 非最后一步: 使用存储的 V(s_{t+1})
                next_val = self.values[t + 1]

            # TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            # done_t=1 时, 下一步价值为0 (回合结束, 没有未来奖励)
            delta = self.rewards[t] + self.gamma * next_val * (1.0 - self.dones[t]) - self.values[t]

            # GAE 递推: A_t = δ_t + (γ * λ) * (1 - done_t) * A_{t+1}
            # done_t=1 时, 不累积未来的优势 (回合已结束)
            gae = delta + self.gamma * self.gae_lambda * (1.0 - self.dones[t]) * gae

            # 存储优势函数值
            self.advantages[t] = gae

            # 回报 = 优势 + 价值: R_t = A_t + V(s_t)
            self.returns[t] = gae + self.values[t]

    def get_minibatches(
        self,
        batch_size: int = 64,
        shuffle: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        """将缓冲区数据分成小批量, 用于 PPO 更新

        每个 minibatch 是一个字典, 包含:
            - robot_batch:     (batch, 10, 4) 归一化后的机器人信息
            - state_batch:     (batch, 11) 归一化后的标量状态
            - chassis_actions: (batch, 2) 底盘速度动作
            - shoot_actions:   (batch,) 整数, 射击动作
            - old_log_probs:   (batch,) 旧策略的 log 概率
            - advantages:      (batch,) 优势函数
            - returns:         (batch,) 回报

        Args:
            batch_size: 每个小批量的大小
            shuffle: 是否打乱顺序 (训练时通常打乱)

        Returns:
            小批量列表
        """
        from .obs_preprocessor import preprocess_obs_batch

        num_steps = len(self.rewards)
        if num_steps == 0 or self.advantages is None:
            return []

        # 生成索引
        indices = np.arange(num_steps)
        if shuffle:
            np.random.shuffle(indices)

        # ---- 预处理所有观测为张量 ----
        # 将原始观测字典列表转为 (robot_batch, state_batch) 张量
        obs_list = [self.observations[i] for i in indices]
        robot_batch, state_batch = preprocess_obs_batch(obs_list, device='cpu')

        # ---- 提取动作张量 ----
        chassis_actions_list = []
        shoot_actions_list = []
        for i in indices:
            action = self.actions[i]
            chassis_actions_list.append(action['chassis_velocity'])
            shoot_actions_list.append(int(action['shoot']))

        chassis_actions = torch.FloatTensor(np.array(chassis_actions_list))  # (num_steps, 2)
        shoot_actions = torch.LongTensor(np.array(shoot_actions_list))       # (num_steps,)

        # ---- 提取旧 log_prob, 优势, 回报 ----
        old_log_probs = torch.FloatTensor(np.array([self.log_probs[i] for i in indices]))
        advantages = torch.FloatTensor(np.array([self.advantages[i] for i in indices]))
        returns = torch.FloatTensor(np.array([self.returns[i] for i in indices]))

        # ---- 优势函数归一化 ----
        # 归一化优势函数可以稳定训练 (减少梯度方差)
        # 注意: 只在优势函数不全为0时归一化
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- 分成小批量 ----
        minibatches = []
        for start in range(0, num_steps, batch_size):
            end = min(start + batch_size, num_steps)
            # 如果最后一批太小, 跳过
            if end - start < batch_size // 2:
                continue

            minibatch = {
                'robot_batch': robot_batch[start:end].clone(),
                'state_batch': state_batch[start:end].clone(),
                'chassis_actions': chassis_actions[start:end].clone(),
                'shoot_actions': shoot_actions[start:end].clone(),
                'old_log_probs': old_log_probs[start:end].clone(),
                'advantages': advantages[start:end].clone(),
                'returns': returns[start:end].clone(),
            }
            minibatches.append(minibatch)

        return minibatches

    def __len__(self) -> int:
        """返回缓冲区中的经验数量"""
        return len(self.rewards)

    def reset(self):
        """清空缓冲区, 开始新的 rollout"""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages = None
        self.returns = None
