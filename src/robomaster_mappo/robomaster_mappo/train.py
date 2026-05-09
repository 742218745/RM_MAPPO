#!/usr/bin/env python3
"""
MAPPO (Multi-Agent PPO) 训练脚本

这是 RoboMaster 强化学习的完整训练入口。
包含: 环境交互、经验收集、PPO 更新、日志记录、模型保存。

=== 使用方法 ===

    # 基本训练 (使用默认参数)
    python -m robomaster_mappo.train

    # 自定义参数训练
    python -m robomaster_mappo.train --num_episodes 1000 --lr 3e-4 --render

    # 从检查点恢复训练
    python -m robomaster_mappo.train --load_checkpoint checkpoints/mappo_ep500.pt

=== PPO 算法核心流程 ===

PPO (Proximal Policy Optimization) 是一种 on-policy 的策略梯度算法,
通过"裁剪"来限制每次更新的幅度, 避免策略变化太大导致训练崩溃。

一个完整的训练迭代 (iteration) 包含以下步骤:

    1. 【收集经验】用当前策略在环境中交互 rollout_steps 步,
       将 (obs, action, reward, done, log_prob, value) 存入 buffer

    2. 【计算优势】用 GAE (广义优势估计) 从 buffer 中的奖励序列
       计算每一步的优势函数 A(s,a) 和回报 R(s)

    3. 【PPO 更新】对 buffer 中的数据重复更新 ppo_epochs 轮:
       - 用新策略计算 log π_new(a|s)
       - 计算重要性采样比率 r = exp(log π_new - log π_old)
       - 裁剪目标: L_clip = min(r*A, clip(r, 1±ε)*A)
       - 更新 Actor 最大化 L_clip
       - 更新 Critic 最小化 (V(s) - R(s))²
       - 额外加入熵正则化鼓励探索

    4. 【清空 buffer】准备下一轮收集

=== 关键超参数说明 ===

    learning_rate:    学习率, 控制每次参数更新的步长
                      - 太大: 训练不稳定, 可能发散
                      - 太小: 收敛太慢
                      - 常用值: 3e-4

    gamma:            折扣因子, 控制未来奖励的衰减
                      - 0.99: 重视长远奖励 (约100步后的奖励衰减到1/e)
                      - 0.9:  只重视近期奖励
                      - 常用值: 0.99

    gae_lambda:       GAE 的 lambda, 控制优势估计的偏差-方差权衡
                      - 0.95: 常用值, 较好的平衡
                      - 1.0:  蒙特卡洛回报 (高方差低偏差)
                      - 0.0:  一步 TD (低方差高偏差)

    clip_epsilon:     PPO 裁剪范围
                      - 0.2: 常用值, 限制策略比率在 [0.8, 1.2] 范围内
                      - 越小: 更新越保守 (稳定但慢)
                      - 越大: 更新越激进 (快但可能不稳定)

    ppo_epochs:       每批数据重复更新的轮数
                      - 10: 常用值
                      - 太大: 可能过拟合这批数据
                      - 太小: 数据利用不充分

    entropy_coef:     熵正则化系数, 鼓励策略探索
                      - 0.01: 常用值
                      - 越大: 策略越随机 (探索越多)
                      - 越小: 策略越确定 (利用越多)

    rollout_steps:    每次收集经验的步数
                      - 2048: 常用值
                      - 越大: 每次更新用的数据越多 (更稳定)
                      - 越小: 更新更频繁 (但每次数据少)

    minibatch_size:   小批量大小, PPO 更新时每次用的数据量
                      - 64: 常用值
                      - 必须能整除 rollout_steps
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List
import threading
import select
import sys

# ---- 导入项目模块 ----
from .actor import MAPPOActor
from .critic import MAPPOCritic
from .rollout_buffer import RolloutBuffer
from .obs_preprocessor import preprocess_obs, preprocess_obs_batch


# ==================== 暂停控制器 ====================

class PauseController:
    """训练暂停控制器

    在独立线程中监听键盘输入, 支持以下按键:
      P: 暂停/恢复训练
      Q: 安全退出训练 (保存检查点后退出)

    使用方式:
        controller = PauseController()
        controller.start()
        ...
        if controller.is_paused:
            # 暂停中, 等待恢复
            time.sleep(0.1)
            continue
        ...
        controller.stop()
    """

    def __init__(self):
        self._paused = False
        self._stop_requested = False
        self._thread = None
        self._lock = threading.Lock()

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop_requested

    def start(self):
        """启动键盘监听线程"""
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("[PauseController] 已启动 (按 P 暂停/恢复, Q 安全退出)")

    def stop(self):
        """停止键盘监听线程"""
        with self._lock:
            self._stop_requested = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _listen_loop(self):
        """键盘监听循环 (在独立线程中运行)"""
        try:
            import tty
            import termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                # 设置终端为非阻塞原始模式
                tty.setcbreak(fd)
                while True:
                    with self._lock:
                        if self._stop_requested:
                            break
                    # 非阻塞检查是否有输入
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        ch = sys.stdin.read(1).upper()
                        if ch == 'P':
                            with self._lock:
                                self._paused = not self._paused
                            if self._paused:
                                print("\n[PauseController] >>> 训练已暂停 (按 P 恢复) <<<")
                            else:
                                print("\n[PauseController] >>> 训练已恢复 <<<")
                        elif ch == 'Q':
                            with self._lock:
                                self._stop_requested = True
                            print("\n[PauseController] >>> 请求安全退出 <<<")
                            break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception as e:
            # 如果无法设置终端 (如在非交互环境中), 回退到信号模式
            print(f"[PauseController] 终端模式不可用({e}), 使用 Ctrl+C 安全退出")
            import signal
            def _signal_handler(sig, frame):
                with self._lock:
                    self._stop_requested = True
                print("\n[PauseController] >>> 收到Ctrl+C, 请求安全退出 <<<")
            signal.signal(signal.SIGINT, _signal_handler)
            while True:
                with self._lock:
                    if self._stop_requested:
                        break
                time.sleep(0.5)


# ==================== PPO 更新器 ====================

class PPOUpdater:
    """PPO 参数更新器

    封装了 PPO-Clip 的损失计算和参数更新逻辑。

    PPO 的核心思想:
      - 用"重要性采样"来复用旧策略收集的数据
      - 用"裁剪"来限制新策略和旧策略的差异
      - 如果新策略和旧策略差异太大, 就停止梯度更新

    裁剪目标函数:
      L_CLIP = E[ min( r(θ) * A,  clip(r(θ), 1-ε, 1+ε) * A ) ]

    其中:
      r(θ) = π_θ(a|s) / π_θ_old(a|s)  (重要性采样比率)
      A = 优势函数
      ε = clip_epsilon (裁剪范围, 通常0.2)
    """

    def __init__(
        self,
        actor: MAPPOActor,
        critic: MAPPOCritic,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        ppo_epochs: int = 10,
        minibatch_size: int = 64,
        entropy_coef: float = 0.05,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        """初始化 PPO 更新器

        Args:
            actor: Actor 网络 (策略网络)
            critic: Critic 网络 (价值网络)
            lr: 学习率
            clip_epsilon: PPO 裁剪范围
            ppo_epochs: 每批数据重复更新的轮数
            minibatch_size: 小批量大小
            entropy_coef: 熵正则化系数 (鼓励探索)
            value_loss_coef: 价值损失系数 (Critic 损失的权重)
            max_grad_norm: 梯度裁剪的最大范数 (防止梯度爆炸)
        """
        self.actor = actor
        self.critic = critic
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        # ---- 优化器 ----
        # Actor 和 Critic 使用同一个优化器 (也可以分开)
        # 使用 Adam 优化器, 自适应学习率
        self.optimizer = optim.Adam(
            list(actor.parameters()) + list(critic.parameters()),
            lr=lr,
            eps=1e-5  # Adam 的 epsilon, 防止除零
        )

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """执行一轮 PPO 更新

        流程:
          1. 从 buffer 取出小批量数据
          2. 对每个小批量计算 PPO 损失
          3. 反向传播更新参数
          4. 重复 ppo_epochs 轮

        Args:
            buffer: 已计算好优势函数的经验缓冲区

        Returns:
            训练统计信息字典, 包含:
                - policy_loss: 策略损失 (PPO-Clip)
                - value_loss:  价值损失 (MSE)
                - entropy:     策略熵 (越大越随机)
                - total_loss:  总损失
                - approx_kl:   近似 KL 散度 (衡量新旧策略差异)
        """
        device = next(self.actor.parameters()).device

        # 累计统计量
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        num_updates = 0

        # ---- 重复 ppo_epochs 轮 ----
        for epoch in range(self.ppo_epochs):
            # 从 buffer 获取小批量
            minibatches = buffer.get_minibatches(
                batch_size=self.minibatch_size,
                shuffle=True
            )

            for batch in minibatches:
                # 将数据移到设备上
                robot_batch = batch['robot_batch'].to(device)
                state_batch = batch['state_batch'].to(device)
                chassis_actions = batch['chassis_actions'].to(device)
                shoot_actions = batch['shoot_actions'].to(device)
                old_log_probs = batch['old_log_probs'].to(device)
                advantages = batch['advantages'].to(device)
                returns = batch['returns'].to(device)

                # ==== 1. 用新策略评估动作 ====
                # 计算新策略下的 log_prob 和 entropy
                new_log_probs, entropy = self.actor.evaluate_actions(
                    robot_batch, state_batch,
                    chassis_actions, shoot_actions
                )

                # 计算新 Critic 的价值估计
                new_values = self.critic(robot_batch, state_batch)

                # ==== 2. 计算 PPO 策略损失 ====
                # 重要性采样比率: r(θ) = exp(log π_new - log π_old)
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                # 近似 KL 散度 (用于监控, 不参与梯度)
                # KL ≈ E[(r - 1) - log(r)]
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()

                # PPO-Clip 目标:
                #   L1 = r(θ) * A
                #   L2 = clip(r(θ), 1-ε, 1+ε) * A
                #   L_CLIP = min(L1, L2)
                # 注意: 我们最大化 L_CLIP, 所以损失取负号
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # ==== 3. 计算价值损失 ====
                # MSE 损失: L_V = E[(V(s) - R(s))²]
                value_loss = ((new_values - returns) ** 2).mean()

                # ==== 4. 计算熵奖励 ====
                # 熵越大 → 策略越随机 → 探索越多
                # 我们最大化熵, 所以损失取负号
                entropy_loss = -entropy.mean()

                # ==== 5. 总损失 ====
                # L_total = L_policy + c1 * L_value + c2 * L_entropy
                total_loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # ==== 6. 反向传播 + 梯度裁剪 + 参数更新 ====
                self.optimizer.zero_grad()
                total_loss.backward()
                # 梯度裁剪: 防止梯度爆炸
                # 如果梯度的 L2 范数超过 max_grad_norm, 就缩放梯度
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                # 累计统计量
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += approx_kl
                num_updates += 1

        # 计算平均值
        if num_updates > 0:
            return {
                'policy_loss': total_policy_loss / num_updates,
                'value_loss': total_value_loss / num_updates,
                'entropy': total_entropy / num_updates,
                'total_loss': (total_policy_loss + total_value_loss) / num_updates,
                'approx_kl': total_kl / num_updates,
            }
        else:
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'total_loss': 0.0,
                'approx_kl': 0.0,
            }


# ==================== 训练主循环 ====================

def train(
    # ---- 环境参数 ----
    num_episodes: int = 1000,       # 训练的总回合数
    rollout_steps: int = 2048,      # 每次收集经验的步数
    max_steps_per_episode: int = 2048,  # 每个回合最大步数
    render: bool = False,           # 是否渲染 (训练时通常关闭以加速)
    render_interval: int = 50,      # 每隔多少回合渲染一次

    # ---- PPO 超参数 ----
    lr: float = 3e-4,               # 学习率
    gamma: float = 0.99,            # 折扣因子
    gae_lambda: float = 0.95,       # GAE lambda
    clip_epsilon: float = 0.2,      # PPO 裁剪范围
    ppo_epochs: int = 10,           # 每批数据更新轮数
    minibatch_size: int = 64,       # 小批量大小
    entropy_coef: float = 0.05,     # 熵正则化系数
    value_loss_coef: float = 0.5,   # 价值损失系数
    max_grad_norm: float = 0.5,     # 梯度裁剪范数

    # ---- 其他参数 ----
    save_interval: int = 100,       # 每隔多少回合保存检查点
    log_interval: int = 10,         # 每隔多少回合打印日志
    checkpoint_dir: str = 'checkpoints',  # 检查点保存目录
    load_checkpoint: str = None,    # 从检查点恢复训练的路径
    device: str = 'auto',           # 计算设备 ('cpu', 'cuda', 'auto')
):
    """MAPPO 训练主函数

    训练流程:
      for episode in range(num_episodes):
          1. 重置环境
          2. 收集 rollout_steps 步经验
          3. 计算 GAE 和回报
          4. PPO 更新 Actor 和 Critic
          5. 记录日志, 保存检查点

    Args:
        (见上方参数说明)
    """
    # ---- 确定计算设备 ----
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Train] 使用设备: {device}")
    if device == 'cpu':
        print(f"[Train] 提示: 未检测到 GPU, 使用 CPU 推理 (~20ms/步 vs GPU ~5ms/步)")
        print(f"[Train]       训练速度约为 GPU 的 1/3, 建议使用 GPU 实例加速")

    # ---- 创建检查点目录 ----
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---- 创建环境 ----
    # 延迟导入, 避免在没有 ROS2 的机器上导入失败
    from robomaster_gym_env import RoboMasterGazeboEnv, GymEnvConfig

    env_config = GymEnvConfig()
    env = RoboMasterGazeboEnv(config=env_config)
    print(f"[Train] 环境创建成功")
    print(f"[Train]   观察空间: {env.observation_space}")
    print(f"[Train]   动作空间: {env.action_space}")

    # ---- 创建网络 ----
    actor = MAPPOActor().to(device)
    critic = MAPPOCritic().to(device)
    print(f"[Train] Actor 参数量: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"[Train] Critic 参数量: {sum(p.numel() for p in critic.parameters()):,}")

    # ---- 创建 PPO 更新器和缓冲区 ----
    ppo_updater = PPOUpdater(
        actor=actor,
        critic=critic,
        lr=lr,
        clip_epsilon=clip_epsilon,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
        entropy_coef=entropy_coef,
        value_loss_coef=value_loss_coef,
        max_grad_norm=max_grad_norm,
    )

    buffer = RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)

    # ---- 从检查点恢复 ----
    start_episode = 0
    episode_rewards: List[float] = []     # 每个回合的总奖励
    episode_lengths: List[int] = []       # 每个回合的步数
    best_avg_reward = -float('inf')       # 最佳平均奖励 (用于保存最佳模型)

    if load_checkpoint is not None and os.path.exists(load_checkpoint):
        checkpoint = torch.load(load_checkpoint, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        ppo_updater.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 恢复训练统计
        if 'episode_rewards' in checkpoint:
            episode_rewards = checkpoint['episode_rewards']
        if 'episode_lengths' in checkpoint:
            episode_lengths = checkpoint['episode_lengths']
        if 'best_avg_reward' in checkpoint:
            best_avg_reward = checkpoint['best_avg_reward']
        # 用检查点中保存的 episode 编号恢复，+1 表示从下一个 episode 继续
        start_episode = checkpoint.get('episode', 0) + 1
        print(f"[Train] 从检查点恢复: {load_checkpoint} (从 episode {start_episode+1} 继续)")
        print(f"[Train] 已恢复训练统计: {len(episode_rewards)} 个已完成回合, best_avg_reward={best_avg_reward:.2f}")
    else:
        if load_checkpoint is not None:
            print(f"[Train] 检查点不存在: {load_checkpoint}, 从头开始训练")

    print(f"\n{'='*60}")
    print(f"  开始 MAPPO 训练")
    print(f"  总回合数: {num_episodes}")
    print(f"  起始回合: {start_episode}")
    print(f"  每回合步数: {rollout_steps}")
    print(f"  学习率: {lr}")
    print(f"  PPO epochs: {ppo_epochs}")
    print(f"  设备: {device}")
    print(f"{'='*60}\n")

    # ---- 启动暂停控制器 ----
    pause_ctrl = PauseController()
    pause_ctrl.start()

    # ==================== 训练主循环 ====================
    for episode in range(start_episode, num_episodes):
        # ---- 检查安全退出请求 ----
        if pause_ctrl.stop_requested:
            print(f"\n[Train] 收到退出请求, 保存检查点后退出...")
            _save_checkpoint(
                checkpoint_dir, episode, actor, critic,
                ppo_updater.optimizer, 'latest',
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                best_avg_reward=best_avg_reward
            )
            break

        # ---- 暂停等待 ----
        while pause_ctrl.is_paused:
            if pause_ctrl.stop_requested:
                break
            # 暂停时更新渲染 (显示暂停状态)
            if render and hasattr(env, '_train_progress'):
                env.render()
            time.sleep(0.2)

        episode_start_time = time.time()

        # ---- 1. 重置环境 ----
        obs, info = env.reset()
        episode_reward = 0.0
        episode_step = 0

        # ---- 2. 收集经验 (rollout) ----
        buffer.reset()  # 清空缓冲区

        for step in range(rollout_steps):
            # 用 Actor 采样动作
            # deterministic=False: 训练时需要探索, 所以随机采样
            _t0 = time.time()
            action, log_prob = actor.get_action(obs, deterministic=False, device=device)

            # 用 Critic 估计价值
            value = critic.get_value(obs, device=device)
            _t1 = time.time()

            # 在环境中执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            _t2 = time.time()
            done = terminated or truncated

            # 仿真不稳定时: 丢弃该步数据, 继续控制
            # (车不动时rtf≈0, 开局rtf=0, 等恢复会死等, 所以不等)
            if info.get('sim_unstable', False):
                obs = next_obs
                if done:
                    obs, info = env.reset()
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_step)
                    episode_reward = 0.0
                    episode_step = 0
                continue

            # 存入缓冲区
            buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                log_prob=log_prob,
                value=value
            )

            # 累计奖励和步数
            episode_reward += reward
            episode_step += 1

            # 终端实时进度 (每100步打印一行)
            if step % 100 == 0 or done:
                own_x, own_y = env._get_own_position_safe()
                if own_x is not None:
                    dist_to_target = np.sqrt(
                        (own_x - env._virtual_blue_x)**2 + (own_y - env._virtual_blue_y)**2
                    )
                    pos_str = f"({own_x:.1f},{own_y:.1f}) dist={dist_to_target:.1f}m"
                else:
                    pos_str = "N/A"
                chassis_vel = action.get('chassis_velocity', np.zeros(2))
                terminated_reason = ""
                if done:
                    if terminated:
                        terminated_reason = " [TERMINATED]"
                    else:
                        terminated_reason = " [TRUNCATED]"
                # rtf 检测
                measured_rtf = env._real_time_factor_measured
                set_rtf = env.real_time_factor
                rtf_str = f"rtf={measured_rtf:.1f}/{set_rtf:.1f}"
                if measured_rtf > 0 and set_rtf > 0:
                    ratio = measured_rtf / set_rtf
                    if ratio < 0.8:
                        rtf_str += f" ⚠SLOW({ratio:.0%})"
                    elif ratio < 0.95:
                        rtf_str += f" (~{ratio:.0%})"
                # 计时诊断
                infer_ms = (_t1 - _t0) * 1000
                step_ms = (_t2 - _t1) * 1000
                print(
                    f"  Ep{episode+1:4d} Step{step:4d}/{rollout_steps} | "
                    f"r={reward:+7.2f} sum_r={episode_reward:+8.1f} | "
                    f"pos={pos_str} | "
                    f"vel=({chassis_vel[0]:+.1f},{chassis_vel[1]:+.1f}) | "
                    f"stage={info.get('curriculum_stage',1)} | "
                    f"{rtf_str} | "
                    f"infer={infer_ms:.1f}ms step={step_ms:.1f}ms"
                    f"{terminated_reason}"
                )

            # 渲染 (仅在--render开启时, 每20步渲染一次)
            if render and step % 20 == 0:
                recent_rewards = episode_rewards[-100:] if episode_rewards else [0]
                env._train_progress = {
                    'episode': episode + 1,
                    'num_episodes': num_episodes,
                    'curriculum_stage': info.get('curriculum_stage', 1),
                    'avg_reward': np.mean(recent_rewards),
                    'best_reward': best_avg_reward,
                    'paused': pause_ctrl.is_paused,
                }
                env.render()

            # 更新观测
            obs = next_obs

            # 如果回合结束, 重置环境
            if done:
                obs, info = env.reset()
                # 记录上一个回合的统计
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_step)
                episode_reward = 0.0
                episode_step = 0

                # 渲染: 重置后立即刷新
                if render:
                    recent_rewards = episode_rewards[-100:] if episode_rewards else [0]
                    env._train_progress = {
                        'episode': episode + 1,
                        'num_episodes': num_episodes,
                        'curriculum_stage': info.get('curriculum_stage', 1),
                        'avg_reward': np.mean(recent_rewards),
                        'best_reward': best_avg_reward,
                        'paused': pause_ctrl.is_paused,
                    }
                    env.render()

        # ---- 3. 计算优势函数和回报 ----
        # 用 Critic 估计最后一个状态的价值 (用于 bootstrap)
        next_value = critic.get_value(obs, device=device)
        buffer.compute_returns(next_value=next_value)

        # ---- 4. PPO 更新 ----
        if render and episode % render_interval == 0:
            print(f"  [PPO] 正在更新策略网络...")
            # PPO更新期间标记为更新状态, 渲染器会显示"更新中"
            env._train_progress = {
                'episode': episode + 1,
                'num_episodes': num_episodes,
                'curriculum_stage': info.get('curriculum_stage', 1),
                'avg_reward': np.mean(episode_rewards[-100:]) if episode_rewards else 0,
                'best_reward': best_avg_reward,
                'paused': False,
                'ppo_updating': True,
            }
            env.render()

        # PPO更新前: 发送零速度让车停下, 避免PPO期间车因惯性滑出边界
        env.ros2_interface.send_chassis_velocity(0.0, 0.0, 0.0)
        update_stats = ppo_updater.update(buffer)

        # PPO更新后: 渲染并打印完成信息
        if render and episode % render_interval == 0:
            print(f"  [PPO] 策略网络更新完成")
            # PPO后重置步数, 让渲染器显示新一轮开始
            env.current_step = 0
            env._train_progress = {
                'episode': episode + 1,
                'num_episodes': num_episodes,
                'curriculum_stage': info.get('curriculum_stage', 1),
                'avg_reward': np.mean(episode_rewards[-100:]) if episode_rewards else 0,
                'best_reward': best_avg_reward,
                'paused': False,
                'ppo_updating': False,
            }
            env.render()

        # PPO更新后: 检查是否出界/翻车, 如果是则重置
        if env.ros2_interface.is_tumbled() or env._check_terminated():
            obs, info = env.reset()
            episode_reward = 0.0
            episode_step = 0

        # ---- 5. 记录日志 ----
        episode_time = time.time() - episode_start_time

        if (episode + 1) % log_interval == 0:
            # 计算最近 log_interval 个回合的平均奖励
            recent_rewards = episode_rewards[-log_interval:] if episode_rewards else [0]
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(episode_lengths[-log_interval:]) if episode_lengths else 0
            curriculum_stage = info.get('curriculum_stage', 1)

            # 物理精度检测: 实测时间流速 vs 设定值
            measured_rtf = env._real_time_factor_measured
            set_rtf = env.real_time_factor
            rtf_status = ""
            if measured_rtf > 0 and set_rtf > 0:
                ratio = measured_rtf / set_rtf
                if ratio < 0.8:
                    rtf_status = f" ⚠SLOW({ratio:.0%})"
                elif ratio < 0.95:
                    rtf_status = f" (~{ratio:.0%})"

            print(
                f"[Episode {episode+1:4d}/{num_episodes}] "
                f"stage={curriculum_stage} | "
                f"avg_reward={avg_reward:8.2f} | "
                f"avg_len={avg_length:6.0f} | "
                f"p_loss={update_stats['policy_loss']:8.4f} | "
                f"v_loss={update_stats['value_loss']:8.4f} | "
                f"entropy={update_stats['entropy']:6.4f} | "
                f"kl={update_stats['approx_kl']:6.4f} | "
                f"rtf={measured_rtf:.1f}/{set_rtf:.1f}{rtf_status} | "
                f"time={episode_time:.1f}s"
            )

        # ---- 6. 保存检查点 ----
        if (episode + 1) % save_interval == 0:
            # 保存最新检查点
            _save_checkpoint(
                checkpoint_dir, episode, actor, critic,
                ppo_updater.optimizer, 'latest',
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                best_avg_reward=best_avg_reward
            )

            # 如果是最佳模型, 额外保存一份
            recent_rewards = episode_rewards[-save_interval:] if episode_rewards else [0]
            avg_reward = np.mean(recent_rewards)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                _save_checkpoint(
                    checkpoint_dir, episode, actor, critic,
                    ppo_updater.optimizer, 'best',
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    best_avg_reward=best_avg_reward
                )
                print(f"  -> 新最佳模型! avg_reward={avg_reward:.2f}")

            # 同时保存带回合号的检查点
            _save_checkpoint(
                checkpoint_dir, episode, actor, critic,
                ppo_updater.optimizer, f'ep{episode+1}',
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                best_avg_reward=best_avg_reward
            )

    # ---- 训练结束 ----
    pause_ctrl.stop()

    print(f"\n{'='*60}")
    print(f"  训练完成!")
    print(f"  总回合数: {num_episodes}")
    if episode_rewards:
        print(f"  最终平均奖励 (最近100回合): {np.mean(episode_rewards[-100:]):.2f}")
        print(f"  最佳平均奖励: {best_avg_reward:.2f}")
    print(f"  检查点保存在: {checkpoint_dir}/")
    print(f"{'='*60}")

    # 关闭环境
    env.close()


def _save_checkpoint(
    checkpoint_dir: str,
    episode: int,
    actor: MAPPOActor,
    critic: MAPPOCritic,
    optimizer: optim.Optimizer,
    tag: str,
    episode_rewards: list = None,
    episode_lengths: list = None,
    best_avg_reward: float = None
):
    """保存训练检查点

    Args:
        checkpoint_dir: 保存目录
        episode: 当前回合数
        actor: Actor 网络
        critic: Critic 网络
        optimizer: 优化器
        tag: 标签名 (如 'latest', 'best', 'ep100')
        episode_rewards: 训练奖励历史
        episode_lengths: 训练步数历史
        best_avg_reward: 最佳平均奖励
    """
    path = os.path.join(checkpoint_dir, f'mappo_{tag}.pt')
    checkpoint = {
        'episode': episode,
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    # 保存训练统计
    if episode_rewards is not None:
        checkpoint['episode_rewards'] = episode_rewards
    if episode_lengths is not None:
        checkpoint['episode_lengths'] = episode_lengths
    if best_avg_reward is not None:
        checkpoint['best_avg_reward'] = best_avg_reward
    torch.save(checkpoint, path)


# ==================== 命令行入口 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MAPPO 训练脚本')

    # 环境参数
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='训练的总回合数 (默认: 1000)')
    parser.add_argument('--rollout_steps', type=int, default=2048,
                        help='每次收集经验的步数 (默认: 2048)')
    parser.add_argument('--render', action='store_true',
                        help='是否渲染 (训练时通常关闭以加速)')
    parser.add_argument('--render_interval', type=int, default=50,
                        help='每隔多少回合渲染一次 (默认: 50)')

    # PPO 超参数
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='学习率 (默认: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='折扣因子 (默认: 0.99)')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda (默认: 0.95)')
    parser.add_argument('--clip_epsilon', type=float, default=0.2,
                        help='PPO 裁剪范围 (默认: 0.2)')
    parser.add_argument('--ppo_epochs', type=int, default=10,
                        help='每批数据更新轮数 (默认: 10)')
    parser.add_argument('--minibatch_size', type=int, default=64,
                        help='小批量大小 (默认: 64)')
    parser.add_argument('--entropy_coef', type=float, default=0.05,
                        help='熵正则化系数 (默认: 0.05)')

    # 其他参数
    parser.add_argument('--save_interval', type=int, default=100,
                        help='每隔多少回合保存检查点 (默认: 100)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='每隔多少回合打印日志 (默认: 10)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='检查点保存目录 (默认: checkpoints)')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help='从检查点恢复训练的路径')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备: cpu/cuda/auto (默认: auto)')

    return parser.parse_args()


def main():
    """命令行入口函数"""
    args = parse_args()

    train(
        num_episodes=args.num_episodes,
        rollout_steps=args.rollout_steps,
        render=args.render,
        render_interval=args.render_interval,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
        load_checkpoint=args.load_checkpoint,
        device=args.device,
    )


if __name__ == '__main__':
    main()
