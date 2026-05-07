"""
RoboMaster 环境渲染器

使用 matplotlib 绘制 2D 俯视图, 在网格世界中显示:
  - 场地边界和网格
  - 所有机器人位置 (红方/蓝方用不同颜色)
  - 前哨站和基地位置
  - 右侧面板显示观察空间各项参数

使用方式:
  方式1: 在环境中调用 env.render() 自动调用
  方式2: 独立使用
      renderer = EnvRenderer()
      renderer.render(obs, env_config)
      renderer.close()
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import time

try:
    import matplotlib
    matplotlib.use('TkAgg')  # 使用 TkAgg 后端, 兼容性最好
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ==================== 场地常量 ====================
# RoboMaster 标准场地尺寸 (单位: 米)
FIELD_LENGTH = 28.0   # 场地长度 (x 方向)
FIELD_WIDTH = 15.0    # 场地宽度 (y 方向)

# 场地关键区域坐标 (基于 RoboMaster 2025 赛季赛场地)
# 红方区域 (x 较小的一侧)
RED_OUTPOST_POS = (11.0, 11.35)    # 红方前哨站 (x, y)
RED_BASE_POS = (2.4, 7.5)          # 红方基地
# 蓝方区域 (x 较大的一侧)
BLUE_OUTPOST_POS = (17.0, 3.65)    # 蓝方前哨站
BLUE_BASE_POS = (25.6, 7.5)        # 蓝方基地


class EnvRenderer:
    """RoboMaster 环境 2D 俯视图渲染器

    绘制内容:
      - 场地边界 (28m x 15m 矩形)
      - 网格线 (每 2m 一条)
      - 红方/蓝方机器人 (圆形标记, 带ID标注)
      - 前哨站 (菱形标记)
      - 基地 (方形标记)
      - 右侧面板: 观察空间各项参数数值

    Attributes:
        fig: matplotlib Figure 对象
        ax_field: 场地子图 (左侧)
        ax_info: 信息面板子图 (右侧)
    """

    def __init__(self, figsize: Tuple[int, int] = (16, 8)):
        """初始化渲染器

        Args:
            figsize: 窗口大小 (宽, 高) 英寸
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib 不可用, 渲染功能已禁用")
            self.fig = None
            return

        # 创建图形窗口, 使用 GridSpec 分左右两栏
        # 左栏: 场地俯视图 (占 2/3 宽度)
        # 右栏: 参数信息面板 (占 1/3 宽度)
        self.fig = plt.figure(figsize=figsize, facecolor='#1a1a2e')
        gs = GridSpec(1, 3, figure=self.fig, wspace=0.05)

        # 场地子图
        self.ax_field = self.fig.add_subplot(gs[0, 0:2])
        # 信息面板子图
        self.ax_info = self.fig.add_subplot(gs[0, 2])

        # 设置窗口标题
        self.fig.canvas.manager.set_window_title('RoboMaster Gym Env Renderer')

        # 上一次渲染时间 (用于控制刷新频率)
        self._last_render_time = 0.0

        # 初始化场地
        self._setup_field()

    def _setup_field(self):
        """设置场地基础元素 (边界、网格、前哨站、基地)

        这些元素每帧都一样, 但为了简单起见, 每帧都重新绘制
        """
        ax = self.ax_field
        ax.set_facecolor('#0f0f23')  # 深色背景

        # 场地边界
        ax.set_xlim(-1, FIELD_LENGTH + 1)
        ax.set_ylim(-1, FIELD_WIDTH + 1)
        ax.set_aspect('equal')  # 等比例, 不变形

        # 网格线 (每 2m 一条)
        ax.set_xticks(np.arange(0, FIELD_LENGTH + 1, 2))
        ax.set_yticks(np.arange(0, FIELD_WIDTH + 1, 2))
        ax.grid(True, alpha=0.15, color='white', linestyle='--')

        # 场地边框
        field_rect = patches.Rectangle(
            (0, 0), FIELD_LENGTH, FIELD_WIDTH,
            linewidth=2, edgecolor='white', facecolor='none', linestyle='-'
        )
        ax.add_patch(field_rect)

        # 坐标轴标签
        ax.set_xlabel('X (m)', color='white', fontsize=10)
        ax.set_ylabel('Y (m)', color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=8)

        # 中线 (场地中央竖线)
        ax.axvline(x=FIELD_LENGTH / 2, color='yellow', alpha=0.3, linestyle='-', linewidth=1)

    def render(
        self,
        obs: Dict[str, Any],
        env_config: Any = None,
        current_step: int = 0,
        max_steps: int = 2100,
        reward: float = 0.0,
        team: str = 'red',
        virtual_blue_pos: Optional[Tuple[float, float]] = None,
        train_progress: Optional[Dict[str, Any]] = None,
    ):
        """渲染一帧

        Args:
            obs: Gym 环境返回的观察字典, 包含:
                - all_robots: (10, 4) 数组 [id, team, x, y]
                    team: 0=己方, 1=敌方, -1=unknown
                - own_hp: 己方血量
                - own_ammo: 己方弹药量
                - team_economy: 我方经济
                - remaining_steps: 剩余步数
                - judge_countdown_steps: 判负步数
                - damage_per_step: 每步伤害
                - outpost_hp: 前哨站血量
                - base_hp: 基地血量
                - base_exposed: 基地展开状态
                - ammo_consumed_per_step: 每步弹药消耗
                - revive_waiting_steps: 复活等待步数
            env_config: 环境配置对象 (GymEnvConfig), 用于获取前哨站/基地坐标
            current_step: 当前步数
            max_steps: 最大步数
            reward: 最近一步的奖励
            team: 自身队伍 ('red' or 'blue')
            virtual_blue_pos: 虚拟蓝方目标位置 (x, y), 若提供则在场地上标记
            train_progress: 训练进度信息字典, 包含:
                - episode: 当前回合
                - num_episodes: 总回合数
                - curriculum_stage: 课程学习阶段
                - avg_reward: 近期平均奖励
                - best_reward: 最佳平均奖励
                - paused: 是否暂停
        """
        if not MATPLOTLIB_AVAILABLE or self.fig is None:
            return

        # ---- 清空子图, 准备重绘 ----
        self.ax_field.cla()
        self.ax_info.cla()

        # ---- 1. 绘制场地基础 ----
        self._setup_field()

        # ---- 2. 绘制前哨站和基地 ----
        self._draw_structures(env_config)

        # ---- 3. 绘制虚拟蓝方目标点 ----
        if virtual_blue_pos is not None:
            self._draw_virtual_target(virtual_blue_pos)

        # ---- 4. 绘制所有机器人 ----
        all_robots = obs.get('all_robots', None)
        if all_robots is not None:
            self._draw_robots(all_robots, team)

        # ---- 5. 绘制信息面板 ----
        self._draw_info_panel(obs, current_step, max_steps, reward, team, train_progress)

        # ---- 6. 刷新显示 ----
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # 非阻塞显示: plt.pause 会处理 GUI 事件
        plt.pause(0.001)

    def _draw_structures(self, env_config: Any = None):
        """绘制前哨站和基地

        Args:
            env_config: 环境配置, 包含前哨站/基地坐标
        """
        ax = self.ax_field

        # 从配置获取坐标, 如果没有则使用默认值
        if env_config is not None:
            red_outpost = (env_config.red_outpost_position[0], env_config.red_outpost_position[1])
            red_base = (env_config.red_base_position[0], env_config.red_base_position[1])
            blue_outpost = (env_config.blue_outpost_position[0], env_config.blue_outpost_position[1])
            blue_base = (env_config.blue_base_position[0], env_config.blue_base_position[1])
        else:
            red_outpost = RED_OUTPOST_POS
            red_base = RED_BASE_POS
            blue_outpost = BLUE_OUTPOST_POS
            blue_base = BLUE_BASE_POS

        # 红方前哨站 (菱形)
        ax.plot(red_outpost[0], red_outpost[1], marker='D', color='red',
                markersize=12, markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        ax.annotate('R-Outpost', (red_outpost[0], red_outpost[1]),
                    textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=7, color='red', fontweight='bold')

        # 红方基地 (方形)
        ax.plot(red_base[0], red_base[1], marker='s', color='red',
                markersize=14, markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        ax.annotate('R-Base', (red_base[0], red_base[1]),
                    textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=7, color='red', fontweight='bold')

        # 蓝方前哨站 (菱形)
        ax.plot(blue_outpost[0], blue_outpost[1], marker='D', color='deepskyblue',
                markersize=12, markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        ax.annotate('B-Outpost', (blue_outpost[0], blue_outpost[1]),
                    textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=7, color='deepskyblue', fontweight='bold')

        # 蓝方基地 (方形)
        ax.plot(blue_base[0], blue_base[1], marker='s', color='deepskyblue',
                markersize=14, markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        ax.annotate('B-Base', (blue_base[0], blue_base[1]),
                    textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=7, color='deepskyblue', fontweight='bold')

    def _draw_virtual_target(self, virtual_blue_pos: Tuple[float, float]):
        """绘制虚拟蓝方目标点标记

        用醒目的十字准星 + 脉冲圆圈标记课程学习的虚拟目标位置,
        并用虚线连接红方机器人到目标点。

        Args:
            virtual_blue_pos: 虚拟蓝方位置 (x, y)
        """
        ax = self.ax_field
        vx, vy = virtual_blue_pos

        # 脉冲圆圈 (3层, 模拟雷达效果)
        for radius, alpha in [(2.0, 0.15), (1.2, 0.25), (0.5, 0.4)]:
            circle = patches.Circle(
                (vx, vy), radius,
                linewidth=1.5, edgecolor='#00ff88',
                facecolor='none', alpha=alpha, linestyle='--', zorder=8
            )
            ax.add_patch(circle)

        # 十字准星
        cross_size = 0.8
        ax.plot([vx - cross_size, vx + cross_size], [vy, vy],
                color='#00ff88', linewidth=2, alpha=0.9, zorder=9)
        ax.plot([vx, vx], [vy - cross_size, vy + cross_size],
                color='#00ff88', linewidth=2, alpha=0.9, zorder=9)

        # 中心点
        ax.plot(vx, vy, marker='o', color='#00ff88',
                markersize=8, markeredgecolor='white',
                markeredgewidth=1.5, zorder=10)

        # 标注 "TARGET"
        ax.annotate('TARGET', (vx, vy),
                    textcoords="offset points", xytext=(0, 16),
                    ha='center', fontsize=8, fontweight='bold',
                    color='#00ff88', zorder=10)
        ax.annotate(f'({vx:.1f}, {vy:.1f})', (vx, vy),
                    textcoords="offset points", xytext=(0, -16),
                    ha='center', fontsize=6, color='#00ff88', zorder=10)

    def _draw_robots(self, all_robots: np.ndarray, own_team: str = 'red'):
        """绘制所有机器人

        Args:
            all_robots: (10, 4) 数组, 每行 [id, team, x, y]
                team: 0=己方(ally), 1=敌方(enemy), -1=unknown
            own_team: 自身队伍颜色 ('red' or 'blue')
        """
        ax = self.ax_field

        for i in range(all_robots.shape[0]):
            robot_id = int(all_robots[i, 0])
            team_code = int(all_robots[i, 1])
            x = all_robots[i, 2]
            y = all_robots[i, 3]

            # 跳过无效机器人 (id=-1 表示 padding)
            if robot_id == -1:
                continue

            # 跳过坐标为 nan 的机器人 (数据缺失)
            if np.isnan(x) or np.isnan(y):
                continue

            # 根据队伍关系选择颜色
            # team_code: 0=己方, 1=敌方, -1=unknown
            if team_code == 0:
                # 己方: 用自身队伍颜色
                color = 'red' if own_team == 'red' else 'deepskyblue'
                edge_color = 'white'
                marker_size = 10
            elif team_code == 1:
                # 敌方: 用对方队伍颜色
                color = 'deepskyblue' if own_team == 'red' else 'red'
                edge_color = 'yellow'
                marker_size = 10
            else:
                # unknown: 灰色
                color = 'gray'
                edge_color = 'gray'
                marker_size = 7

            # 绘制机器人 (圆形标记)
            ax.plot(x, y, marker='o', color=color,
                    markersize=marker_size,
                    markeredgecolor=edge_color,
                    markeredgewidth=1.5, zorder=10)

            # 标注机器人 ID
            label = f'#{robot_id}'
            if team_code == 0:
                label += '(ally)'
            elif team_code == 1:
                label += '(enemy)'
            ax.annotate(label, (x, y),
                        textcoords="offset points", xytext=(0, -14),
                        ha='center', fontsize=6, color=edge_color)

    def _draw_info_panel(
        self,
        obs: Dict[str, Any],
        current_step: int,
        max_steps: int,
        reward: float,
        team: str,
        train_progress: Optional[Dict[str, Any]] = None,
    ):
        """绘制右侧信息面板, 列出观察空间各项参数

        Args:
            obs: 观察字典
            current_step: 当前步数
            max_steps: 最大步数
            reward: 最近一步的奖励
            team: 自身队伍
            train_progress: 训练进度信息字典
        """
        ax = self.ax_info
        ax.set_facecolor('#16213e')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')  # 隐藏坐标轴

        # 标题
        title_color = '#ff6b6b' if team == 'red' else '#4ecdc4'
        ax.text(0.5, 0.97, f'RoboMaster Gym Env',
                ha='center', va='top', fontsize=13,
                fontweight='bold', color=title_color,
                transform=ax.transAxes)

        # 队伍和步数
        ax.text(0.5, 0.92, f'Team: {team.upper()} | Step: {current_step}/{max_steps}',
                ha='center', va='top', fontsize=9, color='white',
                transform=ax.transAxes)

        # 进度条
        progress = current_step / max_steps if max_steps > 0 else 0
        bar_y = 0.885
        ax.barh(bar_y, progress, height=0.012, left=0.05,
                color=title_color, alpha=0.8, transform=ax.transAxes)
        ax.barh(bar_y, 1.0, height=0.012, left=0.05,
                color='gray', alpha=0.2, transform=ax.transAxes)

        # ---- 观察空间参数列表 ----
        # 每个参数一行, 格式: "参数名: 值"
        # y 坐标从上往下递减
        params = [
            ('--- Robot Status ---', None, 'yellow'),
            ('own_hp', obs.get('own_hp', '?'), '#ff6b6b'),
            ('own_ammo', obs.get('own_ammo', '?'), '#ffa502'),
            ('--- Game State ---', None, 'yellow'),
            ('team_economy', obs.get('team_economy', '?'), '#2ed573'),
            ('remaining_steps', obs.get('remaining_steps', '?'), '#70a1ff'),
            ('judge_countdown', obs.get('judge_countdown_steps', '?'), '#ff4757'),
            ('--- Combat ---', None, 'yellow'),
            ('damage_per_step', self._format_damage(obs.get('damage_per_step', '?')), '#eccc68'),
            ('outpost_hp', obs.get('outpost_hp', '?'), '#ff6348'),
            ('base_hp', obs.get('base_hp', '?'), '#ff6348'),
            ('base_exposed', obs.get('base_exposed', '?'), '#a4b0be'),
            ('ammo_consumed', obs.get('ammo_consumed_per_step', '?'), '#ffa502'),
            ('revive_wait', obs.get('revive_waiting_steps', '?'), '#747d8c'),
            ('--- Reward ---', None, 'yellow'),
            ('reward', f'{reward:.4f}', '#2ed573'),
        ]

        y_start = 0.84
        y_step = 0.042  # 每行间距

        for i, (name, value, color) in enumerate(params):
            y = y_start - i * y_step

            if value is None:
                # 分隔行
                ax.text(0.5, y, name, ha='center', va='top',
                        fontsize=8, fontweight='bold', color=color,
                        transform=ax.transAxes)
            else:
                # 参数行: 左边名称, 右边值
                ax.text(0.05, y, f'{name}:', ha='left', va='top',
                        fontsize=8, color='#a4b0be',
                        transform=ax.transAxes)
                ax.text(0.95, y, str(value), ha='right', va='top',
                        fontsize=8, fontweight='bold', color=color,
                        transform=ax.transAxes)

        # ---- all_robots 摘要 ----
        all_robots = obs.get('all_robots', None)
        if all_robots is not None:
            y_robots = y_start - len(params) * y_step - 0.02
            ax.text(0.5, y_robots, '--- Robots on Field ---',
                    ha='center', va='top', fontsize=8,
                    fontweight='bold', color='yellow',
                    transform=ax.transAxes)

            # 统计有效机器人数
            valid_mask = all_robots[:, 0] != -1  # id != -1 表示有效
            ally_count = np.sum((all_robots[:, 1] == 0) & valid_mask)
            enemy_count = np.sum((all_robots[:, 1] == 1) & valid_mask)
            unknown_count = np.sum((all_robots[:, 1] == -1) & valid_mask)

            y_detail = y_robots - 0.04
            ax.text(0.05, y_detail, f'Ally: {ally_count}', ha='left', va='top',
                    fontsize=8, color='#2ed573', transform=ax.transAxes)
            ax.text(0.35, y_detail, f'Enemy: {enemy_count}', ha='left', va='top',
                    fontsize=8, color='#ff4757', transform=ax.transAxes)
            ax.text(0.7, y_detail, f'Unknown: {unknown_count}', ha='left', va='top',
                    fontsize=8, color='gray', transform=ax.transAxes)

        # ---- 训练进度面板 ----
        if train_progress is not None:
            y_train = y_start - len(params) * y_step - 0.14
            ax.text(0.5, y_train, '--- Training Progress ---',
                    ha='center', va='top', fontsize=8,
                    fontweight='bold', color='#00ff88',
                    transform=ax.transAxes)

            ep = train_progress.get('episode', 0)
            num_ep = train_progress.get('num_episodes', 0)
            stage = train_progress.get('curriculum_stage', 1)
            avg_r = train_progress.get('avg_reward', 0.0)
            best_r = train_progress.get('best_reward', 0.0)
            paused = train_progress.get('paused', False)

            y_tp = y_train - 0.04
            # Episode 进度
            ax.text(0.05, y_tp, f'Episode:', ha='left', va='top',
                    fontsize=8, color='#a4b0be', transform=ax.transAxes)
            ax.text(0.95, y_tp, f'{ep}/{num_ep}', ha='right', va='top',
                    fontsize=8, fontweight='bold', color='#00ff88',
                    transform=ax.transAxes)

            # Episode 进度条
            y_tp -= 0.025
            ep_progress = ep / num_ep if num_ep > 0 else 0
            ax.barh(y_tp, ep_progress, height=0.012, left=0.05,
                    color='#00ff88', alpha=0.8, transform=ax.transAxes)
            ax.barh(y_tp, 1.0, height=0.012, left=0.05,
                    color='gray', alpha=0.2, transform=ax.transAxes)

            # 课程学习阶段
            y_tp -= 0.035
            stage_names = {1: 'Near(3-6m)', 2: 'Mid(6-12m)',
                           3: 'Far(12-20m)', 4: 'Full(3-25m)'}
            stage_name = stage_names.get(stage, f'Stage {stage}')
            ax.text(0.05, y_tp, f'Curriculum:', ha='left', va='top',
                    fontsize=8, color='#a4b0be', transform=ax.transAxes)
            ax.text(0.95, y_tp, stage_name, ha='right', va='top',
                    fontsize=8, fontweight='bold', color='#ffa502',
                    transform=ax.transAxes)

            # 平均奖励
            y_tp -= 0.04
            ax.text(0.05, y_tp, f'Avg Reward:', ha='left', va='top',
                    fontsize=8, color='#a4b0be', transform=ax.transAxes)
            ax.text(0.95, y_tp, f'{avg_r:.2f}', ha='right', va='top',
                    fontsize=8, fontweight='bold', color='#2ed573',
                    transform=ax.transAxes)

            # 最佳奖励
            y_tp -= 0.04
            ax.text(0.05, y_tp, f'Best Reward:', ha='left', va='top',
                    fontsize=8, color='#a4b0be', transform=ax.transAxes)
            ax.text(0.95, y_tp, f'{best_r:.2f}', ha='right', va='top',
                    fontsize=8, fontweight='bold', color='#ff6b6b',
                    transform=ax.transAxes)

            # 暂停状态
            if paused:
                y_tp -= 0.05
                ax.text(0.5, y_tp, '[ PAUSED - Press P to Resume ]',
                        ha='center', va='top', fontsize=9,
                        fontweight='bold', color='#ff4757',
                        transform=ax.transAxes)

            # PPO更新状态
            ppo_updating = train_progress.get('ppo_updating', False)
            if ppo_updating:
                y_tp -= 0.05
                ax.text(0.5, y_tp, '[ PPO Updating... ]',
                        ha='center', va='top', fontsize=9,
                        fontweight='bold', color='#ffa502',
                        transform=ax.transAxes)

    @staticmethod
    def _format_damage(value) -> str:
        """格式化 damage_per_step (可能是数组)"""
        if isinstance(value, np.ndarray):
            return f'{float(value.squeeze()):.1f}'
        elif isinstance(value, (int, float)):
            return f'{float(value):.1f}'
        return str(value)

    def close(self):
        """关闭渲染器, 释放资源"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
