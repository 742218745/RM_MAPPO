#!/usr/bin/env python3
"""
测试训练好的 MAPPO 模型效果 (离散动作空间版本)

使用方法:
    python3 test_model.py --checkpoint checkpoints_nav_train/mappo_best.pt
    python3 test_model.py --checkpoint checkpoints_nav_train/mappo_ep550.pt
"""
import os
import sys
import argparse
import numpy as np
import torch
import time
from collections import defaultdict

# 设置 DDS 环境变量
os.environ['ROS_DISABLE_FASTRTPS_SHM'] = '1'
os.environ['RMW_IMPLEMENTATION'] = 'rmw_fastrtps_cpp'

# Source 工作空间
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'install', 'lib', 'python3.10', 'site-packages'))

from robomaster_gym_env import RoboMasterGazeboEnv, GymEnvConfig
from robomaster_mappo.actor import MAPPOActor


def load_model(checkpoint_path: str, device: str = 'auto'):
    """加载训练好的模型"""
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"[加载模型] 设备: {device}")
    print(f"[加载模型] 检查点: {checkpoint_path}")
    
    actor = MAPPOActor(
        robot_embed_dim=64,
        state_embed_dim=64,
        hidden_dim=128,
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'actor_state_dict' in checkpoint:
        actor.load_state_dict(checkpoint['actor_state_dict'])
        print(f"[加载模型] 训练回合: {checkpoint.get('episode', 'unknown')}")
        print(f"[加载模型] 最佳平均奖励: {checkpoint.get('best_avg_reward', 'unknown')}")
    else:
        actor.load_state_dict(checkpoint)
    
    actor.eval()
    print("[加载模型] 成功!\n")
    
    return actor, device


def test_episode(env, actor, device, episode_num, deterministic=True, verbose=True):
    """测试一个回合"""
    
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0
    episode_info = defaultdict(list)
    
    start_pos = env._get_own_position_safe()
    if start_pos is None or start_pos[0] is None:
        start_pos = (8.64, 3.65)
    target_pos = (14.0, 7.5)
    waypoint_pos = (4.81, 2.47)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"测试回合 {episode_num}")
        print(f"{'='*60}")
        print(f"起点: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
        print(f"中间点: ({waypoint_pos[0]:.2f}, {waypoint_pos[1]:.2f})")
        print(f"目标: ({target_pos[0]:.2f}, {target_pos[1]:.2f})")
        print(f"{'-'*60}")
    
    while not done:
        with torch.no_grad():
            action, log_prob = actor.get_action(obs, deterministic=deterministic, device=device)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        
        current_pos = env._get_own_position_safe()
        if current_pos is None or current_pos[0] is None:
            current_pos = start_pos
        dist_to_target = np.sqrt((current_pos[0]-target_pos[0])**2 + (current_pos[1]-target_pos[1])**2)
        dist_to_waypoint = np.sqrt((current_pos[0]-waypoint_pos[0])**2 + (current_pos[1]-waypoint_pos[1])**2)
        
        # 获取实际速度 (从动作索引映射)
        vel_levels = [-2.0, -1.0, 0.0, 1.0, 2.0]
        vel_idx = action['chassis_velocity']
        actual_vx = vel_levels[int(vel_idx[0])]
        actual_vy = vel_levels[int(vel_idx[1])]
        
        episode_info['positions'].append(current_pos)
        episode_info['rewards'].append(reward)
        episode_info['dist_to_target'].append(dist_to_target)
        episode_info['dist_to_waypoint'].append(dist_to_waypoint)
        episode_info['velocities'].append((actual_vx, actual_vy))
        
        if verbose and step_count % 100 == 0:
            # 出界检测
            boundary_margin = 1.5
            out_of_bounds = (current_pos[0] < boundary_margin or 
                           current_pos[0] > 28 - boundary_margin or 
                           current_pos[1] < boundary_margin or 
                           current_pos[1] > 15 - boundary_margin)
            bound_warn = " ⚠️出界!" if out_of_bounds else ""
            
            print(f"Step {step_count:4d} | pos=({current_pos[0]:5.2f}, {current_pos[1]:5.2f}) | "
                  f"dist_wp={dist_to_waypoint:5.2f}m | dist_tgt={dist_to_target:5.2f}m | "
                  f"vel=({actual_vx:5.1f}, {actual_vy:5.1f}) | "
                  f"r={reward:6.2f} | sum_r={total_reward:7.2f}{bound_warn}")
    
    final_pos = episode_info['positions'][-1] if episode_info['positions'] else start_pos
    min_dist_wp = min(episode_info['dist_to_waypoint']) if episode_info['dist_to_waypoint'] else 0
    min_dist_tgt = min(episode_info['dist_to_target']) if episode_info['dist_to_target'] else 0
    final_dist_wp = episode_info['dist_to_waypoint'][-1] if episode_info['dist_to_waypoint'] else 0
    final_dist_tgt = episode_info['dist_to_target'][-1] if episode_info['dist_to_target'] else 0
    
    result = {
        'total_reward': total_reward,
        'step_count': step_count,
        'start_pos': start_pos,
        'final_pos': final_pos,
        'min_dist_to_waypoint': min_dist_wp,
        'min_dist_to_target': min_dist_tgt,
        'final_dist_to_waypoint': final_dist_wp,
        'final_dist_to_target': final_dist_tgt,
        'reached_waypoint': min_dist_wp < 1.0,
        'reached_target': min_dist_tgt < 2.0,
        'terminated': terminated,
        'truncated': truncated,
    }
    
    if verbose:
        print(f"{'-'*60}")
        print(f"回合结束:")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  步数: {step_count}")
        print(f"  最终位置: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
        print(f"  最小距离(中间点): {min_dist_wp:.2f}m")
        print(f"  最小距离(目标): {min_dist_tgt:.2f}m")
        print(f"  到达中间点: {'✅ 是' if result['reached_waypoint'] else '❌ 否'}")
        print(f"  到达目标: {'✅ 是' if result['reached_target'] else '❌ 否'}")
        print(f"  结束原因: {'终止(翻车/出界)' if terminated else '截断(超时)'}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='测试训练好的 MAPPO 模型')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints_nav_train/mappo_best.pt',
                       help='模型检查点路径')
    parser.add_argument('--num_episodes', type=int, default=3,
                       help='测试回合数')
    parser.add_argument('--deterministic', action='store_true',
                       help='使用确定性策略')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备: cpu, cuda, auto')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出')
    
    args = parser.parse_args()
    
    checkpoint_path = os.path.join(script_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到检查点文件 {checkpoint_path}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("MAPPO 模型测试 (离散动作空间)")
    print("="*60)
    
    actor, device = load_model(checkpoint_path, args.device)
    
    print("[创建环境] 初始化中...")
    config = GymEnvConfig()
    env = RoboMasterGazeboEnv(config=config)
    print(f"[创建环境] 动作空间: {env.action_space}")
    print("[创建环境] 成功!\n")
    
    results = []
    for i in range(args.num_episodes):
        result = test_episode(
            env, actor, device, 
            episode_num=i+1,
            deterministic=args.deterministic,
            verbose=args.verbose
        )
        results.append(result)
    
    # 统计结果
    print("\n" + "="*60)
    print("测试统计")
    print("="*60)
    
    avg_reward = np.mean([r['total_reward'] for r in results])
    avg_steps = np.mean([r['step_count'] for r in results])
    avg_min_dist_wp = np.mean([r['min_dist_to_waypoint'] for r in results])
    avg_min_dist_tgt = np.mean([r['min_dist_to_target'] for r in results])
    wp_rate = np.mean([r['reached_waypoint'] for r in results]) * 100
    tgt_rate = np.mean([r['reached_target'] for r in results]) * 100
    term_rate = np.mean([r['terminated'] for r in results]) * 100
    
    print(f"测试回合数: {args.num_episodes}")
    print(f"平均总奖励: {avg_reward:.2f}")
    print(f"平均步数: {avg_steps:.1f}")
    print(f"平均最小距离(中间点): {avg_min_dist_wp:.2f}m")
    print(f"平均最小距离(目标): {avg_min_dist_tgt:.2f}m")
    print(f"到达中间点成功率: {wp_rate:.1f}%")
    print(f"到达目标成功率: {tgt_rate:.1f}%")
    print(f"异常终止率: {term_rate:.1f}%")
    
    print("\n各回合详情:")
    for i, r in enumerate(results):
        wp = "✅" if r['reached_waypoint'] else "❌"
        tgt = "✅" if r['reached_target'] else "❌"
        print(f"  回合 {i+1}: reward={r['total_reward']:7.2f}, "
              f"steps={r['step_count']:4d}, "
              f"min_wp={r['min_dist_to_waypoint']:5.2f}m, "
              f"min_tgt={r['min_dist_to_target']:5.2f}m, "
              f"wp={wp} tgt={tgt}")
    
    env.close()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n测试出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
