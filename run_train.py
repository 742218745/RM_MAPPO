#!/usr/bin/env python3
"""
训练启动脚本

重要: 启动前必须在shell中设置与Gazebo仿真完全一致的DDS环境变量!

终端1 - 启动仿真:
  source install/setup.bash
  export ROS_DISABLE_FASTRTPS_SHM=1
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  bash start_sim.sh

终端2 - 启动训练 (等仿真完全启动后):
  source install/setup.bash
  export ROS_DISABLE_FASTRTPS_SHM=1
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  python3 run_train.py
"""
import os
import sys

# 检查DDS配置是否与仿真一致
shm = os.environ.get('ROS_DISABLE_FASTRTPS_SHM', '')
rmw = os.environ.get('RMW_IMPLEMENTATION', '')
if shm != '1' or rmw != 'rmw_fastrtps_cpp':
    print("警告: DDS配置可能与Gazebo仿真不一致!")
    print(f"  当前: ROS_DISABLE_FASTRTPS_SHM={shm}, RMW_IMPLEMENTATION={rmw}")
    print(f"  建议: export ROS_DISABLE_FASTRTPS_SHM=1 && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp")
    print()

# 自动检测工作目录
script_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = os.path.join(script_dir, 'checkpoints_nav_train')
latest_checkpoint = os.path.join(checkpoint_dir, 'mappo_latest.pt')

# 自动检测 GPU
import torch
has_gpu = torch.cuda.is_available()
device = 'auto'  # 让 train.py 自动决定
if not has_gpu:
    print("[run_train] 未检测到 GPU, 将使用 CPU 推理 (速度约为 GPU 的 1/3)")
    print()

sys.argv = [
    'robomaster_mappo.train',
    '--num_episodes', '3000',           # 增加训练回合 (分阶段引导需更多episodes收敛)
    '--rollout_steps', '2048',
    '--ppo_epochs', '8',                # 略微降低加速训练
    '--minibatch_size', '64',           # 默认值
    '--log_interval', '10',
    '--save_interval', '50',
    '--checkpoint_dir', checkpoint_dir,
    '--device', device,
]

# 自动加载最新检查点继续训练
if os.path.exists(latest_checkpoint):
    sys.argv.extend(['--load_checkpoint', latest_checkpoint])

from robomaster_mappo.train import main
main()
