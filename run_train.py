#!/usr/bin/env python3
"""
训练启动脚本

重要: 启动前必须在shell中设置与Gazebo仿真完全一致的DDS环境变量!

终端1 - 启动仿真:
  source /home/xufurui/ros_ws/install/setup.bash
  export ROS_DISABLE_FASTRTPS_SHM=1
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  bash /home/xufurui/ros_ws/start_sim.sh

终端2 - 启动训练 (等仿真完全启动后):
  source /home/xufurui/ros_ws/install/setup.bash
  export ROS_DISABLE_FASTRTPS_SHM=1
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  python3 /home/xufurui/ros_ws/run_train.py
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

sys.argv = [
    'robomaster_mappo.train',
    '--num_episodes', '1500',           # 增加训练回合
    '--rollout_steps', '2048',          # 适合 RTX 3060 6GB
    '--ppo_epochs', '8',                # 略微降低加速训练
    '--minibatch_size', '64',           # 默认值
    '--log_interval', '10',
    '--save_interval', '50',
    '--checkpoint_dir', '/home/xufurui/ros_ws/checkpoints_nav_train',
    # 自动加载最新检查点继续训练
    '--load_checkpoint', '/home/xufurui/ros_ws/checkpoints_nav_train/mappo_latest.pt',
]

from robomaster_mappo.train import main
main()
