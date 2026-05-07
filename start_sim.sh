#!/bin/bash
# 启动仿真脚本 - 解决 libEGL 警告

# 设置软件渲染（消除 libEGL 警告）
# 如果遇到 GPU 渲染问题 (libEGL warning, Segmentation fault in rendering),
# 取消下面两行的注释，使用软件渲染
# export LIBGL_ALWAYS_SOFTWARE=1
# export MESA_GL_VERSION_OVERRIDE=3.3

# 禁用ROS2共享内存传输 (解决RTPS_TRANSPORT_SHM错误)
# 重要: 训练脚本也必须设置相同的环境变量, 否则新进程加入会导致Gazebo crash
export ROS_DISABLE_FASTRTPS_SHM=1
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Source ROS2 工作空间
source /home/xufurui/ros_ws/install/setup.bash

# 清理残留进程和共享内存
cleanup_before() {
    echo "清理残留进程和共享内存..."
    pkill -9 -f "ign gazebo" 2>/dev/null
    pkill -9 -f "gz sim" 2>/dev/null
    pkill -9 -f "ros_gz_bridge" 2>/dev/null
    pkill -9 -f "parameter_bridge" 2>/dev/null
    pkill -9 -f "rmua19_robot_base" 2>/dev/null
    pkill -9 -f "robot_state_publisher" 2>/dev/null
    pkill -9 -f "gazebo_auto_play" 2>/dev/null
    rm -rf /dev/shm/fastrtps_* 2>/dev/null
    sleep 2
}

# 退出时清理
cleanup_after() {
    echo ""
    echo "仿真退出，清理残留进程..."
    pkill -9 -f "ign gazebo" 2>/dev/null
    pkill -9 -f "gz sim" 2>/dev/null
    pkill -9 -f "ros_gz_bridge" 2>/dev/null
    pkill -9 -f "parameter_bridge" 2>/dev/null
    pkill -9 -f "rmua19_robot_base" 2>/dev/null
    pkill -9 -f "robot_state_publisher" 2>/dev/null
    rm -rf /dev/shm/fastrtps_* 2>/dev/null
}

cleanup_before
trap cleanup_after EXIT

# 启动仿真
echo "启动 Gazebo 仿真..."
# echo "已设置软件渲染模式，消除 libEGL 警告"
echo ""

ros2 launch rmu_gazebo_simulator bringup_sim.launch.py "$@"
