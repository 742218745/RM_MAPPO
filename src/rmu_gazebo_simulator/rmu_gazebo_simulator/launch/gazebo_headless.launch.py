# Copyright 2025 Lihan Chen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
)
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    pkg_simulator = get_package_share_directory("rmu_gazebo_simulator")

    world_sdf_path = LaunchConfiguration("world_sdf_path")

    declare_world_sdf_path = DeclareLaunchArgument(
        "world_sdf_path",
        default_value=os.path.join(
            pkg_simulator, "resource", "worlds", "rmul_2024_world.sdf"
        ),
        description="Path to the world SDF file",
    )

    # 启动Gazebo服务器 (无GUI)
    # 使用 ign gazebo -s 启动服务器模式, -r 立即运行仿真
    # 无UI模式需要Xvfb提供虚拟X显示(在start_sim_headless.sh中启动)
    gazebo_server = ExecuteProcess(
        cmd=[
            'ign', 'gazebo', '-s',  # -s: server only, no GUI
            '-r',  # -r: run simulation
            world_sdf_path,
        ],
        output='screen',
    )

    # 时钟桥接
    robot_ign_bridge = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        output='screen',
    )

    # 延迟auto-play: 等待Gazebo完全加载后发送play命令
    # 与有GUI模式一致, 确保仿真正确启动
    auto_play = ExecuteProcess(
        cmd=[
            'bash',
            os.path.join(pkg_simulator, 'launch', 'gazebo_auto_play.sh'),
            '10',  # delay seconds
        ],
        output='screen',
    )

    ld = LaunchDescription()

    ld.add_action(declare_world_sdf_path)
    ld.add_action(gazebo_server)
    ld.add_action(robot_ign_bridge)
    ld.add_action(auto_play)

    return ld
