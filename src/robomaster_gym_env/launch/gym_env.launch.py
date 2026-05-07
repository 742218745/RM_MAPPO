"""
Launch文件: 启动Gym环境测试
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """生成launch描述"""

    # 参数
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='red_standard_robot1',
        description='Robot name'
    )

    team_arg = DeclareLaunchArgument(
        'team',
        default_value='red',
        description='Team color (red or blue)'
    )

    # 测试节点
    test_node = Node(
        package='robomaster_gym_env',
        executable='gym_test_node',
        name='gym_test_node',
        output='screen',
        parameters=[{
            'robot_name': LaunchConfiguration('robot_name'),
            'team': LaunchConfiguration('team'),
        }]
    )

    return LaunchDescription([
        robot_name_arg,
        team_arg,
        test_node,
    ])
