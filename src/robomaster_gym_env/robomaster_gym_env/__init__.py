"""
RoboMaster Gazebo Gym Environment
基于ROS2 Gazebo仿真的RoboMaster强化学习环境
"""

from .robomaster_env import RoboMasterGazeboEnv
from .multi_agent_env import RoboMasterMultiAgentEnv
from .ros2_interface import ROS2Interface
from .observation_space import ObservationSpace, ObservationConfig
from .action_space import ActionSpace
from .config import (
    GymEnvConfig,
    DEFAULT_GYM_CONFIG,
)
from .unknown_state_handler import UnknownStateHandler
from .interface_adapter import InterfaceAdapter
from .data_processor import DataProcessor
from .env_renderer import EnvRenderer

__version__ = "2.0.0"
__all__ = [
    # 环境
    "RoboMasterGazeboEnv",
    "RoboMasterMultiAgentEnv",
    # 接口
    "ROS2Interface",
    # 观察空间
    "ObservationSpace",
    "ObservationConfig",
    # 动作空间
    "ActionSpace",
    # 配置
    "GymEnvConfig",
    "DEFAULT_GYM_CONFIG",
    # 新模块
    "UnknownStateHandler",
    "InterfaceAdapter",
    "DataProcessor",
    "EnvRenderer",
]
