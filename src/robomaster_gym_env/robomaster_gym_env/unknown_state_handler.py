"""
Unknown状态处理模块

该模块定义了所有观察项的unknown状态常量和处理方法。
用于处理数据缺失或无效的情况，增强系统容错性。

Unknown状态标识规则:
- float类型: 使用 np.nan
- int类型: 使用 -1
- 数组类型: id=-1 表示无效项
"""

import numpy as np
from typing import Any, Union


class UnknownStateHandler:
    """Unknown状态处理器

    提供统一的unknown状态定义、检测和处理方法。
    """

    # ==================== Unknown状态常量定义 ====================

    # float类型的unknown值
    UNKNOWN_FLOAT: float = np.nan

    # int类型的unknown值
    UNKNOWN_INT: int = -1

    # 位置坐标的unknown值 [x, y, yaw]
    UNKNOWN_POSITION: np.ndarray = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    # 单个机器人的unknown值 [id, team, x, y]
    # id=-1 表示该位置无效
    UNKNOWN_ROBOT: np.ndarray = np.array([-1, -1, np.nan, np.nan], dtype=np.float32)

    # 可移动范围的unknown值（空数组）
    UNKNOWN_RANGE: np.ndarray = np.array([], dtype=np.float32)

    # ==================== 检测方法 ====================

    @staticmethod
    def is_unknown(value: Any) -> bool:
        """检查值是否为unknown状态

        Args:
            value: 待检查的值，可以是float、int、np.ndarray等

        Returns:
            bool: True表示是unknown状态，False表示不是

        Examples:
            >>> UnknownStateHandler.is_unknown(np.nan)
            True
            >>> UnknownStateHandler.is_unknown(-1)
            True
            >>> UnknownStateHandler.is_unknown(100)
            False
        """
        if value is None:
            return True

        if isinstance(value, (float, np.floating)):
            return np.isnan(value)

        if isinstance(value, (int, np.integer)):
            return value == UnknownStateHandler.UNKNOWN_INT

        if isinstance(value, np.ndarray):
            if value.size == 0:
                return True
            # 检查数组中是否包含nan或-1
            if np.issubdtype(value.dtype, np.floating):
                return np.any(np.isnan(value))
            elif np.issubdtype(value.dtype, np.integer):
                return np.any(value == UnknownStateHandler.UNKNOWN_INT)

        return False

    @staticmethod
    def is_robot_unknown(robot_data: np.ndarray) -> bool:
        """检查机器人数据是否为unknown状态

        Args:
            robot_data: 机器人数据数组 [id, team, x, y]

        Returns:
            bool: True表示是unknown状态（id=-1），False表示不是
        """
        if robot_data.size == 0:
            return True
        return robot_data[0] == UnknownStateHandler.UNKNOWN_INT

    # ==================== 获取方法 ====================

    @staticmethod
    def get_unknown_value(data_type: str) -> Any:
        """获取指定类型的unknown值

        Args:
            data_type: 数据类型，可选值：
                - 'float': float类型
                - 'int': int类型
                - 'position': 位置坐标 [x, y, yaw]
                - 'robot': 机器人数据 [id, team, x, y]
                - 'range': 可移动范围

        Returns:
            对应类型的unknown值

        Raises:
            ValueError: 不支持的数据类型

        Examples:
            >>> UnknownStateHandler.get_unknown_value('float')
            nan
            >>> UnknownStateHandler.get_unknown_value('int')
            -1
        """
        type_mapping = {
            'float': UnknownStateHandler.UNKNOWN_FLOAT,
            'int': UnknownStateHandler.UNKNOWN_INT,
            'position': UnknownStateHandler.UNKNOWN_POSITION,
            'robot': UnknownStateHandler.UNKNOWN_ROBOT,
            'range': UnknownStateHandler.UNKNOWN_RANGE,
        }

        if data_type not in type_mapping:
            raise ValueError(f"不支持的数据类型: {data_type}，支持的类型: {list(type_mapping.keys())}")

        return type_mapping[data_type]

    # ==================== 处理方法 ====================

    @staticmethod
    def handle_missing_data(data: Any, data_type: str) -> Any:
        """处理缺失数据，返回对应类型的unknown值

        Args:
            data: 原始数据，可能为None或无效
            data_type: 数据类型

        Returns:
            如果数据有效则返回原数据，否则返回对应类型的unknown值
        """
        if data is None or UnknownStateHandler.is_unknown(data):
            return UnknownStateHandler.get_unknown_value(data_type)
        return data

    @staticmethod
    def create_unknown_robot_array(num_robots: int = 10) -> np.ndarray:
        """创建包含unknown机器人的数组

        Args:
            num_robots: 机器人数量，默认为10

        Returns:
            np.ndarray: 形状为 (num_robots, 4) 的数组，所有项都是unknown状态
        """
        return np.array([UnknownStateHandler.UNKNOWN_ROBOT] * num_robots, dtype=np.float32)

    @staticmethod
    def validate_and_handle(
        value: Any,
        data_type: str,
        valid_range: tuple = None
    ) -> tuple[bool, Any]:
        """验证数据并处理unknown状态

        Args:
            value: 待验证的值
            data_type: 数据类型
            valid_range: 有效范围 (min, max)，仅对数值类型有效

        Returns:
            tuple[bool, Any]: (是否有效, 处理后的值)
                - 如果有效，返回 (True, 原值)
                - 如果无效，返回 (False, unknown值)
        """
        # 检查是否为None或unknown
        if value is None or UnknownStateHandler.is_unknown(value):
            return False, UnknownStateHandler.get_unknown_value(data_type)

        # 检查数值范围
        if valid_range is not None and isinstance(value, (int, float, np.number)):
            min_val, max_val = valid_range
            if not (min_val <= value <= max_val):
                return False, UnknownStateHandler.get_unknown_value(data_type)

        return True, value
