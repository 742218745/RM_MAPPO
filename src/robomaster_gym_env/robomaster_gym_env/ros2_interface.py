"""
ROS2接口管理器
负责与ROS2节点的通信,包括订阅、发布、服务调用等
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading
import time
from typing import Dict, Any, Optional, Callable, Tuple
import numpy as np

# ROS2消息类型
from geometry_msgs.msg import Twist, TransformStamped, Pose, Accel
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
from example_interfaces.msg import UInt8
from tf2_msgs.msg import TFMessage

# 自定义消息类型 (需要根据实际包名调整)
try:
    from rmoss_interfaces.msg import (
        ChassisCmd, GimbalCmd, ShootCmd, Gimbal,
        RefereeCmd, RfidStatusArray,
        RobotStatus, GameStatus
    )
    from rmoss_interfaces.srv import ControlTask, ExchangeAmmon
    RMOSS_AVAILABLE = True
except ImportError:
    RMOSS_AVAILABLE = False
    print("Warning: rmoss_interfaces not available, using placeholder types")


class ROS2Interface:
    """ROS2接口管理器"""

    def __init__(
        self,
        robot_name: str,
        robot_namespace: str,
        team: str = "red",
        use_sim_time: bool = True
    ):
        """
        初始化ROS2接口

        Args:
            robot_name: 机器人名称
            robot_namespace: 机器人命名空间
            team: 队伍 ('red' or 'blue')
            use_sim_time: 是否使用仿真时间
        """
        self.robot_name = robot_name
        self.robot_namespace = robot_namespace
        self.team = team
        self.use_sim_time = use_sim_time

        # 初始化ROS2
        if not rclpy.ok():
            rclpy.init()

        # 创建节点
        self.node = Node(
            f'{robot_name}_gym_interface',
            automatically_declare_parameters_from_overrides=True
        )

        # 设置仿真时间
        if use_sim_time:
            try:
                self.node.declare_parameter('use_sim_time', True)
            except Exception:
                # 参数已声明,忽略
                pass
            self.node.set_parameters([
                rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
            ])

        # QoS配置
        # 修复: 使用RELIABLE以匹配Gazebo桥接的QoS配置
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 回调组
        self.callback_group = MutuallyExclusiveCallbackGroup()

        # 数据存储
        self.sensor_data = {}
        self.state_data = {}
        self.referee_data = {}

        # 自身队伍颜色缓存 (从仿真动态读取)
        self._own_team = team

        # 发布器字典
        self.publishers = {}

        # 订阅器字典
        self.subscribers = {}

        # 服务客户端字典
        self.service_clients = {}

        # 执行器和线程
        self.executor = None
        self.spin_thread = None
        self.is_spinning = False

        # 初始化通信
        self._init_publishers()
        self._init_subscribers()
        self._init_services()

    def _init_publishers(self):
        """初始化所有发布器"""
        ns = self.robot_namespace

        '''
        1. 底盘控制命令发布器

        线速度：
        linear.x: 前进/后退速度，正值前进，负值后退
        linear.y: 横向移动速度，正值向左，负值向右（全向底盘）
        linear.z: 垂直速度, 通常为0(地面机器人)
        角速度：(旋转)
        angular.x: roll角速度, 通常为0
        angular.y: pitch角速度, 通常为0
        angular.z: yaw角速度, 正值逆时针旋转，负值顺时针旋转
        '''
        self.publishers['cmd_vel'] = self.node.create_publisher(
            Twist,
            f'{ns}/cmd_vel',
            self.qos_profile
        )

        if RMOSS_AVAILABLE:
            self.publishers['chassis_cmd'] = self.node.create_publisher(
                ChassisCmd,
                f'{ns}/robot_base/chassis_cmd',
                self.qos_profile
            )


        '''
        2. 云台控制命令发布器
        
        yaw: yaw轴旋转角度值或速度值
        pitch: pitch轴旋转角度值或速度值
        yaw_type: yaw轴旋转控制类型(1-绝对角度, 2-相对角度, 3-旋转速度)
        pitch_type: pitch轴旋转控制类型(同上)
        '''
        if RMOSS_AVAILABLE:
            self.publishers['gimbal_cmd'] = self.node.create_publisher(
                GimbalCmd,
                f'{ns}/robot_base/gimbal_cmd',
                self.qos_profile
            )

        self.publishers['cmd_gimbal_joint'] = self.node.create_publisher(
            JointState,
            f'{ns}/cmd_gimbal_joint',
            self.qos_profile
        )


        '''
        3. 射击命令发布器

        type: 射击类型(1=单发, 2=连发等)
        projectile_num: 发射的弹丸数量
        projectile_velocity: 弹丸初速度
        '''
        if RMOSS_AVAILABLE:
            self.publishers['shoot_cmd'] = self.node.create_publisher(
                ShootCmd,
                f'{ns}/robot_base/shoot_cmd',
                self.qos_profile
            )

        self.publishers['cmd_shoot'] = self.node.create_publisher(
            UInt8,
            f'{ns}/cmd_shoot',
            self.qos_profile
        )

        # 4. 裁判系统命令发布器
        if RMOSS_AVAILABLE:
            self.publishers['referee_cmd'] = self.node.create_publisher(
                RefereeCmd,
                '/referee_system/referee_cmd',
                self.qos_profile
            )

        self.publishers['set_pose'] = self.node.create_publisher(
            TransformStamped,
            '/referee_system/set_pose',
            self.qos_profile
        )

    def _init_subscribers(self):
        """初始化所有订阅器"""
        ns = self.robot_namespace

        # 1. 状态数据订阅
        '''
        里程计(传感器估计, 存在误差)

        Child Frame ID(子坐标系ID): 通常是 "base_link"（机器人本体坐标系), 
            速度数据相对于此坐标系
        
        pose:    

        Position(位置):
        x: x 轴位置（前进方向，米）
        y: y 轴位置（左右方向，米）
        z: z 轴位置（垂直方向，米）
        单位：米
        参考系：相对于 frame_id 坐标系
        
        Orientation(姿态):
        x, y, z, w: 四元数表示的旋转
        单位：归一化四元数
        用途：表示机器人的朝向
        
        Covariance(协方差矩阵):
        pose.covariance[36]:6x6 协方差矩阵
        顺序：[x, y, z, roll, pitch, yaw]
        用途：表示位姿估计的不确定性

        twist:

        Linear Velocity(线速度):
        x: 前进速度
        y: 横向速度
        z: 垂直速度
        单位：米/秒
        参考系：相对于 child_frame_id 坐标系
        
        Angular Velocity(角速度):
        x: roll 角速度
        y: pitch 角速度
        z: yaw 角速度
        单位：弧度/秒

        Covariance(协方差矩阵):
        twist.covariance[36]: 6x6 协方差矩阵
        顺序：[vx, vy, vz, vroll, vpitch, vyaw]
        '''
        self.subscribers['odom'] = self.node.create_subscription(
            Odometry,
            f'{ns}/robot_base/odom',
            self._odom_callback,
            self.qos_profile,
            callback_group=self.callback_group
        )

        # 底盘里程计真值(同上, 但理想值)
        self.subscribers['chassis_odometry_gt'] = self.node.create_subscription(
            Odometry,
            f'{ns}/chassis_odometry_gt',
            self._chassis_odom_gt_callback,
            self.qos_profile,
            callback_group=self.callback_group
        )

        '''
        云台状态

        Yaw(偏航角):
        定义：云台相对于底盘的水平旋转角度
        单位：弧度
        范围：通常 -π ~ +π 弧度(-180° ~ +180°)
        正值：向左旋转
        用途：控制云台水平瞄准方向
        
        Pitch(俯仰角):
        定义：云台相对于水平面的俯仰角度
        单位：弧度
        范围：通常 -π/6 ~ +π/3 弧度(-30° ~ +60°)
        正值：向上仰起(抬头)
        用途：控制云台垂直瞄准方向
        '''
        if RMOSS_AVAILABLE:
            self.subscribers['gimbal_state'] = self.node.create_subscription(
                Gimbal,
                f'{ns}/robot_base/gimbal_state',
                self._gimbal_state_callback,
                self.qos_profile,
                callback_group=self.callback_group
            )

        # 3. 裁判系统数据订阅
        '''
        机器人状态

        id: 机器人唯一标识符(0-7)
        level: 机器人等级/类型
            0: 哨兵机器人
            1: 英雄机器人
            2: 工程机器人
            3: 步兵机器人
            4: 无人机
        name: 机器人名称字符串, 如 "red_standard_robot1"
        remain_hp: 当前剩余血量
        max_hp: 最大血量上限
        total_projectiles: 初始总弹丸数
        used_projectiles: 已发射的弹丸数
        hit_projectiles: 命中敌人的弹丸数
        '''
        if RMOSS_AVAILABLE:
            self.subscribers['robot_status'] = self.node.create_subscription(
                RobotStatus,
                f'/referee_system/{self.robot_name}/robot_status',
                self._robot_status_callback,
                self.qos_profile,
                callback_group=self.callback_group
            )

        # 电源使能(bool类型, False表示断电, 可用于判断死亡)
        self.subscribers['enable_power'] = self.node.create_subscription(
            Bool,
            f'/referee_system/{self.robot_name}/enable_power',
            self._enable_power_callback,
            self.qos_profile,
            callback_group=self.callback_group
        )

        # 控制使能(bool, 能否接收控制命令)
        self.subscribers['enable_control'] = self.node.create_subscription(
            Bool,
            f'/referee_system/{self.robot_name}/enable_control',
            self._enable_control_callback,
            self.qos_profile,
            callback_group=self.callback_group
        )

        '''
        RFID信息

        supplier_area_is_triggered:是否在补给区
        center_area_is_triggered:是否在中心区
        '''
        if RMOSS_AVAILABLE:
            self.subscribers['rfid_info'] = self.node.create_subscription(
                RfidStatusArray,
                '/referee_system/rfid_info',
                self._rfid_info_callback,
                self.qos_profile,
                callback_group=self.callback_group
            )
        
        '''
        攻击信息

        shooter_model_name: 射击者机器人模型名称("red_standard_robot1")
        shooter_name: 射击者名称("shooter")
        target_model_name: 目标机器人模型名称("blue_standard_robot1")
        target_link_name: 目标部位("armor_front")
        target_collision_name: 碰撞名称("target_collision")
        '''
        self.subscribers['attack_info'] = self.node.create_subscription(
            String,
            '/referee_system/attack_info',
            self._attack_info_callback,
            self.qos_profile,
            callback_group=self.callback_group
        )

        '''
        射击信息

        shooter_model_name: 射击者机器人模型名称("red_standard_robot1")
        shooter_name: 射击者名称("shooter")
        velocity: 弹丸初速度
        '''
        self.subscribers['shoot_info'] = self.node.create_subscription(
            String,
            '/referee_system/shoot_info',
            self._shoot_info_callback,
            self.qos_profile,
            callback_group=self.callback_group
        )

        # IMU数据订阅
        self.subscribers['imu'] = self.node.create_subscription(
            Imu,
            f'{ns}/livox/imu',
            self._imu_callback,
            self.qos_profile,
            callback_group=self.callback_group
        )

        # 翻车检测参数 (基于 IMU 数据内部判断, 不依赖外部节点)
        self._tumble_tilt_threshold = 0.785  # 姿态倾斜阈值 (rad), 约 45°
        self._tumble_accel_threshold = 15.0  # 加速度异常阈值 (m/s^2)
        self._tumble_consecutive_count = 0  # 连续超过阈值的次数
        self._tumble_consecutive_threshold = 5  # 连续超过阈值5次才判定翻车

        # 位姿信息
        self.subscribers['pose_info'] = self.node.create_subscription(
            TFMessage,
            '/referee_system/pose_info',
            self._pose_info_callback,
            self.qos_profile,
            callback_group=self.callback_group
        )

        # 前哨站状态
        if RMOSS_AVAILABLE:
            self.subscribers['outpost_status'] = self.node.create_subscription(
                RobotStatus,
                '/referee_system/outpost_status',
                self._outpost_status_callback,
                self.qos_profile,
                callback_group=self.callback_group
            )

        # 基地状态
        if RMOSS_AVAILABLE:
            self.subscribers['base_status'] = self.node.create_subscription(
                RobotStatus,
                '/referee_system/base_status',
                self._base_status_callback,
                self.qos_profile,
                callback_group=self.callback_group
            )

        # 游戏状态(包含经济等信息)
        if RMOSS_AVAILABLE:
            self.subscribers['game_status'] = self.node.create_subscription(
                GameStatus,
                '/referee_system/game_status',
                self._game_status_callback,
                self.qos_profile,
                callback_group=self.callback_group
            )

    def _init_services(self):
        """初始化服务客户端"""
        if RMOSS_AVAILABLE:
            # 弹丸兑换服务
            self.service_clients['exchange_ammo'] = self.node.create_client(
                ExchangeAmmon,
                '/exchange_ammo'
            )

            # 任务控制服务
            self.service_clients['control_task'] = self.node.create_client(
                ControlTask,
                f'{self.robot_namespace}/robot_base/control_task'
            )

    # ==================== 回调函数 ====================

    def _imu_callback(self, msg: Imu):
        """IMU回调"""
        self.sensor_data['imu'] = {
            'header': msg.header,
            'orientation': [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w
            ],
            'angular_velocity': [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ],
            'linear_acceleration': [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ],
            'timestamp': self._get_timestamp(msg.header)
        }

    def _odom_callback(self, msg: Odometry):
        """里程计回调"""
        self.state_data['odom'] = {
            'header': msg.header,
            'pose': {
                'position': [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z
                ],
                'orientation': [
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w
                ]
            },
            'twist': {
                'linear': [
                    msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.linear.z
                ],
                'angular': [
                    msg.twist.twist.angular.x,
                    msg.twist.twist.angular.y,
                    msg.twist.twist.angular.z
                ]
            },
            'timestamp': self._get_timestamp(msg.header)
        }

    def _chassis_odom_gt_callback(self, msg: Odometry):
        """底盘里程计真值回调"""
        self.state_data['chassis_odometry_gt'] = {
            'header': msg.header,
            'pose': {
                'position': [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z
                ],
                'orientation': [
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w
                ]
            },
            'twist': {
                'linear': [
                    msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.linear.z
                ],
                'angular': [
                    msg.twist.twist.angular.x,
                    msg.twist.twist.angular.y,
                    msg.twist.twist.angular.z
                ]
            },
            'timestamp': self._get_timestamp(msg.header)
        }

    def _gimbal_state_callback(self, msg):
        """云台状态回调"""
        self.state_data['gimbal_state'] = {
            'yaw': msg.yaw,
            'pitch': msg.pitch,
            'timestamp': time.time()  # Gimbal消息没有header，使用当前时间
        }

    def _robot_status_callback(self, msg):
        """机器人状态回调"""
        self.referee_data['robot_status'] = {
            'id': msg.id,
            'level': msg.level,
            'name': msg.name,
            'remain_hp': msg.remain_hp,
            'max_hp': msg.max_hp,
            'total_projectiles': msg.total_projectiles,
            'used_projectiles': msg.used_projectiles,
            'hit_projectiles': msg.hit_projectiles,
            'gt_tf': {
                'translation': [
                    msg.gt_tf.translation.x,
                    msg.gt_tf.translation.y,
                    msg.gt_tf.translation.z
                ],
                'rotation': [
                    msg.gt_tf.rotation.x,
                    msg.gt_tf.rotation.y,
                    msg.gt_tf.rotation.z,
                    msg.gt_tf.rotation.w
                ]
            },
            'timestamp': time.time()  # RobotStatus消息没有header，使用当前时间
        }

        # 从机器人名称中提取自身队伍颜色
        # 名称格式如 'red_standard_robot1' 或 'blue_standard_robot3'
        robot_name = msg.name if hasattr(msg, 'name') else ''
        if 'red' in robot_name.lower():
            self._own_team = 'red'
        elif 'blue' in robot_name.lower():
            self._own_team = 'blue'

    def _enable_power_callback(self, msg: Bool):
        """电源使能回调"""
        self.referee_data['enable_power'] = msg.data

    def _enable_control_callback(self, msg: Bool):
        """控制使能回调"""
        self.referee_data['enable_control'] = msg.data

    def _rfid_info_callback(self, msg):
        """RFID信息回调"""
        self.referee_data['rfid_info'] = {
            'status_array': msg.robot_rfid_status,  # 正确的字段名
            'timestamp': time.time()  # RfidStatusArray消息没有header，使用当前时间
        }

    def _attack_info_callback(self, msg: String):
        """攻击信息回调"""
        self.referee_data['attack_info'] = msg.data

    def _shoot_info_callback(self, msg: String):
        """射击信息回调"""
        self.referee_data['shoot_info'] = msg.data

    def _pose_info_callback(self, msg: TFMessage):
        """位姿信息回调"""
        self.referee_data['pose_info'] = msg.transforms

    def _outpost_status_callback(self, msg):
        """前哨站状态回调"""
        self.referee_data['outpost_status'] = {
            'remain_hp': msg.remain_hp,
            'max_hp': msg.max_hp,
            'timestamp': time.time()
        }

    def _base_status_callback(self, msg):
        """基地状态回调"""
        self.referee_data['base_status'] = {
            'remain_hp': msg.remain_hp,
            'max_hp': msg.max_hp,
            'base_exposed': getattr(msg, 'base_exposed', False),  # 基地展开状态
            'timestamp': time.time()
        }

    def _game_status_callback(self, msg):
        """游戏状态回调"""
        self.referee_data['game_status'] = {
            'game_status': msg.status,
            'remaining_time': msg.total_time - msg.current_time,
            'team_economy': getattr(msg, 'team_economy', 0),  # 队伍经济
            'judge_countdown': getattr(msg, 'judge_countdown', 0),  # 判负时间
            'timestamp': time.time()
        }

    # ==================== 辅助函数 ====================

    def _get_timestamp(self, header) -> float:
        """获取时间戳"""
        return header.stamp.sec + header.stamp.nanosec * 1e-9


    # ==================== 控制接口 ====================

    def send_chassis_velocity(self, linear_x: float, linear_y: float, angular_z: float):
        """
        发送底盘速度命令

        Args:
            linear_x: 前进速度 (m/s)
            linear_y: 横向速度 (m/s)
            angular_z: 旋转角速度 (rad/s)
        """
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.linear.y = float(linear_y)
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(angular_z)
        self.publishers['cmd_vel'].publish(msg)

    def send_gimbal_angle(self, yaw: float, pitch: float, yaw_type: int = 1, pitch_type: int = 1):
        """
        发送云台角度命令

        Args:
            yaw: yaw角度 (rad)
            pitch: pitch角度 (rad)
            yaw_type: yaw控制类型 (1=绝对角度, 2=相对角度, 3=速度)
            pitch_type: pitch控制类型
        """
        if RMOSS_AVAILABLE and 'gimbal_cmd' in self.publishers:
            msg = GimbalCmd()
            msg.tid = 0
            msg.yaw_type = yaw_type
            msg.pitch_type = pitch_type
            msg.position.yaw = yaw
            msg.position.pitch = pitch
            msg.velocity.yaw = 0.0
            msg.velocity.pitch = 0.0
            self.publishers['gimbal_cmd'].publish(msg)

    def send_shoot_command(self, projectile_num: int = 1, velocity: float = 25.0):
        """
        发送射击命令

        Args:
            projectile_num: 弹丸数量
            velocity: 弹丸初速度 (m/s)
        """
        if RMOSS_AVAILABLE and 'shoot_cmd' in self.publishers:
            msg = ShootCmd()
            msg.tid = 0
            msg.type = 1
            msg.projectile_num = projectile_num
            msg.projectile_velocity = velocity
            self.publishers['shoot_cmd'].publish(msg)

        # 同时发送简化命令
        msg_simple = UInt8()
        msg_simple.data = projectile_num
        self.publishers['cmd_shoot'].publish(msg_simple)

    def set_robot_pose(self, x: float, y: float, z: float, yaw: float):
        """
        设置机器人位姿

        Args:
            x, y, z: 位置坐标
            yaw: 偏航角
        """
        # 使用原始机器人名称(不带_0后缀)
        # Gazebo的set_pose接口通过原始名称映射到实际模型
        msg = TransformStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.child_frame_id = self.robot_name
        msg.transform.translation.x = x
        msg.transform.translation.y = y
        msg.transform.translation.z = z
        # 设置四元数 (简化,只考虑yaw)
        msg.transform.rotation.x = 0.0
        msg.transform.rotation.y = 0.0
        msg.transform.rotation.z = np.sin(yaw / 2)
        msg.transform.rotation.w = np.cos(yaw / 2)
        self.publishers['set_pose'].publish(msg)

    def reset_simulation(self, initial_x: float = 7.0, initial_y: float = 7.5, initial_yaw: float = 0.0):
        """重置 Gazebo 仿真状态

        通过 ROS2 命令通知 Gazebo 重置比赛和机器人位姿:
        1. 发送裁判系统重置命令 (START_PREPARATION → START_GAME)
        2. 发送机器人位姿重置命令

        Args:
            initial_x: 初始 x 坐标 (m)
            initial_y: 初始 y 坐标 (m)
            initial_yaw: 初始朝向 (rad)
        """
        # 1. 重置裁判系统: 直接发START_GAME重新开始比赛
        # 注意: 跳过START_PREPARATION, 因为它会导致Gazebo崩溃
        # PREPARATION会关闭电源+重置位姿+重新开启电源, 这个过程不稳定
        if RMOSS_AVAILABLE and 'referee_cmd' in self.publishers:
            # START_GAME: 开始新比赛 (如果已在比赛中则重新开始)
            start_msg = RefereeCmd()
            start_msg.cmd = RefereeCmd.START_GAME
            start_msg.robot_name = ''
            self.publishers['referee_cmd'].publish(start_msg)
            time.sleep(0.3)

        # 2. 重置机器人位姿 (回到初始位置, z=0.1 为底盘正常高度)
        self.set_robot_pose(initial_x, initial_y, 0.1, initial_yaw)

    def set_robot_pose_by_name(self, robot_name: str, x: float, y: float, z: float, yaw: float):
        """
        设置指定名称机器人的位姿(可用于设置蓝方等非自身机器人位置)

        Args:
            robot_name: 机器人名称(如 'blue_standard_robot1')
            x, y, z: 位置坐标
            yaw: 偏航角
        """
        # 使用原始机器人名称(不带_0后缀)
        # Gazebo的set_pose接口通过原始名称映射到实际模型
        msg = TransformStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.child_frame_id = robot_name
        msg.transform.translation.x = x
        msg.transform.translation.y = y
        msg.transform.translation.z = z
        msg.transform.rotation.x = 0.0
        msg.transform.rotation.y = 0.0
        msg.transform.rotation.z = np.sin(yaw / 2)
        msg.transform.rotation.w = np.cos(yaw / 2)
        self.publishers['set_pose'].publish(msg)

    # ==================== 数据获取接口 ====================

    def get_sensor_data(self) -> Dict[str, Any]:
        """获取所有传感器数据"""
        return self.sensor_data.copy()

    def get_state_data(self) -> Dict[str, Any]:
        """获取所有状态数据"""
        return self.state_data.copy()

    def get_referee_data(self) -> Dict[str, Any]:
        """获取所有裁判系统数据"""
        return self.referee_data.copy()

    def get_imu(self) -> Optional[Dict]:
        """获取IMU数据"""
        return self.sensor_data.get('imu')

    def is_tumbled(self) -> bool:
        """基于 IMU 数据判断是否翻车

        通过姿态角 (roll/pitch) 和加速度异常综合判断:
        1. roll 或 pitch 超过阈值 (默认 90°)
        2. 加速度异常 (重力方向严重偏离)
        3. 连续超过阈值 10 次才判定翻车 (避免瞬时抖动误判)

        Returns:
            bool: True 表示翻车
        """
        imu = self.sensor_data.get('imu')
        if imu is None:
            self._tumble_consecutive_count = 0
            return False

        try:
            # 从四元数提取 roll 和 pitch
            q = imu['orientation']
            x, y, z, w = q[0], q[1], q[2], q[3]

            # Roll
            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            # Pitch
            sinp = 2.0 * (w * y - z * x)
            sinp = max(-1.0, min(1.0, sinp))
            pitch = np.arcsin(sinp)

            # 姿态角检测
            tilt_exceeded = abs(roll) > self._tumble_tilt_threshold or abs(pitch) > self._tumble_tilt_threshold

            # 加速度异常检测
            a = imu['linear_acceleration']
            gravity_deviation = abs(a[2] - 9.8)
            accel_exceeded = gravity_deviation > self._tumble_accel_threshold

            # 连续超过阈值才判定翻车
            if tilt_exceeded or accel_exceeded:
                self._tumble_consecutive_count += 1
            else:
                self._tumble_consecutive_count = 0

            return self._tumble_consecutive_count >= self._tumble_consecutive_threshold

        except (IndexError, TypeError):
            self._tumble_consecutive_count = 0
            return False

    def get_odom(self) -> Optional[Dict]:
        """获取里程计数据"""
        return self.state_data.get('odom')

    def get_gimbal_state(self) -> Optional[Dict]:
        """获取云台状态"""
        return self.state_data.get('gimbal_state')

    def get_robot_status(self) -> Optional[Dict]:
        """获取机器人状态"""
        return self.referee_data.get('robot_status')

    def get_own_team(self) -> str:
        """获取自身队伍颜色

        优先从仿真中动态读取的值（由robot_status回调更新），
        如果尚未收到数据则返回初始化时的默认值。

        Returns:
            str: 自身队伍颜色 ('red' or 'blue')
        """
        return self._own_team

    def get_hp(self) -> int:
        """获取当前HP"""
        if 'robot_status' in self.referee_data:
            return self.referee_data['robot_status']['remain_hp']
        return 0

    def get_ammo_count(self) -> int:
        """获取剩余弹丸数"""
        if 'robot_status' in self.referee_data:
            status = self.referee_data['robot_status']
            return status['total_projectiles'] - status['used_projectiles']
        return 0

    def get_robot_position(self) -> Tuple[float, float, float]:
        """获取机器人位置坐标

        Returns:
            Tuple[float, float, float]: (x, y, z) 坐标，如果无法获取返回(0.0, 0.0, 0.0)
        """
        # 优先从里程计真值获取
        if 'chassis_odometry_gt' in self.state_data:
            odom = self.state_data['chassis_odometry_gt']
            position = odom['pose']['position']
            return position[0], position[1], position[2]

        # 其次从里程计获取
        if 'odom' in self.state_data:
            odom = self.state_data['odom']
            position = odom['pose']['position']
            return position[0], position[1], position[2]

        # 无法获取，返回(0, 0, 0)
        return 0.0, 0.0, 0.0

    # ==================== 生命周期管理 ====================

    def start_spinning(self):
        """启动ROS2 spinning
        
        不再使用后台线程的 executor.spin()，因为它会长时间持有 GIL
        阻塞主线程的 PyTorch 推理 (actor.get_action / critic.get_value)。
        
        改为在主线程中按需调用 spin_once() 处理回调:
        - 每次调用前由主线程主动触发
        - spin_once(timeout_sec=0) 非阻塞，只处理当前就绪的回调
        - 数据立即可用，不存在线程间同步延迟
        """
        if self.is_spinning:
            return

        from rclpy.executors import SingleThreadedExecutor
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.is_spinning = True

    def spin_once(self):
        """处理一次就绪的 ROS2 回调
        
        在主线程中调用，非阻塞。应在每次 env.step() 前调用
        以确保 DDS 消息被及时处理。
        """
        if self.executor is not None:
            try:
                self.executor.spin_once(timeout_sec=0)
            except Exception:
                pass

    def stop_spinning(self):
        """停止ROS2 spinning"""
        if not self.is_spinning:
            return

        if self.executor:
            self.executor.shutdown()

        self.is_spinning = False

    def destroy(self):
        """销毁接口"""
        self.stop_spinning()

        # 销毁所有订阅器
        for sub in self.subscribers.values():
            self.node.destroy_subscription(sub)

        # 销毁所有发布器
        for pub in self.publishers.values():
            self.node.destroy_publisher(pub)

        # 销毁所有服务客户端
        for client in self.service_clients.values():
            self.node.destroy_client(client)

        # 销毁节点
        self.node.destroy_node()

    def __del__(self):
        """析构函数"""
        self.destroy()
