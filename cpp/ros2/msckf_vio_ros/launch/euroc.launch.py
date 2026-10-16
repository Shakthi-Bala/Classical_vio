"""Launch the VIO node for EuRoC bags (/imu0, /cam0/image_raw, /cam1/image_raw).

  ros2 launch msckf_vio_ros euroc.launch.py config:=/path/to/euroc_fixed.yaml backend:=swo
  ros2 bag play MH_01_easy_ros2
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument('config', description='VIO YAML (cpp/config/*.yaml)'),
        DeclareLaunchArgument('backend', default_value='msckf', description='msckf or swo'),
        DeclareLaunchArgument('output_dir', default_value='', description='TUM + timing output'),
        DeclareLaunchArgument('skip_seconds', default_value='0.0',
                              description='ignore data before first IMU stamp + this'),
        DeclareLaunchArgument('opencv_threads', default_value='1'),
        DeclareLaunchArgument('reliable_qos', default_value='true',
                              description='false for drivers that publish best effort'),
    ]
    node = Node(
        package='msckf_vio_ros',
        executable='vio_node',
        name='msckf_vio',
        output='screen',
        parameters=[{
            'config': LaunchConfiguration('config'),
            'backend': LaunchConfiguration('backend'),
            'output_dir': LaunchConfiguration('output_dir'),
            # Typed explicitly: "skip_seconds:=40" would otherwise arrive as an int.
            'skip_seconds': ParameterValue(LaunchConfiguration('skip_seconds'), value_type=float),
            'opencv_threads': ParameterValue(LaunchConfiguration('opencv_threads'), value_type=int),
            'reliable_qos': ParameterValue(LaunchConfiguration('reliable_qos'), value_type=bool),
        }],
    )
    return LaunchDescription(args + [node])
