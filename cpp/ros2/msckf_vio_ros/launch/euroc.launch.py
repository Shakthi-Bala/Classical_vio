"""Launch the VIO node for EuRoC bags (/imu0, /cam0/image_raw, /cam1/image_raw).

  ros2 launch msckf_vio_ros euroc.launch.py config:=/path/to/euroc_fixed.yaml backend:=swo
  ros2 bag play MH_01_easy_ros2
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('config', description='VIO YAML (cpp/config/*.yaml)'),
        DeclareLaunchArgument('backend', default_value='msckf', description='msckf or swo'),
        DeclareLaunchArgument('output_dir', default_value='', description='TUM + timing output'),
        DeclareLaunchArgument('skip_seconds', default_value='0.0',
                              description='ignore data before first IMU stamp + this'),
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
            'skip_seconds': LaunchConfiguration('skip_seconds'),
        }],
        remappings=[('imu0', '/imu0'), ('cam0/image_raw', '/cam0/image_raw'),
                    ('cam1/image_raw', '/cam1/image_raw')],
    )
    return LaunchDescription(args + [node])
