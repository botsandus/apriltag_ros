from os import getenv
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import AnyLaunchDescriptionSource
ARGUMENTS = []

def generate_launch_description():
    use_sim_time = LaunchConfiguration(
        'use_sim_time', default=(getenv('SIMULATION') == 'true'))

    # april_tag_detector
    detector_launch_file = "/launch/elp_36h11.launch.yml"
    sim_detector_launch_file = "/launch/gazebo_elp_36h11.launch.yml"

    tag_format = "arri"

    # Define LaunchDescription variable
    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(IncludeLaunchDescription(AnyLaunchDescriptionSource(
        get_package_share_directory('apriltag_ros') +
        detector_launch_file),launch_arguments=[("tag_format", tag_format)],
        condition=UnlessCondition(use_sim_time)
    ))
    ld.add_action(IncludeLaunchDescription(AnyLaunchDescriptionSource(
        get_package_share_directory('apriltag_ros') +
        sim_detector_launch_file),launch_arguments=[("tag_format", tag_format)],
        condition=IfCondition(use_sim_time)
    ))
    return ld
