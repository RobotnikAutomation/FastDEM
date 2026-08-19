"""Launch file for FastDEM elevation mapping node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _launch_setup(context):
    global_mapping = LaunchConfiguration('global_mapping').perform(context) == 'true'
    input_scan = LaunchConfiguration('input_scan').perform(context)

    # Package path
    pkg_share = FindPackageShare('fastdem_ros2')

    # Config file
    config_name = 'global_mapping.yaml' if global_mapping else 'local_mapping.yaml'
    rviz_name = 'fastdem_global.rviz' if global_mapping else 'fastdem_local.rviz'
    config_file = PathJoinSubstitution([pkg_share, 'config', config_name])
    rviz_config = PathJoinSubstitution([pkg_share, 'launch', 'rviz', rviz_name])
    collision_monitor_config = PathJoinSubstitution(
        [pkg_share, 'config', 'collision_monitor_fastdem.yaml']
    )

    # Node parameters
    node_params = {'config_file': config_file}
    if input_scan:
        node_params['input_scan'] = input_scan

    # FastDEM mapping node
    fastdem_node = Node(
        package='fastdem_ros2',
        executable='fastdem_node',
        name='fastdem',
        output='screen',
        parameters=[node_params],
    )

    # RViz2 (optional)
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('rviz')),
    )

    # Nav2 collision monitor: uses FastDEM drop cloud as obstacle source
    collision_monitor_node = Node(
        package='nav2_collision_monitor',
        executable='collision_monitor',
        name='collision_monitor',
        output='screen',
        parameters=[collision_monitor_config],
        condition=IfCondition(LaunchConfiguration('collision_monitor')),
    )

    lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_collision_monitor',
        output='screen',
        parameters=[{
            'autostart': True,
            'node_names': ['collision_monitor'],
        }],
        condition=IfCondition(LaunchConfiguration('collision_monitor')),
    )

    return [fastdem_node, rviz_node, collision_monitor_node, lifecycle_manager_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'global_mapping', default_value='false',
            description='Enable global (fixed-origin) mapping mode'),
        DeclareLaunchArgument(
            'input_scan', default_value='',
            description='Override input topic (empty = use config)'),
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description='Launch RViz2 for visualization'),
        DeclareLaunchArgument(
            'collision_monitor', default_value='true',
            description='Launch Nav2 collision monitor using FastDEM drop cloud'),
        OpaqueFunction(function=_launch_setup),
    ])
