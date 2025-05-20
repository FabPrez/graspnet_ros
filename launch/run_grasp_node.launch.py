#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_prefix, get_package_share_directory

def generate_launch_description():
    pkg_prefix = get_package_share_directory('graspnet_ros')
    venv_path = os.path.join(pkg_prefix, 'venv_graspnet')
    python_bin = os.path.join(venv_path, 'bin', 'python')

    return LaunchDescription([
        Node(
            package='graspnet_ros',
            executable='graspnet_node.py',
            name='graspnet_node',
            output='screen',
            prefix=f'{python_bin}'
        )
    ])
