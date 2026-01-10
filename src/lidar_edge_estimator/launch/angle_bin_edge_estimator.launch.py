from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('lidar_edge_estimator')
    param_path = os.path.join(pkg_share, 'config', 'angle_bin_edge_estimator.yaml')

    node = Node(
        package='lidar_edge_estimator',
        executable='angle_bin_edge_estimator',
        name='angle_bin_edge_estimator',
        output='screen',
        parameters=[param_path],
    )

    return LaunchDescription([node])

