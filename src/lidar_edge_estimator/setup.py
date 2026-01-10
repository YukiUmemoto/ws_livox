from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'lidar_edge_estimator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # ★追加：launchファイルをinstall/shareへ配置
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),

        # ★追加：パラメータyamlをinstall/shareへ配置
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='agx-orin-07',
    maintainer_email='agx-orin-07@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'angle_bin_edge_estimator = lidar_edge_estimator.angle_bin_edge_estimator:main',
        ],
    },
)

