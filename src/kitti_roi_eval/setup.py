from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'kitti_roi_eval'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'params'), glob('params/*.yaml')),
        (f"share/{package_name}/params", glob("params/*.yaml")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='agx-orin-07',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'kitti_player_with_gt = kitti_roi_eval.kitti_player_with_gt:main',
            'roi_eval_iou = kitti_roi_eval.roi_eval_iou:main',
            "roi_min_logger = kitti_roi_eval.roi_min_logger:main",
            "offline_generate_gt_bbox = kitti_roi_eval.offline_generate_gt_bbox:main",
            'kitti_player_finish_first = kitti_roi_eval.kitti_player_finish_first:main',
            'roi_finish_logger = kitti_roi_eval.roi_finish_logger:main',
            'roi_cover_logger = kitti_roi_eval.roi_cover_logger:main',
        ],
    },
)
