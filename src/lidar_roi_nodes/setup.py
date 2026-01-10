from setuptools import find_packages, setup

package_name = 'lidar_roi_nodes'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='agx-orin-07',
    maintainer_email='agx-orin-07@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pointcloud_perturber = lidar_roi_nodes.pointcloud_perturber:main',
            'important_roi_estimator = lidar_roi_nodes.important_roi_estimator:main',
        ],
    },
)
