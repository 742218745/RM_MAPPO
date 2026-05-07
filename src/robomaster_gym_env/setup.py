from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robomaster_gym_env'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'shapely',
        'gymnasium',
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='OpenAI Gym interface for RoboMaster Gazebo simulation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gym_test_node = robomaster_gym_env.test.test_node:main',
            'sim_data_publisher = robomaster_gym_env.test.sim_data_publisher:main',
        ],
    },
)
