from setuptools import setup, find_packages

setup(
    name='robomaster_mappo',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
    ],
    description='MAPPO reinforcement learning for RoboMaster',
    entry_points={
        'console_scripts': [
            'mappo_train = robomaster_mappo.train:main',
        ],
    },
)
