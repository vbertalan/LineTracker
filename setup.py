from setuptools import setup, find_packages

setup(
    name='linetracker',
    version='0.1.0',
    packages=['find_packages()'],
    install_requires=[
        # List your dependencies here
    ],
    entry_points={
        'console_scripts': [
            'my_script = my_package.my_module:main',
        ],
    },
    description='Allow to group log lines together',
    url='https://github.com/vbertalan/LineTracker',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: POSIX :: Linux',
    ],
)