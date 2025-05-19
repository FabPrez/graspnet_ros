from setuptools import setup

package_name = 'graspnet_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Tuo Nome',
    maintainer_email='tuo@email.com',
    description='Nodo graspnet con ROS 2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'graspnet_node = scripts.graspnet_node:main'
        ],
    },
)
