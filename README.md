# graspnet_ros

ROS 2 wrapper for [graspnet-baseline](https://github.com/graspnet/graspnet-baseline.git).
I have forked also a stable version for ubuntu22.04 and ros2 humble for graspnet-baseline, there are some minor fixes -> [graspnet-baseline](https://github.com/FabPrez/graspnet-baseline)

## Installation

Move to your ROS 2 workspace and clone the repository:

```bash
cd src
git clone https://github.com/FabPrez/graspnet_ros.git
```

Move to the scripts folder and create the virtual environment:

```bash
cd graspnet_ros/scripts
python3 -m venv venv_graspnet
```
<details>
<summary><strong>Setting of venv for ROS</strong></summary>

To use the virtual environment inside your ROS 2 workspace:

1. Open the `setup.cfg` file of the package.
2. Add the following section (or edit it if it already exists):

   ```ini
   [build_scripts]
   executable = path/to/your/venv_graspnet/bin/env python3
   ```
</details>

Inside the same folder, clone the [graspnet-baseline](https://github.com/graspnet/graspnet-baseline.git) repository and install following its ReadMe in the virtual environment.
Remember to activate the virtual environment before installing the requirements:
```bash
source venv_graspnet/bin/activate
cd graspnet-baseline
pip install -r requirements.txt
```


<details>

Inside the same folder, clone the [graspnet-baseline](https://github.com/graspnet/graspnet-baseline.git) repository and install following its ReadMe in the virtual environment.
Remember to activate the virtual environment before installing the requirements:
```bash
source venv_graspnet/bin/activate
cd graspnet-baseline
pip install -r requirements.txt
```

<summary><strong>Troubleshooting during graspnet-baseline installation</strong></summary>

- Use the correct version of PyTorch for your CUDA version.
- ERRORS during installing pointnet2 and knn: Set the compilers GCC 9 and G++ 9 before running the installation:
    ```bash
    export CC=gcc-9
    export CXX=g++-9
    ```
- Typeguard may miss in the requirements, so install it by yourself: "pip install typeguard"
- The checkpoint folder should not be extracted. Leave it as a compressed file in the graspnet-baseline folder.
- ERROR "no modul name Torch._six": Inside graspnet-baseline/dataset/_dataset.py substitute:
    ```bash
    from torch.six import container_abcs
    ```
    with:
   ```bash
    import collections.abc as container_abcs
    ```
</details>


Finally, return to your ROS 2 workspace root, deactivate the venv, and build the package:
```bash
cd ~/your_ros_ws
colcon build --symlink-install
source install/local_setup.bash
```

## Quick Start

To launch the ROS 2 node, you do not need to acrtivate the venv, simply run:

```bash
ros2 launch graspnet_ros run_grasp_node.launch.py
```

