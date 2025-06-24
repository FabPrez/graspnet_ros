#!/usr/bin/env python3
import os
import sys
import numpy as np
import parameters as params
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
import threading
from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from scipy.spatial.transform import Rotation
from nbv_interfaces.srv import UpdateInterestMap

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"




current_dir = os.path.dirname(os.path.abspath(__file__))
import graspnet_pipeline

class GraspNetNode(Node):
    def __init__(self):
        super().__init__('graspnet_node')
        
        self.subscription = self.create_subscription(
            PointCloud2, '/pc_octomap_filtered', self.pointcloud_callback, 10)
        self.get_logger().info("GraspNetNode started, listening for pointclouds...")
        self.cli = self.create_client(UpdateInterestMap, '/update_interest_map')
        self.req = UpdateInterestMap.Request()
        self.num_iterations = 0
        
        # Structure to store the best grasp and the score for each iteration
        self.best_grasp_history = []
        self.score_history = []
        

    def pointcloud_callback(self, msg):
        self.get_logger().info("Received PointCloud2")
        
        # Print the number of iterations
        print(f"{GREEN} ============== Iteration n.{self.num_iterations+1} ============== {RESET}", flush=True)

        pc = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

        if len(pc) == 0:
            self.get_logger().warn("Empty pointcloud received")
            return

        # Convert the list of tuples into a 2D array (N, 3)
        points = np.array([ [x, y, z] for x, y, z in pc ], dtype=np.float32)

        # Run the pipeline
        gg = graspnet_pipeline.run_graspnet_pipeline(points)
        
        grasp_pose = [] # the num_best_grasp I want to send to the service
        scores = [] # the scores of the grasps I want to send to the service
        num_grasps = min(len(gg), params.num_best_grasps)
        
        for k in range(num_grasps):
            grasp = gg[k]
            
            R = np.array(grasp.rotation_matrix)
            q = Rotation.from_matrix(R)
            q_xyzw = q.as_quat()
        
            p = Pose()
            p.position = Point(x = float(grasp.translation[0]), y = float(grasp.translation[1]), z = float(grasp.translation[2]))
            p.orientation = Quaternion(x = float(q_xyzw[0]), y = float(q_xyzw[1]), z = float(q_xyzw[2]), w = float(q_xyzw[3]))
            
            grasp_pose.append(p) 
            scores.append(grasp.score)
        self.num_iterations += 1
        
        try:
            self.best_grasp_history.append(grasp_pose[0]) # Save only the best grasp
            self.score_history.append(scores[0]) # Save only the score of the best grasp
        except IndexError as e:
            self.get_logger().warn(f'No grasp generated: grasp_pose or scores are empty. Error: {e}')
        
        self.call_srv_update_interest_map(grasp_pose, scores)

    
    def call_srv_update_interest_map(self,poses,scores):
        self.req.grasps = poses
        self.req.scores = scores

        future = self.cli.call_async(self.req)
        self.get_logger().info(f'Calling service /update_interest_map')
        # rclpy.spin_until_future_complete(self, future)

        # if future.result() is not None:
        #     self.get_logger().info(f'Service response: success = {future.result().success}')
        # else:
        #     self.get_logger().error('Service call failed')


def main(args=None):
    rclpy.init(args=args)
    node = GraspNetNode()

    # Start ROS2 spin in a secondary (daemon) THREAD
    def ros_spin():
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass

    spin_thread = threading.Thread(target=ros_spin, daemon=True)
    spin_thread.start()
    
    # Visualize the grasps in Open3D in the main thread
    try:
        graspnet_pipeline.visualization_in_open3d(params.num_best_grasps)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
    spin_thread.join()


if __name__ == '__main__':
    main()