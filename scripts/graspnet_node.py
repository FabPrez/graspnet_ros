#!/usr/bin/env python3
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

import graspnet_pipeline
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class GraspNetNode(Node):
    def __init__(self):
        super().__init__('graspnet_node')
        
        # #! -- inizio: PROVA demo_pcd --
        # pcd_path = '/home/vignofede/grasp_NBV_ws/saved_pointclouds/trial_simulated_pointcloud.pcd'
        # graspnet_pipeline.demo_pcd(pcd_path)
        # #! -- fine: PROVA demo_pcd --
        
        self.subscription = self.create_subscription(
            PointCloud2, '/octomap_point_cloud_centers', self.pointcloud_callback, 10)
        self.get_logger().info("GraspNetNode started, listening for point clouds...")

    def pointcloud_callback(self,msg):
        self.get_logger().info("Received PointCloud2")

        pc = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

        if len(pc) == 0:
            self.get_logger().warn("Empty pointcloud received")
            return

        # Convert the list of tuples into a 2D array (N, 3)
        points = np.array([ [x, y, z] for x, y, z in pc ], dtype=np.float32)

        # Run the pipeline without color
        graspnet_pipeline.run_graspnet_pipeline(points)
        


if __name__=='__main__':
    print("Starting GraspNetNode...")
    rclpy.init()
    node = GraspNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
