#!/usr/bin/env python3
import os
import sys
import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
import threading
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

current_dir = os.path.dirname(os.path.abspath(__file__))
import graspnet_pipeline

class GraspNetNode(Node):
    def __init__(self):
        super().__init__('graspnet_node')
        
        #! -- inizio: [DEBUG] Function to be used to debug (demo_pcd) --
        # pcd_path = '/home/vignofede/grasp_NBV_ws/saved_pointclouds/trial_simulated_pointcloud.pcd'
        # pcd_path = '/home/vignofede/grasp_NBV_ws/saved_pointclouds/prova_box.pcd'
        pcd_path = '/home/vignofede/grasp_NBV_ws/saved_pointclouds/prova_box_5x5x5cm.pcd'
        # pcd_path = '/home/vignofede/grasp_NBV_ws/saved_pointclouds/prova_box_10000pts.pcd'
        # pcd_path = '/home/vignofede/grasp_NBV_ws/saved_pointclouds/prova_sphere.pcd'
        # pcd_path = '/home/vignofede/grasp_NBV_ws/saved_pointclouds/prova_sphere_5cm.pcd'
        # pcd_path = '/home/vignofede/grasp_NBV_ws/saved_pointclouds/prova_sphere_10000pts.pcd'
        graspnet_pipeline.demo_pcd(pcd_path)
        #! -- fine: [DEBUG] Function to be used to debug (demo_pcd) --
        
        # self.subscription = self.create_subscription(
        #     PointCloud2, '/octomap_point_cloud_centers', self.pointcloud_callback, 10)
        # self.get_logger().info("GraspNetNode started, listening for point clouds...")

    # def pointcloud_callback(self, msg):
    #     self.get_logger().info("Received PointCloud2")

    #     pc = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

    #     if len(pc) == 0:
    #         self.get_logger().warn("Empty pointcloud received")
    #         return

    #     # Convert the list of tuples into a 2D array (N, 3)
    #     points = np.array([ [x, y, z] for x, y, z in pc ], dtype=np.float32)

    #     # Run the pipeline without color
    #     graspnet_pipeline.run_graspnet_pipeline(points)


def main(args=None):
    rclpy.init(args=args)
    node = GraspNetNode()

    # Start the visualization thread (daemon=True → it dies with the main thread)
    vis_thread = threading.Thread(target=graspnet_pipeline.visualizer_loop, daemon=True)
    vis_thread.start()

    # Start ROS2 spin
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        with graspnet_pipeline.lock:
            graspnet_pipeline.terminate = True
        vis_thread.join()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()