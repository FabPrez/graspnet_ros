#!/usr/bin/env python3
import os
import sys
import argparse
import importlib
import numpy as np
import open3d as o3d
import scipy.io as scio
import time
import torch
import threading
from PIL import Image
from graspnetAPI import GraspGroup

current_dir = os.path.dirname(os.path.abspath(__file__))
graspnet_baseline_dir = os.path.join(current_dir, 'graspnet-baseline')
doc_dir = os.path.join(graspnet_baseline_dir, 'doc', 'example_data')
sys.path.append(current_dir)
sys.path.append(graspnet_baseline_dir)
sys.path.append(os.path.join(graspnet_baseline_dir, 'models'))
sys.path.append(os.path.join(graspnet_baseline_dir, 'dataset'))
sys.path.append(os.path.join(graspnet_baseline_dir, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

# -----------------------------------------
# Parameters
# -----------------------------------------
checkpoint_path = os.path.join(graspnet_baseline_dir, 'checkpoint-rs.tar') #! CANNOT be changed
hmin = -0.00 # default: -0.02
hmax_list = [0.04, 0.06, 0.08, 0.10] # default: [0.01, 0.02, 0.03, 0.04]
num_point = 100 # default: 20000
num_view = 300 # default: 300 #! CANNOT be changed
num_angle = 12 # default: 12 #! CANNOT be changed
num_depth = len(hmax_list) # default: 4 #! CANNOT be changed
cylinder_radius = 1.00 # default: 0.05
collision_thresh = 0.01 # default: 0.01
voxel_size = 0.01 # default: 0.01

# -----------------------------------------
# Global variables for the visualization
# -----------------------------------------
vis = None
pcd_vis = None
gripper_list = []
net = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
latest_points = None
latest_gg = None
pending_update = False
terminate = False
lock = threading.Lock()
num_best_grasps = 50 # default: 50


def get_net():
    """
    Function to load the graspnet model (parameters + checkpoint folder).
    :return: net
    """
    global net
    if net is not None:
        return net
    
    # Initialize the net
    net = GraspNet(input_feature_dim=0, num_view=num_view, num_angle=num_angle, num_depth=num_depth,
                   cylinder_radius=cylinder_radius, hmin=hmin, hmax_list=hmax_list, is_training=False)
    net.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    print("-> loaded checkpoint %s"%(checkpoint_path), flush=True)
    net.eval()
    return net


def get_grasps(net, end_points):
    """
    Function to get the grasps.
    :param net: graspnet model
    :param end_points: dictionary with the input data (pointcloud and colors)
    :return: GraspGroup
    """
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg


def collision_detection(gg, cloud):
    """
    Function to detect collisions between the grasps and the point cloud.
    :param gg: GraspGroup (predicted grasps)
    :param cloud: pointcloud
    :return: GraspGroup (filtered grasps)
    """
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]
    return gg


def create_ground_plane(x_range, y_range, step=1.0):
    """
    Function to create a ground plane in Open3D.
    :param x_range: (x_min, x_max)
    :param y_range: (y_min, y_max)
    :param step: step size for the grid
    :return: mesh (TriangleMesh) and grid (LineSet)
    """
    # 1) Generate the vertices of the regular grid
    xs = np.arange(x_range[0], x_range[1] + step, step)
    ys = np.arange(y_range[0], y_range[1] + step, step)
    verts = [[x, y, 0.0] for x in xs for y in ys]
    
    # 2) Build triangles for the filled mesh
    nx, ny = len(xs), len(ys)
    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            idx = i * ny + j
            tris.append([idx,     idx+1,   idx+ny])
            tris.append([idx+1,   idx+ny+1,idx+ny])
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(verts))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tris))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.8, 0.8, 0.8]) # light gray plane

    # 3) Build the LineSet for the grid lines
    lines = []
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            if i < nx - 1:
                lines.append([idx, idx + ny])
            if j < ny - 1:
                lines.append([idx, idx + 1])
    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(verts)),
        lines=o3d.utility.Vector2iVector(np.array(lines))
    )
    grid.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in lines]) # black lines

    return mesh, grid


def init_visualizer():
    """
    Function to initialize (ONLY ONCE) the Open3D window and to add:
    - the reference axis
    - the ground plane
    - an empty PointCloud that will be updated in update_visualization().
    """
    global vis, pcd_vis, gripper_list
    if vis is not None:
        return vis, pcd_vis, gripper_list

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='GraspNet Live', width=1280, height=720)

    # Create the reference axis
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(axis)

    # Create the ground plane
    plane_mesh, plane_grid = create_ground_plane(x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), step=1.0)
    vis.add_geometry(plane_mesh)
    vis.add_geometry(plane_grid)

    # Create the PointCloud placeholder (empty, will be populated in later loops)
    pcd_vis = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_vis)

    # Create the list for the grippers (empty, will be populated in later loops)
    gripper_list = []

    # Store in global variables
    globals()['vis'] = vis
    globals()['pcd_vis'] = pcd_vis
    globals()['gripper_list'] = gripper_list

    return vis, pcd_vis, gripper_list


def update_visualization(points_np, gg):
    """
    Function to update the Open3D window with the new pointcloud and the new grasps.
    This method is called ONLY by the visualization thread!
    points_np: numpy array (Nx3) with the new pointcloud
    gg: GraspGroup (predicted grasps)
    """
    global vis, pcd_vis, gripper_list

    # If the visualizer is not yet initialized, do it now
    if vis is None or pcd_vis is None:
        init_visualizer()

    # Update the existing pointcloud
    pcd_vis.points = o3d.utility.Vector3dVector(points_np.astype(np.float32))
    vis.update_geometry(pcd_vis)

    # Remove the previous grippers
    for g in gripper_list:
        vis.remove_geometry(g, reset_bounding_box=False)
    gripper_list.clear()
    
    # Generate the new grippers and add them to the visualizer
    gg.nms()
    gg.sort_by_score()
    print(f"Total number of grasps generated: {len(gg)}", flush=True)
    gg = gg[:num_best_grasps] # Limit to the top num_best_grasps grasps
    new_grippers = gg.to_open3d_geometry_list()
    if len(gg) < num_best_grasps:
        print(f"Visualize the best {len(gg)} grasps", flush=True)
    else:
        print(f"Visualize the best {num_best_grasps} grasps", flush=True)
    for g in new_grippers:
        vis.add_geometry(g)
        gripper_list.append(g)

    # Update the visualizer
    vis.poll_events()
    vis.update_renderer()


def visualizer_loop():
    """
    Thread that stays in a loop:
    - if pending_update == True, it reads the latest_points/latest_gg and calls update_visualization()
    - runs vis.poll_events() and vis.update_renderer() to keep the window responsive
    """
    global latest_points, latest_gg, pending_update, terminate

    # If the visualizer is not yet initialized, do it now
    if vis is None or pcd_vis is None:
        init_visualizer()

    while not terminate:
        data_to_apply = None
        with lock:
            if pending_update and latest_points is not None and latest_gg is not None:
                # Copy the data locally
                data_to_apply = (latest_points.copy(), latest_gg)
                pending_update = False

        if data_to_apply is not None:
            pts_np, gg_local = data_to_apply
            update_visualization(pts_np, gg_local)

        # Poll + render to keep the window alive
        vis.poll_events()
        vis.update_renderer()

        # Sleep for a short time to avoid high CPU usage
        time.sleep(0.01)

    # When terminate becomes True, exit and destroy the window
    if vis is not None:
        vis.destroy_window()


def run_graspnet_pipeline(points):
    """
    This function is called for each new pointcloud received (numpy array Nx3).
    It computes the grasps and calls update_visualization() to update the window.
    :param points: np.ndarray of shape (N, 3) with XYZ coordinates
    :return: GraspGroup (predicted grasps)
    """
    global latest_points, latest_gg, pending_update
    
    # Load the graspnet model
    net = get_net()
    
    # Sampling to get num_point points
    if len(points) >= num_point:
        idxs = np.random.choice(len(points), num_point, replace=False)
    else:
        idxs1 = np.arange(len(points))
        idxs2 = np.random.choice(len(points), num_point - len(points), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = points[idxs]
    color_sampled = np.ones_like(cloud_sampled, dtype=np.float32) * 0.5 # Colori fittizi (grigio)

    # Create the dictionary end_points (pointcloud + colors)
    tensor_points = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points = {
        'point_clouds': tensor_points,
        'cloud_colors': color_sampled
    }

    # Compute the grasps
    gg = get_grasps(net, end_points)
    # if collision_thresh > 0:
    #     gg = collision_detection(gg, points)
    
    # Put the results in the shared buffer, signaling the visualization thread
    with lock:
        latest_points = points.copy()
        latest_gg = gg
        pending_update = True


#! -- inizio: [DEBUG] Functions to be used to debug --
# def demo(doc_dir):
#     '''
#     Function to be used to DEBUG.
#     Demo to visualize the grasps from the 'color.png', 'depth.png', 'workspace_mask.png' in doc_dir
#     :param doc_dir: directory containing the 'color.png', 'depth.png', 'workspace_mask.png'
#     :return: None
#     '''
#     net = get_net()
#     end_points, cloud = get_and_process_data(doc_dir)
#     gg = get_grasps(net, end_points)
#     if collision_thresh > 0:
#         gg = collision_detection(gg, np.array(cloud.points))
#     visualization_in_open3d(gg, cloud)
    

# def demo_pcd(pcd_path):
#     '''
#     Function to be used to DEBUG.
#     Demo to visualize the grasps from the pcd file
#     :param pcd_path: path to the pcd file
#     :return: None
#     '''
#     net = get_net()
    
#     # 1) Read the PCD
#     pcd = o3d.io.read_point_cloud(pcd_path)
#     pts = np.asarray(pcd.points, dtype=np.float32)
#     print("-> loaded pointcloud %s"%(pcd_path))
    
#     # 2) Sample exactly as in get_and_process_data()
#     if len(pts) >= num_point:
#         idxs = np.random.choice(len(pts), num_point, replace=False)
#     else:
#         idxs1 = np.arange(len(pts))
#         idxs2 = np.random.choice(len(pts), num_point - len(pts), replace=True)
#         idxs = np.concatenate([idxs1, idxs2], axis=0)
#     pts_sampled = pts[idxs]
    
#     # 3) Build end_points
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     pts_tensor = torch.from_numpy(pts_sampled[np.newaxis].astype(np.float32)).to(device)
#     end_points = {'point_clouds': pts_tensor, 'cloud_colors': np.ones_like(pts_sampled)}
#     cloud = pcd
    
#     gg = get_grasps(net, end_points)
#     if collision_thresh > 0:
#         gg = collision_detection(gg, np.array(cloud.points))
    
#     # 4) Visualize in Open3D
#     visualization_in_open3d(gg, cloud)


# def get_and_process_data(doc_dir):
#     # load data
#     color = np.array(Image.open(os.path.join(doc_dir, 'color.png')), dtype=np.float32) / 255.0
#     depth = np.array(Image.open(os.path.join(doc_dir, 'depth.png')))
#     workspace_mask = np.array(Image.open(os.path.join(doc_dir, 'workspace_mask.png')))
#     meta = scio.loadmat(os.path.join(doc_dir, 'meta.mat'))
#     intrinsic = meta['intrinsic_matrix']
#     factor_depth = meta['factor_depth']

#     # generate cloud
#     camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
#     cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

#     # get valid points
#     mask = (workspace_mask & (depth > 0))
#     cloud_masked = cloud[mask]
#     color_masked = color[mask]

#     # sample points
#     if len(cloud_masked) >= num_point:
#         idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
#     else:
#         idxs1 = np.arange(len(cloud_masked))
#         idxs2 = np.random.choice(len(cloud_masked), num_point-len(cloud_masked), replace=True)
#         idxs = np.concatenate([idxs1, idxs2], axis=0)
#     cloud_sampled = cloud_masked[idxs]
#     color_sampled = color_masked[idxs]

#     # convert data
#     cloud = o3d.geometry.PointCloud()
#     cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
#     cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
#     end_points = dict()
#     cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     cloud_sampled = cloud_sampled.to(device)
#     end_points['point_clouds'] = cloud_sampled
#     end_points['cloud_colors'] = color_sampled

#     return end_points, cloud


# def visualization_in_open3d(gg, cloud):
#     # Create and visualize the origin frame O(0,0,0)
#     axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
#     # Create and visualize the ground plane
#     plane_mesh, plane_grid = create_ground_plane(x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), step=1.0)
    
#     # Create and visualize the grasps and the gripper
#     gg.nms()
#     gg.sort_by_score()
#     gg = gg[:50]
#     grippers = gg.to_open3d_geometry_list()
    
#     # Visualize in Open3D
#     o3d.visualization.draw_geometries([axis, cloud, *grippers, plane_mesh, plane_grid],
#         mesh_show_wireframe=False, mesh_show_back_face=True)


# if __name__=='__main__':
#     demo(doc_dir)
#     # demo_pcd(pcd_path)
#! -- fine: [DEBUG] Functions to be used to debug --