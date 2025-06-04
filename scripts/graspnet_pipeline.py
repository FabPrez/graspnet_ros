#!/usr/bin/env python3
import os
import sys
import argparse
import importlib
import numpy as np
import open3d as o3d
import scipy.io as scio
import signal
import time
import torch
import threading
from PIL import Image
from graspnetAPI import GraspGroup

current_dir = os.path.dirname(os.path.abspath(__file__))
graspnet_baseline_dir = os.path.join(current_dir, 'graspnet-baseline')
checkpoint_path = os.path.join(graspnet_baseline_dir, 'checkpoint-rs.tar') #! CANNOT be changed
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
from parameters import *


def get_net():
    """
    Function to load the graspnet model (parameters + checkpoint folder).
    ----- Output parameters -----
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
    print(f"-> loaded checkpoint {checkpoint_path}", flush=True)
    net.eval()
    return net


def get_grasps(net, end_points):
    """
    Function to get the grasps.
    ----- Input parameters -----
    :param net: graspnet model
    :param end_points: dictionary with the input data (pointcloud and colors)
    ----- Output parameters -----
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
    ----- Input parameters -----
    :param gg: GraspGroup (predicted grasps)
    :param cloud: pointcloud
    ----- Output parameters -----
    :return: GraspGroup (filtered grasps)
    """
    mfcdetector = ModelFreeCollisionDetector(cloud, finger_width=finger_width, finger_length=finger_length, voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=approach_dist, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]
    return gg


def create_ground_plane(x_range, y_range, step=1.0):
    """
    Function to create a ground plane in Open3D.
    ----- Input parameters -----
    :param x_range: (x_min, x_max)
    :param y_range: (y_min, y_max)
    :param step: step size for the grid
    ----- Output parameters -----
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
    ----- Output parameters -----
    :return vis = Open3D Visualizer
    :return pcd_vis = PointCloud placeholder
    :return gripper_list = list of grippers
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
    ----- Input parameters -----
    :param points_np = numpy array (Nx3) with the new pointcloud
    :param gg = GraspGroup (predicted grasps)
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
    print(f"Visualize the best {len(gg)} grasps", flush=True)
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


def run_graspnet_pipeline(object_pts):
    """
    This function is called for each new pointcloud received (numpy array Nx3).
    It computes the grasps and calls update_visualization() to update the window.
    ----- Input parameters -----
    :param object_pts: np.ndarray of shape (N, 3) with XYZ coordinates
    ----- Output parameters -----
    :return: GraspGroup (predicted grasps)
    """
    global latest_points, latest_gg, pending_update
    
    # TODO: manually add the plane UNDER the objects
    # Find the lowest point (minimum z)
    idx_max_z = np.argmax(object_pts[:, 2])
    z_min = object_pts[idx_max_z, 2]

    #! GENERATE THE PLANE CENTERED AT C(center[0], center[1]) AT A HEIGHT Z = z_min
    # Compute the center
    center = np.mean(object_pts, axis=0)
    side_length = 0.20
    half_side = side_length / 2.0
    num_samples = 20

    xs = np.linspace(center[0] - half_side, center[0] + half_side, num_samples)
    ys = np.linspace(center[1] - half_side, center[1] + half_side, num_samples)
    xx, yy = np.meshgrid(xs, ys)

    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    zz_flat = np.full_like(xx_flat, z_min)

    plane_pts = np.stack([xx_flat, yy_flat, zz_flat], axis=1)

    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(plane_pts)
    print(f"-> generated plane with {plane_pts.shape[0]} points")
    
    
    #! COMBINE THE TWO POINTCLOUDS
    all_pts = np.vstack([object_pts, plane_pts])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    
    
    # Load the graspnet model
    net = get_net()
    
    # Sampling to get num_point points
    if len(all_pts) >= num_point:
        idxs = np.random.choice(len(all_pts), num_point, replace=False)
    else:
        idxs1 = np.arange(len(all_pts))
        idxs2 = np.random.choice(len(all_pts), num_point - len(all_pts), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = all_pts[idxs]
    color_sampled = np.ones_like(cloud_sampled, dtype=np.float32) * 0.5  # Dummy colors (gray)

    # Create the dictionary end_points (pointcloud + colors)
    tensor_points = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points = {
        'point_clouds': tensor_points,
        'cloud_colors': color_sampled
    }

    # Compute the grasps
    gg = get_grasps(net, end_points)
    if collision_thresh > 0:
        gg = collision_detection(gg, all_pts)
    
    # Put the results in the shared buffer, signaling the visualization thread
    with lock:
        latest_points = all_pts.copy()
        latest_gg = gg
        pending_update = True





#! -- inizio: [DEBUG] Functions to be used to debug --
def demo(doc_dir):
    """
    Function to be used to DEBUG.
    Demo to visualize the grasps from the 'color.png', 'depth.png', 'workspace_mask.png' in doc_dir
    ----- Input parameters -----
    :param doc_dir: directory containing the 'color.png', 'depth.png', 'workspace_mask.png'
    ----- Output parameters -----
    :return: None
    """
    net = get_net()
    end_points, cloud = get_and_process_data(doc_dir)
    gg = get_grasps(net, end_points)
    if collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    visualization_in_open3d(gg, cloud)
    

def demo_pcd(pcd_path):
    """
    Function to be used to DEBUG.
    Demo to visualize the grasps from the pcd file
    ----- Input parameters -----
    :param pcd_path: path to the pcd file
    ----- Output parameters -----
    :return: None
    """
    net = get_net()

    
    #! LOAD POINT CLOUD FROM .PCD FILE (or create it afterwards)
    object_pcd = o3d.io.read_point_cloud(pcd_path)
    object_pts = np.asarray(object_pcd.points, dtype=np.float32)
    print(f"-> loaded pointcloud with {object_pts.shape[0]} points for the OBJECT ONLY", flush=True)
    
    
    #! GENERATE THE PLANE CENTERED AT C(center[0], center[1]) AT A HEIGHT Z = z_plane
    # Compute the center
    center = np.mean(object_pts, axis=0)
    z_plane = 0.50
    
    # Compute the plane parameters
    N = 400 # number of points for the plane cloud
    min_x, max_x = np.min(object_pts[:, 0]), np.max(object_pts[:, 0])
    min_y, max_y = np.min(object_pts[:, 1]), np.max(object_pts[:, 1])
    dx = max_x - min_x
    dy = max_y - min_y
    if (dx < dy):
        side_length_x = 3.0 * dx
        side_length_y = 1.5 * dy
    elif(dx == dy):
        side_length_x = 1.5 * dx
        side_length_y = 1.5 * dy 
    else:
        side_length_x = 1.5 * dx
        side_length_y = 3.0 * dy
    
    ratio = side_length_x / side_length_y
    num_x = int(round(np.sqrt(N * ratio)))
    num_x = max(num_x, 2)
    num_y = int(np.ceil(N / num_x))
    num_y = max(num_y, 2)
    
    xs = np.linspace(center[0] - side_length_x / 2.0, center[0] + side_length_x / 2.0, num_x)
    ys = np.linspace(center[1] - side_length_y / 2.0, center[1] + side_length_y / 2.0, num_y)
    xx, yy = np.meshgrid(xs, ys)

    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    zz_flat = np.full_like(xx_flat, z_plane)

    plane_pts = np.stack([xx_flat, yy_flat, zz_flat], axis=1)

    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(plane_pts)
    print(f"-> generated plane with {plane_pts.shape[0]} points")
    
    
    #! COMBINE THE TWO POINTCLOUDS
    all_pts = np.vstack([object_pts, plane_pts])
    
    
    #! ROTATE THE TWO POINTCLOUDS
    # Extract the rotated pointcloud only
    all_pts_rotated, _, _, _ = rotate_pointcloud_180(all_pts, center)
    # Extract the axis of rotation only
    _, rotation_axis, min_proj, max_proj = rotate_pointcloud_180(all_pts, center)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts_rotated)
    
    
    # 2) Sample exactly as in get_and_process_data()
    if len(all_pts_rotated) >= num_point:
        idxs = np.random.choice(len(all_pts_rotated), num_point, replace=False)
    else:
        idxs1 = np.arange(len(all_pts_rotated))
        idxs2 = np.random.choice(len(all_pts_rotated), num_point - len(all_pts_rotated), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    pts_sampled = all_pts_rotated[idxs]
    
    # 3) Build end_points
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pts_tensor = torch.from_numpy(pts_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points = {'point_clouds': pts_tensor, 'cloud_colors': np.ones_like(pts_sampled)}
    
    # 4) Generate the grasps
    gg = get_grasps(net, end_points)
    print(f"Total number of grasps generated: {len(gg)}", flush=True)
    if collision_thresh > 0:
        gg = collision_detection(gg, np.array(pcd.points))
    print(f"Total number of grasps AFTER collision check: {len(gg)}", flush=True)
    gg = gg[:num_best_grasps] # Limit to the top num_best_grasps grasps
    print(f"Visualize the best {len(gg)} grasps", flush=True)
    
    # 5) Visualize in Open3D
    visualization_in_open3d(gg, pcd, rotation_axis, center, min_proj, max_proj)


def rotate_pointcloud_180(all_pts, center):
    """
    Function to rotate the pointcloud of 180 degrees around its main axis.
    ----- Input parameters -----
    :param all_pts = points of the combined pointcloud (object_pts + plane_pts)
    ----- Output parameters -----
    :return all_rot = points of the combined pointcloud (object_pts + plane_pts) rotated by 180 degrees
    """
    # Center all the points in the origin
    pts_centered = all_pts - center

    # Compute the covariance matrix
    cov = (pts_centered.T @ pts_centered) / pts_centered.shape[0]

    # Eigen-decomposition to find eigenvectors/eigenvalues.
    # Then take the index of the largest eigenvalue = first principal component
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_idx = np.argmax(eigvals)
    rotation_axis = eigvecs[:, principal_idx]
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # Compute the projections to visualize in Open3D the rotation_axis of the pointcloud 
    projs = pts_centered.dot(rotation_axis)
    min_proj = projs.min()
    max_proj = projs.max()

    # Rotation matrix of 180° (θ = π) around "rotation_axis"
    k = rotation_axis.reshape(3, 1)
    R = -np.eye(3) + 2 * (k @ k.T)

    # Apply the rotation and shift all the points to the initial position 
    all_rot = (R @ pts_centered.T).T + center
    
    return all_rot, rotation_axis, min_proj, max_proj


def get_and_process_data(doc_dir):
    """
    Function to load the data from the doc_dir and process it to create the point cloud.
    ----- Input parameters -----
    :param doc_dir: directory containing the 'color.png', 'depth.png', 'workspace_mask.png', 'meta.mat'
    ----- Output parameters -----
    :return: end_points (dict with point cloud and colors), cloud (open3d PointCloud)
    """
    # load data
    color = np.array(Image.open(os.path.join(doc_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(doc_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(doc_dir, 'workspace_mask.png')))
    meta = scio.loadmat(os.path.join(doc_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud


def visualization_in_open3d(gg, cloud, rotation_axis, center, min_proj, max_proj):
    """
    Function to visualize the grasps and the point cloud in Open3D.
    ----- Input parameters -----
    :param gg: GraspGroup (predicted grasps)
    :param cloud: open3d PointCloud with the point cloud
    """
    global terminate_visualization
    terminate_visualization = False
    
    signal.signal(signal.SIGINT, _on_sigint_visual)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='GraspNet Live', width=1280, height=720)
    
    # Create and visualize the origin frame O(0,0,0)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    # Create and visualize the ground plane
    plane_mesh, plane_grid = create_ground_plane(x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), step=1.0)
    
    # Create and visualize the grasps and the gripper
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    
    # Create and visualize the rotation rotation_axis of the pointcloud
    end1 = center + rotation_axis * min_proj
    end2 = center + rotation_axis * max_proj
    points = np.vstack([end1, end2])
    lines = [[0, 1]]
    rotation_axis = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points), 
        lines=o3d.utility.Vector2iVector(lines))
    
    # Visualize in Open3D
    vis.add_geometry(axis)
    vis.add_geometry(cloud)
    vis.add_geometry(plane_mesh)
    vis.add_geometry(plane_grid)
    vis.add_geometry(rotation_axis)
    for g in grippers:
        vis.add_geometry(g)
    
    while True:
        try:
            if terminate_visualization:
                break
            vis.poll_events()
            vis.update_renderer()
        except Exception:
            break
        time.sleep(0.01) # Short sleep to avoid 100% CPU usage
    vis.destroy_window()


def _on_sigint_visual(signum, frame):
    """
    SIGINT handler to stop the visualization loop.
    It is registered only when the window is opened.
    """
    global terminate_visualization
    terminate_visualization = True

#! -- fine: [DEBUG] Functions to be used to debug --
