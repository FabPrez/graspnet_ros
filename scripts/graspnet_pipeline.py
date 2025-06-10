#!/usr/bin/env python3
import os
import sys
import copy
import numpy as np
import open3d as o3d
import signal
import time
import torch
import threading
from graspnetAPI import GraspGroup
from scipy.spatial.transform import Rotation

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
from collision_detector import ModelFreeCollisionDetector
from parameters import *


# ----- Global variables -----
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vis = None
net = None
new_pc_flag = False
_window_lock = threading.Lock()


def get_net():
    """
    Loads and initializes the GraspNet model with the required parameters and checkpoint.
    
    :return: net = initialized GraspNet model
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
    Performs inference using the GraspNet model to generate grasp candidates from input data.

    :param net: Pre-loaded GraspNet model for inference
    :param end_points: Dictionary containing the point cloud and associated color information

    :return: GraspGroup = collection of predicted grasps
    """
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg


def collision_detection(gg, cloud):
    """
    Detects and removes grasps that are in collision with the input point cloud.
    
    :param gg: GraspGroup containing predicted grasp candidates
    :param cloud: point cloud as a numpy array or Open3D PointCloud
    
    :return: GraspGroup with only collision-free grasps
    """
    mfcdetector = ModelFreeCollisionDetector(cloud, finger_width=finger_width, finger_length=finger_length, voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=approach_dist, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]
    return gg


def run_graspnet_pipeline(object_pts):
    """
    Called for each new incoming point cloud (numpy array Nx3).
    Computes grasp candidates and updates the visualization window.
    
    :param object_pts: np.ndarray of shape (N, 3) with XYZ coordinates
    
    :return: gg_up = GraspGroup containing the top-ranked grasp candidates.
    """
    net = get_net()
    
    
    #! GENERATE THE PLANE CENTERED AT C(center[0], center[1]) AT A HEIGHT Z = z_plane
    # Compute the center
    center = np.mean(object_pts, axis=0)
    z_plane = 0.20
    
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
    print(f"-> generated plane with {plane_pts.shape[0]} points")
    
    
    #! COMBINE THE TWO POINTCLOUDS
    pts_up = np.vstack([object_pts, plane_pts])
    pcd_up = o3d.geometry.PointCloud()
    pcd_up.points = o3d.utility.Vector3dVector(pts_up)
    
    
    #! ROTATE THE TWO POINTCLOUDS
    result = rotate_pointcloud_180(pts_up, center)
    # Extract the rotated pointcloud only
    pts_down = result[0]
    # Extract the rotation matrix
    R = result[1]
    # Extract the rotation_axis, min_proj and max_proj only
    rotation_axis = result[2]
    min_proj = result[3]
    max_proj = result[4]
    
    pcd_down = o3d.geometry.PointCloud()
    pcd_down.points = o3d.utility.Vector3dVector(pts_down)
    
    
    #! SAMPLE AND GENERATE THE GRASPS
    # 2) Sample the points to provide always num_point to the net
    if len(pts_down) >= num_point:
        idxs = np.random.choice(len(pts_down), num_point, replace=False)
    else:
        idxs1 = np.arange(len(pts_down))
        idxs2 = np.random.choice(len(pts_down), num_point - len(pts_down), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    pts_sampled = pts_down[idxs]
    
    # 3) Build end_points
    pts_tensor = torch.from_numpy(pts_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points = {'point_clouds': pts_tensor, 'cloud_colors': np.ones_like(pts_sampled)}
    
    # 4) Generate the grasps
    gg_down = get_grasps(net, end_points)
    print(f"Total number of grasps generated: {len(gg_down)}", flush=True)
    if collision_thresh > 0:
        gg_down = collision_detection(gg_down, np.array(pcd_down.points))
    print(f"Total number of grasps AFTER collision check: {len(gg_down)}", flush=True)
    gg_down = gg_down[:num_best_grasps] # Limit to the top num_best_grasps grasps
    
    
    #! ROTATE THE GRASPS
    gg_up = copy.deepcopy(gg_down)
    gg_up = rotate_grasps_180(gg_up, center, R)
    
    
    #! VISUALIZE IN OPEN3D
    global latest_pointcloud_shared, latest_grippers_shared, new_pc_flag
    latest_pointcloud_shared = None
    latest_grippers_shared = None
    with _window_lock:
        latest_pointcloud_shared = copy.deepcopy(pcd_up)
        latest_grippers_shared = copy.deepcopy(gg_up)
        new_pc_flag = True
    
    return gg_up


def rotate_pointcloud_180(points, center):
    """
    Rotates the input point cloud by 180 degrees around its principal axis (first principal component).

    :param points: (N, 3) numpy array containing the 3D coordinates of the combined point cloud (object points + plane points).
    :param center: (3,) numpy array representing the centroid of the object point cloud, used as the rotation origin.

    :return all_rot: (N, 3) numpy array of the rotated point cloud (object + plane), after 180-degree rotation about the principal axis.
    :return R: (3, 3) numpy array, the rotation matrix representing a 180-degree rotation around the principal axis.
    :return rotation_axis: (3,) numpy array, the unit vector of the principal axis (first principal component) of the point cloud.
    :return min_proj: float, minimum projection value of the centered points along the principal axis (for visualization).
    :return max_proj: float, maximum projection value of the centered points along the principal axis (for visualization).
    """
    # Center all the points in the origin
    pts_centered = points - center

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

    # Rotation matrix of 180° (θ terminate_visualization = False= π) around "rotation_axis"
    k = rotation_axis.reshape(3, 1)
    R = -np.eye(3) + 2 * (k @ k.T)

    # Apply the rotation and shift all the points to the initial position 
    all_rot = (R @ pts_centered.T).T + center
    
    return all_rot, R, rotation_axis, min_proj, max_proj


def rotate_grasps_180(grasp_group, center, R):
    """
    Rotates all grasps in the given GraspGroup by 180 degrees around the principal axis of the point cloud.
    
    :param grasp_group: GraspGroup object containing the set of grasps to be rotated.
    :param center: numpy array of shape (3,) representing the center point (cx, cy, cz) about which the rotation is performed.
    :param R: numpy array of shape (3, 3), the rotation matrix representing a 180° rotation around the principal axis.
    
    :return: GraspGroup with all grasps rotated by 180 degrees around the specified principal axis.
    """
    translations = np.asarray(grasp_group.translations)
    rot_matrices = np.asarray(grasp_group.rotation_matrices)
    
    num_grasps = translations.shape[0]
    for i in range(num_grasps):
        # Extract the center position (translations) and the orientation (rot_matrices)
        t_orig = translations[i]
        R_orig = rot_matrices[i]

        # Center all the grasps at the origin, apply the rotation, then shift all the grasps by the "center" vector
        t_rotated = R.dot(t_orig - center) + center
        
        # Update the rotation matrix of the grasp
        R_rotated = R.dot(R_orig)

        # Save the results in the grasp_group
        grasp_group.translations[i] = t_rotated
        grasp_group.rotation_matrices[i] = R_rotated
    
    return grasp_group


def visualization_in_open3d(num_best_grasps):
    """
    Visualizes the results in Open3D.
    In particular, it creates a new window 
    and prints the reference frame (origin frame), the ground plane, the pointcloud, the grasps (grippers).
    
    :param num_best_grasps: int, number of best grasps to visualize
    """
    global latest_pointcloud_shared, latest_grippers_shared, new_pc_flag

    # Create the Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='GraspNet', width=1280, height=720)
    
    # Create and visualize the origin frame O(0,0,0)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(axis)
    
    # Create and visualize the ground plane
    plane_mesh, plane_grid = create_ground_plane(x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), step=1.0)
    vis.add_geometry(plane_mesh)
    vis.add_geometry(plane_grid)
    
    while True:
        if not vis.poll_events():
            break

        with _window_lock:
            if new_pc_flag and latest_pointcloud_shared is not None and latest_grippers_shared is not None:
                # Clear everything from the scene
                vis.clear_geometries()

                # Visualize the origin frame O(0,0,0) again
                vis.add_geometry(axis)
                
                # Visualize the ground plane
                vis.add_geometry(plane_mesh)
                vis.add_geometry(plane_grid)

                # Visualize the new point cloud
                vis.add_geometry(latest_pointcloud_shared)

                # Visualize the new grasps (grippers)
                latest_grippers_shared = latest_grippers_shared[:num_best_grasps]  # Limit to the top num_best_grasps grasps
                print(f"Visualize the best {len(latest_grippers_shared)} grasps", flush=True)
                for g in latest_grippers_shared:
                    vis.add_geometry(g.to_open3d_geometry())

                # Update the flag
                new_pc_flag = False
        
        vis.update_renderer()
        time.sleep(0.01)

    # When exiting (window closed), destroy everything
    vis.destroy_window()


def create_ground_plane(x_range, y_range, step=1.0):
    """
    Creates a ground plane mesh and grid lines using Open3D.
    
    :param x_range: tuple (x_min, x_max) specifying the X-axis limits
    :param y_range: tuple (y_min, y_max) specifying the Y-axis limits
    :param step: grid spacing for both axes
    
    :return: mesh (TriangleMesh) and grid (LineSet) representing the plane
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





# #! ----- inizio debug: visualizzo a schermo pointcloud ruotata + grasp ruotati + pointcloud non ruotata + grasp non ruotati -----
# ----- Global variables -----
# terminate_visualization = False

# def DEBUG_visualization_in_open3d(gg_up, pcd_up, gg_down, pcd_down, rotation_axis, center, min_proj, max_proj):
    
#     global terminate_visualization
#     terminate_visualization = False
    
#     signal.signal(signal.SIGINT, _on_sigint_visual)
    
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name='GraspNet Live', width=1280, height=720)
    
#     # Create and visualize the origin frame O(0,0,0)
#     axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
#     # Create and visualize the ground plane
#     plane_mesh, plane_grid = create_ground_plane(x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), step=1.0)
    
#     # Create and visualize the grasps and the gripper
#     gg_up.nms()
#     gg_up.sort_by_score()
#     gg_up = gg_up[:50]
#     grippers_up = gg_up.to_open3d_geometry_list()
    
#     # Create and visualize the grasps and the gripper
#     gg_down.nms()
#     gg_down.sort_by_score()
#     gg_down = gg_down[:50]
#     grippers_down = gg_down.to_open3d_geometry_list()
    
#     # Create and visualize the rotation rotation_axis of the pointcloud
#     end1 = center + rotation_axis * min_proj
#     end2 = center + rotation_axis * max_proj
#     points = np.vstack([end1, end2])
#     lines = [[0, 1]]
#     rotation_axis = o3d.geometry.LineSet(
#         points=o3d.utility.Vector3dVector(points), 
#         lines=o3d.utility.Vector2iVector(lines))
    
#     # Colora pcd_up di rosso
#     pcd_up.paint_uniform_color([1.0, 0.0, 0.0])
#     # Colora pcd_down di blu
#     pcd_down.paint_uniform_color([0.0, 0.0, 1.0])
    
#     # Visualize in Open3D
#     vis.add_geometry(axis)
#     vis.add_geometry(pcd_up)
#     vis.add_geometry(pcd_down)
#     vis.add_geometry(plane_mesh)
#     vis.add_geometry(plane_grid)
#     vis.add_geometry(rotation_axis)
    
#     for g_up in grippers_up:
#         if isinstance(g_up, o3d.geometry.Geometry): # vale per LineSet, TriangleMesh, ecc.
#             g_up.paint_uniform_color([1.0, 0.0, 0.0]) # Colora ogni gripper di rosso
#         vis.add_geometry(g_up)
    
#     for g_down in grippers_down:
#         if isinstance(g_down, o3d.geometry.Geometry): # vale per LineSet, TriangleMesh, ecc.
#             g_down.paint_uniform_color([0.0, 0.0, 1.0]) # Colora ogni gripper di blu
#         vis.add_geometry(g_down)
    
#     while True:
#         try:
#             if terminate_visualization:
#                 break
#             vis.poll_events()
#             vis.update_renderer()
#         except Exception:
#             break
#         time.sleep(0.01) # Short sleep to avoid 100% CPU usage
#     vis.destroy_window()
# #! ----- fine debug: visualizzo a schermo pointcloud ruotata + grasp ruotati + pointcloud non ruotata + grasp non ruotati -----
