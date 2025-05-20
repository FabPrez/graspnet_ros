#!/usr/bin/env python3
import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
import torch
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

checkpoint_path = os.path.join(graspnet_baseline_dir, 'checkpoint-rs.tar')
num_point = 20000
num_view = 300
collision_thresh = 0.01
voxel_size = 0.01


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    print("-> loaded checkpoint %s"%(checkpoint_path))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(doc_dir):
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

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]
    return gg


def visualization_in_open3d(gg, cloud):
    # Create and visualize the origin frame O(0,0,0)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    # Create and visualize the ground plane
    plane_mesh, plane_grid = create_ground_plane(x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), step=1.0)
    
    # Create and visualize the grasps and the gripper
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    
    # Visualize in Open3D
    o3d.visualization.draw_geometries([axis, cloud, *grippers, plane_mesh, plane_grid],
        mesh_show_wireframe=False, mesh_show_back_face=True)
    

def create_ground_plane(x_range, y_range, step=1.0):
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


def demo(doc_dir):
    '''
    Function to be used to DEBUG.
    Demo to visualize the grasps from the 'color.png', 'depth.png', 'workspace_mask.png' in doc_dir
    :param doc_dir: directory containing the 'color.png', 'depth.png', 'workspace_mask.png'
    :return: None
    '''
    net = get_net()
    end_points, cloud = get_and_process_data(doc_dir)
    gg = get_grasps(net, end_points)
    if collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    visualization_in_open3d(gg, cloud)
    

def demo_pcd(pcd_path):
    '''
    Function to be used to DEBUG.
    Demo to visualize the grasps from the pcd file
    :param pcd_path: path to the pcd file
    :return: None
    '''
    net = get_net()
    
    # 1) Read the PCD
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    print("-> loaded pointcloud %s"%(pcd_path))
    
    # 2) Sample exactly as in get_and_process_data()
    if len(pts) >= num_point:
        idxs = np.random.choice(len(pts), num_point, replace=False)
    else:
        idxs1 = np.arange(len(pts))
        idxs2 = np.random.choice(len(pts), num_point - len(pts), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    pts_sampled = pts[idxs]
    
    # 3) Build end_points
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pts_tensor = torch.from_numpy(pts_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points = {'point_clouds': pts_tensor, 'cloud_colors': np.ones_like(pts_sampled)}
    cloud = pcd
    
    gg = get_grasps(net, end_points)
    if collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    
    # 4) Visualize in Open3D
    visualization_in_open3d(gg, cloud)
    
    
def run_graspnet_pipeline(points):
    """
    :param points: np.ndarray di shape (N, 3) con coordinate XYZ
    :return: GraspGroup
    """
    net = get_net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if len(points) >= num_point:
        idxs = np.random.choice(len(points), num_point, replace=False)
    else:
        idxs1 = np.arange(len(points))
        idxs2 = np.random.choice(len(points), num_point - len(points), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = points[idxs]

    # Colori fittizi (grigio) se non disponibili
    color_sampled = np.ones_like(cloud_sampled, dtype=np.float32) * 0.5

    # Prepara dizionario end_points
    cloud_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points = {
        'point_clouds': cloud_tensor,
        'cloud_colors': color_sampled
    }

    gg = get_grasps(net, end_points)
    if collision_thresh > 0:
        gg = collision_detection(gg, points)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    print("[DEBUG] Eseguita graspnet_pipeline", flush=True)
    visualization_in_open3d(gg, cloud)

if __name__=='__main__':
    demo(doc_dir)
    # demo_pcd(pcd_path)