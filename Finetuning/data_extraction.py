#!/usr/bin/env python3
"""
Point Cloud to Camera Projection System

This module processes TLS point clouds and generates depth maps and RGB images
from multiple camera perspectives using ground truth poses.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CameraParameters:
    """Camera intrinsic and extrinsic parameters."""
    K: np.ndarray  # Intrinsic matrix
    D: np.ndarray  # Distortion coefficients
    T_cam_lidar: np.ndarray  # Transformation from lidar to camera frame


@dataclass
class ProjectionConfig:
    """Configuration parameters for the projection system."""
    pcd_path: str
    gt_tum_path: str
    output_dir: str
    png_dir: str
    img_width: int = 1440
    img_height: int = 1080
    voxel_size: float = 0.05
    frame_skip: int = 100  # Process every N-th frame


class GroundTruthLoader:
    """Handles loading and parsing of ground truth pose data."""
    
    def __init__(self, gt_path: str):
        self.gt_path = gt_path
        
    def load_poses(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load ground truth poses from TUM format file.
        
        Returns:
            Tuple of (timestamps, positions, quaternions)
        """
        timestamps, poses, quats = [], [], []
        
        with open(self.gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:  # Ensure valid line
                    timestamps.append(parts[0])
                    poses.append([float(x) for x in parts[1:4]])
                    quats.append([float(x) for x in parts[4:8]])
        
        return (np.array(timestamps, dtype=str), 
                np.array(poses, dtype=np.float64), 
                np.array(quats, dtype=np.float64))


class PointCloudProcessor:
    """Handles point cloud loading and preprocessing."""
    
    def __init__(self, pcd_path: str, voxel_size: float = 0.05):
        self.pcd_path = pcd_path
        self.voxel_size = voxel_size
        self.points = None
        self.colors = None
        
    def load_and_preprocess(self) -> None:
        """Load point cloud and apply voxel downsampling."""
        pcd = o3d.io.read_point_cloud(self.pcd_path)
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        self.points = np.asarray(pcd.points)
        self.colors = np.asarray(pcd.colors)
        
        print(f"Loaded point cloud with {len(self.points)} points")


class CameraProjector:
    """Handles projection of 3D points to camera images."""
    
    def __init__(self, cameras: Dict[str, CameraParameters], img_width: int, img_height: int):
        self.cameras = cameras
        self.img_width = img_width
        self.img_height = img_height
        
    def project_points(self, points_3d: np.ndarray, colors: np.ndarray, 
                      camera_params: CameraParameters) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points to camera image plane.
        
        Args:
            points_3d: 3D points in lidar frame (N x 4, homogeneous)
            colors: Point colors (N x 3)
            camera_params: Camera parameters
            
        Returns:
            Tuple of (depth_map, rgb_image)
        """
        # Transform points to camera frame
        points_cam = camera_params.T_cam_lidar @ points_3d
        points_cam = points_cam[:3].T  # Convert to (N x 3)
        
        # Project to image plane
        uv_homo = (camera_params.K @ points_cam.T).T
        uv = uv_homo[:, :2] / points_cam[:, 2:3]  # Normalize by depth
        
        # Filter valid projections
        valid_mask = self._get_valid_mask(uv, points_cam[:, 2])
        uv_valid = uv[valid_mask]
        depth_valid = points_cam[valid_mask, 2]
        colors_valid = colors[valid_mask]
        
        # Create depth map and RGB image
        depth_map = np.full((self.img_height, self.img_width), np.inf)
        rgb_image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        
        # Fill pixel values (handle occlusions with z-buffer)
        for (u, v), z, c in zip(uv_valid, depth_valid, colors_valid):
            u_int, v_int = int(round(u)), int(round(v))
            u_int = np.clip(u_int, 0, self.img_width - 1)
            v_int = np.clip(v_int, 0, self.img_height - 1)
            
            if z < depth_map[v_int, u_int]:
                depth_map[v_int, u_int] = z
                rgb_image[v_int, u_int] = (c * 255).astype(np.uint8)
        
        # Set infinite depths to zero
        depth_map[depth_map == np.inf] = 0
        
        return depth_map, rgb_image
    
    def _get_valid_mask(self, uv: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """Get mask for valid projections (within image bounds and positive depth)."""
        return ((uv[:, 0] >= 0) & (uv[:, 0] < self.img_width) & 
                (uv[:, 1] >= 0) & (uv[:, 1] < self.img_height) & 
                (depths > 0))


class ProjectionSystem:
    """Main system orchestrating the point cloud to image projection pipeline."""
    
    def __init__(self, config: ProjectionConfig):
        self.config = config
        self.gt_loader = GroundTruthLoader(config.gt_tum_path)
        self.pc_processor = PointCloudProcessor(config.pcd_path, config.voxel_size)
        self.cameras = self._initialize_cameras()
        self.projector = CameraProjector(self.cameras, config.img_width, config.img_height)
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.png_dir, exist_ok=True)
    
    def _initialize_cameras(self) -> Dict[str, CameraParameters]:
        """Initialize camera parameters for all cameras."""
        cameras = {
            "cam0": CameraParameters(
                K=np.array([
                    [698.3946772032825, 0.0, 718.431814292891],
                    [0.0, 698.4629665854908, 559.7771701003702],
                    [0.0, 0.0, 1.0]
                ]),
                D=np.array([-0.04707469981321094, 0.011515146787673071, 
                           -0.008908394423298474, 0.0018561099343051583]),
                T_cam_lidar=np.array([
                    [0.007194409453692541, 0.9999715416500337, 0.002270762573978069, 0.003242860366163889],
                    [0.004589905134308125, 0.0022377749953642057, -0.9999869624819755, -0.07368532755947366],
                    [-0.999963585958744, 0.007204738241246789, -0.004573675043599734, -0.05485800045216396],
                    [0.0, 0.0, 0.0, 1.0]
                ])
            ),
            "cam1": CameraParameters(
                K=np.array([
                    [698.7077057399946, 0.0, 720.4468046683089],
                    [0.0, 699.7213977428826, 539.6639557186915],
                    [0.0, 0.0, 1.0]
                ]),
                D=np.array([-0.04010020519601471, -0.00044301230334375594, 
                           -7.327926294282181e-05, -0.00017811009939676002]),
                T_cam_lidar=np.array([
                    [-0.999964543690561, 0.00816136472446813, 0.0020744848903766324, 0.002445973115576358],
                    [-0.0020990354958363975, -0.003000734812745681, -0.9999932947978047, -0.07369303566179404],
                    [-0.008155085021838376, -0.9999621931435289, 0.003017759412014723, -0.057386869899386704],
                    [0.0, 0.0, 0.0, 1.0]
                ])
            ),
            "cam2": CameraParameters(
                K=np.array([
                    [698.6954304056306, 0.0, 714.6369903593052],
                    [0.0, 697.3997961335664, 532.530974606606],
                    [0.0, 0.0, 1.0]
                ]),
                D=np.array([-0.036603475396788196, -0.007341373813717986, 
                           0.005371575306265293, -0.0018297510848552124]),
                T_cam_lidar=np.array([
                    [0.9999969590188027, -0.0019062966654495807, -0.00156460415799267, -0.002246894712629628],
                    [-0.0015620417693603859, 0.0013449191744065846, -0.9999978756067066, -0.07344206996606042],
                    [0.001908396881858144, 0.9999972786090163, 0.0013419373695164598, -0.05028255821151065],
                    [0.0, 0.0, 0.0, 1.0]
                ])
            )
        }
        return cameras
    
    def process(self) -> None:
        """Run the complete projection pipeline."""
        print("Loading point cloud...")
        self.pc_processor.load_and_preprocess()
        
        print("Loading ground truth poses...")
        timestamps, poses, quats = self.gt_loader.load_poses()
        
        print(f"Processing {len(timestamps)} poses (every {self.config.frame_skip}th frame)...")
        
        # Process frames
        for i in tqdm(range(0, len(timestamps), self.config.frame_skip), 
                     desc="Generating depth & RGB maps"):
            self._process_frame(i, timestamps[i], poses[i], quats[i])
    
    def _process_frame(self, frame_idx: int, timestamp: str, pose: np.ndarray, quat: np.ndarray) -> None:
        """Process a single frame for all cameras."""
        # Create transformation from TLS to lidar frame
        R_lidar_tls = R.from_quat(quat).as_matrix()
        T_lidar_tls = np.eye(4)
        T_lidar_tls[:3, :3] = R_lidar_tls
        T_lidar_tls[:3, 3] = pose
        T_tls_lidar = np.linalg.inv(T_lidar_tls)
        
        # Transform points to lidar frame
        points_homo = np.hstack([self.pc_processor.points, 
                                np.ones((self.pc_processor.points.shape[0], 1))]).T
        points_lidar = T_tls_lidar @ points_homo
        
        # Project for each camera
        for cam_name, cam_params in self.cameras.items():
            depth_map, rgb_image = self.projector.project_points(
                points_lidar, self.pc_processor.colors, cam_params
            )
            
            # Save outputs
            self._save_outputs(timestamp, cam_name, depth_map, rgb_image)
    
    def _save_outputs(self, timestamp: str, cam_name: str, 
                     depth_map: np.ndarray, rgb_image: np.ndarray) -> None:
        """Save depth map and RGB image to files."""
        # Save numpy depth map
        depth_filename = f"{timestamp}_{cam_name}_blenheim_depth.npy"
        np.save(os.path.join(self.config.output_dir, depth_filename), depth_map)
        
        # Save depth visualization
        depth_png = f"{timestamp}_{cam_name}_blenheim_depth.png"
        plt.imsave(os.path.join(self.config.png_dir, depth_png), depth_map, cmap='plasma')
        
        # Save RGB image
        rgb_filename = f"{timestamp}_{cam_name}_blenheim_rgb.png"
        Image.fromarray(rgb_image).save(os.path.join(self.config.output_dir, rgb_filename))


def main():
    """Main entry point."""
    # Configuration
    config = ProjectionConfig(
        pcd_path="Bodelian/pointclouds.pcd",
        gt_tum_path="Bodelian/gt-tum.txt",
        output_dir="Bodelian/bodeliandata",
        png_dir="Bodelian/png",
        img_width=1440,
        img_height=1080,
        voxel_size=0.05,
        frame_skip=100
    )
    
    # Run processing
    system = ProjectionSystem(config)
    system.process()
    
    print("Processing complete!")


if __name__ == "__main__":
    main()