""" A simple robotics navigation code including SLAM, exploration, planning"""

import time

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])
        
        # score parameters
        # N: number of random draws without improvement before stopping
        self.score_iterations = 20
        # Standard deviation for the random search (x, y, theta)
        self.localise_std = np.array([2.0, 2.0, 0.05], dtype=float)
        
    @staticmethod
    def _wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
        return (theta + np.pi) % (2 * np.pi) - np.pi
    
    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        pose = np.asarray(pose, dtype=float)
        x_ref, y_ref, theta_ref = pose

        lidar_ranges = np.asarray(lidar.get_sensor_values(), dtype=float)
        ray_angles = np.asarray(lidar.get_ray_angles(), dtype=float)

        # 1) Remove max-range (no obstacle) and invalid values
        valid = (lidar_ranges > 0.0) & (lidar_ranges < float(lidar.max_range))
        if not np.any(valid):
            return -float("inf")

        lidar_ranges = lidar_ranges[valid]
        ray_angles = ray_angles[valid]

        # 2) Endpoints in absolute/world frame
        angles = ray_angles + theta_ref
        x = x_ref + lidar_ranges * np.cos(angles)
        y = y_ref + lidar_ranges * np.sin(angles)

        # 3) Convert to map indices and filter out-of-map
        x_px, y_px = self.grid.conv_world_to_map(x, y)
        inside = (
            (x_px >= 0)
            & (x_px < self.grid.x_max_map)
            & (y_px >= 0)
            & (y_px < self.grid.y_max_map)
        )
        if not np.any(inside):
            return -float("inf")

        x_px = x_px[inside]
        y_px = y_px[inside]

        # 4) Sum grid values
        return float(np.sum(self.grid.occupancy_map[x_px, y_px]))


    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        ref = self.odom_pose_ref if odom_pose_ref is None else np.asarray(odom_pose_ref, dtype=float)
        odom_pose = np.asarray(odom_pose, dtype=float)

        x_ref, y_ref, theta_ref = ref
        x_o, y_o, theta_o = odom_pose

        c = np.cos(theta_ref)
        s = np.sin(theta_ref)
        x = x_ref + c * x_o - s * y_o
        y = y_ref + s * x_o + c * y_o
        theta = self._wrap_angle(theta_ref + theta_o)

        return np.array([x, y, theta], dtype=float)



    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        return best_score : best score found
        """
        raw_odom_pose = np.asarray(raw_odom_pose, dtype=float)

        best_ref = np.asarray(self.odom_pose_ref, dtype=float).copy()
        best_pose = self.get_corrected_pose(raw_odom_pose, best_ref)
        best_score = self._score(lidar, best_pose)

        draws_without_improve = 0
        while draws_without_improve < int(self.score_iterations):
            offset = np.random.normal(loc=0.0, scale=self.localise_std, size=3)
            cand_ref = best_ref + offset
            cand_pose = self.get_corrected_pose(raw_odom_pose, cand_ref)
            cand_score = self._score(lidar, cand_pose)

            if cand_score > best_score:
                best_score = cand_score
                best_ref = cand_ref
                draws_without_improve = 0
            else:
                draws_without_improve += 1

        self.odom_pose_ref = best_ref
        return best_score
    
    
    
        
    def polar_to_cartesian(self, ranges, ray_angles):
        """
        Convert polar coordinates to cartesian coordinates in the robot frame
        """
        points = []
        for i in range(len(ranges)):
            pt_x = ranges[i] * np.cos(ray_angles[i])
            pt_y = ranges[i] * np.sin(ray_angles[i])
            points.append([pt_x, pt_y])
        return np.array(points).T

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        x_ref, y_ref, theta_ref = pose
        
        # 1. Convert lidar data to cartesian coordinates
        lidar_ranges = np.asarray(lidar.get_sensor_values())
        ray_angles = np.asarray(lidar.get_ray_angles())
        
        # 1.1 Filter out invalid ranges
        valid_mask = (lidar_ranges > 0.0) & (lidar_ranges < lidar.max_range)
        lidar_ranges = lidar_ranges[valid_mask]
        ray_angles = ray_angles[valid_mask]
        
        
        # 2. Obtain real robot coordinates(world frame)
        x = x_ref + lidar_ranges * np.cos(ray_angles + theta_ref)
        y = y_ref + lidar_ranges * np.sin(ray_angles + theta_ref)
        theta = theta_ref + ray_angles
            
    
        # 3. Update occupancy grid
        for i in range(len(x)):
            self.grid.add_value_along_line(x_ref, y_ref, x[i], y[i], val=-0.95)  
        self.grid.add_map_points(x, y, val=6)  # Mark observed points as occupied
        
        np.clip(self.grid.occupancy_map, -40, 40, out=self.grid.occupancy_map) # Clip values to [0, 100]
        
        print(f"pose: {pose}, num points: {len(x)}")
        
        # 4. plot for debug
        self.grid.display_cv(pose)



    def compute(self):
        """ Useless function, just for the exercise on using the profiler """
        # Remove after TP1

        ranges = np.random.rand(3600)
        ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

        # Poor implementation of polar to cartesian conversion
        # points = []
        # for i in range(3600):
        #     pt_x = ranges[i] * np.cos(ray_angles[i])
        #     pt_y = ranges[i] * np.sin(ray_angles[i])
        #     points.append([pt_x, pt_y])

