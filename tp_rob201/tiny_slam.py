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

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4

        score = 0

        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4
        corrected_pose = odom_pose

        return corrected_pose

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        best_score = 0

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
        
        # Filter out invalid ranges
        valid_mask = np.isfinite(lidar_ranges) & (lidar_ranges > 0.0) & (lidar_ranges < lidar.max_range)
        lidar_ranges = lidar_ranges[valid_mask]
        ray_angles = ray_angles[valid_mask]
        
        
        # 2. Obtain real robot coordinates(world frame)
        x = x_ref + lidar_ranges * np.cos(ray_angles + theta_ref)
        y = y_ref + lidar_ranges * np.sin(ray_angles + theta_ref)
        theta = theta_ref + ray_angles
        
        # 2.1 Adjust the lidar to the map
        points = self.polar_to_cartesian(lidar_ranges, ray_angles)
        
    
        # 3. Update occupancy grid
        for i in range(len(x)):
            self.grid.add_value_along_line(x_ref, y_ref, points[0,i], points[1,i], val=-0.95)  
        self.grid.add_map_points(x, y, val=6)  # Mark observed points as occupied
        
        np.clip(self.grid.occupancy_map, -40, 40, out=self.grid.occupancy_map)  # Clip values to [0, 100]
        
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

