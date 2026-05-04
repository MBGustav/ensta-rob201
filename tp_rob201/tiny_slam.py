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

        # Optional improvements
        self.use_bilinear_score_TP4 = True
        # - localisation method: "random" (default) or "cem"
        self.localise_method = "cem"

        # ===================
        # CEM parameters 
        # ===================
        self.cem_population  = 40    # era 20 — cobertura do espaço
        self.cem_iterations  = 15    # era 10 — tempo para convergir
        self.cem_elite_frac  = 0.25  # era 0.3 → nro. elites (mais seletivo)
        self.cem_alpha       = 0.7   # era 0.4 — converge mais rápido para elites
        self.cem_min_std     = np.array([0.1, 0.1, 0.005], dtype=float)

        # Permite correções maiores
        self.max_ref_update  = np.array([40.0, 40.0, 0.5], dtype=float)  # era [25, 25, 0.35]
        
    @staticmethod
    def _wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
        return (theta + np.pi) % (2 * np.pi) - np.pi
    
    def score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        if self.use_bilinear_score_TP4:
            return self._score_bilinear(lidar, pose)  # New version with bilinear interpolation on the grid
        else:
            return self._score_nearest(lidar, pose)

    def _score_nearest(self, lidar, pose):
        """Naive score using nearest neighbor on the grid."""
        pose = np.asarray(pose, dtype=float)
        x_ref, y_ref, theta_ref = pose

        lidar_ranges = np.asarray(lidar.get_sensor_values(), dtype=float)
        ray_angles = np.asarray(lidar.get_ray_angles(), dtype=float)

        valid = (lidar_ranges > 0.0) & (lidar_ranges < float(lidar.max_range))
        if not np.any(valid):
            return -float("inf")
        lidar_ranges = lidar_ranges[valid]
        ray_angles = ray_angles[valid]

        angles = ray_angles + theta_ref
        x = x_ref + lidar_ranges * np.cos(angles)
        y = y_ref + lidar_ranges * np.sin(angles)

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

        return float(np.sum(self.grid.occupancy_map[x_px, y_px]))
    
    def _score_bilinear(self, lidar, pose):
        """Same as _score_nearest, but using bilinear interpolation on the grid."""
        pose = np.asarray(pose, dtype=float)
        x_ref, y_ref, theta_ref = pose

        lidar_ranges = np.asarray(lidar.get_sensor_values(), dtype=float)
        ray_angles = np.asarray(lidar.get_ray_angles(), dtype=float)

        valid = (lidar_ranges > 0.0) & (lidar_ranges < float(lidar.max_range))
        if not np.any(valid):
            return -float("inf")
        lidar_ranges = lidar_ranges[valid]
        ray_angles = ray_angles[valid]

        angles = ray_angles + theta_ref
        x = x_ref + lidar_ranges * np.cos(angles)
        y = y_ref + lidar_ranges * np.sin(angles)

        # Float map coordinates
        u = (x - self.grid.x_min_world) / self.grid.resolution
        v = (y - self.grid.y_min_world) / self.grid.resolution

        i0 = np.floor(u).astype(int)
        j0 = np.floor(v).astype(int)
        a = u - i0
        b = v - j0

        # Need i0+1 and j0+1 to be inside
        inside_mask = (
            (i0 >= 0)
            & (i0 < int(self.grid.x_max_map) - 1)
            & (j0 >= 0)
            & (j0 < int(self.grid.y_max_map) - 1)
        )
        if not np.any(inside_mask):
            return -float("inf")

        i0 = i0[inside_mask]
        j0 = j0[inside_mask]
        a = a[inside_mask]
        b = b[inside_mask]

        m = self.grid.occupancy_map
        v00 = m[i0, j0]
        v10 = m[i0 + 1, j0]
        v01 = m[i0, j0 + 1]
        v11 = m[i0 + 1, j0 + 1]

        vals = (1 - a) * (1 - b) * v00 + a * (1 - b) * v10 + (1 - a) * b * v01 + a * b * v11
        return float(np.sum(vals))



    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose.
        """
        ref = self.odom_pose_ref if odom_pose_ref is None else np.asarray(odom_pose_ref, dtype=float)
        odom_pose = np.asarray(odom_pose, dtype=float)

        x_ref, y_ref, theta_ref = ref
        x_o, y_o, theta_o = odom_pose

        c, s = np.cos(theta_ref), np.sin(theta_ref)
        
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
        
        if self.localise_method == "cem":
            return self._localise_cem(lidar, raw_odom_pose)
        
        
        raw_odom_pose = np.asarray(raw_odom_pose, dtype=float)

        best_ref = np.asarray(self.odom_pose_ref, dtype=float).copy()
        best_pose = self.get_corrected_pose(raw_odom_pose, best_ref)
        best_score = self.score(lidar, best_pose)

        draws_without_improve = 0
        while draws_without_improve < int(self.score_iterations):
            offset = np.random.normal(loc=0.0, scale=self.localise_std, size=3)
            cand_ref = best_ref + offset
            cand_pose = self.get_corrected_pose(raw_odom_pose, cand_ref)
            cand_score = self.score(lidar, cand_pose)

            if cand_score > best_score:
                best_score = cand_score
                best_ref = cand_ref
                draws_without_improve = 0
            else:
                draws_without_improve += 1

        self.odom_pose_ref = best_ref
        return best_score

    def _localise_cem(self, lidar, raw_odom_pose):
        """Cross-Entropy Method optimisation of the odom->map reference pose."""
        raw_odom_pose = np.asarray(raw_odom_pose, dtype=float)

        base_ref = np.asarray(self.odom_pose_ref, dtype=float).copy()
        best_delta = np.zeros(3, dtype=float)
        best_pose = self.get_corrected_pose(raw_odom_pose, base_ref)
        best_score = self.score(lidar, best_pose)

        mu = np.zeros(3, dtype=float)
        std = np.asarray(self.localise_std, dtype=float).copy()

        k = int(self.cem_population)
        elite_n = max(1, int(np.ceil(self.cem_elite_frac * k)))

        for _ in range(int(self.cem_iterations)):
            deltas = np.random.normal(loc=mu, scale=std, size=(k, 3))
            # Always evaluate current mean and current best
            if k >= 1:
                deltas[0] = mu
            if k >= 2:
                deltas[1] = best_delta

            scores = np.empty(k, dtype=float)
            for i in range(k):
                cand_ref = base_ref + deltas[i]
                cand_pose = self.get_corrected_pose(raw_odom_pose, cand_ref)
                scores[i] = self.score(lidar, cand_pose)

            elite_idx = np.argsort(scores)[-elite_n:]
            elite = deltas[elite_idx]
            elite_scores = scores[elite_idx]

            # Keep best seen candidate
            top_i = elite_idx[int(np.argmax(elite_scores))]
            if scores[top_i] > best_score:
                best_score = scores[top_i]
                best_delta = deltas[top_i]

            mu_new = elite.mean(axis=0)
            std_new = elite.std(axis=0)
            mu = (1.0 - self.cem_alpha) * mu + self.cem_alpha * mu_new
            std = (1.0 - self.cem_alpha) * std + self.cem_alpha * std_new
            std = np.maximum(std, self.cem_min_std)

        # Apply a bounded update to avoid sudden teleportation when scan matching fails.
        applied_delta = np.asarray(best_delta, dtype=float)
        applied_delta[0:2] = np.clip(applied_delta[0:2], -self.max_ref_update[0:2], self.max_ref_update[0:2])
        applied_delta[2] = float(np.clip(applied_delta[2], -self.max_ref_update[2], self.max_ref_update[2]))

        new_ref = base_ref + applied_delta
        new_ref[2] = self._wrap_angle(new_ref[2])
        self.odom_pose_ref = new_ref
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
        x_ref, y_ref, theta_ref = np.asarray(pose, dtype=float)
        
        # 1. Convert lidar data to numpy arrays
        lidar_ranges = np.asarray(lidar.get_sensor_values(), dtype=float)
        ray_angles = np.asarray(lidar.get_ray_angles(), dtype=float)
        max_range = float(lidar.max_range)
        
        # 1.1 Filter out invalid ranges
        # Keep "no-hit" rays reported at exactly max_range so we can still mark free space.
        valid_mask = (lidar_ranges > 0.0) & np.isfinite(lidar_ranges) & (lidar_ranges <= max_range)
        if not np.any(valid_mask):
            return

        lidar_ranges = lidar_ranges[valid_mask]
        ray_angles = ray_angles[valid_mask]

        # Separate "hits" (actual obstacle return) from "no return" (at max range)
        hit_mask = lidar_ranges < max_range
        
        
        # 2. Rays in world frame
        angles = ray_angles + theta_ref
        c = np.cos(angles)
        s = np.sin(angles)

        # Used to get rid of unknown cells in front of the robot.
        # If the lidar range is 100, and the obstacle is at 90, we want to mark free space up to ~80, not all the way to 100.
        free_end_offset = 2.0 * float(self.grid.resolution)
        free_ranges = np.maximum(lidar_ranges - free_end_offset, 0.0)
        
        
        x_free = x_ref + free_ranges * c
        y_free = y_ref + free_ranges * s

        x_occ = x_ref + lidar_ranges[hit_mask] * c[hit_mask]
        y_occ = y_ref + lidar_ranges[hit_mask] * s[hit_mask]
            
    
        # 3. Update occupancy grid (log-odds style increments)
        # Free space along all rays
        for x1, y1 in zip(x_free, y_free):
            self.grid.add_value_along_line(x_ref, y_ref, float(x1), float(y1), val=-0.6)

        # Occupied only when we really hit something
        if len(x_occ) > 0:
            self.grid.add_map_points(x_occ, y_occ, val=2.0)
        
        np.clip(self.grid.occupancy_map, -40, 40, out=self.grid.occupancy_map) # Clip values to [0, 100]
    
        # 4. plot for debug
        self.grid.display_cv(pose)
    


    def compute(self):
        """ Useless function, just for the exercise on using the profiler """
        # Remove after TP1

        ranges = np.random.rand(3600)
        ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

