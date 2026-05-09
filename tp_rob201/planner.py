"""
Planner class
Implementation of A*
"""

import math
import heapq
from collections import defaultdict
import cv2
import numpy as np

from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def get_neighbors(self, current_cell):
        """ 8-connected neighbors """
        x, y = current_cell
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid.x_max_map and 0 <= ny < self.grid.y_max_map:
                    # check obstacle
                    if self.map_walls[int(nx), int(ny)] < 0.5:
                        neighbors.append((nx, ny))
        return neighbors

    def heuristic(self, cell_1, cell_2):
        """ Euclidean distance """
        return math.hypot(cell_1[0] - cell_2[0], cell_1[1] - cell_2[1])

    def plan(self, start, goal, mu=1.0):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        mu : float, weight of the heuristic function (weighted A*)
        """
        # Threshold map without dilation to allow passage through narrow doorways
        # Using 1x1 kernel (essentially no dilation) - let potential field handle wall avoidance
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        self.map_walls = cv2.filter2D((self.grid.occupancy_map > 0).astype(np.float32), -1, kernel)

        start_cell = self.grid.conv_world_to_map(start[0], start[1])
        goal_cell = self.grid.conv_world_to_map(goal[0], goal[1])
        start_cell = (int(start_cell[0]), int(start_cell[1]))
        goal_cell = (int(goal_cell[0]), int(goal_cell[1]))

        # Validate start and goal cells are within bounds
        if not (0 <= start_cell[0] < self.grid.x_max_map and 0 <= start_cell[1] < self.grid.y_max_map):
            print(f"ERROR: Start cell {start_cell} out of bounds")
            return []
        
        if not (0 <= goal_cell[0] < self.grid.x_max_map and 0 <= goal_cell[1] < self.grid.y_max_map):
            print(f"ERROR: Goal cell {goal_cell} out of bounds")
            return []

        # Check if goal is in obstacle (blocked)
        if self.map_walls[int(goal_cell[0]), int(goal_cell[1])] > 0.5:
            print(f"WARNING: Goal cell {goal_cell} is in obstacle, adjusting...")
            # Try to find nearby free cell
            found_free = False
            for radius in range(1, 20):
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        nx, ny = goal_cell[0] + dx, goal_cell[1] + dy
                        if (0 <= nx < self.grid.x_max_map and 0 <= ny < self.grid.y_max_map and
                            self.map_walls[int(nx), int(ny)] < 0.5):
                            goal_cell = (int(nx), int(ny))
                            found_free = True
                            print(f"Adjusted goal to nearby free cell: {goal_cell}")
                            break
                    if found_free:
                        break
                if found_free:
                    break
            
            if not found_free:
                print(f"ERROR: Cannot find free cell near goal")
                return []

        openSet = []
        heapq.heappush(openSet, (0.0, start_cell))
        cameFrom = {}
        
        gScore = defaultdict(lambda: math.inf)
        gScore[start_cell] = 0.0
        
        fScore = defaultdict(lambda: math.inf)
        fScore[start_cell] = mu * self.heuristic(start_cell, goal_cell)

        openSet_hash = {start_cell}

        while openSet:
            _, current = heapq.heappop(openSet)
            
            if current in openSet_hash:
                openSet_hash.remove(current)

            if current == goal_cell:
                # reconstruct
                path_cells = [current]
                while current in cameFrom:
                    current = cameFrom[current]
                    path_cells.insert(0, current)
                
                # convert cells to world
                path_world = []
                for cell in path_cells:
                    xw, yw = self.grid.conv_map_to_world(cell[0], cell[1])
                    path_world.append([xw, yw, 0.0])
                print(f"Path found with {len(path_world)} waypoints")
                return path_world

            for neighbor in self.get_neighbors(current):
                # distance between connected cells
                d = 1.414 if abs(neighbor[0]-current[0]) + abs(neighbor[1]-current[1]) == 2 else 1.0
                tentative_gScore = gScore[current] + d
                
                if tentative_gScore < gScore[neighbor]:
                    cameFrom[neighbor] = current
                    gScore[neighbor] = tentative_gScore
                    fScore[neighbor] = tentative_gScore + mu * self.heuristic(neighbor, goal_cell)
                    if neighbor not in openSet_hash:
                        openSet_hash.add(neighbor)
                        heapq.heappush(openSet, (fScore[neighbor], neighbor))

        print(f"ERROR: No path found from {start_cell} to {goal_cell}")
        return []  # Return empty path instead of invalid fallback

    def explore_frontiers(self, current_pose):
        grid = self.grid.occupancy_map

        # 1. Free / unknown masks
        free = grid < -0.01
        unknown = (grid >= -0.01) & (grid <= 0.1)

        # 2. Frontier = unknown next to free
        kernel = np.ones((3, 3), np.uint8)
        free_dilated = cv2.dilate(free.astype(np.uint8), kernel)
        frontiers = unknown & (free_dilated > 0)

        points = np.column_stack(np.where(frontiers))

        if len(points) == 0:
            print(f"DEBUG: No frontier points found")
            return np.array([0., 0., 0.])

        # 3. Robot position in map frame
        rx, ry = self.grid.conv_world_to_map(current_pose[0], current_pose[1])
        robot = np.array([rx, ry])

        # 4. Filter frontier points - relax constraints for better exploration
        valid_frontiers = []
        for point in points:
            dist = np.linalg.norm(point - robot)
            # Looser distance constraint: 5 to 800 units
            # Don't check if in free space - frontier detection already ensures border between free/unknown
            if 5 < dist < 800:
                valid_frontiers.append(point)
        
        if len(valid_frontiers) == 0:
            print(f"DEBUG: No valid frontiers within distance range. Total points: {len(points)}")
            if len(points) > 0:
                # If we have points but none pass filters, just take closest anyway
                dists = np.linalg.norm(points - robot, axis=1)
                best_idx = np.argmin(dists)
                best = points[best_idx]
                print(f"DEBUG: Taking closest point anyway at distance {dists[best_idx]:.1f}")
            else:
                return np.array([0., 0., 0.])
        else:
            # Pick nearest valid frontier
            valid_frontiers = np.array(valid_frontiers)
            dists = np.linalg.norm(valid_frontiers - robot, axis=1)
            best_idx = np.argmin(dists)
            best = valid_frontiers[best_idx]
            print(f"DEBUG: Found {len(valid_frontiers)} valid frontiers, closest at distance {dists[best_idx]:.1f}")

        # 5. Convert back to world
        xw, yw = self.grid.conv_map_to_world(best[0], best[1])
        return np.array([xw, yw, 0.0])