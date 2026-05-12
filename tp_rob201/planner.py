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
        x, y = current_cell
        neighbors = []
        directions = [  # (dx, dy)
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)
        ]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid.x_max_map and 0 <= ny < self.grid.y_max_map:
                if self.map_walls[nx, ny] < 0.5:
                    # For diagonal moves, ensure both adjacent sides are free (no corner cutting)
                    if abs(dx) == 1 and abs(dy) == 1:
                        if self.map_walls[x + dx, y] > 0.5 or self.map_walls[x, y + dy] > 0.5:
                            continue
                    neighbors.append((nx, ny))
        return neighbors

    def heuristic(self, cell_1, cell_2):
        """ Euclidean distance """
        return math.hypot(cell_1[0] - cell_2[0], cell_1[1] - cell_2[1])

    def setup_wall_map(self):
        occ_threshold = 2
        inflation_kernel_size = 16  # era 15 — muito grande, bloqueava corredores
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflation_kernel_size, inflation_kernel_size))
        occ_map_bin = (self.grid.occupancy_map > occ_threshold).astype(np.uint8)
        self.map_walls = cv2.dilate(occ_map_bin, kernel)
        cv2.imwrite("inflated_walls.png", self.map_walls * 255)
        
        
    def plan(self, start, goal, mu=1.0):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        mu : float, weight of the heuristic function (weighted A*)
        """
        
        
        # save inflated walls for debug
        self.setup_wall_map()
        
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

        free    = grid < -0.01
        unknown = (grid >= -0.01) & (grid <= 0.1)

        kernel = np.ones((3, 3), np.uint8)
        free_dilated = cv2.dilate(free.astype(np.uint8), kernel)
        frontier_mask = unknown & (free_dilated > 0)

        # Garante que map_walls está atualizado antes de filtrar
        self.setup_wall_map()
        frontier_mask = frontier_mask & (self.map_walls < 0.5)

        points = np.column_stack(np.where(frontier_mask))

        if len(points) == 0:
            return np.array([0., 0., 0.])

        rx, ry = self.grid.conv_world_to_map(current_pose[0], current_pose[1])
        robot  = np.array([rx, ry])
        dists  = np.linalg.norm(points - robot, axis=1)
        best   = points[np.argmin(dists)]

        xw, yw = self.grid.conv_map_to_world(best[0], best[1])
        return np.array([xw, yw, 0.0])
    
    