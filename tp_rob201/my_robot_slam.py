"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.simulation.robot.robot_abstract import RobotAbstract
from place_bot.simulation.robot.odometer import OdometerParams
from place_bot.simulation.ray_sensors.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid, path_following_control
from occupancy_grid import OccupancyGrid
from planner import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])
        self.localisation_score_threshold = 5.0
        self.localisation_warmup_steps = 80
        self.goal_tolerance = 55.0
    def control(self):
        """
        Main control function executed at each time step
        """
        return self.control_tp5()

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        
        pose = self.odometer_values()

        if self.counter >= self.localisation_warmup_steps:
            localisation_score = self.tiny_slam.localise(self.lidar(), pose)
        else:
            localisation_score = self.localisation_score_threshold

        self.corrected_pose = self.tiny_slam.get_corrected_pose(pose)

        if localisation_score >= self.localisation_score_threshold:
            self.tiny_slam.update_map(self.lidar(), self.corrected_pose)

        if self.counter % 10 == 0:
            self.occupancy_grid.display_cv(self.corrected_pose)

        self.counter += 1

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.odometer_values()
        goal = [0,0,0]

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), pose, goal)

        return command

    def control_tp5(self):
        """
        Control function for TP5: Frontier exploration with path planning
        - SLAM with localization
        - Frontier-based exploration
        - A* path planning
        - Path following with lookahead
        """
        pose = self.odometer_values()

        # Always try localization
        try:
            localisation_score = self.tiny_slam.localise(self.lidar(), pose)
        except:
            localisation_score = self.localisation_score_threshold

        try:
            self.corrected_pose = self.tiny_slam.get_corrected_pose(pose)
        except:
            self.corrected_pose = pose

        # Always update map to ensure consistency
        try:
            self.tiny_slam.update_map(self.lidar(), self.corrected_pose)
        except Exception as e:
            print("Mapping error:", e)

        # Initialize exploration state on first run
        if getattr(self, 'exploration_state', None) is None:
            self.exploration_state = 'explore'
            self.current_goal = np.array([0., 0., 0.])
            self.planned_path = None
            self.force_replan = True
            self.ticks_failed = 0

        # Warmup phase: build initial map
        if self.counter < 100:
            command = reactive_obst_avoid(self.lidar())
            if self.counter % 5 == 0:
                try:
                    self.occupancy_grid.display_cv(self.corrected_pose)
                except:
                    pass
            self.counter += 1
            return command

        # Replan when necessary: no path, force replan, or periodically
        if self.planned_path is None or self.force_replan or self.counter % 300 == 0:
            if self.exploration_state == 'explore':
                # Find nearest frontier
                self.current_goal = self.planner.explore_frontiers(self.corrected_pose)
                # If no frontier found, switch to return state
                # Only mark exploration done if frontier is truly [0,0,0] or very close
                frontier_dist = np.linalg.norm(self.current_goal[:2])
                if frontier_dist < 1.0:
                    print("No more reachable frontiers! Exploration finished.")
                    self.exploration_state = 'return'
                    self.current_goal = np.array([0., 0., 0.])
                else:
                    print(f"Found frontier at distance {frontier_dist:.1f}")

            print(f"[{self.exploration_state}] Planning to: {self.current_goal[:2]}")
            try:
                self.planned_path = self.planner.plan(self.corrected_pose, self.current_goal, mu=1.0)
            except Exception as e:
                print(f"Planning error: {e}")
                self.planned_path = None
            self.force_replan = False
        
        # =============
        # Execute plan
        # =============
        if self.planned_path is not None and len(self.planned_path) > 0:
            dist_to_goal = np.linalg.norm(self.corrected_pose[:2] - self.current_goal[:2])
            
            # Check if reached goal area
            if dist_to_goal < self.goal_tolerance:
                print(f"Reached goal area: {self.current_goal[:2]}")
                if self.exploration_state == 'return':
                    command = {"forward": 0.0, "rotation": 0.0}
                    print("Exploration complete. Home reached.")
                else:
                    self.force_replan = True
                    command = reactive_obst_avoid(self.lidar())
            else:
                # Follow path with enhanced path following control
                d_pursuit = 35.0
                dists = [np.linalg.norm(self.corrected_pose[:2] - pt[:2]) for pt in self.planned_path]
                closest_idx = int(np.argmin(dists))
                
                try:
                    # Use new path following control with lateral force
                    command = path_following_control(self.lidar(), self.corrected_pose, self.planned_path, lidar_weight=0.25)
                    # Detect stuck robot
                    if command['forward'] < 0.1 and abs(command['rotation']) < 0.1:
                        self.ticks_failed += 1
                        if self.ticks_failed > 20:
                            self.force_replan = True
                            self.ticks_failed = 0
                    else:
                        self.ticks_failed = 0
                except Exception as e:
                    print(f"Control error: {e}")
                    command = {"forward": 0.0, "rotation": 0.0}
        else:
            # No valid path, fallback to obstacle avoidance
            command = reactive_obst_avoid(self.lidar())
            self.force_replan = True

        # Display trajectory
        if self.counter % 5 == 0:
            try:
                if self.planned_path is not None and len(self.planned_path) > 0:
                    traj_array = np.array(self.planned_path).T
                    self.occupancy_grid.display_cv(self.corrected_pose, traj=traj_array)
                else:
                    self.occupancy_grid.display_cv(self.corrected_pose)
            except Exception as e:
                print(f"Display error: {e}")

        self.counter += 1
        return command
                
            