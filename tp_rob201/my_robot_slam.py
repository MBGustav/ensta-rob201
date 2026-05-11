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
        self.goal_tolerance = 30.0 
        
        self.exploration_state = "explore"
        self.current_goal = np.array([0., 0., 0.])
        self.planned_path = None
        self.force_replan = True
        self.ticks_failed = 0
        
        self.consecutive_path_failures = 0
        self.max_path_failures = 5
        
        self.last_dist_to_goal = np.inf
        self.no_progress_ticks = 0
        self.max_no_progress_ticks = 100
        
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

    def step_check_progress(self):
        """Força replan se o robô não está progredindo em direção ao goal."""
        if self.planned_path is None or self.exploration_state == "return":
            return

        dist = np.linalg.norm(self.corrected_pose[:2] - self.current_goal[:2])

        if dist < self.last_dist_to_goal - 5.0:  # progrediu pelo menos 5 unidades
            self.last_dist_to_goal = dist
            self.no_progress_ticks = 0
        else:
            self.no_progress_ticks += 1

        if self.no_progress_ticks >= self.max_no_progress_ticks:
            print(f"Sem progresso há {self.max_no_progress_ticks} ticks → novo frontier")
            self.no_progress_ticks = 0
            self.last_dist_to_goal = np.inf
            self.force_replan = True
            self.planned_path = None
            
    def step_location(self, pose):
        """
        Step function for localization only, to test the localization part of the SLAM
        """
        
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

        
    def step_replanning(self):
        if not (
            self.planned_path is None
            or self.force_replan
            or self.counter % 150 == 0  # era 50 — menos replanning
        ):
            return

        if self.exploration_state == "explore":
            frontier_goal = self.planner.explore_frontiers(self.corrected_pose)

            if np.allclose(frontier_goal[:2], [0.0, 0.0], atol=1.0):
                print("Sem frontier → retornando")
                self.exploration_state = "return"
                self.current_goal = np.array([0., 0., 0.])
            else:
                # Só atualiza o goal se o frontier mudou bastante
                dist_to_current_goal = np.linalg.norm(frontier_goal[:2] - self.current_goal[:2])
                if dist_to_current_goal < 20.0 and not self.force_replan:
                    return  # frontier quase igual, mantém plano atual
                
                self.current_goal = frontier_goal
                occ_map = self.occupancy_grid.occupancy_map
                frac_unknown = np.sum((occ_map >= -0.1) & (occ_map <= 0.5)) / occ_map.size
                print(f"Frontier: {self.current_goal[:2]} (unknown={frac_unknown:.4f})")

        elif self.exploration_state == "return":
            self.current_goal = np.array([0., 0., 0.])

        print(f"[{self.exploration_state}] Planejando para {self.current_goal[:2]}")

        try:
            self.planned_path = self.planner.plan(
                self.corrected_pose, self.current_goal, mu=1.0
            )
            if not self.planned_path:
                self.consecutive_path_failures += 1
                print(f"Falha {self.consecutive_path_failures}/{self.max_path_failures}")
                
                if self.consecutive_path_failures >= self.max_path_failures:
                    print("Muitas falhas → retornando para casa")
                    self.exploration_state = "return"
                    self.current_goal = np.array([0., 0., 0.])
                    self.consecutive_path_failures = 0
                    self.planned_path = self.planner.plan(
                        self.corrected_pose, self.current_goal, mu=1.0
                    )
                else:
                    self.force_replan = True
                return
        except Exception as e:
            print("Erro no planejamento:", e)
            self.planned_path = None
            self.force_replan = True
            return

        self.consecutive_path_failures = 0  # reset ao encontrar path válido
        self.force_replan = False


    def step_execute_plan(self):
        if self.planned_path is not None and len(self.planned_path) > 0:
            dist_to_goal = np.linalg.norm(self.corrected_pose[:2] - self.current_goal[:2])

            if dist_to_goal < self.goal_tolerance:
                if self.exploration_state == 'return':
                    print("Home reached. Stopping.")
                    return {"forward": 0.0, "rotation": 0.0}
                else:
                    self.force_replan = True
                    command = reactive_obst_avoid(self.lidar())
            else:
                try:
                    command = path_following_control(
                        self.lidar(), self.corrected_pose, self.planned_path, lidar_weight=0.25
                    )
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
            command = reactive_obst_avoid(self.lidar())
            self.force_replan = True

        return command
    
    
    def step_display(self):
        if self.counter % 5 == 0:
            try:
                if self.planned_path is not None and len(self.planned_path) > 0:
                    traj_array = np.array(self.planned_path).T
                    self.occupancy_grid.display_cv(self.corrected_pose, traj=traj_array)
                else:
                    self.occupancy_grid.display_cv(self.corrected_pose)
            except Exception as e:
                print(f"Display error: {e}")

    def step_validate_path(self):
        if self.planned_path is None or len(self.planned_path) == 0:
            return

        lookahead = min(10, len(self.planned_path))
        for waypoint in self.planned_path[:lookahead]:
            cell = self.occupancy_grid.conv_world_to_map(waypoint[0], waypoint[1])
            cx, cy = int(cell[0]), int(cell[1])

            if not (0 <= cx < self.occupancy_grid.x_max_map and
                    0 <= cy < self.occupancy_grid.y_max_map):
                continue

            # Verifica parede inflada
            if hasattr(self.planner, 'map_walls') and self.planner.map_walls[cx, cy] > 0.5:
                print("Waypoint em parede inflada → replanando")
                self.force_replan = True
                self.planned_path = None
                return

            # Verifica parede recém descoberta no mapa bruto
            if self.occupancy_grid.occupancy_map[cx, cy] > 2.0:
                print("Waypoint em parede recém descoberta → replanando")
                self.force_replan = True
                self.planned_path = None
                return
    
    def control_tp5(self):
        pose = self.odometer_values()

        self.step_location(pose)
        self.step_validate_path()
        self.step_check_progress()   # ← novo
        self.step_replanning()
        command = self.step_execute_plan()
        self.step_display()

        self.counter += 1
        return command
            