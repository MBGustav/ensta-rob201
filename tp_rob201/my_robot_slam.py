"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.simulation.robot.robot_abstract import RobotAbstract
from place_bot.simulation.robot.odometer import OdometerParams
from place_bot.simulation.ray_sensors.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid
from occupancy_grid import OccupancyGrid
from planner import Planner

itr_debug = 0
def dbg_print(*args, **kwargs):
    global itr_debug
    if kwargs.pop('info', None) or itr_debug % 100 == 0:
        print(*args, **kwargs)
    itr_debug += 1
    

# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(lidar_params=lidar_params,
                         odometer_params=odometer_params)


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


        # Path planner
        self.planner = Planner(self.occupancy_grid)
        
        self.tiny_slam = TinySlam(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])
        
        
        # ================================ 
        # Storage for goals (user-defined)
        # ================================  
        self.goals_list = [ 
                           np.array([-260, -480, 0]), 
                           np.array([-460, -480, 0]), # Small room - down left
                           np.array([-260, -480, 0]), #comme back to corridor
                           np.array([-250, -189, 0]), 
                           np.array([-400, -20,  0]),
                           np.array([-400, -50,  0]),
                           np.array([-210, -25,  0]),
                           np.array([-800, -40,  0]),
                           np.array([-800, -230, 0]),
                          ]
        self.curr_goal_idx = 0
        self.goal_tol = 100
        self.itr_debug = 0
        
        
        self.map_filename = "maps/final_map"
        # self.planner.A_start(np.array([0, 0, 0]), self.goals_list[-1])
        # self.goal_planner = goals_list[-1]
    def get_curr_goal(self):
        if self.curr_goal_idx < len(self.goals_list):
            return self.goals_list[self.curr_goal_idx]
        else:
            return None
    
    def check_goal_reached(self, current_pose):
        goal = self.get_curr_goal()
        if goal is not None:
            dist_to_goal = np.linalg.norm(current_pose[:2] - goal[:2])
            if dist_to_goal < self.goal_tol:
                print(f"Goal {self.curr_goal_idx} reached at position {current_pose[:2]} with distance {dist_to_goal:.2f}")
                self.curr_goal_idx += 1
                return True
        return False
    
    def control(self):
        """
        Main control function executed at each time step
        """
        pose = self.odometer_values()
        lidar = self.lidar()
        
            # 1) Scan-matching localisation (measurement)
        self.tiny_slam.localise(lidar, pose)
        self.corrected_pose = self.tiny_slam.get_corrected_pose(pose)

        # 3) Compute command using filtered pose
        command = potential_field_control(lidar=lidar, current_pose=pose, goal_pose= self.get_curr_goal())

        if self.check_goal_reached(self.corrected_pose):
            print(f"New goal: {self.get_curr_goal()}")
            
        
        # 4) Update map with filtered pose
        self.tiny_slam.update_map(lidar, self.corrected_pose)
        

        return command
    
    
    def slam_compute(self):
        self.tiny_slam.compute()

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        self.tiny_slam.compute()

        # Compute new command speed to perform obstacle avoidance
        return command
