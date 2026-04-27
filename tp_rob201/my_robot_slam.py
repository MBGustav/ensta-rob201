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

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

    def control(self):
        """
        Main control function executed at each time step
        """
        #For TP1:
        # return self.control_tp1()
        
        #For TP3:
        self.tiny_slam.localise(self.lidar(), self.odometer_values())
        return self.control_tp3()
    
    def slam_compute(self):
        self.tiny_slam.compute()

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        self.tiny_slam.compute()

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command

    def control_tp3(self):
        """
        Control function for TP3
        Main control function with full SLAM, random exploration and path planning
        """
        
        
        pose = self.odometer_values()
        
        
        # 1. Compute new command speed to perform obstacle avoidance
        command = self.control_tp1()
        
        # 2. Update odometry reference by scan-matching, then compute corrected pose
        self.tiny_slam.localise(self.lidar(), pose)
        self.corrected_pose = self.tiny_slam.get_corrected_pose(pose)

        # 3. Update map with corrected pose
        self.tiny_slam.update_map(self.lidar(), self.corrected_pose)
        
    
        
        return command
    
    
    

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        
        pose = self.odometer_values()

        # Update odometry reference by scan-matching, then compute corrected pose
        self.tiny_slam.localise(self.lidar(), pose)
        self.corrected_pose = self.tiny_slam.get_corrected_pose(pose)
        
        
        
        goal = -1. * np.array([100,100])
    

        # Compute new command speed to perform obstacle avoidance
        potential_command = potential_field_control(self.lidar(), pose, goal)
        
        # reactive_command = reactive_obst_avoid(self.lidar())
        
        
        #TODO: fuse the two commands by weighted averaging 
        # fusing to avoid obstacles while going to goal
        
        #Debug prints)
        
        return potential_command
