""" A set of robotics control functions """
import time
import numpy as np



def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1
    # laser_dist = lidar.get_sensor_values()
    # speed = 1
    # rotation_speed = 1

    # command = {"forward": speed,
    #            "rotation": rotation_speed}

    # return command
    return avoidance_method_TP1(lidar)


def avoidance_method_TP1(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """

    # Default speeds
    speed = 50           # velocidade para frente
    rotation_speed = 0   # sem rotação inicialmente
    
    # Lidar readings
    laser_dist = np.array(lidar.get_sensor_values())
    
    # Definir FOV 
    main_direction = 180
    threshold_angle = 80
    # FOV angles 
    fov_angles = np.arange(main_direction - threshold_angle, main_direction + threshold_angle)
    
    # Filtrar leituras dentro do FOV
    laser_dist_fov = laser_dist[fov_angles]
    
    # Distâncias mínimas e máximas
    max_dist = np.max(laser_dist_fov)
    min_dist = np.min(laser_dist_fov)
    max_index = np.argmax(laser_dist_fov)
    
    
    # --- Simple obstacle avoidance logic ---
    safe_distance = 30  # distância mínima segura para obstáculo
    
    
    if min_dist < safe_distance:
        # Se um obstáculo estiver muito próximo, gira para evitar
        rotation_speed = 50  # gira para a direita
        speed = 0           # para de avançar
    elif max_dist > safe_distance:
        # Se houver um caminho livre, avança
        speed = 50          # avança para frente
        rotation_speed = 0   # sem rotação
    else:
        # Se estiver em uma situação intermediária, gira levemente para o lado com mais espaço
        if max_index < len(laser_dist_fov) / 2:
            rotation_speed = -25  # gira para a esquerda
        else:
            rotation_speed = 25   # gira para a direita
        speed = 25              # avança lentamente    
    
    
    
    # Normalize speed and velocity:
    speed = speed / 50  
    rotation_speed = rotation_speed / 50
    
    command = {"forward": speed, "rotation": rotation_speed}
    
    # Debug prints
    # print("FOV max dist:", max_dist, "min dist:", min_dist)
    # print("Overall max dist:", np.max(laser_dist), "min dist:", np.min(laser_dist))
    
    return command

    

def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    # TODO for TP2
    

    
    print(current_pose)
    
    
    command = {"forward": 0,
               "rotation": 0}

    return command


