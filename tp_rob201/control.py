""" A set of robotics control functions """
import time
import numpy as np



def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1    
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
            rotation_speed = -10  # gira para a esquerda
        else:
            rotation_speed = 10   # gira para a direita
        speed = 25                # avança lentamente    
    
    
    # Normalize commands to controller ranges:
    # - forward in [0.0, 1.0]
    # - rotation in [-1.0, 1.0]
    forward = np.clip(speed / 50.0, 0.0, 1.0)
    rotation = np.clip(rotation_speed / 50.0, -1.0, 1.0)

    command = {"forward": float(forward), "rotation": float(rotation)}
    # Debug prints
    # print("FOV max dist:", max_dist, "min dist:", min_dist)
    # print("Overall max dist:", np.max(laser_dist), "min dist:", np.min(laser_dist))
    
    return command

def avoidance_method_TP1(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """

    # Default speeds
    speed_factor = 0.7   # velocidade para frente
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
            rotation_speed = -10  # gira para a esquerda
        else:
            rotation_speed = 10   # gira para a direita
        speed = 25                # avança lentamente    
    
    
    # Normalize commands to controller ranges:
    # - forward in [0.0, 1.0]
    # - rotation in [-1.0, 1.0]
    forward = np.clip(speed / 50.0, 0.0, 1.0)
    rotation = np.clip(rotation_speed / 50.0, -1.0, 1.0)

    command = {"forward": float(forward), "rotation": float(rotation)}
    # Debug prints
    # print("FOV max dist:", max_dist, "min dist:", min_dist)
    # print("Overall max dist:", np.max(laser_dist), "min dist:", np.min(laser_dist))
    
    return command

    

import numpy as np

import numpy as np

import numpy as np

def potential_field_control(lidar, current_pose, goal_pose):

    # =========================
    # PARAMETERS
    # =========================
    k_att = 1.0
    k_rep = 100.0
    dist_parameter = 45
    speed_factor = 0.6
    rotation_factor = 2.0
    safety_dist = 18
    goal_tol = 0.3

    # =========================
    # POSE
    # =========================
    x, y, theta = current_pose
    gx, gy, _ = goal_pose

    dx = gx - x
    dy = gy - y

    dist_goal = np.hypot(dx, dy)

    if dist_goal < goal_tol:
        return {"forward": 0.0, "rotation": 0.0}

    # =========================
    # WORLD -> ROBOT FRAME
    # =========================
    c = np.cos(theta)
    s = np.sin(theta)

    dx_r =  c * dx + s * dy
    dy_r = -s * dx + c * dy

    # =========================
    # ATTRACTIVE FORCE
    # =========================
    F_att_x = k_att * dx_r
    F_att_y = k_att * dy_r

    # =========================
    # REPULSIVE FORCE + SAFETY DISTANCE
    # =========================
    laser = np.asarray(lidar.get_sensor_values(), dtype=float)
    angles = np.asarray(lidar.get_ray_angles(), dtype=float)

    eps = 1e-3

    F_rep_x = 0.0
    F_rep_y = 0.0

    for d, a in zip(laser, angles):
        if eps < d < dist_parameter:
            # classical potential field gradient
            rep = k_rep / (1/d**2) 
            
            # rep = k_rep / (1.0 / d - 1.0 / dist_parameter)**2

            grad = rep / (d**3)

            # Avoid colisions by increasing the force
            if d < safety_dist:
                risk = (safety_dist * (1/(d + eps)))**2
            else:
                risk = 1.0

            grad *= risk

            F_rep_x -= grad * np.cos(a)
            F_rep_y -= grad * np.sin(a)

    # =========================
    # FORCE CALCULATION
    # =========================
    F_x = F_att_x + F_rep_x
    F_y = F_att_y + F_rep_y

    force_mag = np.hypot(F_x, F_y)

    # =========================
    # CONTROL LAW
    # =========================
    final_angle = np.arctan2(F_y, F_x)

    angle_error = np.arctan2(np.sin(final_angle), np.cos(final_angle))

    # angular velocity
    rotation_cmd = 2.0 * angle_error

    # forward depends on alignment AND force magnitude
    alignment = np.exp(-abs(angle_error))
    forward_cmd = alignment * np.tanh(force_mag)

    return {
        "forward": float(np.clip(forward_cmd * speed_factor, 0.0, 1.0)),
        "rotation": float(np.clip(rotation_cmd * rotation_factor, -1.0, 1.0))
    }