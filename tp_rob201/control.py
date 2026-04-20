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
    # --- Parameters (tune as needed) ---
    k_att = 4.0          # attractive gain
    k_rep = 2500.0       # repulsive gain
    d0 = 60.0            # influence distance for obstacles
    goal_tol = 5.0      # stop radius around goal

    max_forward = 50.0
    max_rot = 1.0        # command "rotation" [-1.0, 1.0]

    # --- Inputs ---
    x, y, theta = float(current_pose[0]), float(current_pose[1]), float(current_pose[2])
    gx, gy = float(goal_pose[0]), float(goal_pose[1])

    # --- Attractive force in odom/world frame ---
    gvec_w = np.array([gx - x, gy - y], dtype=float)
    dist_goal = np.linalg.norm(gvec_w)

    if dist_goal < goal_tol:
        return {"forward": 0.0, "rotation": 0.0}

    # convert attractive vector to robot frame (x forward, y left)
    c, s = np.cos(theta), np.sin(theta)
    R_wr = np.array([[c, s],
                     [-s, c]], dtype=float)  # world -> robot
    F_att_r = k_att * (R_wr @ gvec_w)

    # --- Repulsive force from lidar in robot frame ---
    ranges = np.asarray(lidar.get_sensor_values(), dtype=float)
    n = ranges.size

    # assume indices map to degrees and "forward" is index 180
    idx = np.arange(n)
    ang = np.deg2rad(idx - 180.0)  # 0 rad = forward
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)

    mask = (ranges > 1e-6) & (ranges < d0)
    r = ranges[mask]
    if r.size > 0:
        # magnitude: k_rep * (1/r - 1/d0) / r^2
        mag = k_rep * (1.0 / r - 1.0 / d0) / (r * r)

        # direction away from obstacle: -[cos, sin]
        Fx = np.sum(-mag * cos_a[mask])
        Fy = np.sum(-mag * sin_a[mask])
        F_rep_r = np.array([Fx, Fy], dtype=float)
    else:
        F_rep_r = np.zeros(2, dtype=float)

    # --- Total force -> commands ---
    F = F_att_r + F_rep_r
    desired_heading = np.arctan2(F[1], F[0])  # in robot frame

    # rotation proportional to heading error (already in robot frame)
    rot_cmd = np.clip(2.0 * desired_heading, -max_rot, max_rot)

    # forward proportional to "how much forward component remains"
    # forward command normalized to [0, 1]
    forward_cmd = np.clip(0.8 * F[0] / max_forward, 0.0, 1.0)
    forward_cmd *= np.clip(1.0 - abs(rot_cmd), 0.0, 1.0)  # slow down while turning
    forward_cmd = np.clip(forward_cmd, 0.0, 1.0)
    
    # Debug prints
    print(f"Goal dist: {dist_goal:.2f}, F_att: {F_att_r}, F_rep: {F_rep_r}, F_total: {F}, rot_cmd: {rot_cmd:.2f}, forward_cmd: {forward_cmd:.2f}, goal: ({gx:.1f}, {gy:.1f}), pose: ({x:.1f}, {y:.1f}, {theta:.2f} rad)")

    return {"forward": float(forward_cmd), "rotation": float(rot_cmd)}


