""" A set of robotics control functions """
import time
import numpy as np


def path_following_control(lidar, current_pose, path, lidar_weight=0.3):
    """
    Enhanced path following with lateral force to keep robot on track
    
    current_pose : [x, y, theta] current robot pose
    path : list of [x, y, theta] waypoints
    lidar_weight : balance between path following (0.7) and obstacle avoidance (0.3)
    
    Returns command dict with forward and rotation
    """
    if path is None or len(path) == 0:
        return {"forward": 0.0, "rotation": 0.0}
    
    x, y, theta = current_pose
    c = np.cos(theta)
    s = np.sin(theta)
    
    # Find closest point on path
    dists = [np.linalg.norm(current_pose[:2] - np.array(pt[:2])) for pt in path]
    closest_idx = int(np.argmin(dists))
    
    # Lookahead distance - find target ahead on path
    lookahead_dist = 40.0
    target = path[-1]
    for i in range(closest_idx, len(path)):
        if np.linalg.norm(current_pose[:2] - np.array(path[i][:2])) >= lookahead_dist:
            target = path[i]
            break
    
    # ===== PATH FOLLOWING FORCE =====
    # Attraction to target
    dx = target[0] - x
    dy = target[1] - y
    
    # Transform to robot frame
    dx_r = c * dx + s * dy
    dy_r = -s * dx + c * dy
    
    # Longitudinal (forward) and lateral (side) components
    dist_to_target = np.hypot(dx_r, dy_r)
    
    if dist_to_target < 0.1:
        return {"forward": 0.0, "rotation": 0.0}
    
    # Strong attraction to path
    k_path = 1200.0
    F_path_x = k_path * dx_r
    F_path_y = k_path * dy_r
    
    # ===== LATERAL CORRECTION FORCE =====
    # If robot drifted from path, apply strong lateral force
    if closest_idx < len(path) - 1:
        # Get previous waypoint to define path direction
        prev_waypoint = path[closest_idx]
        next_waypoint = path[min(closest_idx + 1, len(path) - 1)]
        
        # Path direction vector
        path_dx = next_waypoint[0] - prev_waypoint[0]
        path_dy = next_waypoint[1] - prev_waypoint[1]
        path_len = np.hypot(path_dx, path_dy)
        
        if path_len > 0.1:
            # Normalize path direction
            path_dx /= path_len
            path_dy /= path_len
            
            # Position relative to path start
            rel_x = x - prev_waypoint[0]
            rel_y = y - prev_waypoint[1]
            
            # Lateral distance (perpendicular to path)
            lateral_dist = -rel_x * path_dy + rel_y * path_dx
            
            # Lateral correction force (strong to keep on path)
            k_lateral = 800.0
            F_lateral = k_lateral * lateral_dist * path_dy  # force perpendicular to path
            F_lateral_x = -F_lateral * path_dy
            F_lateral_y = F_lateral * path_dx
            
            # Transform to robot frame
            F_lat_x_r = c * F_lateral_x + s * F_lateral_y
            F_lat_y_r = -s * F_lateral_x + c * F_lateral_y
        else:
            F_lat_x_r = 0.0
            F_lat_y_r = 0.0
    else:
        F_lat_x_r = 0.0
        F_lat_y_r = 0.0
    
    # ===== OBSTACLE AVOIDANCE (lighter) =====
    laser = np.asarray(lidar.get_sensor_values(), dtype=float)
    angles = np.asarray(lidar.get_ray_angles(), dtype=float)
    
    eps = 1e-3
    F_rep_x = 0.0
    F_rep_y = 0.0
    
    dist_param = 40.0
    k_rep = 80.0
    safety_dist = 8.0
    
    for d, a in zip(laser, angles):
        if eps < d < dist_param:
            rep = k_rep / (1.0 / d - 1.0 / dist_param)**2
            grad = rep / (d**3)
            
            if d < safety_dist:
                risk = (safety_dist * (1/(d + eps)))**2
            else:
                risk = 1.0
            
            grad *= risk
            F_rep_x -= grad * np.cos(a)
            F_rep_y -= grad * np.sin(a)
    
    # ===== COMBINE FORCES =====
    # Path following dominates, obstacles modulate
    F_x = (1.0 - lidar_weight) * (F_path_x + F_lat_x_r) + lidar_weight * F_rep_x
    F_y = (1.0 - lidar_weight) * (F_path_y + F_lat_y_r) + lidar_weight * F_rep_y
    
    force_mag = np.hypot(F_x, F_y)
    
    # ===== CONTROL LAW =====
    final_angle = np.arctan2(F_y, F_x)
    angle_error = np.arctan2(np.sin(final_angle), np.cos(final_angle))
    
    rotation_cmd = 2.5 * angle_error
    alignment = np.exp(-abs(angle_error))
    forward_cmd = alignment * np.tanh(force_mag / 2000.0)
    
    return {
        "forward": float(np.clip(forward_cmd * 0.85, 0.0, 1.0)),
        "rotation": float(np.clip(rotation_cmd * 2.0, -1.0, 1.0))
    }


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1    
    return avoidance_method_TP1(lidar)


def avoidance_method_TP1(lidar):
    """
    Otimized obstacle avoidance for door passage
    lidar : placebot object with lidar data
    """

    # ====== OPTIMIZED PARAMETERS FOR DOOR PASSAGE ======
    # Default speeds
    speed_factor = 0.8              # velocidade para frente (aumentado)
    rotation_speed = 0              # sem rotação inicialmente
    
    # Lidar readings
    laser_dist = np.array(lidar.get_sensor_values())
    
    # ====== EXPANDED FOV FOR BETTER DOOR DETECTION ======
    main_direction = 180
    threshold_angle = 110            # Expanded from 80 to 110 for wider view
    # FOV angles 
    fov_angles = np.arange(main_direction - threshold_angle, main_direction + threshold_angle)
    
    # Filtrar leituras dentro do FOV
    laser_dist_fov = laser_dist[fov_angles]
    
    # Distâncias mínimas e máximas
    max_dist = np.max(laser_dist_fov)
    min_dist = np.min(laser_dist_fov)
    max_index = np.argmax(laser_dist_fov)
    
    # ====== REDUCED SAFE DISTANCE FOR NARROW SPACES ======
    safe_distance = 18              # Reduced from 30/10 to allow passage through doors
    critical_distance = 12           # Emergency stop distance
    
    
    # --- Optimized obstacle avoidance logic ---
    if min_dist < critical_distance:
        # Emergency: obstacle very close
        rotation_speed = 60          # gira rapidamente para evitar
        speed = 0                    # para de avançar
    elif min_dist < safe_distance:
        # Obstacle close: turn toward open space
        open_space_center = fov_angles[max_index]
        center_fov = main_direction
        turn_direction = open_space_center - center_fov
        
        # Adaptive rotation based on opening location
        if turn_direction > 0:
            rotation_speed = -20     # Turn left (negative rotation)
        else:
            rotation_speed = 20      # Turn right (positive rotation)
        speed = 20                   # Move forward slowly while turning
    elif max_dist > safe_distance * 1.5:
        # Clear path: advance forward
        speed = 60                   # avança para frente
        rotation_speed = 0           # sem rotação
    else:
        # Intermediate: gentle adjustment toward best opening
        if max_index < len(laser_dist_fov) / 2:
            rotation_speed = -8      # Slight left
        else:
            rotation_speed = 8       # Slight right
        speed = 35                   # avança moderadamente    
    
    
    # Normalize commands to controller ranges:
    # - forward in [0.0, 1.0]
    # - rotation in [-1.0, 1.0]
    forward = np.clip(speed / 60.0, 0.0, 1.0)
    rotation = np.clip(rotation_speed / 60.0, -1.0, 1.0)

    command = {"forward": float(forward), "rotation": float(rotation)}
    
    # Debug prints (uncomment to debug)
    # print(f"FOV max dist: {max_dist:.1f}, min dist: {min_dist:.1f}, cmd: F={forward:.2f} R={rotation:.2f}")
    
    return command


def potential_field_control(lidar, current_pose, goal_pose):

    # =========================
    # PARAMETERS
    # =========================
    k_att = 850.0
    k_rep = 50.0
    dist_parameter = 25
    speed_factor = 0.6
    rotation_factor = 2.0
    safety_dist = 10
    goal_tol = 10

    # =========================
    # POSE
    # =========================
    x, y, theta = current_pose
    if goal_pose is None:
        return {"forward": 0.0, "rotation": 0.0}
    
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
            # rep = k_rep / (1/d**2) 
            
            rep = k_rep / (1.0 / d - 1.0 / dist_parameter)**2

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