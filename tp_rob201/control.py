""" A set of robotics control functions """
import time
import numpy as np


def path_following_control(lidar, current_pose, path, lidar_weight=0.3):
    """
    Enhanced path following with lateral force to keep robot on track
    
    current_pose : [x, y, theta] current robot pose
    path : list of [x, y, theta] waypoints
    lidar_weight : balance between path following (0.7) and obstacle avoidance (0.3)
    """
    if path is None or len(path) == 0:
        return {"forward": 0.0, "rotation": 0.0}
    
    speed_factor = 0.6

    
    x, y, theta = current_pose
    c = np.cos(theta)
    s = np.sin(theta)
    
    # Find closest point on path
    dists = [np.linalg.norm(current_pose[:2] - np.array(pt[:2])) for pt in path]
    closest_idx = int(np.argmin(dists))
    
    # Lookahead distance - find target ahead on path
    lookahead_dist = 20.0
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
    k_path = 1500.0
    k_lateral = 500.0
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
    
    dist_param = 20.0
    k_rep = 60.0
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
        "forward":  float(np.clip(forward_cmd *speed_factor, 0.0, 1.0)),
        "rotation": float(np.clip(rotation_cmd * 2.0, -1.0, 1.0))
    }


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1    
    return avoidance_method_TP1(lidar)
import numpy as np


def avoidance_method_TP1(lidar):
    """
    Reactive exploration + door finding
    """
    speed_factor = 0.6
    rotation_factor = 1.0

    # Lidar readings
    laser_dist = np.array(lidar.get_sensor_values())

    # Forward field of view
    center = 180
    fov = 100

    angles = np.arange(center - fov, center + fov)
    scan = laser_dist[angles]

    # Parameters
    safe_distance = 25
    free_threshold = 40

    # Default motion
    speed = 0.5
    rotation = 0.0

    # -----------------------------------
    # 1. Emergency obstacle avoidance
    # -----------------------------------
    front_zone = scan[fov-20:fov+20]

    if np.min(front_zone) < safe_distance:

        # Compare left/right clearance
        left_clearance = np.mean(scan[:fov])
        right_clearance = np.mean(scan[fov:])

        speed = 0

        if left_clearance > right_clearance:
            rotation = -1.0
        else:
            rotation = 1.0

    else:

        # -----------------------------------
        # 2. Detect free gaps (doors)
        # -----------------------------------
        free_space = scan > free_threshold

        segments = []
        start = None

        for i, val in enumerate(free_space):

            if val and start is None:
                start = i

            elif not val and start is not None:
                segments.append((start, i))
                start = None

        if start is not None:
            segments.append((start, len(free_space)))

        # -----------------------------------
        # 3. Choose best opening
        # -----------------------------------
        if segments:

            # Prefer widest opening
            widths = [end-start for start, end in segments]
            best_idx = np.argmax(widths)

            start, end = segments[best_idx]

            target = (start + end) / 2

            error = target - len(scan)/2

            # Steering toward opening
            rotation = error / (len(scan)/2)

            # Larger opening = faster motion
            speed = 0.8

        else:

            # No opening → wall following
            left_mean = np.mean(scan[:fov])
            right_mean = np.mean(scan[fov:])

            if left_mean > right_mean:
                rotation = -0.5
            else:
                rotation = 0.5

            speed = 0.3

    return {
        "forward": speed * speed_factor,
        "rotation": rotation * rotation_factor
    }
    
    
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