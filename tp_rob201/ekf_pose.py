"""
EKF Class for pose estimation (Additional method)
State: x = [x, y, theta] in the map/world frame.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(theta: float) -> float:
    return float((theta + np.pi) % (2.0 * np.pi) - np.pi)


class PoseEKF:
    def __init__(self,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        R: np.ndarray | None = None,
        q_xy: float = 2.0,
        q_theta: float = 0.05,
        gate_mahalanobis: float | None = 11.34,
    ):
        """
        Initialize an Extended Kalman Filter for pose estimation.
        Args:
            x0: Initial state vector [x, y, theta]. 
            P0: Initial state covariance matrix (3x3). 
            R: Measurement noise covariance matrix (3x3). 
            q_xy: Process noise scalar for xy motion (default: 2.0).
            q_theta: Process noise scalar for angular motion (default: 0.05).
        Attributes:
            x: Current state estimate [x, y, theta].
            P: State covariance matrix.
            R: Measurement noise covariance.
            q_xy, q_theta: Process noise parameters.
            last_update_accepted: Flag indicating if last measurement was accepted.
            
        """
        
        if x0 is None:
            self.x = np.zeros(3, dtype=float)
        else:
            self.x = np.asarray(x0, dtype=float).copy()
        
        
        self.x[2] = wrap_angle(self.x[2])

        if P0 is None:
            self.P = np.diag([500.0, 500.0, 10.5])
        else:
            self.P = np.asarray(P0, dtype=float).copy()

        # Measurement noise
        if R is None:
            self.R = np.diag([0.5, 0.5, 0.05])
        else:
            self.R = np.asarray(R, dtype=float).copy()

        # Base process noise scalars; scaled by step motion magnitude.
        self.q_xy = float(q_xy)
        self.q_theta = float(q_theta)

        # Chi-square gate on innovation (3 dof). Typical values:
        #  - 7.81  (95%)
        #  - 11.34 (99%)
        #  - 16.27 (99.9%)
        self.gate_mahalanobis = None if gate_mahalanobis is None else float(gate_mahalanobis)

        # Debug/telemetry
        self.last_innovation = np.zeros(3, dtype=float)
        self.last_update_accepted = True

        self._prev_odom_pose: np.ndarray | None = None
        self._initialised = x0 is not None

    @property
    def initialised(self) -> bool:
        return bool(self._initialised)

    def reset(self, x0: np.ndarray, P0: np.ndarray | None = None) -> None:
        self.x = np.asarray(x0, dtype=float).copy()
        self.x[2] = wrap_angle(self.x[2])
        if P0 is not None:
            self.P = np.asarray(P0, dtype=float).copy()
        self._prev_odom_pose = None
        self._initialised = True

    def set_prev_odom(self, odom_pose: np.ndarray) -> None:
        self._prev_odom_pose = np.asarray(odom_pose, dtype=float).copy()

    def predict_from_odom(self, odom_pose: np.ndarray) -> None:
        """Predict step using delta extracted from successive odometry poses.

        Notes:
        - We approximate translation as forward motion of length ds.
        - Rotation is taken directly from odometry delta dtheta.
        """
        odom_pose = np.asarray(odom_pose, dtype=float)

        if self._prev_odom_pose is None:
            self._prev_odom_pose = odom_pose.copy()
            return

        dx_o = float(odom_pose[0] - self._prev_odom_pose[0])
        dy_o = float(odom_pose[1] - self._prev_odom_pose[1])
        dtheta = wrap_angle(float(odom_pose[2] - self._prev_odom_pose[2]))
        self._prev_odom_pose = odom_pose.copy()

        ds = float(np.hypot(dx_o, dy_o))

        theta = float(self.x[2])
        c, s = float(np.cos(theta)), float(np.sin(theta))
        # ===================
        # Motion model:
        # [x']     =[x]     + [ds*cos(theta)] + noise
        # [y']     =[y]     + [ds*sin(theta)] + noise
        # [theta'] =[theta] + [dtheta]        + noise
        # ===================
        self.x[0] += ds * c
        self.x[1] += ds * s
        self.x[2] = wrap_angle(self.x[2] + dtheta)

        # Jacobian w.r.t state
        
        F = np.array(
            [
                [1.0, 0.0, -ds * s],
                [0.0, 1.0,  ds * c],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        # Process noise
        Q = np.diag(
            [
                self.q_xy    * (ds**2 + 1.0),
                self.q_xy    * (ds**2 + 1.0),
                self.q_theta * (dtheta**2 + 1.0),
            ]
        )

        self.P = F @ self.P @ F.T + Q

    def update_pose_measurement(self, z: np.ndarray) -> bool:
        """Update with pose measurement z=[x,y,theta] (e.g. scan-matching).

        Returns True if the update was accepted (passes the gate), else False.
        """
        z = np.asarray(z, dtype=float)

        H = np.eye(3, dtype=float)

        y = z - self.x
        y[2] = wrap_angle(float(y[2]))

        S = self.P + self.R

        Sinv = np.linalg.inv(S)
        maha = float(y.T @ Sinv @ y)
        self.last_innovation = y.copy()

        K = self.P @ Sinv

        self.x = self.x + K @ y
        self.x[2] = wrap_angle(float(self.x[2]))

        I = np.eye(3, dtype=float)
        self.P = (I - K @ H) @ self.P
        self._initialised = True
        self.last_update_accepted = True
        return True
