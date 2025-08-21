import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class VehicleParams:
    """Physical parameters of the differential drive vehicle."""

    # Dimensions (meters)
    length: float = 0.4
    width: float = 0.3
    wheelbase: float = 0.25  # Distance between wheel centers
    wheel_radius: float = 0.05

    # Performance limits
    max_linear_velocity: float = 2.0  # m/s
    max_angular_velocity: float = 3.0  # rad/s
    max_linear_acceleration: float = 4.0  # m/s²
    max_angular_acceleration: float = 8.0  # rad/s²

    # Dynamics (transient response)
    tau_linear: float = 0.2  # Linear velocity time constant (s)
    tau_angular: float = 0.15  # Angular velocity time constant (s)


class DifferentialDriveVehicle:
    """
    Differential drive vehicle with realistic transient dynamics.

    State Space Model:
    - State: [x, y, θ, v_actual, ω_actual]
    - Input: [v_cmd, ω_cmd]
    - Dynamics: First-order response with configurable time constants
    """

    def __init__(self, params: VehicleParams, dt: float = 0.02):
        """
        Initialize vehicle with given parameters.

        Args:
            params: Vehicle physical parameters
            dt: Simulation time step (seconds)
        """
        self.params = params
        self.dt = dt

        # State variables
        self.x = 0.0  # Position x (m)
        self.y = 0.0  # Position y (m)
        self.theta = 0.0  # Heading angle (rad)
        self.v_actual = 0.0  # Actual linear velocity (m/s)
        self.omega_actual = 0.0  # Actual angular velocity (rad/s)

        # Command history for analysis
        self.v_cmd_history = []
        self.omega_cmd_history = []
        self.state_history = []
        self.time_history = []
        self.current_time = 0.0

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """Reset vehicle to initial state."""
        self.x = x
        self.y = y
        self.theta = theta
        self.v_actual = 0.0
        self.omega_actual = 0.0
        self.current_time = 0.0

        # Clear history
        self.v_cmd_history.clear()
        self.omega_cmd_history.clear()
        self.state_history.clear()
        self.time_history.clear()

    def step(self, v_cmd: float, omega_cmd: float) -> Tuple[float, float, float]:
        """
        Advance vehicle dynamics by one time step.

        Args:
            v_cmd: Commanded linear velocity (m/s)
            omega_cmd: Commanded angular velocity (rad/s)

        Returns:
            Current position and heading (x, y, theta)
        """
        # Saturate commands to physical limits
        v_cmd = np.clip(
            v_cmd, -self.params.max_linear_velocity, self.params.max_linear_velocity
        )
        omega_cmd = np.clip(
            omega_cmd,
            -self.params.max_angular_velocity,
            self.params.max_angular_velocity,
        )

        # Store command history
        self.v_cmd_history.append(v_cmd)
        self.omega_cmd_history.append(omega_cmd)

        # Apply first-order dynamics for velocity response
        # τ * dv/dt = v_cmd - v_actual
        # Discrete: v_actual[k+1] = v_actual[k] + (dt/τ) * (v_cmd - v_actual[k])
        alpha_v = self.dt / self.params.tau_linear
        alpha_omega = self.dt / self.params.tau_angular

        # Update actual velocities with first-order response
        self.v_actual += alpha_v * (v_cmd - self.v_actual)
        self.omega_actual += alpha_omega * (omega_cmd - self.omega_actual)

        # Apply acceleration limits (rate limiting)
        max_dv = self.params.max_linear_acceleration * self.dt
        max_domega = self.params.max_angular_acceleration * self.dt

        # This would require storing previous velocities for proper rate limiting
        # For simplicity, we'll skip this step in the basic implementation

        # Update vehicle pose using actual velocities (differential drive kinematics)
        self.x += self.v_actual * np.cos(self.theta) * self.dt
        self.y += self.v_actual * np.sin(self.theta) * self.dt
        self.theta += self.omega_actual * self.dt

        # Normalize theta to [-π, π]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        # Update time and store state history
        self.current_time += self.dt
        self.time_history.append(self.current_time)
        self.state_history.append(self.get_state().copy())

        return self.x, self.y, self.theta

    def get_state(self) -> np.ndarray:
        """Get current state vector [x, y, theta, v_actual, omega_actual]."""
        return np.array([self.x, self.y, self.theta, self.v_actual, self.omega_actual])

    def get_position(self) -> np.ndarray:
        """Get current position [x, y]."""
        return np.array([self.x, self.y])

    def get_heading(self) -> float:
        """Get current heading angle (radians)."""
        return self.theta

    def get_velocities(self) -> Tuple[float, float]:
        """Get current actual velocities (v, omega)."""
        return self.v_actual, self.omega_actual

    def get_wheel_velocities(self) -> Tuple[float, float]:
        """
        Convert current velocities to individual wheel velocities.

        Returns:
            Left and right wheel velocities (rad/s)
        """
        # Differential drive kinematics: v = (v_left + v_right) * R / 2
        # omega = (v_right - v_left) * R / L
        # Solving: v_left = (2*v - omega*L) / (2*R)
        #          v_right = (2*v + omega*L) / (2*R)

        R = self.params.wheel_radius
        L = self.params.wheelbase

        v_left = (2 * self.v_actual - self.omega_actual * L) / (2 * R)
        v_right = (2 * self.v_actual + self.omega_actual * L) / (2 * R)

        return v_left, v_right



    def get_dynamics_info(self) -> dict:
        """Get information about vehicle dynamics and current state."""
        return {
            "time_constants": {
                "tau_linear": self.params.tau_linear,
                "tau_angular": self.params.tau_angular,
            },
            "max_velocities": {
                "linear": self.params.max_linear_velocity,
                "angular": self.params.max_angular_velocity,
            },
            "physical_params": {
                "length": self.params.length,
                "width": self.params.width,
                "wheelbase": self.params.wheelbase,
            },
            "current_state": self.get_state(),
            "wheel_velocities": self.get_wheel_velocities(),
        }
