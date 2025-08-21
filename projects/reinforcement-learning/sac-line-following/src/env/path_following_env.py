"""
Path following environment for differential drive vehicle.

This environment provides a Gymnasium-compatible interface for training
reinforcement learning agents to follow a predefined path using a
differential drive vehicle model.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Optional, Dict, Any
import pygame

from ..vehicle_model.vehicle import DifferentialDriveVehicle, VehicleParams


class PathFollowingEnv(gym.Env):
    """
    Gymnasium environment for path following with a differential drive vehicle.
    
    Observation space: [x, y, heading, dist_to_next_point_x, dist_to_next_point_y]
    Action space: [linear_velocity_cmd, angular_velocity_cmd]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        path_points: List[Tuple[float, float]],
        vehicle_params: Optional[VehicleParams] = None,
        max_episode_steps: int = 1000,
        goal_tolerance: float = 0.2,
        max_distance_from_path: float = 5.0,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the path following environment.
        
        Args:
            path_points: List of (x, y) coordinates defining the path
            vehicle_params: Vehicle configuration parameters
            max_episode_steps: Maximum steps per episode
            goal_tolerance: Distance tolerance to consider path point reached
            max_distance_from_path: Maximum allowed distance from path before termination
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        super().__init__()
        
        # Environment parameters
        self.path_points = np.array(path_points)
        self.max_episode_steps = max_episode_steps
        self.goal_tolerance = goal_tolerance
        self.max_distance_from_path = max_distance_from_path
        self.render_mode = render_mode
        
        # Initialize vehicle
        if vehicle_params is None:
            vehicle_params = VehicleParams()
        self.vehicle = DifferentialDriveVehicle(vehicle_params)
        
        # Define action and observation spaces
        # Action: [linear_velocity_cmd, angular_velocity_cmd]
        self.action_space = spaces.Box(
            low=np.array([-2.0, -np.pi]),  # Max reverse speed, max turn rate
            high=np.array([2.0, np.pi]),   # Max forward speed, max turn rate
            dtype=np.float32
        )
        
        # Observation: [x, y, heading, dist_to_next_point_x, dist_to_next_point_y]
        obs_low = np.array([-np.inf, -np.inf, -np.pi, -np.inf, -np.inf])
        obs_high = np.array([np.inf, np.inf, np.pi, np.inf, np.inf])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Environment state
        self.current_step = 0
        self.current_target_idx = 0
        self.total_distance_traveled = 0.0
        self.previous_position = None
        
        # Rendering - no need to store matplotlib objects anymore
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset environment state
        self.current_step = 0
        self.current_target_idx = 0
        self.total_distance_traveled = 0.0
        
        # Reset vehicle to start of path with some random variation
        if options and 'start_position' in options:
            start_x, start_y = options['start_position']
        else:
            start_x, start_y = self.path_points[0]
            # Add small random variation to starting position
            start_x += np.random.uniform(-0.5, 0.5)
            start_y += np.random.uniform(-0.5, 0.5)
        
        # Initialize heading towards first path point
        if len(self.path_points) > 1:
            dx = self.path_points[1][0] - start_x
            dy = self.path_points[1][1] - start_y
            initial_heading = np.arctan2(dy, dx)
        else:
            initial_heading = 0.0
        
        self.vehicle.reset(start_x, start_y, initial_heading)
        self.previous_position = np.array([start_x, start_y])
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation.astype(np.float32), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        v_cmd, omega_cmd = action
        
        # Update vehicle dynamics
        self.vehicle.step(v_cmd, omega_cmd)
        self.current_step += 1
        
        # Update distance traveled
        current_position = np.array([self.vehicle.x, self.vehicle.y])
        if self.previous_position is not None:
            distance_step = np.linalg.norm(current_position - self.previous_position)
            self.total_distance_traveled += distance_step
        self.previous_position = current_position.copy()
        
        # Update target point
        self._update_target_point()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation.astype(np.float32), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        # Find next target point using the specified search method
        target_point, _ = self._find_next_target_point()
        
        # Calculate distance to next target point
        dist_x = target_point[0] - self.vehicle.x
        dist_y = target_point[1] - self.vehicle.y
        
        observation = np.array([
            self.vehicle.x,
            self.vehicle.y,
            self.vehicle.theta,
            dist_x,
            dist_y
        ])
        
        return observation
    
    def _find_next_target_point(self) -> Tuple[np.ndarray, int]:
        """
        Find the next target point using the specified search strategy.
        
        First searches in -90 to 90 degree range relative to vehicle heading.
        If none found, returns closest point overall.
        
        Returns:
            target_point: (x, y) coordinates of target
            target_idx: Index of target point in path
        """
        vehicle_pos = np.array([self.vehicle.x, self.vehicle.y])
        vehicle_heading = self.vehicle.theta
        
        # Calculate vectors to all remaining path points
        remaining_points = self.path_points[self.current_target_idx:]
        if len(remaining_points) == 0:
            # Reached end of path, return last point
            return self.path_points[-1], len(self.path_points) - 1
        
        vectors_to_points = remaining_points - vehicle_pos
        distances = np.linalg.norm(vectors_to_points, axis=1)
        
        # Calculate angles relative to vehicle heading
        angles_to_points = np.arctan2(vectors_to_points[:, 1], vectors_to_points[:, 0])
        angle_diff = angles_to_points - vehicle_heading
        
        # Normalize angles to [-pi, pi]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        # Find points within -90 to 90 degree range (pi/2 radians)
        forward_mask = np.abs(angle_diff) <= np.pi/2
        
        if np.any(forward_mask):
            # Select closest point within forward range
            forward_distances = distances.copy()
            forward_distances[~forward_mask] = np.inf
            min_idx = np.argmin(forward_distances)
            target_idx = self.current_target_idx + min_idx
        else:
            # No points in forward range, select closest overall
            min_idx = np.argmin(distances)
            target_idx = self.current_target_idx + min_idx
        
        return self.path_points[target_idx], target_idx
    
    def _update_target_point(self):
        """Update the current target point based on vehicle progress."""
        if self.current_target_idx >= len(self.path_points):
            return
        
        # Check if vehicle is close enough to current target
        current_target = self.path_points[self.current_target_idx]
        distance_to_target = np.linalg.norm([
            self.vehicle.x - current_target[0],
            self.vehicle.y - current_target[1]
        ])
        
        if distance_to_target < self.goal_tolerance:
            # Move to next target point
            self.current_target_idx += 1
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward for current state and action."""
        reward = 0.0
        
        # Progress reward: reward forward movement along the path
        if self.current_target_idx < len(self.path_points):
            target_point = self.path_points[self.current_target_idx]
            distance_to_target = np.linalg.norm([
                self.vehicle.x - target_point[0],
                self.vehicle.y - target_point[1]
            ])
            
            # Reward being close to target
            reward += max(0, 2.0 - distance_to_target)
            
            # Reward reaching waypoints
            if distance_to_target < self.goal_tolerance:
                reward += 10.0
        
        # Penalize excessive angular velocity (encourage smooth motion)
        angular_penalty = -0.1 * abs(action[1])
        reward += angular_penalty
        
        # Penalize being far from path
        min_distance_to_path = self._distance_to_path()
        if min_distance_to_path > 1.0:
            reward -= min_distance_to_path
        
        # Bonus for completing the path
        if self.current_target_idx >= len(self.path_points):
            reward += 100.0
        
        return reward
    
    def _distance_to_path(self) -> float:
        """Calculate minimum distance from vehicle to any path segment."""
        if len(self.path_points) < 2:
            return np.linalg.norm([
                self.vehicle.x - self.path_points[0][0],
                self.vehicle.y - self.path_points[0][1]
            ])
        
        vehicle_pos = np.array([self.vehicle.x, self.vehicle.y])
        min_distance = float('inf')
        
        # Check distance to all path segments
        for i in range(len(self.path_points) - 1):
            p1 = self.path_points[i]
            p2 = self.path_points[i + 1]
            
            # Calculate distance from point to line segment
            segment_vec = p2 - p1
            point_vec = vehicle_pos - p1
            
            if np.dot(segment_vec, segment_vec) == 0:
                # Degenerate segment (single point)
                distance = np.linalg.norm(point_vec)
            else:
                # Project point onto line segment
                t = max(0, min(1, np.dot(point_vec, segment_vec) / np.dot(segment_vec, segment_vec)))
                projection = p1 + t * segment_vec
                distance = np.linalg.norm(vehicle_pos - projection)
            
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _is_terminated(self) -> bool:
        """Check if episode should be terminated."""
        # Terminate if too far from path
        if self._distance_to_path() > self.max_distance_from_path:
            return True
        
        # Terminate if completed the path
        if self.current_target_idx >= len(self.path_points):
            return True
        
        return False
    
    def _get_info(self) -> Dict:
        """Get additional information about current state."""
        return {
            "current_target_idx": self.current_target_idx,
            "distance_to_path": self._distance_to_path(),
            "total_distance_traveled": self.total_distance_traveled,
            "path_completion": self.current_target_idx / len(self.path_points),
            "vehicle_x": self.vehicle.x,
            "vehicle_y": self.vehicle.y,
            "vehicle_theta": self.vehicle.theta
        }
    
    def render(self):
        """Render the environment using pygame."""
        if self.render_mode is None:
            return
        
        # Convert path points to list of tuples for pygame
        path_points_list = [(float(x), float(y)) for x, y in self.path_points]
        
        # Render using pygame
        success = self.vehicle.render_pygame(
            path_points=path_points_list,
            current_target_idx=self.current_target_idx,
            show_trajectory=True,
            show_info=True
        )
        
        if not success:
            # Pygame window was closed
            return None
        
        if self.render_mode == "rgb_array":
            # Convert pygame surface to numpy array
            string_image = pygame.image.tostring(self.vehicle.screen, 'RGB')
            temp_surf = pygame.image.fromstring(string_image, 
                                               (self.vehicle.screen_width, self.vehicle.screen_height), 'RGB')
            return pygame.surfarray.array3d(temp_surf).transpose([1, 0, 2])
    
    def close(self):
        """Close the environment."""
        self.vehicle.close_pygame()