import pygame
import numpy as np
import math
from typing import List, Tuple, Optional

from src.vehicle_model import DifferentialDriveVehicle


class VehicleVisualizer:
    """Pygame-based visualizer for DifferentialDriveVehicle."""
    
    def __init__(self, screen_width=800, screen_height=600, scale=50):
        """
        Initialize the visualizer.
        
        Args:
            screen_width: Width of the pygame window
            screen_height: Height of the pygame window
            scale: Pixels per meter for world-to-screen conversion
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.scale = scale
        self.center_x = screen_width // 2
        self.center_y = screen_height // 2
        
        # Pygame components
        self.pygame_initialized = False
        self.screen = None
        self.clock = None
        self.font = None
    
    def init_pygame(self):
        """Initialize pygame for visualization."""
        if not self.pygame_initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Differential Drive Vehicle Simulation")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.pygame_initialized = True
    
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        screen_x = int(self.center_x + x * self.scale)
        screen_y = int(self.center_y - y * self.scale)  # Flip Y axis
        return screen_x, screen_y
    
    def get_vehicle_corners(self, vehicle: DifferentialDriveVehicle) -> np.ndarray:
        """
        Get vehicle corner positions for visualization.
        
        Args:
            vehicle: The vehicle to get corners for
            
        Returns:
            Array of corner positions [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        # Vehicle corners in local coordinates (center at origin)
        half_length = vehicle.params.length / 2
        half_width = vehicle.params.width / 2
        
        local_corners = np.array([
            [-half_length, -half_width],  # Rear left
            [half_length, -half_width],   # Front left
            [half_length, half_width],    # Front right
            [-half_length, half_width],   # Rear right
        ])
        
        # Rotation matrix
        cos_theta = np.cos(vehicle.theta)
        sin_theta = np.sin(vehicle.theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        
        # Transform to global coordinates
        global_corners = local_corners @ rotation_matrix.T + np.array([vehicle.x, vehicle.y])
        
        return global_corners
    
    def draw_vehicle(self, surface, vehicle: DifferentialDriveVehicle, 
                     color=(0, 100, 255), outline_color=(0, 50, 200)):
        """Draw the vehicle on pygame surface."""
        # Get vehicle corners
        corners = self.get_vehicle_corners(vehicle)
        
        # Convert to screen coordinates
        screen_corners = [self.world_to_screen(corner[0], corner[1]) for corner in corners]
        
        # Draw vehicle body
        pygame.draw.polygon(surface, color, screen_corners)
        pygame.draw.polygon(surface, outline_color, screen_corners, 2)
        
        # Draw center point
        center_screen = self.world_to_screen(vehicle.x, vehicle.y)
        pygame.draw.circle(surface, (0, 0, 0), center_screen, 3)
        
        # Draw heading arrow
        arrow_length = vehicle.params.length * 0.8
        end_x = vehicle.x + arrow_length * np.cos(vehicle.theta)
        end_y = vehicle.y + arrow_length * np.sin(vehicle.theta)
        end_screen = self.world_to_screen(end_x, end_y)
        
        pygame.draw.line(surface, (255, 0, 0), center_screen, end_screen, 3)
        
        # Draw arrow head
        arrow_head_length = 0.1
        arrow_head_angle = 0.5
        
        # Left arrow head line
        left_x = end_x - arrow_head_length * np.cos(vehicle.theta - arrow_head_angle)
        left_y = end_y - arrow_head_length * np.sin(vehicle.theta - arrow_head_angle)
        left_screen = self.world_to_screen(left_x, left_y)
        pygame.draw.line(surface, (255, 0, 0), end_screen, left_screen, 2)
        
        # Right arrow head line
        right_x = end_x - arrow_head_length * np.cos(vehicle.theta + arrow_head_angle)
        right_y = end_y - arrow_head_length * np.sin(vehicle.theta + arrow_head_angle)
        right_screen = self.world_to_screen(right_x, right_y)
        pygame.draw.line(surface, (255, 0, 0), end_screen, right_screen, 2)
    
    def draw_trajectory(self, surface, vehicle: DifferentialDriveVehicle, 
                       max_points=100, color=(100, 100, 255)):
        """Draw the vehicle trajectory."""
        if len(vehicle.state_history) < 2:
            return
        
        # Get recent trajectory points
        trajectory_data = np.array(vehicle.state_history)
        if len(trajectory_data) > max_points:
            trajectory_data = trajectory_data[-max_points:]
        
        # Convert to screen coordinates
        screen_points = []
        for state in trajectory_data:
            x, y = state[0], state[1]
            screen_x, screen_y = self.world_to_screen(x, y)
            screen_points.append((screen_x, screen_y))
        
        # Draw trajectory lines with fading effect
        if len(screen_points) > 1:
            for i in range(1, len(screen_points)):
                alpha = int((i / len(screen_points)) * 255)
                fade_color = (*color, alpha)
                
                # Create a temporary surface for alpha blending
                temp_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
                pygame.draw.line(temp_surface, fade_color, screen_points[i-1], screen_points[i], 2)
                surface.blit(temp_surface, (0, 0))
    
    def draw_path(self, surface, path_points: List[Tuple[float, float]], 
                  path_color=(0, 255, 0), waypoint_color=(255, 255, 0), 
                  current_target_idx=None, target_color=(255, 0, 0)):
        """Draw a path with waypoints."""
        if len(path_points) < 2:
            return
        
        # Convert path points to screen coordinates
        screen_points = [self.world_to_screen(x, y) for x, y in path_points]
        
        # Draw path lines
        for i in range(len(screen_points) - 1):
            pygame.draw.line(surface, path_color, screen_points[i], screen_points[i+1], 3)
        
        # Draw waypoints
        for i, point in enumerate(screen_points):
            if current_target_idx is not None and i == current_target_idx:
                # Highlight current target
                pygame.draw.circle(surface, target_color, point, 8)
                pygame.draw.circle(surface, (255, 255, 255), point, 8, 2)
            else:
                pygame.draw.circle(surface, waypoint_color, point, 5)
                pygame.draw.circle(surface, (0, 0, 0), point, 5, 1)
    
    def draw_info(self, surface, vehicle: DifferentialDriveVehicle):
        """Draw vehicle information on screen."""
        info_lines = [
            f"Position: ({vehicle.x:.2f}, {vehicle.y:.2f})",
            f"Heading: {math.degrees(vehicle.theta):.1f}°",
            f"Linear Vel: {vehicle.v_actual:.2f} m/s",
            f"Angular Vel: {vehicle.omega_actual:.2f} rad/s",
            f"Time: {vehicle.current_time:.1f}s"
        ]
        
        # Draw background rectangle
        text_height = len(info_lines) * 25 + 10
        pygame.draw.rect(surface, (255, 255, 255, 200), (10, 10, 250, text_height))
        pygame.draw.rect(surface, (0, 0, 0), (10, 10, 250, text_height), 2)
        
        # Draw text lines
        assert self.font is not None, "Font should be initialized"
        for i, line in enumerate(info_lines):
            text_surface = self.font.render(line, True, (0, 0, 0))
            surface.blit(text_surface, (15, 15 + i * 25))
    
    def draw_grid(self, surface, grid_size=1.0, color=(200, 200, 200)):
        """Draw a grid on the screen."""
        # Vertical lines
        start_x = int(-self.center_x / self.scale / grid_size) * grid_size
        for x in np.arange(start_x, start_x + self.screen_width / self.scale + grid_size, grid_size):
            screen_x, _ = self.world_to_screen(x, 0)
            if 0 <= screen_x <= self.screen_width:
                pygame.draw.line(surface, color, (screen_x, 0), (screen_x, self.screen_height), 1)
        
        # Horizontal lines
        start_y = int(-self.center_y / self.scale / grid_size) * grid_size
        for y in np.arange(start_y, start_y + self.screen_height / self.scale + grid_size, grid_size):
            _, screen_y = self.world_to_screen(0, y)
            if 0 <= screen_y <= self.screen_height:
                pygame.draw.line(surface, color, (0, screen_y), (self.screen_width, screen_y), 1)
    
    def render(self, vehicle: DifferentialDriveVehicle, path_points=None, 
               current_target_idx=None, show_trajectory=True, show_info=True):
        """
        Render the vehicle and scene.
        
        Args:
            vehicle: The vehicle to render
            path_points: List of (x, y) path points to visualize
            current_target_idx: Index of current target point
            show_trajectory: Whether to show trajectory history
            show_info: Whether to show vehicle information
        
        Returns:
            True if rendering should continue, False if window was closed
        """
        if not self.pygame_initialized:
            self.init_pygame()
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Clear screen
        assert self.screen is not None, "Screen should be initialized"
        self.screen.fill((240, 240, 240))  # Light gray background
        
        # Draw grid
        self.draw_grid(self.screen)
        
        # Draw path if provided
        if path_points:
            self.draw_path(self.screen, path_points, current_target_idx=current_target_idx)
        
        # Draw trajectory
        if show_trajectory:
            self.draw_trajectory(self.screen, vehicle)
        
        # Draw vehicle
        self.draw_vehicle(self.screen, vehicle)
        
        # Draw information
        if show_info:
            self.draw_info(self.screen, vehicle)
        
        # Update display
        pygame.display.flip()
        assert self.clock is not None, "Clock should be initialized"
        self.clock.tick(60)  # 60 FPS
        
        return True
    
    def close(self):
        """Close pygame window."""
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False