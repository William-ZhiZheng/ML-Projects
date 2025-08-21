import pygame
import numpy as np
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

from src.vehicle_model import DifferentialDriveVehicle, VehicleParams
from src.visualizer.vehicle_visualizer import VehicleVisualizer

# Setup - Configure vehicle with physical specifications
# Robot can do 360° in 0.5s: ω_max = 2π / 0.5 = 4π ≈ 12.57 rad/s
# Max speed is 3 m/s
params = VehicleParams(
    max_linear_velocity=3.0,      # 3 m/s max speed
    max_angular_velocity=12.57,   # 360° in 0.5s = 4π rad/s
    tau_linear=0.05,              # Much faster response (50ms instead of 200ms)
    tau_angular=0.03,             # Much faster response (30ms instead of 150ms)
    max_linear_acceleration=15.0, # Higher acceleration for responsiveness
    max_angular_acceleration=50.0 # Higher angular acceleration
)
vehicle = DifferentialDriveVehicle(params, dt=1/60)  # Match 60Hz rendering
visualizer = VehicleVisualizer(screen_width=1280, screen_height=720, scale=100)
visualizer.init_pygame()  # Initialize pygame

running = True
while running:
    # Handle input
    keys = pygame.key.get_pressed()

    # Vehicle command [linear_speed, angular_speed]
    vehicle_cmd = np.array([0.0, 0.0])

    # Linear speed (forward/backward)
    if keys[pygame.K_w]:
        vehicle_cmd[0] = 1.0  # Forward
    elif keys[pygame.K_s]:
        vehicle_cmd[0] = -1.0  # Backward

    # Angular speed (turn left/right)
    if keys[pygame.K_a]:
        vehicle_cmd[1] = 10.0  # Turn left
    elif keys[pygame.K_d]:
        vehicle_cmd[1] = -10.0  # Turn right

    # Update vehicle
    vehicle.step(vehicle_cmd[0], vehicle_cmd[1])

    # Render
    running = visualizer.render(vehicle, show_trajectory=True, show_info=True)

visualizer.close()
