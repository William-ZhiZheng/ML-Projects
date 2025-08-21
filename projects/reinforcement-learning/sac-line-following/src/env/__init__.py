"""
Environment package for differential drive vehicle simulation.

This package provides:
- Vehicle state space model with realistic dynamics
- Visualization and testing tools
- Ready for integration with reinforcement learning frameworks
"""

from ..vehicle_model.vehicle import DifferentialDriveVehicle, VehicleParams
from .path_following_env import PathFollowingEnv

__all__ = [
    'DifferentialDriveVehicle',
    'VehicleParams',
    'PathFollowingEnv'
]

__version__ = "1.0.0"
__author__ = "Claude"
__description__ = "Differential Drive Vehicle Environment"