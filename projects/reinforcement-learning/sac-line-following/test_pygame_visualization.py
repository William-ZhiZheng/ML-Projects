"""
Test script for pygame visualization with the vehicle and path following environment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import time
from env import DifferentialDriveVehicle, VehicleParams, PathFollowingEnv


def test_vehicle_pygame():
    """Test basic vehicle pygame visualization."""
    print("Testing vehicle pygame visualization...")
    
    vehicle = DifferentialDriveVehicle(VehicleParams())
    vehicle.reset(0, 0, 0)
    
    # Simple test path
    path_points = [(0, 0), (2, 0), (4, 2), (4, 4), (2, 4), (0, 4), (0, 2)]
    
    print("Vehicle pygame test started. Press ESC or close window to exit.")
    print("Vehicle will follow a simple control pattern.")
    
    running = True
    step_count = 0
    
    try:
        while running and step_count < 500:
            # Simple circular motion
            v_cmd = 1.0
            omega_cmd = 0.5 * np.sin(step_count * 0.05)
            
            # Update vehicle
            vehicle.step(v_cmd, omega_cmd)
            
            # Render with path
            success = vehicle.render_pygame(
                path_points=path_points,
                current_target_idx=None,
                show_trajectory=True,
                show_info=True
            )
            
            if not success:
                print("Pygame window closed by user")
                break
            
            step_count += 1
            time.sleep(0.02)  # ~50 FPS
    
    except KeyboardInterrupt:
        print("Test interrupted by user")
    
    finally:
        vehicle.close_pygame()
    
    print("[OK] Vehicle pygame test completed")





def main():
    """Run all pygame visualization tests."""
    print("=== Pygame Visualization Test Suite ===\n")
    
    try:
        # Test basic vehicle visualization
        test_vehicle_pygame()
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)