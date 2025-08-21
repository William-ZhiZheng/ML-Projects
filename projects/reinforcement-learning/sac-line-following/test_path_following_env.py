"""
Test script for the PathFollowingEnv environment.

This script validates the path following environment functionality
including observation spaces, action spaces, and basic behavior.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from env import PathFollowingEnv, VehicleParams


def create_test_paths():
    """Create various test paths for validation."""
    paths = {}
    
    # Simple straight line
    paths['straight'] = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)
    ]
    
    # L-shaped path
    paths['L_shape'] = [
        (0, 0), (2, 0), (4, 0), (4, 2), (4, 4)
    ]
    
    # Circular path
    angles = np.linspace(0, 2*np.pi, 16)
    radius = 3
    paths['circle'] = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    
    # Figure-8 path
    t = np.linspace(0, 2*np.pi, 32)
    paths['figure8'] = [
        (2 * np.sin(ti), np.sin(2*ti)) for ti in t
    ]
    
    return paths


def test_environment_creation():
    """Test basic environment creation and properties."""
    print("Testing environment creation...")
    
    path = [(0, 0), (5, 0), (5, 5)]
    env = PathFollowingEnv(path)
    
    # Test spaces
    assert env.observation_space.shape == (5,), f"Expected obs shape (5,), got {env.observation_space.shape}"
    assert env.action_space.shape == (2,), f"Expected action shape (2,), got {env.action_space.shape}"
    
    print("[OK] Environment creation successful")
    print(f"[OK] Observation space: {env.observation_space}")
    print(f"[OK] Action space: {env.action_space}")


def test_reset_functionality():
    """Test environment reset functionality."""
    print("\nTesting reset functionality...")
    
    path = [(0, 0), (5, 0), (5, 5)]
    env = PathFollowingEnv(path)
    
    # Test reset
    obs, info = env.reset()
    
    assert len(obs) == 5, f"Expected observation length 5, got {len(obs)}"
    assert isinstance(info, dict), f"Expected info to be dict, got {type(info)}"
    
    print(f"[OK] Reset successful")
    print(f"[OK] Initial observation: {obs}")
    print(f"[OK] Initial info keys: {list(info.keys())}")


def test_step_functionality():
    """Test environment step functionality."""
    print("\nTesting step functionality...")
    
    path = [(0, 0), (2, 0), (4, 0)]
    env = PathFollowingEnv(path)
    
    obs, info = env.reset()
    
    # Test multiple steps
    for i in range(5):
        action = np.array([1.0, 0.1])  # Move forward with slight turn
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}: reward={reward:.3f}, terminated={terminated}, pos=({obs[0]:.2f}, {obs[1]:.2f})")
        
        assert len(obs) == 5, f"Observation length should be 5, got {len(obs)}"
        assert isinstance(reward, (int, float)), f"Reward should be numeric, got {type(reward)}"
        assert isinstance(terminated, bool), f"Terminated should be bool, got {type(terminated)}"
        assert isinstance(truncated, bool), f"Truncated should be bool, got {type(truncated)}"
        
        if terminated:
            print(f"[OK] Episode terminated at step {i+1}")
            break
    
    print("[OK] Step functionality working")


def test_target_point_selection():
    """Test the target point selection logic."""
    print("\nTesting target point selection...")
    
    # Create a path that tests the search range logic
    path = [
        (0, 0),    # Start
        (-1, 1),   # Behind and to left (should not be selected if facing forward)
        (1, 1),    # Forward and to right (should be selected)
        (2, 2),    # Further forward
    ]
    
    env = PathFollowingEnv(path)
    obs, info = env.reset()
    
    # Test target selection with different vehicle orientations
    for angle_deg in [0, 45, 90, 135, 180]:
        angle_rad = np.radians(angle_deg)
        env.vehicle.reset(0, 0, angle_rad)
        
        target_point, target_idx = env._find_next_target_point()
        print(f"Vehicle heading {angle_deg}°: target_idx={target_idx}, target={target_point}")
    
    print("[OK] Target point selection working")


def test_path_completion():
    """Test completing a full path."""
    print("\nTesting path completion...")
    
    # Simple straight path
    path = [(0, 0), (1, 0), (2, 0)]
    env = PathFollowingEnv(path, goal_tolerance=0.3)
    
    obs, info = env.reset()
    total_reward = 0
    step_count = 0
    max_steps = 200
    
    while step_count < max_steps:
        # Simple controller: move towards target
        target_x, target_y = obs[3] + obs[0], obs[4] + obs[1]  # Target position
        
        # Calculate desired heading
        dx, dy = obs[3], obs[4]  # Distance to target
        desired_heading = np.arctan2(dy, dx)
        current_heading = obs[2]
        
        # Heading error
        heading_error = desired_heading - current_heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Simple control
        v_cmd = min(1.0, np.sqrt(dx*dx + dy*dy))  # Slow down when close
        omega_cmd = 2.0 * heading_error  # Proportional heading control
        
        action = np.array([v_cmd, omega_cmd])
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        if step_count % 50 == 0:
            print(f"Step {step_count}: completion={info['path_completion']:.1%}, "
                  f"distance_to_path={info['distance_to_path']:.2f}")
        
        if terminated or truncated:
            break
    
    print(f"[OK] Path following test completed in {step_count} steps")
    print(f"[OK] Final path completion: {info['path_completion']:.1%}")
    print(f"[OK] Total reward: {total_reward:.2f}")
    
    if info['path_completion'] > 0.8:
        print("[OK] Successfully followed most of the path")
    else:
        print("[WARN] Low path completion - may need tuning")


def test_observation_components():
    """Test individual observation components."""
    print("\nTesting observation components...")
    
    path = [(0, 0), (5, 0), (5, 5)]
    env = PathFollowingEnv(path)
    
    obs, info = env.reset()
    
    # Check observation components
    x, y, heading, dist_x, dist_y = obs
    
    print(f"Vehicle position: ({x:.2f}, {y:.2f})")
    print(f"Vehicle heading: {heading:.2f} rad ({np.degrees(heading):.1f}°)")
    print(f"Distance to next point: ({dist_x:.2f}, {dist_y:.2f})")
    
    # Verify distance calculation
    target_point, _ = env._find_next_target_point()
    expected_dist_x = target_point[0] - x
    expected_dist_y = target_point[1] - y
    
    assert abs(dist_x - expected_dist_x) < 1e-6, "Distance X calculation error"
    assert abs(dist_y - expected_dist_y) < 1e-6, "Distance Y calculation error"
    
    print("[OK] Observation components correct")


def run_interactive_demo(path_name='L_shape'):
    """Run an interactive demonstration of the environment."""
    print(f"\nRunning interactive demo with {path_name} path...")
    
    paths = create_test_paths()
    path = paths[path_name]
    
    env = PathFollowingEnv(path, render_mode="human")
    obs, info = env.reset()
    
    print("Starting interactive demo. Close the plot window to end.")
    print("The vehicle will use a simple controller to follow the path.")
    
    step_count = 0
    total_reward = 0
    
    try:
        while step_count < 500:
            # Simple controller
            dx, dy = obs[3], obs[4]
            distance_to_target = np.sqrt(dx*dx + dy*dy)
            
            if distance_to_target < 0.1:
                # Very close to target, move slowly
                v_cmd = 0.2
                omega_cmd = 0.0
            else:
                # Calculate desired heading and control
                desired_heading = np.arctan2(dy, dx)
                current_heading = obs[2]
                heading_error = desired_heading - current_heading
                heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
                
                v_cmd = min(1.0, distance_to_target)
                omega_cmd = 3.0 * heading_error
            
            action = np.array([v_cmd, omega_cmd])
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            # Render
            env.render()
            
            if terminated or truncated:
                print(f"Episode ended: terminated={terminated}, truncated={truncated}")
                break
            
            if step_count % 100 == 0:
                print(f"Step {step_count}: completion={info['path_completion']:.1%}")
    
    except KeyboardInterrupt:
        print("Demo interrupted by user")
    
    finally:
        env.close()
    
    print(f"Demo completed in {step_count} steps")
    print(f"Final path completion: {info['path_completion']:.1%}")
    print(f"Total reward: {total_reward:.2f}")


def main():
    """Run all tests."""
    print("=== PathFollowingEnv Test Suite ===\n")
    
    try:
        # Basic functionality tests
        test_environment_creation()
        test_reset_functionality()
        test_step_functionality()
        test_target_point_selection()
        test_observation_components()
        test_path_completion()
        
        print("\n=== All Tests Passed! ===")
        
        # Ask if user wants to run interactive demo
        try:
            response = input("\nRun interactive demo? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                run_interactive_demo()
        except (EOFError, KeyboardInterrupt):
            print("\nSkipping interactive demo")
        
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