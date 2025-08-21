#!/usr/bin/env python3
"""
Interactive demonstration of the differential drive vehicle.

This script provides:
1. Real-time keyboard control
2. Parameter adjustment interface
3. Multiple demonstration scenarios
4. Live plotting and analysis
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.animation import FuncAnimation
import threading
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ..vehicle_model.vehicle import DifferentialDriveVehicle, VehicleParams


class VehicleDemo:
    """Interactive vehicle demonstration with real-time control and visualization."""

    def __init__(self):
        self.vehicle = DifferentialDriveVehicle(VehicleParams())

        # Control state
        self.v_cmd = 0.0
        self.omega_cmd = 0.0
        self.auto_mode = False
        self.demo_scenario = "manual"

        # Demo parameters
        self.demo_time = 0.0
        self.demo_patterns = {
            "circle": self.circle_pattern,
            "figure8": self.figure8_pattern,
            "square": self.square_pattern,
            "random": self.random_pattern,
        }

        # Animation data
        self.max_history = 300
        self.time_history = []
        self.v_cmd_history = []
        self.v_actual_history = []
        self.omega_cmd_history = []
        self.omega_actual_history = []

        # Setup GUI
        self.setup_gui()

        # Control instructions
        self.print_instructions()

    def setup_gui(self):
        """Setup the GUI with plots and controls."""
        self.fig = plt.figure(figsize=(16, 10))

        # Main vehicle plot
        self.ax_vehicle = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        self.ax_vehicle.set_xlim(-3, 3)
        self.ax_vehicle.set_ylim(-3, 3)
        self.ax_vehicle.set_aspect("equal")
        self.ax_vehicle.set_title("Vehicle State (Press keys for control)")

        # Velocity plots
        self.ax_velocities = plt.subplot2grid((3, 3), (0, 2))
        self.ax_velocities.set_title("Velocities")
        self.ax_velocities.set_ylabel("Linear (m/s)")

        self.ax_angular = plt.subplot2grid((3, 3), (1, 2), sharex=self.ax_velocities)
        self.ax_angular.set_title("Angular Velocity")
        self.ax_angular.set_ylabel("Angular (rad/s)")
        self.ax_angular.set_xlabel("Time (s)")

        # Control panel
        self.ax_controls = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        self.ax_controls.set_xlim(0, 1)
        self.ax_controls.set_ylim(0, 1)
        self.ax_controls.axis("off")

        # Add sliders for parameter adjustment
        self.setup_sliders()

        # Connect keyboard events
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self.on_key_release)

        # Make sure the figure can receive focus
        self.fig.canvas.set_window_title(
            "Vehicle Demo - Click here for keyboard control"
        )

    def setup_sliders(self):
        """Setup parameter adjustment sliders."""
        # Slider axes
        slider_height = 0.03
        slider_spacing = 0.05

        ax_tau_linear = plt.axes([0.15, 0.25, 0.3, slider_height])
        ax_tau_angular = plt.axes([0.15, 0.20, 0.3, slider_height])
        ax_max_vel = plt.axes([0.15, 0.15, 0.3, slider_height])

        # Create sliders
        self.slider_tau_linear = widgets.Slider(
            ax_tau_linear,
            "τ_linear",
            0.05,
            0.5,
            valinit=self.vehicle.params.tau_linear,
            valstep=0.01,
        )
        self.slider_tau_angular = widgets.Slider(
            ax_tau_angular,
            "τ_angular",
            0.05,
            0.5,
            valinit=self.vehicle.params.tau_angular,
            valstep=0.01,
        )
        self.slider_max_vel = widgets.Slider(
            ax_max_vel,
            "Max velocity",
            0.5,
            5.0,
            valinit=self.vehicle.params.max_linear_velocity,
            valstep=0.1,
        )

        # Connect slider events
        self.slider_tau_linear.on_changed(self.update_parameters)
        self.slider_tau_angular.on_changed(self.update_parameters)
        self.slider_max_vel.on_changed(self.update_parameters)

        # Demo mode buttons
        ax_demo_buttons = plt.axes([0.55, 0.15, 0.4, 0.15])
        ax_demo_buttons.axis("off")

        # Add demo scenario buttons
        button_width = 0.08
        button_height = 0.04
        button_spacing = 0.1

        scenarios = ["manual", "circle", "figure8", "square", "random"]
        self.demo_buttons = {}

        for i, scenario in enumerate(scenarios):
            x_pos = 0.55 + (i % 3) * button_spacing
            y_pos = 0.25 - (i // 3) * 0.06

            ax_button = plt.axes([x_pos, y_pos, button_width, button_height])
            button = widgets.Button(ax_button, scenario.title())
            button.on_clicked(lambda event, s=scenario: self.set_demo_scenario(s))
            self.demo_buttons[scenario] = button

    def update_parameters(self, val):
        """Update vehicle parameters from sliders."""
        self.vehicle.params.tau_linear = self.slider_tau_linear.val
        self.vehicle.params.tau_angular = self.slider_tau_angular.val
        self.vehicle.params.max_linear_velocity = self.slider_max_vel.val
        self.vehicle.params.max_angular_velocity = self.slider_max_vel.val

    def set_demo_scenario(self, scenario):
        """Set the demo scenario."""
        self.demo_scenario = scenario
        self.auto_mode = scenario != "manual"
        self.demo_time = 0.0

        if scenario == "manual":
            self.v_cmd = 0.0
            self.omega_cmd = 0.0

        print(f"Demo mode: {scenario}")

    def print_instructions(self):
        """Print control instructions."""
        print("\n" + "=" * 60)
        print("VEHICLE DEMO - INTERACTIVE CONTROL")
        print("=" * 60)
        print("KEYBOARD CONTROLS:")
        print("  Arrow Keys:")
        print("    ↑/↓    - Forward/Backward")
        print("    ←/→    - Turn Left/Right")
        print("  Other Keys:")
        print("    SPACE  - Stop/Brake")
        print("    'r'    - Reset vehicle position")
        print("    'c'    - Clear trajectory")
        print("    'q'    - Quit demo")
        print("\nSLIDERS:")
        print("  Adjust time constants and max velocity in real-time")
        print("\nDEMO MODES:")
        print("  Click buttons to run automated patterns")
        print("\nClick on the plot window to activate keyboard control!")
        print("=" * 60)

    def on_key_press(self, event):
        """Handle key press events."""
        if event.key == "up":
            self.v_cmd = min(self.v_cmd + 0.2, 2.0)
            self.auto_mode = False
        elif event.key == "down":
            self.v_cmd = max(self.v_cmd - 0.2, -2.0)
            self.auto_mode = False
        elif event.key == "left":
            self.omega_cmd = min(self.omega_cmd + 0.3, 3.0)
            self.auto_mode = False
        elif event.key == "right":
            self.omega_cmd = max(self.omega_cmd - 0.3, -3.0)
            self.auto_mode = False
        elif event.key == " ":  # Space bar
            self.v_cmd = 0.0
            self.omega_cmd = 0.0
            self.auto_mode = False
        elif event.key == "r":
            self.vehicle.reset()
            self.clear_history()
            print("Vehicle reset to origin")
        elif event.key == "c":
            self.clear_history()
            print("Trajectory cleared")
        elif event.key == "q":
            plt.close("all")
            return

    def on_key_release(self, event):
        """Handle key release events."""
        # For this demo, we'll use press-and-hold behavior
        # You could implement different behavior here
        pass

    def clear_history(self):
        """Clear all history data."""
        self.time_history.clear()
        self.v_cmd_history.clear()
        self.v_actual_history.clear()
        self.omega_cmd_history.clear()
        self.omega_actual_history.clear()
        self.vehicle.state_history.clear()
        self.vehicle.time_history.clear()

    def circle_pattern(self, t):
        """Generate circular motion pattern."""
        return 1.0, 1.0  # Constant forward and angular velocity

    def figure8_pattern(self, t):
        """Generate figure-8 pattern."""
        v = 0.8
        omega = 1.5 * np.sin(t * 0.8)  # Varying angular velocity
        return v, omega

    def square_pattern(self, t):
        """Generate square pattern."""
        cycle_time = 8.0
        phase = (t % cycle_time) / cycle_time

        if phase < 0.4:  # Forward
            return 1.0, 0.0
        elif phase < 0.5:  # Turn
            return 0.0, 1.57  # 90 degrees in 0.1 * cycle_time
        elif phase < 0.9:  # Forward
            return 1.0, 0.0
        else:  # Turn
            return 0.0, 1.57

    def random_pattern(self, t):
        """Generate random motion pattern."""
        # Smooth random walk
        v = 1.0 + 0.5 * np.sin(0.3 * t) * np.sin(0.7 * t)
        omega = 1.0 * np.sin(0.2 * t) * np.cos(0.5 * t)
        return v, omega

    def update_commands(self):
        """Update commands based on current mode."""
        if self.auto_mode and self.demo_scenario in self.demo_patterns:
            self.v_cmd, self.omega_cmd = self.demo_patterns[self.demo_scenario](
                self.demo_time
            )
            self.demo_time += 0.05  # 20 Hz update rate

    def update_history(self):
        """Update history data for plotting."""
        self.time_history.append(self.vehicle.current_time)
        self.v_cmd_history.append(self.v_cmd)
        self.v_actual_history.append(self.vehicle.v_actual)
        self.omega_cmd_history.append(self.omega_cmd)
        self.omega_actual_history.append(self.omega_actual)

        # Limit history length
        if len(self.time_history) > self.max_history:
            self.time_history.pop(0)
            self.v_cmd_history.pop(0)
            self.v_actual_history.pop(0)
            self.omega_cmd_history.pop(0)
            self.omega_actual_history.pop(0)

    def update_plots(self):
        """Update all plots with current data."""
        # Clear axes
        self.ax_vehicle.clear()
        self.ax_velocities.clear()
        self.ax_angular.clear()

        # Vehicle plot
        self.vehicle.plot(self.ax_vehicle, show_trajectory=True, show_velocities=True)
        self.ax_vehicle.set_xlim(-3, 3)
        self.ax_vehicle.set_ylim(-3, 3)

        # Add command display
        cmd_text = f"Commands: v={self.v_cmd:.2f} m/s, ω={self.omega_cmd:.2f} rad/s"
        mode_text = (
            f'Mode: {self.demo_scenario} {"(AUTO)" if self.auto_mode else "(MANUAL)"}'
        )
        self.ax_vehicle.text(
            0.02,
            0.02,
            f"{cmd_text}\n{mode_text}",
            transform=self.ax_vehicle.transAxes,
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
        )

        # Velocity plots
        if len(self.time_history) > 1:
            self.ax_velocities.plot(
                self.time_history,
                self.v_cmd_history,
                "r--",
                label="Commanded",
                alpha=0.8,
            )
            self.ax_velocities.plot(
                self.time_history,
                self.v_actual_history,
                "b-",
                label="Actual",
                linewidth=2,
            )
            self.ax_velocities.set_ylabel("Linear (m/s)")
            self.ax_velocities.legend()
            self.ax_velocities.grid(True, alpha=0.3)

            self.ax_angular.plot(
                self.time_history,
                self.omega_cmd_history,
                "r--",
                label="Commanded",
                alpha=0.8,
            )
            self.ax_angular.plot(
                self.time_history,
                self.omega_actual_history,
                "g-",
                label="Actual",
                linewidth=2,
            )
            self.ax_angular.set_ylabel("Angular (rad/s)")
            self.ax_angular.set_xlabel("Time (s)")
            self.ax_angular.legend()
            self.ax_angular.grid(True, alpha=0.3)

        # Update titles with current values
        self.ax_velocities.set_title(
            f"Linear Velocity (τ={self.vehicle.params.tau_linear:.2f}s)"
        )
        self.ax_angular.set_title(
            f"Angular Velocity (τ={self.vehicle.params.tau_angular:.2f}s)"
        )

    def animation_update(self, frame):
        """Animation update function."""
        # Update commands
        self.update_commands()

        # Step vehicle simulation
        self.vehicle.step(self.v_cmd, self.omega_cmd)

        # Update history
        self.update_history()

        # Update plots
        self.update_plots()

        return []

    def run(self):
        """Run the interactive demo."""
        print("Starting interactive demo...")
        print("Click on the plot window and use keyboard controls!")

        # Create animation
        self.anim = FuncAnimation(
            self.fig, self.animation_update, interval=50, blit=False, repeat=True
        )

        # Show the plot
        plt.tight_layout()
        plt.show()


def run_benchmark_comparison():
    """Run a benchmark comparison of different vehicle configurations."""
    print("\nRunning benchmark comparison...")

    # Different configurations to compare
    configs = {
        "Fast Response": VehicleParams(tau_linear=0.1, tau_angular=0.08),
        "Default": VehicleParams(),
        "Slow Response": VehicleParams(tau_linear=0.4, tau_angular=0.3),
        "High Performance": VehicleParams(
            max_linear_velocity=3.0, max_angular_velocity=5.0
        ),
        "Conservative": VehicleParams(
            max_linear_velocity=1.0, max_angular_velocity=1.5
        ),
    }

    # Test scenario: circle with radius 1m
    test_duration = 10.0
    circle_v = 1.0
    circle_omega = 1.0

    results = {}

    for config_name, params in configs.items():
        print(f"Testing {config_name}...")

        vehicle = DifferentialDriveVehicle(params)

        # Run test
        for _ in range(int(test_duration / vehicle.dt)):
            vehicle.step(circle_v, circle_omega)

        # Calculate metrics
        final_pos = vehicle.get_position()
        distance_from_origin = np.linalg.norm(final_pos)

        # Expected circle radius for perfect tracking
        expected_radius = circle_v / circle_omega
        radius_error = abs(distance_from_origin - expected_radius)

        # Velocity tracking error
        v_error = abs(vehicle.v_actual - circle_v)
        omega_error = abs(vehicle.omega_actual - circle_omega)

        results[config_name] = {
            "radius_error": radius_error,
            "velocity_error": v_error,
            "angular_error": omega_error,
            "final_position": final_pos,
        }

        print(f"  Radius error: {radius_error:.3f}m")
        print(f"  Velocity error: {v_error:.3f}m/s")
        print(f"  Angular error: {omega_error:.3f}rad/s")

    # Plot comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    config_names = list(results.keys())
    radius_errors = [results[name]["radius_error"] for name in config_names]
    velocity_errors = [results[name]["velocity_error"] for name in config_names]
    angular_errors = [results[name]["angular_error"] for name in config_names]

    # Bar plots
    x_pos = np.arange(len(config_names))

    ax1.bar(x_pos, radius_errors)
    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Radius Error (m)")
    ax1.set_title("Circle Tracking - Radius Error")
    ax1.set_xticklabels(config_names, rotation=45)

    ax2.bar(x_pos, velocity_errors)
    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Velocity Error (m/s)")
    ax2.set_title("Velocity Tracking Error")
    ax2.set_xticklabels(config_names, rotation=45)

    ax3.bar(x_pos, angular_errors)
    ax3.set_xlabel("Configuration")
    ax3.set_ylabel("Angular Error (rad/s)")
    ax3.set_title("Angular Velocity Tracking Error")
    ax3.set_xticklabels(config_names, rotation=45)

    # Trajectory comparison
    ax4.set_xlim(-2, 2)
    ax4.set_ylim(-2, 2)
    ax4.set_aspect("equal")

    # Draw expected circle
    theta = np.linspace(0, 2 * np.pi, 100)
    expected_x = expected_radius * np.cos(theta)
    expected_y = expected_radius * np.sin(theta)
    ax4.plot(expected_x, expected_y, "k--", label="Expected", linewidth=2)

    # Plot actual final positions
    colors = plt.cm.tab10(np.linspace(0, 1, len(config_names)))
    for i, (name, result) in enumerate(results.items()):
        pos = result["final_position"]
        ax4.plot(pos[0], pos[1], "o", color=colors[i], markersize=10, label=name)

    ax4.set_xlabel("X Position (m)")
    ax4.set_ylabel("Y Position (m)")
    ax4.set_title("Final Positions After Circle Test")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\nBenchmark Summary:")
    print("-" * 50)
    best_config = min(results.items(), key=lambda x: x[1]["radius_error"])
    print(f"Best overall tracking: {best_config[0]}")
    print(f"Radius error: {best_config[1]['radius_error']:.3f}m")


if __name__ == "__main__":
    print("Vehicle Interactive Demo")
    print("=" * 50)

    print("\nSelect demo mode:")
    print("1. Interactive Control Demo")
    print("2. Benchmark Comparison")
    print("3. Both")

    try:
        choice = input("Enter choice (1-3): ").strip()

        if choice == "1":
            demo = VehicleDemo()
            demo.run()
        elif choice == "2":
            run_benchmark_comparison()
        elif choice == "3":
            run_benchmark_comparison()
            input("\nPress Enter to start interactive demo...")
            demo = VehicleDemo()
            demo.run()
        else:
            print("Invalid choice, running interactive demo...")
            demo = VehicleDemo()
            demo.run()

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback

        traceback.print_exc()

    print("Demo completed!")
