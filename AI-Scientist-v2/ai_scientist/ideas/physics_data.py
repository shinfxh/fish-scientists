# generate data for a double pendulum
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def double_pendulum_equations(t, y, L1, L2, m1, m2, g):
    """
    Equations of motion for a double pendulum system.
    
    Parameters:
    t: time
    y: state vector [theta1, z1, theta2, z2] where z1 = dtheta1/dt, z2 = dtheta2/dt
    L1, L2: lengths of pendulum arms
    m1, m2: masses of pendulum bobs
    g: gravitational acceleration
    """
    theta1, z1, theta2, z2 = y
    
    # Calculate cos and sin differences for efficiency
    cos_diff = np.cos(theta1 - theta2)
    sin_diff = np.sin(theta1 - theta2)
    
    # Denominators
    den1 = (m1 + m2) * L1 - m2 * L1 * cos_diff * cos_diff
    den2 = (L2 / L1) * den1
    
    # First pendulum angular acceleration
    num1 = (-m2 * L1 * z1**2 * sin_diff * cos_diff +
            m2 * g * np.sin(theta2) * cos_diff -
            m2 * L2 * z2**2 * sin_diff -
            (m1 + m2) * g * np.sin(theta1))
    
    dz1_dt = num1 / den1
    
    # Second pendulum angular acceleration
    num2 = (m2 * L2 * z2**2 * sin_diff * cos_diff +
            (m1 + m2) * (g * np.sin(theta1) * cos_diff + L1 * z1**2 * sin_diff) -
            (m1 + m2) * g * np.sin(theta2))
    
    dz2_dt = num2 / den2
    
    return [z1, dz1_dt, z2, dz2_dt]

def simulate_double_pendulum(L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81,
                           theta1_0=np.pi/2, theta2_0=np.pi/2,
                           omega1_0=0.0, omega2_0=0.0,
                           t_span=(0, 20), dt=0.01):
    """
    Simulate a double pendulum system.
    
    Parameters:
    L1, L2: lengths of pendulum arms (m)
    m1, m2: masses of pendulum bobs (kg)
    g: gravitational acceleration (m/s^2)
    theta1_0, theta2_0: initial angles (rad)
    omega1_0, omega2_0: initial angular velocities (rad/s)
    t_span: time span for simulation (s)
    dt: time step (s)
    
    Returns:
    DataFrame with time, angles, and cartesian coordinates
    """
    
    # Initial conditions
    y0 = [theta1_0, omega1_0, theta2_0, omega2_0]
    
    # Time points
    t_eval = np.arange(t_span[0], t_span[1], dt)
    
    # Solve the differential equation
    sol = solve_ivp(double_pendulum_equations, t_span, y0, 
                    args=(L1, L2, m1, m2, g), t_eval=t_eval, 
                    method='RK45', rtol=1e-8)
    
    # Extract solution
    theta1 = sol.y[0]
    theta2 = sol.y[2]
    
    # Convert to Cartesian coordinates
    # First pendulum bob
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    
    # Second pendulum bob
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    
    # Create DataFrame
    data = pd.DataFrame({
        'time': sol.t,
        'theta1': theta1,
        'theta2': theta2,
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2
    })
    
    return data

def generate_multiple_trajectories(n_trajectories=5, noise_level=0.1):
    """
    Generate multiple double pendulum trajectories with slightly different initial conditions.
    
    Parameters:
    n_trajectories: number of different trajectories to generate
    noise_level: amount of random variation in initial conditions
    
    Returns:
    DataFrame with multiple trajectories
    """
    all_data = []
    
    # Base initial conditions
    base_theta1 = np.pi/2
    base_theta2 = np.pi/2
    
    for i in range(n_trajectories):
        # Add small random perturbations to initial conditions
        theta1_0 = base_theta1 + np.random.normal(0, noise_level)
        theta2_0 = base_theta2 + np.random.normal(0, noise_level)
        omega1_0 = np.random.normal(0, 0.1)
        omega2_0 = np.random.normal(0, 0.1)
        
        # Simulate this trajectory
        data = simulate_double_pendulum(
            theta1_0=theta1_0, theta2_0=theta2_0,
            omega1_0=omega1_0, omega2_0=omega2_0,
            t_span=(0, 10), dt=0.02
        )
        
        # Add trajectory ID
        data['trajectory_id'] = i
        all_data.append(data)
    
    return pd.concat(all_data, ignore_index=True)

def plot_pendulum_trajectory(data, save_path='double_pendulum_trajectory.png'):
    """
    Plot the trajectory of both pendulum bobs in the x-y plane.
    
    Parameters:
    data: DataFrame with pendulum simulation data
    save_path: path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create a colormap based on time for both trajectories
    colors = plt.cm.viridis(data['time'] / data['time'].max())
    
    # Plot trajectory of first pendulum bob
    scatter1 = ax.scatter(data['x1'], data['y1'], c=data['time'], 
                         cmap='viridis', s=1, alpha=0.7, label='First bob')
    
    # Plot trajectory of second pendulum bob
    scatter2 = ax.scatter(data['x2'], data['y2'], c=data['time'], 
                         cmap='plasma', s=1, alpha=0.7, label='Second bob')
    
    # Add starting points
    ax.plot(data['x1'].iloc[0], data['y1'].iloc[0], 'go', markersize=8, label='Start (bob 1)')
    ax.plot(data['x2'].iloc[0], data['y2'].iloc[0], 'ro', markersize=8, label='Start (bob 2)')
    
    # Add ending points
    ax.plot(data['x1'].iloc[-1], data['y1'].iloc[-1], 'g^', markersize=8, label='End (bob 1)')
    ax.plot(data['x2'].iloc[-1], data['y2'].iloc[-1], 'r^', markersize=8, label='End (bob 2)')
    
    # Formatting
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Double Pendulum Trajectory in X-Y Plane', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    # Add colorbar for time
    cbar = plt.colorbar(scatter1, ax=ax, shrink=0.8)
    cbar.set_label('Time (s)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Trajectory plot saved to '{save_path}'")
    plt.show()

def main():
    """
    Main function to generate double pendulum data and save to CSV.
    """
    print("Generating double pendulum data...")
    
    # Generate a single trajectory with detailed parameters
    print("Simulating single trajectory...")
    single_data = simulate_double_pendulum(
        L1=1.0, L2=1.0,  # Slightly different lengths
        m1=1.0, m2=1.0,  # Different masses
        theta1_0=np.pi/4, theta2_0=np.pi/3,  # Interesting initial angles
        omega1_0=0., omega2_0=0.,  # Initial angular velocities
        t_span=(0, 15), dt=0.01
    )
    
    # Save single trajectory
    single_data.to_csv('physics_data.csv', index=False)
    print(f"Single trajectory saved to 'physics_data.csv' ({len(single_data)} points)")
    
    # Plot the single trajectory
    print("Creating trajectory plot...")
    plot_pendulum_trajectory(single_data)
    
    # Generate multiple trajectories to show chaotic behavior
    print("Simulating multiple trajectories...")
    multi_data = generate_multiple_trajectories(n_trajectories=100, noise_level=0.05)
    
    # Save multiple trajectories
    multi_data.to_csv('double_pendulum_multiple.csv', index=False)
    print(f"Multiple trajectories saved to 'double_pendulum_multiple.csv' ({len(multi_data)} points)")
    
    # Print summary statistics
    print("\nData summary:")
    print(f"Time range: {single_data['time'].min():.2f} to {single_data['time'].max():.2f} seconds")
    print(f"X1 range: {single_data['x1'].min():.3f} to {single_data['x1'].max():.3f}")
    print(f"Y1 range: {single_data['y1'].min():.3f} to {single_data['y1'].max():.3f}")
    print(f"X2 range: {single_data['x2'].min():.3f} to {single_data['x2'].max():.3f}")
    print(f"Y2 range: {single_data['y2'].min():.3f} to {single_data['y2'].max():.3f}")
    
    return single_data, multi_data

if __name__ == "__main__":
    single_data, multi_data = main()