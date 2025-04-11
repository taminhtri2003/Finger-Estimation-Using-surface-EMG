import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

# Load the .mat file
data_path = '/home/ubuntu/emg_project/data/s4_full.mat'
mat_data = sio.loadmat(data_path)

# Print the keys in the .mat file to understand its structure
print("Keys in the .mat file:")
for key in mat_data.keys():
    if not key.startswith('__'):  # Skip metadata keys
        print(f"Key: {key}")
        if isinstance(mat_data[key], np.ndarray):
            print(f"  Shape: {mat_data[key].shape}")
            print(f"  Type: {mat_data[key].dtype}")
            
            # If it's a cell array, print more details
            if mat_data[key].dtype == 'object':
                if len(mat_data[key].shape) == 2:
                    rows, cols = mat_data[key].shape
                    print(f"  Cell array with {rows} rows and {cols} columns")
                    
                    # Print details of the first cell to understand structure
                    if rows > 0 and cols > 0:
                        first_cell = mat_data[key][0, 0]
                        if isinstance(first_cell, np.ndarray):
                            print(f"  First cell shape: {first_cell.shape}")
                            print(f"  First cell type: {first_cell.dtype}")

# Create output directory for plots
os.makedirs('/home/ubuntu/emg_project/visualizations/data_exploration', exist_ok=True)

# Function to explore and visualize data from a specific cell
def explore_cell_data(data_key, row_idx=0, col_idx=0, sample_size=1000):
    if data_key not in mat_data:
        print(f"Key {data_key} not found in the .mat file")
        return
    
    if mat_data[data_key].dtype != 'object':
        print(f"Key {data_key} is not a cell array")
        return
    
    if row_idx >= mat_data[data_key].shape[0] or col_idx >= mat_data[data_key].shape[1]:
        print(f"Indices out of range. Max: {mat_data[data_key].shape}")
        return
    
    cell_data = mat_data[data_key][row_idx, col_idx]
    
    if not isinstance(cell_data, np.ndarray):
        print(f"Cell data is not a numpy array. Type: {type(cell_data)}")
        return
    
    print(f"\nExploring {data_key} at cell [{row_idx}, {col_idx}]:")
    print(f"  Shape: {cell_data.shape}")
    print(f"  Type: {cell_data.dtype}")
    
    # Plot a subset of the data
    plt.figure(figsize=(15, 8))
    
    # Determine how many columns to plot
    n_cols = min(8, cell_data.shape[1])
    
    # Plot each column
    for i in range(n_cols):
        plt.subplot(n_cols, 1, i+1)
        plt.plot(cell_data[:sample_size, i])
        plt.title(f'{data_key} [{row_idx}, {col_idx}] - Column {i}')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'/home/ubuntu/emg_project/visualizations/data_exploration/{data_key}_r{row_idx}_c{col_idx}.png')
    plt.close()
    
    return cell_data

# Explore EMG data from the first trial and task
emg_data = explore_cell_data('dsfilt_emg', 0, 0)

# Explore finger kinematics data from the first trial and task
kinematics_data = explore_cell_data('finger_kinematics', 0, 0)

# Explore joint angles data from the first trial and task
joint_angles_data = explore_cell_data('joint_angles', 0, 0)

# Print summary statistics
if emg_data is not None:
    print("\nEMG Data Summary Statistics:")
    print(f"  Min: {np.min(emg_data)}")
    print(f"  Max: {np.max(emg_data)}")
    print(f"  Mean: {np.mean(emg_data)}")
    print(f"  Std: {np.std(emg_data)}")

if joint_angles_data is not None:
    print("\nJoint Angles Data Summary Statistics:")
    print(f"  Min: {np.min(joint_angles_data)}")
    print(f"  Max: {np.max(joint_angles_data)}")
    print(f"  Mean: {np.mean(joint_angles_data)}")
    print(f"  Std: {np.std(joint_angles_data)}")

# Plot correlation between EMG and joint angles for the first trial and task
if emg_data is not None and joint_angles_data is not None:
    plt.figure(figsize=(15, 10))
    
    # Select a subset of data points for visualization
    subset_size = 1000
    
    # For each joint angle, find the EMG channel with highest correlation
    for j in range(min(5, joint_angles_data.shape[1])):  # Plot first 5 joint angles
        correlations = []
        for i in range(emg_data.shape[1]):
            corr = np.corrcoef(emg_data[:subset_size, i], joint_angles_data[:subset_size, j])[0, 1]
            correlations.append((i, corr))
        
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        best_emg_idx = correlations[0][0]
        
        plt.subplot(5, 1, j+1)
        plt.plot(emg_data[:subset_size, best_emg_idx], label=f'EMG Channel {best_emg_idx}')
        plt.plot(joint_angles_data[:subset_size, j], label=f'Joint Angle {j}')
        plt.title(f'EMG Channel {best_emg_idx} vs Joint Angle {j} (Correlation: {correlations[0][1]:.3f})')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/emg_project/visualizations/data_exploration/emg_joint_correlation.png')
    plt.close()

print("\nData exploration completed. Visualizations saved to /home/ubuntu/emg_project/visualizations/data_exploration/")
