import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

# Create output directories
os.makedirs('/home/ubuntu/emg_project/data/processed', exist_ok=True)
os.makedirs('/home/ubuntu/emg_project/visualizations/preprocessed', exist_ok=True)

# Load the .mat file
data_path = '/home/ubuntu/emg_project/data/s4_full.mat'
mat_data = sio.loadmat(data_path)

# Function to preprocess EMG data
def preprocess_emg(emg_data):
    """
    Preprocess EMG data:
    1. Normalize to [0, 1] range
    2. Apply any additional filtering if needed
    """
    # Create a scaler for each channel
    n_samples, n_channels = emg_data.shape
    normalized_data = np.zeros_like(emg_data)
    
    # Normalize each channel separately
    for i in range(n_channels):
        scaler = MinMaxScaler()
        normalized_data[:, i] = scaler.fit_transform(emg_data[:, i].reshape(-1, 1)).flatten()
    
    return normalized_data

# Function to preprocess joint angle data
def preprocess_angles(angle_data):
    """
    Preprocess joint angle data:
    1. Normalize to [0, 1] range for consistent scaling with EMG
    """
    # Create a scaler for each angle
    n_samples, n_angles = angle_data.shape
    normalized_data = np.zeros_like(angle_data)
    
    # Normalize each angle separately
    for i in range(n_angles):
        scaler = MinMaxScaler()
        normalized_data[:, i] = scaler.fit_transform(angle_data[:, i].reshape(-1, 1)).flatten()
    
    return normalized_data, angle_data  # Return both normalized and original data

# Function to prepare data for the model
def prepare_data_for_model(emg_cells, angle_cells, window_size=50, stride=10):
    """
    Prepare data for the model by creating windowed sequences
    
    Args:
        emg_cells: Cell array of EMG data
        angle_cells: Cell array of joint angle data
        window_size: Size of the sliding window
        stride: Stride for the sliding window
        
    Returns:
        X: Input sequences (windows of EMG data)
        y: Target values (corresponding joint angles)
    """
    X_list = []
    y_list = []
    y_orig_list = []  # Original (non-normalized) angles for evaluation
    
    # Process each trial and task
    for trial in range(emg_cells.shape[0]):
        for task in range(emg_cells.shape[1]):
            # Get EMG and angle data for this trial and task
            emg_data = emg_cells[trial, task]
            angle_data = angle_cells[trial, task]
            
            # Skip if data is empty
            if emg_data.size == 0 or angle_data.size == 0:
                continue
            
            # Preprocess data
            emg_norm = preprocess_emg(emg_data)
            angle_norm, angle_orig = preprocess_angles(angle_data)
            
            # Create windows
            for i in range(0, len(emg_norm) - window_size, stride):
                X_list.append(emg_norm[i:i+window_size])
                # Use the angle at the end of the window as the target
                y_list.append(angle_norm[i+window_size-1])
                y_orig_list.append(angle_orig[i+window_size-1])
    
    # Convert to numpy arrays
    X = np.array(X_list)
    y = np.array(y_list)
    y_orig = np.array(y_orig_list)
    
    return X, y, y_orig

# Process all trials and tasks
X_all = []
y_all = []
y_orig_all = []

# Process trial 1 (standard) separately
print("Processing trial 1 (standard) data...")
trial_idx = 0  # Trial 1 (0-indexed)
for task_idx in range(mat_data['dsfilt_emg'].shape[1]):
    emg_data = mat_data['dsfilt_emg'][trial_idx, task_idx]
    angle_data = mat_data['joint_angles'][trial_idx, task_idx]
    
    # Skip if data is empty
    if emg_data.size == 0 or angle_data.size == 0:
        print(f"  Task {task_idx+1}: No data")
        continue
    
    # Preprocess and create windows
    emg_norm = preprocess_emg(emg_data)
    angle_norm, angle_orig = preprocess_angles(angle_data)
    
    # Visualize preprocessed data for the first task
    if task_idx == 0:
        plt.figure(figsize=(15, 10))
        
        # Plot EMG channels
        plt.subplot(2, 1, 1)
        for i in range(min(8, emg_norm.shape[1])):
            plt.plot(emg_norm[:1000, i], label=f'EMG {i}')
        plt.title('Normalized EMG Channels (First 1000 samples)')
        plt.legend()
        plt.grid(True)
        
        # Plot joint angles
        plt.subplot(2, 1, 2)
        for i in range(min(5, angle_norm.shape[1])):
            plt.plot(angle_norm[:1000, i], label=f'Angle {i}')
        plt.title('Normalized Joint Angles (First 1000 samples)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('/home/ubuntu/emg_project/visualizations/preprocessed/trial1_task1_preprocessed.png')
        plt.close()
    
    print(f"  Task {task_idx+1}: EMG shape {emg_data.shape}, Angle shape {angle_data.shape}")
    
    # Create windows with a smaller window size for the standard trial
    window_size = 50
    stride = 10
    
    for i in range(0, len(emg_norm) - window_size, stride):
        X_all.append(emg_norm[i:i+window_size])
        y_all.append(angle_norm[i+window_size-1])
        y_orig_all.append(angle_orig[i+window_size-1])

# Convert to numpy arrays
X_all = np.array(X_all)
y_all = np.array(y_all)
y_orig_all = np.array(y_orig_all)

print(f"\nTotal dataset size:")
print(f"  X shape: {X_all.shape}")
print(f"  y shape: {y_all.shape}")
print(f"  y_orig shape: {y_orig_all.shape}")

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Save preprocessed data
np.save('/home/ubuntu/emg_project/data/processed/X_train.npy', X_train)
np.save('/home/ubuntu/emg_project/data/processed/y_train.npy', y_train)
np.save('/home/ubuntu/emg_project/data/processed/X_val.npy', X_val)
np.save('/home/ubuntu/emg_project/data/processed/y_val.npy', y_val)
np.save('/home/ubuntu/emg_project/data/processed/X_test.npy', X_test)
np.save('/home/ubuntu/emg_project/data/processed/y_test.npy', y_test)

print("\nPreprocessed data saved to /home/ubuntu/emg_project/data/processed/")
