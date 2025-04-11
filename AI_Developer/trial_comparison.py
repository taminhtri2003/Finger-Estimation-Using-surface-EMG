import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import shap
from tensorflow.keras.models import load_model
import matplotlib.cm as cm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Create output directories
os.makedirs('/home/ubuntu/emg_project/visualizations/trial_comparison', exist_ok=True)

# Load the .mat file
data_path = '/home/ubuntu/emg_project/data/s4_full.mat'
mat_data = sio.loadmat(data_path)

# Load the trained model
model_path = '/home/ubuntu/emg_project/models/multi_head_attention_model.keras'
model = load_model(model_path)
print("Model loaded successfully.")

# Define finger groups for visualization
finger_groups = [
    ('Thumb', 0, 2),
    ('Index', 2, 5),
    ('Middle', 5, 8),
    ('Ring', 8, 11),
    ('Little', 11, 14)
]

# Define task names
task_names = [
    "Thumb Flexion/Extension",
    "Index Flexion/Extension",
    "Middle Flexion/Extension",
    "Ring Flexion/Extension",
    "Little Flexion/Extension",
    "All Fingers Flexion/Extension",
    "Random Free Movement"
]

# Function to preprocess EMG data
def preprocess_emg(emg_data):
    """
    Preprocess EMG data:
    1. Normalize to [0, 1] range
    """
    from sklearn.preprocessing import MinMaxScaler
    
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
    from sklearn.preprocessing import MinMaxScaler
    
    # Create a scaler for each angle
    n_samples, n_angles = angle_data.shape
    normalized_data = np.zeros_like(angle_data)
    
    # Normalize each angle separately
    for i in range(n_angles):
        scaler = MinMaxScaler()
        normalized_data[:, i] = scaler.fit_transform(angle_data[:, i].reshape(-1, 1)).flatten()
    
    return normalized_data, angle_data  # Return both normalized and original data

# Function to create windowed sequences from data
def create_windows(emg_data, window_size=50, stride=10):
    """
    Create windowed sequences from EMG data
    """
    windows = []
    for i in range(0, len(emg_data) - window_size, stride):
        windows.append(emg_data[i:i+window_size])
    return np.array(windows)

# Function to compare trials against the standard (trial 1)
def compare_trials_with_standard(standard_trial_idx=0):
    """
    Compare all trials against the standard (trial 1)
    """
    # Process standard trial (trial 1) for each task
    standard_emg = {}
    standard_angles = {}
    standard_predictions = {}
    
    print(f"Processing standard trial (Trial {standard_trial_idx+1})...")
    
    for task_idx in range(mat_data['dsfilt_emg'].shape[1]):
        # Get EMG and angle data for the standard trial
        emg_data = mat_data['dsfilt_emg'][standard_trial_idx, task_idx]
        angle_data = mat_data['joint_angles'][standard_trial_idx, task_idx]
        
        # Skip if data is empty
        if emg_data.size == 0 or angle_data.size == 0:
            print(f"  Task {task_idx+1}: No data")
            continue
        
        # Preprocess data
        emg_norm = preprocess_emg(emg_data)
        angle_norm, angle_orig = preprocess_angles(angle_data)
        
        # Create windows for prediction
        emg_windows = create_windows(emg_norm)
        
        # Make predictions using the model
        if emg_windows.shape[0] > 0:
            predictions = model.predict(emg_windows)
            
            # Store data for this task
            standard_emg[task_idx] = emg_norm
            standard_angles[task_idx] = angle_norm
            standard_predictions[task_idx] = predictions
            
            print(f"  Task {task_idx+1} ({task_names[task_idx]}): EMG shape {emg_data.shape}, Angle shape {angle_data.shape}, Predictions shape {predictions.shape}")
        else:
            print(f"  Task {task_idx+1} ({task_names[task_idx]}): Not enough data for windowing")
    
    # Compare each trial against the standard
    results = {}
    
    for trial_idx in range(mat_data['dsfilt_emg'].shape[0]):
        # Skip the standard trial
        if trial_idx == standard_trial_idx:
            continue
        
        print(f"\nComparing Trial {trial_idx+1} against standard (Trial {standard_trial_idx+1})...")
        trial_results = {}
        
        for task_idx in range(mat_data['dsfilt_emg'].shape[1]):
            # Skip if standard data is not available for this task
            if task_idx not in standard_emg:
                continue
            
            # Get EMG and angle data for this trial
            emg_data = mat_data['dsfilt_emg'][trial_idx, task_idx]
            angle_data = mat_data['joint_angles'][trial_idx, task_idx]
            
            # Skip if data is empty
            if emg_data.size == 0 or angle_data.size == 0:
                print(f"  Task {task_idx+1}: No data")
                continue
            
            # Preprocess data
            emg_norm = preprocess_emg(emg_data)
            angle_norm, angle_orig = preprocess_angles(angle_data)
            
            # Create windows for prediction
            emg_windows = create_windows(emg_norm)
            
            # Make predictions using the model
            if emg_windows.shape[0] > 0:
                predictions = model.predict(emg_windows)
                
                # Calculate metrics for each finger group
                task_metrics = {}
                
                for finger_name, start_idx, end_idx in finger_groups:
                    # Calculate metrics between this trial's predictions and standard trial's predictions
                    # Use the minimum length to ensure comparison is valid
                    min_length = min(predictions.shape[0], standard_predictions[task_idx].shape[0])
                    
                    # Extract predictions for this finger group
                    trial_pred = predictions[:min_length, start_idx:end_idx]
                    standard_pred = standard_predictions[task_idx][:min_length, start_idx:end_idx]
                    
                    # Calculate metrics
                    mae = mean_absolute_error(standard_pred, trial_pred)
                    mse = mean_squared_error(standard_pred, trial_pred)
                    rmse = np.sqrt(mse)
                    
                    # Calculate correlation
                    corr = np.zeros(end_idx - start_idx)
                    for i in range(end_idx - start_idx):
                        corr[i] = np.corrcoef(standard_pred[:, i], trial_pred[:, i])[0, 1]
                    
                    # Store metrics
                    task_metrics[finger_name] = {
                        'mae': mae,
                        'rmse': rmse,
                        'correlation': corr.mean(),
                        'trial_pred': trial_pred,
                        'standard_pred': standard_pred
                    }
                
                # Store results for this task
                trial_results[task_idx] = {
                    'metrics': task_metrics,
                    'emg': emg_norm,
                    'angles': angle_norm,
                    'predictions': predictions
                }
                
                print(f"  Task {task_idx+1} ({task_names[task_idx]}): Processed")
            else:
                print(f"  Task {task_idx+1} ({task_names[task_idx]}): Not enough data for windowing")
        
        # Store results for this trial
        results[trial_idx] = trial_results
    
    return standard_emg, standard_angles, standard_predictions, results

# Function to visualize trial comparisons
def visualize_trial_comparisons(standard_emg, standard_angles, standard_predictions, results):
    """
    Create visualizations for trial comparisons
    """
    # Create a summary heatmap of metrics across all trials and tasks
    for metric_name in ['mae', 'correlation']:
        plt.figure(figsize=(15, 10))
        
        # Create a matrix for the heatmap
        trials = sorted(results.keys())
        tasks = sorted(list(set([task for trial in results.values() for task in trial.keys()])))
        
        # Initialize metric matrices for each finger
        for finger_name, _, _ in finger_groups:
            metric_matrix = np.zeros((len(trials), len(tasks)))
            metric_matrix.fill(np.nan)  # Fill with NaN for missing data
            
            # Fill the matrix with metrics
            for i, trial_idx in enumerate(trials):
                for j, task_idx in enumerate(tasks):
                    if task_idx in results[trial_idx] and finger_name in results[trial_idx][task_idx]['metrics']:
                        metric_matrix[i, j] = results[trial_idx][task_idx]['metrics'][finger_name][metric_name]
            
            # Create heatmap
            plt.subplot(len(finger_groups), 1, finger_groups.index((finger_name, *[fg[1:] for fg in finger_groups if fg[0] == finger_name][0])) + 1)
            im = plt.imshow(metric_matrix, cmap='coolwarm', aspect='auto')
            plt.colorbar(im, label=f'{metric_name.upper()}')
            plt.title(f'{finger_name} Finger - {metric_name.upper()} Comparison')
            plt.xlabel('Task')
            plt.ylabel('Trial')
            plt.xticks(range(len(tasks)), [f"{task_idx+1}: {task_names[task_idx]}" for task_idx in tasks], rotation=45, ha='right')
            plt.yticks(range(len(trials)), [f"Trial {trial_idx+1}" for trial_idx in trials])
        
        plt.tight_layout()
        plt.savefig(f'/home/ubuntu/emg_project/visualizations/trial_comparison/{metric_name}_heatmap.png')
        plt.close()
    
    # Create detailed comparison plots for each trial and task
    for trial_idx in results:
        for task_idx in results[trial_idx]:
            plt.figure(figsize=(15, 12))
            
            for i, (finger_name, start_idx, end_idx) in enumerate(finger_groups):
                if finger_name in results[trial_idx][task_idx]['metrics']:
                    plt.subplot(len(finger_groups), 1, i+1)
                    
                    # Get predictions for this finger
                    trial_pred = results[trial_idx][task_idx]['metrics'][finger_name]['trial_pred']
                    standard_pred = results[trial_idx][task_idx]['metrics'][finger_name]['standard_pred']
                    
                    # Plot the first joint angle for this finger
                    plt.plot(standard_pred[:100, 0], 'b-', label=f'Standard (Trial 1)')
                    plt.plot(trial_pred[:100, 0], 'r-', label=f'Trial {trial_idx+1}')
                    
                    # Add metrics to the title
                    mae = results[trial_idx][task_idx]['metrics'][finger_name]['mae']
                    corr = results[trial_idx][task_idx]['metrics'][finger_name]['correlation']
                    plt.title(f'{finger_name} Finger - Task {task_idx+1} ({task_names[task_idx]}) - MAE: {mae:.4f}, Corr: {corr:.4f}')
                    
                    plt.ylabel('Normalized Angle')
                    plt.xlabel('Sample')
                    plt.legend()
                    plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'/home/ubuntu/emg_project/visualizations/trial_comparison/trial{trial_idx+1}_task{task_idx+1}_comparison.png')
            plt.close()
    
    # Create EMG signal comparison plots
    for trial_idx in results:
        for task_idx in results[trial_idx]:
            plt.figure(figsize=(15, 10))
            
            # Get EMG data
            trial_emg = results[trial_idx][task_idx]['emg']
            standard_emg_data = standard_emg[task_idx]
            
            # Plot EMG channels
            n_channels = min(8, trial_emg.shape[1])
            for i in range(n_channels):
                plt.subplot(n_channels, 1, i+1)
                
                # Use a smaller window for visualization
                window = 500
                plt.plot(standard_emg_data[:window, i], 'b-', label=f'Standard (Trial 1)')
                plt.plot(trial_emg[:window, i], 'r-', label=f'Trial {trial_idx+1}')
                
                # Calculate correlation
                corr = np.corrcoef(standard_emg_data[:min(len(standard_emg_data), len(trial_emg)), i], 
                                  trial_emg[:min(len(standard_emg_data), len(trial_emg)), i])[0, 1]
                
                plt.title(f'EMG Channel {i+1} - Correlation: {corr:.4f}')
                plt.grid(True)
                if i == 0:
                    plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'/home/ubuntu/emg_project/visualizations/trial_comparison/emg_trial{trial_idx+1}_task{task_idx+1}_comparison.png')
            plt.close()
    
    # Create a summary plot of isotonic, isometric, and recovery process evaluation
    # We'll use the correlation metric as an indicator of similarity between trials
    plt.figure(figsize=(15, 8))
    
    # Calculate average correlation across all fingers for each trial and task
    trials = sorted(results.keys())
    tasks = sorted(list(set([task for trial in results.values() for task in trial.keys()])))
    
    avg_correlations = np.zeros((len(trials), len(tasks)))
    avg_correlations.fill(np.nan)  # Fill with NaN for missing data
    
    for i, trial_idx in enumerate(trials):
        for j, task_idx in enumerate(tasks):
            if task_idx in results[trial_idx]:
                # Calculate average correlation across all fingers
                finger_corrs = []
                for finger_name, _, _ in finger_groups:
                    if finger_name in results[trial_idx][task_idx]['metrics']:
                        finger_corrs.append(results[trial_idx][task_idx]['metrics'][finger_name]['correlation'])
                
                if finger_corrs:
                    avg_correlations[i, j] = np.mean(finger_corrs)
    
    # Plot average correlations
    plt.bar(range(len(tasks)), np.nanmean(avg_correlations, axis=0), color='blue', alpha=0.7)
    plt.xlabel('Task')
    plt.ylabel('Average Correlation with Standard Trial')
    plt.title('Evaluation of Isotonic, Isometric, and Recovery Process')
    plt.xticks(range(len(tasks)), [f"{task_idx+1}: {task_names[task_idx]}" for task_idx in tasks], rotation=45, ha='right')
    plt.grid(True, axis='y')
    
    # Add a horizontal line at correlation = 0.7 as a reference
    plt.axhline(y=0.7, color='r', linestyle='--', label='Reference Threshold (0.7)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/emg_project/visualizations/trial_comparison/isotonic_isometric_recovery_evaluation.png')
    plt.close()
    
    # Create a radar chart for finger performance across trials
    plt.figure(figsize=(12, 10))
    
    # Calculate average correlation for each finger across all tasks
    finger_avg_corr = {}
    for finger_name, _, _ in finger_groups:
        finger_corrs = []
        for trial_idx in trials:
            for task_idx in results[trial_idx]:
                if finger_name in results[trial_idx][task_idx]['metrics']:
                    finger_corrs.append(results[trial_idx][task_idx]['metrics'][finger_name]['correlation'])
        
        if finger_corrs:
            finger_avg_corr[finger_name] = np.mean(finger_corrs)
    
    # Create radar chart
    angles = np.linspace(0, 2*np.pi, len(finger_groups), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    finger_names = [fg[0] for fg in finger_grou
(Content truncated due to size limit. Use line ranges to read in chunks)