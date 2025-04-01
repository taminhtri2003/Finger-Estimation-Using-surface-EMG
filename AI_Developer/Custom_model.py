# Description: This script contains the code for the custom model used in the EMG to joint angle estimation pipeline.
from function_custom import load_data, process_data_for_model, create_custom_emg_model, visualize_joint_angles, visualize_model_architecture, visualize_feature_importance, visualize_hand_kinematics_3d
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping

def main(file_path):
    """Main function to run the EMG to joint angle estimation pipeline"""
    # Load data
    data = load_data(file_path)
    if data is None:
        return
    
    # Extract variables
    dsfilt_emg = data.get('dsfilt_emg')
    finger_kinematics = data.get('finger_kinematics')
    joint_angles = data.get('joint_angles')
    
    if dsfilt_emg is None or joint_angles is None:
        print("Required data not found in file.")
        return
    
    # Create a figure to visualize all available trials and tasks
    n_trials = dsfilt_emg.shape[0]
    n_tasks = dsfilt_emg.shape[1]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    for trial in range(n_trials):
        for task in range(n_tasks):
            # Plot a point for each available trial and task
            ax.scatter(task, trial, s=100, marker='o')
            # Add sample count as text
            sample_count = dsfilt_emg[trial, task].shape[0] if dsfilt_emg[trial, task].size > 0 else 0
            if sample_count > 0:
                ax.text(task, trial, str(sample_count), ha='center', va='center')
    
    # Set axis labels and title
    ax.set_xlabel('Task Index')
    ax.set_ylabel('Trial Index')
    ax.set_title('Available Data Samples per Trial and Task')
    
    # Customize x-axis tick labels with task descriptions
    task_descriptions = [
        "Thumb Flex/Ext", "Index Flex/Ext", "Middle Flex/Ext", 
        "Ring Flex/Ext", "Little Flex/Ext", "All Fingers Flex/Ext", 
        "Random Movements"
    ]
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(task_descriptions, rotation=45, ha='right')
    
    ax.set_yticks(range(n_trials))
    ax.set_yticklabels([f"Trial {i+1}" for i in range(n_trials)])
    
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('available_data_visualization.png')
    plt.close(fig)
    
    # Example: Process the first trial of the first task
    trial_idx = 0  # First trial
    task_idx = 0   # First task (thumb flexion/extension)
    
    # Get EMG data for the selected trial and task
    emg_data = dsfilt_emg[trial_idx, task_idx]
    
    # Get joint angle data for the selected trial and task
    joint_angle_data = joint_angles[trial_idx, task_idx]
    
    # Get kinematics data for visualization
    kinematics_data = finger_kinematics[trial_idx, task_idx]
    
    # Print data information
    print(f"EMG data shape: {emg_data.shape}")
    print(f"Joint angle data shape: {joint_angle_data.shape}")
    print(f"Kinematics data shape: {kinematics_data.shape}")
    
    # Visualize sample EMG data for each channel
    plt.figure(figsize=(15, 10))
    n_channels = emg_data.shape[1]
    time_segment = emg_data[:500, :]  # First 500 samples
    
    # Define muscle names
    muscle_names = ['APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR']
    
    for i in range(n_channels):
        plt.subplot(n_channels, 1, i+1)
        plt.plot(time_segment[:, i])
        plt.title(f'EMG Channel: {muscle_names[i]}')
        plt.ylabel('Amplitude')
        if i == n_channels-1:
            plt.xlabel('Time (samples)')
    
    plt.tight_layout()
    plt.savefig('emg_channels_visualization.png')
    plt.close()
    
    # Visualize joint angles
    plt.figure(figsize=(15, 10))
    n_joints = joint_angle_data.shape[1]
    
    joint_names = [
        "Thumb 1", "Thumb 2", 
        "Index 1", "Index 2", "Index 3",
        "Middle 1", "Middle 2", "Middle 3",
        "Ring 1", "Ring 2", "Ring 3",
        "Little 1", "Little 2", "Little 3"
    ]
    
    for i in range(n_joints):
        plt.subplot(n_joints//2, 2, i+1)
        plt.plot(joint_angle_data[:500, i])
        plt.title(f'Joint Angle: {joint_names[i]}')
        plt.ylabel('Angle (degrees)')
        plt.xlabel('Time (samples)')
    
    plt.tight_layout()
    plt.savefig('joint_angles_visualization.png')
    plt.close()
    
    # Process data for model
    X_channels, y, n_features_per_channel = process_data_for_model(emg_data, joint_angle_data)
    
    # Print processed data information
    print(f"Number of EMG channels: {len(X_channels)}")
    print(f"Features per channel: {X_channels[0].shape[1]}")
    print(f"Number of joint angles: {y.shape[1]}")
    
    # Split data into training and test sets
    X_train_channels = []
    X_test_channels = []
    
    indices = np.arange(X_channels[0].shape[0])
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Split each channel's features
    for X_channel in X_channels:
        X_train_channels.append(X_channel[train_indices])
        X_test_channels.append(X_channel[test_indices])
    
    # Split targets
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Normalize data
    X_train_norm_channels = []
    X_test_norm_channels = []
    scalers = []
    
    for X_train, X_test in zip(X_train_channels, X_test_channels):
        # Fit scaler on training data
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_test_norm = scaler.transform(X_test)
        
        X_train_norm_channels.append(X_train_norm)
        X_test_norm_channels.append(X_test_norm)
        scalers.append(scaler)
    
    # Normalize targets
    y_scaler = StandardScaler()
    y_train_norm = y_scaler.fit_transform(y_train)
    y_test_norm = y_scaler.transform(y_test)
    
    # Create model
    n_emg_channels = len(X_channels)
    n_features_per_channel = X_channels[0].shape[1]
    n_outputs = y.shape[1]
    
    model = create_custom_emg_model(n_emg_channels, n_features_per_channel, n_outputs)
    
    # Print model summary
    model.summary()
    
    # Train model
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        X_train_norm_channels, 
        y_train_norm,
        validation_data=(X_test_norm_channels, y_test_norm),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    y_pred_norm = model.predict(X_test_norm_channels)
    y_pred = y_scaler.inverse_transform(y_pred_norm)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Joint angle names for visualization
    joint_names = [
        "Thumb 1", "Thumb 2", 
        "Index 1", "Index 2", "Index 3",
        "Middle 1", "Middle 2", "Middle 3",
        "Ring 1", "Ring 2", "Ring 3",
        "Little 1", "Little 2", "Little 3"
    ]
    
    # Visualize training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Visualize joint angle predictions
    fig = visualize_joint_angles(y_test, y_pred, joint_names)
    plt.savefig('joint_angle_predictions.png')
    plt.close(fig)
    
    # Visualize feature importance for each channel
    feature_names = [
        'MAV', 'RMS', 'WL', 'ZC', 'SSC', 'VAR', 'IEMG',  # Time domain
        'Mean Freq', 'Median Freq', 'Band 0-50', 'Band 50-100', 'Band 100-150', 'Band 150-200',  # Frequency domain
        'Wavelet1', 'Wavelet2', 'Wavelet3', 'Wavelet4',  # Just showing first few wavelet features
        'Wavelet5', 'Wavelet6', 'Wavelet7', 'Wavelet8'
    ]
    
    # Create a simplified model for feature importance visualization
    simplified_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(n_emg_channels * n_features_per_channel,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(n_outputs, activation='linear')
    ])
    
    # Combine all features for simplified model
    X_train_combined = np.concatenate(X_train_norm_channels, axis=1)
    simplified_model.compile(optimizer='adam', loss='mse')
    simplified_model.fit(X_train_combined, y_train_norm, epochs=10, verbose=0)
    
    visualize_model_architecture(simplified_model)

    # Visualize feature importance
    importance_fig = visualize_feature_importance(simplified_model, X_train_combined, y_train_norm, feature_names, n_emg_channels)
    if importance_fig:
        plt.savefig('feature_importance.png')
        plt.close(importance_fig)
    
    # Visualize hand kinematics in 3D
    fig = visualize_hand_kinematics_3d(kinematics_data, joint_angle_data, time_step=100)
    plt.savefig('hand_kinematics_3d.png')
    plt.close(fig)
    

    
    print("Analysis complete. Visualizations saved.")
    
    return model, history, y_test, y_pred, joint_names

# Example usage
if __name__ == "__main__":
    file_path = "s4_full.mat"  # Replace with actual file path
    main(file_path)