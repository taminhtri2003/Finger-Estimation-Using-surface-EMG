import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import scipy.io as sio
import os

# Create output directories
os.makedirs('/home/ubuntu/emg_project/models/generative', exist_ok=True)
os.makedirs('/home/ubuntu/emg_project/visualizations/generative', exist_ok=True)

# Load the .mat file
data_path = '/home/ubuntu/emg_project/data/s4_full.mat'
mat_data = sio.loadmat(data_path)

# Define finger groups for visualization
finger_groups = [
    ('Thumb', 0, 2),
    ('Index', 2, 5),
    ('Middle', 5, 8),
    ('Ring', 8, 11),
    ('Little', 11, 14)
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
    scalers = []
    
    # Normalize each angle separately
    for i in range(n_angles):
        scaler = MinMaxScaler()
        normalized_data[:, i] = scaler.fit_transform(angle_data[:, i].reshape(-1, 1)).flatten()
        scalers.append(scaler)
    
    return normalized_data, angle_data, scalers  # Return normalized, original data, and scalers

# Function to prepare data for the generative model
def prepare_data_for_generative_model(standard_trial_idx=0):
    """
    Prepare data for the generative model using the standard trial
    """
    # Process standard trial (trial 1) for each task
    X_train = []
    y_train = []
    angle_scalers = []
    
    print(f"Processing standard trial (Trial {standard_trial_idx+1}) for generative model...")
    
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
        angle_norm, _, task_scalers = preprocess_angles(angle_data)
        
        # Store scalers for later use
        angle_scalers.append(task_scalers)
        
        # Create sequences for training
        seq_length = 50  # Same as the window size used in the prediction model
        
        for i in range(len(emg_norm) - seq_length):
            # Input: target angles
            # Output: corresponding EMG signals
            X_train.append(angle_norm[i:i+seq_length])
            y_train.append(emg_norm[i:i+seq_length])
        
        print(f"  Task {task_idx+1}: Added {len(emg_norm) - seq_length} sequences")
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"Total training data: {X_train.shape}, {y_train.shape}")
    
    return X_train, y_train, angle_scalers

# Function to create the generative model
def create_generative_model(input_shape, output_shape):
    """
    Create a generative model that generates EMG signals from target joint angles
    """
    # Input layer for target joint angles
    inputs = Input(shape=input_shape)
    
    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    
    # Dense layers for EMG signal generation
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer with sigmoid activation to ensure [0, 1] range
    outputs = Dense(output_shape, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Function to train the generative model
def train_generative_model(X_train, y_train):
    """
    Train the generative model
    """
    # Create the model
    input_shape = X_train.shape[1:]  # (seq_length, n_angles)
    output_shape = y_train.shape[2]  # n_emg_channels
    
    model = create_generative_model(input_shape, output_shape)
    
    # Print model summary
    model.summary()
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='/home/ubuntu/emg_project/models/generative/generative_model.keras',
            monitor='val_loss',
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Split data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train the model
    history = model.fit(
        X_train_split, y_train_split,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the model
    model.save('/home/ubuntu/emg_project/models/generative/generative_model.keras')
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Generative Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True)
    
    # Plot training & validation mean absolute error
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Generative Model Mean Absolute Error')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/emg_project/visualizations/generative/training_history.png')
    plt.close()
    
    return model

# Function to evaluate the generative model
def evaluate_generative_model(model, X_test, y_test):
    """
    Evaluate the generative model
    """
    # Evaluate the model
    loss, mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    # Generate EMG signals from target angles
    y_pred = model.predict(X_test)
    
    # Plot generated vs actual EMG signals
    plt.figure(figsize=(15, 10))
    
    # Plot for each EMG channel
    n_channels = min(8, y_test.shape[2])
    for i in range(n_channels):
        plt.subplot(n_channels, 1, i+1)
        
        # Plot actual and generated EMG for a sample sequence
        sample_idx = 0
        plt.plot(y_test[sample_idx, :, i], 'b-', label='Actual EMG')
        plt.plot(y_pred[sample_idx, :, i], 'r-', label='Generated EMG')
        
        plt.title(f'EMG Channel {i+1}')
        plt.ylabel('Normalized Amplitude')
        plt.grid(True)
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/emg_project/visualizations/generative/generated_vs_actual_emg.png')
    plt.close()
    
    return y_pred

# Function to generate EMG signals for target angles
def generate_emg_for_target_angles(model, target_angles, seq_length=50):
    """
    Generate EMG signals for target angles
    """
    # Ensure target_angles has the right shape
    if len(target_angles.shape) == 2:
        # Add batch dimension if needed
        target_angles = np.expand_dims(target_angles, axis=0)
    
    # Generate EMG signals
    generated_emg = model.predict(target_angles)
    
    return generated_emg

# Function to create target angle trajectories
def create_target_angle_trajectories(n_angles=14, seq_length=50, n_trajectories=5):
    """
    Create synthetic target angle trajectories
    """
    trajectories = []
    
    for i in range(n_trajectories):
        # Create a smooth trajectory for each angle
        trajectory = np.zeros((seq_length, n_angles))
        
        for j in range(n_angles):
            # Create a sinusoidal pattern with random frequency and phase
            freq = np.random.uniform(0.1, 0.5)
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.3, 0.7)
            offset = np.random.uniform(0.2, 0.5)
            
            # Generate the trajectory
            t = np.linspace(0, 2*np.pi, seq_length)
            trajectory[:, j] = offset + amplitude * np.sin(freq * t + phase)
        
        trajectories.append(trajectory)
    
    return np.array(trajectories)

# Function to visualize generated EMG signals for target angles
def visualize_generated_emg(target_angles, generated_emg, finger_groups):
    """
    Visualize generated EMG signals for target angles
    """
    n_trajectories = target_angles.shape[0]
    
    for traj_idx in range(n_trajectories):
        plt.figure(figsize=(15, 12))
        
        # Plot target angles for each finger
        plt.subplot(2, 1, 1)
        for i, (finger_name, start_idx, end_idx) in enumerate(finger_groups):
            # Plot the first joint angle for each finger
            joint_idx = start_idx
            plt.plot(target_angles[traj_idx, :, joint_idx], label=f'{finger_name}')
        
        plt.title('Target Joint Angles')
        plt.ylabel('Normalized Angle')
        plt.xlabel('Time Step')
        plt.legend()
        plt.grid(True)
        
        # Plot generated EMG signals
        plt.subplot(2, 1, 2)
        n_channels = min(8, generated_emg.shape[2])
        for i in range(n_channels):
            plt.plot(generated_emg[traj_idx, :, i], label=f'EMG {i+1}')
        
        plt.title('Generated EMG Signals')
        plt.ylabel('Normalized Amplitude')
        plt.xlabel('Time Step')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'/home/ubuntu/emg_project/visualizations/generative/trajectory_{traj_idx+1}.png')
        plt.close()

# Function to evaluate the full pipeline (generative model + prediction model)
def evaluate_full_pipeline(generative_model, prediction_model, target_angles):
    """
    Evaluate the full pipeline:
    1. Generate EMG signals from target angles using the generative model
    2. Predict joint angles from generated EMG using the prediction model
    3. Compare predicted angles with target angles
    """
    # Generate EMG signals
    generated_emg = generative_model.predict(target_angles)
    
    # Predict joint angles from generated EMG
    predicted_angles = prediction_model.predict(generated_emg)
    
    # Plot target vs predicted angles
    plt.figure(figsize=(15, 12))
    
    for i, (finger_name, start_idx, end_idx) in enumerate(finger_groups):
        plt.subplot(len(finger_groups), 1, i+1)
        
        # Plot target and predicted angles for the first joint of each finger
        joint_idx = start_idx
        
        # Use the first trajectory
        traj_idx = 0
        plt.plot(target_angles[traj_idx, :, joint_idx], 'b-', label='Target')
        plt.plot(predicted_angles[traj_idx, :, joint_idx], 'r-', label='Predicted')
        
        plt.title(f'{finger_name} Finger - Joint Angle')
        plt.ylabel('Normalized Angle')
        plt.xlabel('Time Step')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/emg_project/visualizations/generative/full_pipeline_evaluation.png')
    plt.close()
    
    # Calculate metrics
    mae = np.mean(np.abs(target_angles - predicted_angles))
    mse = np.mean(np.square(target_angles - predicted_angles))
    rmse = np.sqrt(mse)
    
    print(f"Full Pipeline Evaluation:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    return mae, rmse

# Main execution
print("Preparing data for generative model...")
X_train, y_train, angle_scalers = prepare_data_for_generative_model()

print("\nTraining generative model...")
generative_model = train_generative_model(X_train, y_train)

print("\nEvaluating generative model...")
# Use a portion of the training data for evaluation
X_test, y_test = X_train[-100:], y_train[-100:]
generated_emg = evaluate_generative_model(generative_model, X_test, y_test)

print("\nCreating target angle trajectories...")
target_trajectories = create_target_angle_trajectories()

print("\nGenerating EMG signals for target trajectories...")
generated_emg_for_targets = generate_emg_for_target_angles(generative_model, target_trajectories)

print("\nVisualizing generated EMG signals...")
visualize_generated_emg(target_trajectories, generated_emg_for_targets, finger_groups)

print("\nEvaluating full pipeline...")
try:
    # Load the prediction model
    prediction_model = load_model('/home/ubuntu/emg_project/models/multi_head_attention_model.keras')
    
    # Evaluate the full pipeline
    mae, rmse = evaluate_full_pipeline(generative_model, prediction_model, target_trajectories)
except Exception as e:
    print(f"Error evaluating full pipeline: {e}")

print("\nGenerative AI implementation completed. Results saved to /home/ubuntu/emg_project/visualizations/generative/")
