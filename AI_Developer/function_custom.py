import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import seaborn as sns
from scipy.signal import butter, filtfilt
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

# Load the .mat file
def load_data(file_path):
    """Load the MATLAB data file"""
    try:
        data = scipy.io.loadmat(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Feature extraction functions for EMG signals
def extract_time_domain_features(emg_channel):
    """Extract time domain features from an EMG channel"""
    # Mean absolute value
    mav = np.mean(np.abs(emg_channel))
    
    # Root mean square
    rms = np.sqrt(np.mean(np.square(emg_channel)))
    
    # Waveform length
    wl = np.sum(np.abs(np.diff(emg_channel)))
    
    # Zero crossings (with threshold to avoid noise)
    threshold = 0.01 * np.std(emg_channel)
    zero_crossings = np.sum((emg_channel[:-1] * emg_channel[1:]) < 0)
    
    # Slope sign changes
    ssc = np.sum(((emg_channel[1:-1] - emg_channel[:-2]) * (emg_channel[1:-1] - emg_channel[2:])) > 0)
    
    # Variance
    var = np.var(emg_channel)
    
    # Integrated EMG
    iemg = np.sum(np.abs(emg_channel))
    
    return np.array([mav, rms, wl, zero_crossings, ssc, var, iemg])

def extract_frequency_domain_features(emg_channel, fs=2000):
    """Extract frequency domain features from an EMG channel"""
    # Fast Fourier Transform
    n = len(emg_channel)
    fft_result = np.abs(np.fft.fft(emg_channel)) / n
    freqs = np.fft.fftfreq(n, 1/fs)
    
    # Only look at positive frequencies up to Nyquist frequency
    idx = np.where(freqs >= 0)[0]
    fft_result = fft_result[idx]
    freqs = freqs[idx]
    
    # Mean frequency
    mean_freq = np.sum(freqs * fft_result) / np.sum(fft_result) if np.sum(fft_result) > 0 else 0
    
    # Median frequency
    cumsum = np.cumsum(fft_result)
    median_freq = freqs[np.where(cumsum >= cumsum[-1] / 2)[0][0]] if len(fft_result) > 0 and cumsum[-1] > 0 else 0
    
    # Frequency band powers
    # Define frequency bands
    bands = [(0, 50), (50, 100), (100, 150), (150, 200)]
    powers = []
    
    for low, high in bands:
        band_idx = np.where((freqs >= low) & (freqs <= high))[0]
        band_power = np.sum(fft_result[band_idx])
        powers.append(band_power)
    
    return np.array([mean_freq, median_freq] + powers)

def extract_wavelet_features(emg_channel, wavelet='db2', level=4):
    """Extract wavelet features from an EMG channel using PyWavelets"""
    try:
        import pywt
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(emg_channel, wavelet, level=level)
        
        # Extract features from each decomposition level
        features = []
        for coef in coeffs:
            # Calculate statistical features from wavelet coefficients
            features.extend([
                np.mean(np.abs(coef)),  # Mean absolute value
                np.std(coef),  # Standard deviation
                stats.kurtosis(coef),  # Kurtosis
                stats.skew(coef)  # Skewness
            ])
        
        return np.array(features)
    except ImportError:
        print("PyWavelets library not available. Skipping wavelet features.")
        # Return zeros if PyWavelets is not available
        return np.zeros(4 * (level + 1))

def normalize_windows(windows, scaler=None, fit=False):
    """Normalize data windows using StandardScaler"""
    # Reshape to 2D for scaling
    original_shape = windows.shape
    windows_2d = windows.reshape(-1, windows.shape[-1])
    
    if fit:
        scaler = StandardScaler()
        windows_scaled = scaler.fit_transform(windows_2d)
        return windows_scaled.reshape(original_shape), scaler
    else:
        windows_scaled = scaler.transform(windows_2d)
        return windows_scaled.reshape(original_shape)

def create_sliding_windows(data, window_size, step_size):
    """Create sliding windows from sequential data"""
    n_samples = data.shape[0]
    n_windows = (n_samples - window_size) // step_size + 1
    
    windows = np.zeros((n_windows, window_size, data.shape[1]))
    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        windows[i] = data[start_idx:end_idx]
    
    return windows

def extract_features_from_all_channels(emg_data, window_size=200, step_size=50):
    """Extract features from all EMG channels using sliding windows"""
    # Create sliding windows
    emg_windows = create_sliding_windows(emg_data, window_size, step_size)
    
    # Number of windows and channels
    n_windows = emg_windows.shape[0]
    n_channels = emg_windows.shape[2]
    
    # Initialize feature array
    # For each channel: 7 time domain + 6 frequency domain + 20 wavelet features = 33 features per channel
    n_features_per_channel = 33
    features = np.zeros((n_windows, n_channels * n_features_per_channel))
    
    # Extract features from each window for each channel
    for i in range(n_windows):
        feature_idx = 0
        for j in range(n_channels):
            # Extract channel data from window
            channel_data = emg_windows[i, :, j]
            
            # Extract features
            time_features = extract_time_domain_features(channel_data)
            freq_features = extract_frequency_domain_features(channel_data)
            wavelet_features = extract_wavelet_features(channel_data)
            
            # Combine features
            channel_features = np.concatenate([time_features, freq_features, wavelet_features])
            
            # Store in feature array
            features[i, feature_idx:feature_idx + len(channel_features)] = channel_features
            feature_idx += len(channel_features)
    
    return features

def create_custom_emg_model(n_emg_channels, n_features_per_channel, n_outputs):
    """Create a custom neural network model for EMG feature processing"""
    # Create separate inputs and processing branches for each EMG channel
    channel_inputs = []
    channel_outputs = []
    
    for i in range(n_emg_channels):
        # Input for channel features
        channel_input = Input(shape=(n_features_per_channel,), name=f'emg_channel_{i+1}_input')
        channel_inputs.append(channel_input)
        
        # First layer - muscle-specific processing
        x = Dense(64, activation='relu')(channel_input)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Second layer - muscle-specific processing
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Third layer - additional muscle-specific processing
        x = Dense(16, activation='relu')(x)
        
        channel_outputs.append(x)
    
    # Concatenate all channel outputs
    if len(channel_outputs) > 1:
        combined = Concatenate()(channel_outputs)
    else:
        combined = channel_outputs[0]
    
    # First shared layer - combine channel features
    x = Dense(256, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Second shared layer
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Third shared layer
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Final shared layer before output
    x = Dense(32, activation='relu')(x)
    
    # Output layer for joint angles
    outputs = Dense(n_outputs, activation='linear', name='joint_angles')(x)
    
    # Create and compile model
    model = Model(inputs=channel_inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='mse', 
        metrics=['mae']
    )
    
    return model

def process_data_for_model(emg_data, joint_angle_data, window_size=200, step_size=50):
    """Process the data for the model by extracting features and creating corresponding targets"""
    # Create sliding windows for joint angles (targets)
    joint_angle_windows = create_sliding_windows(joint_angle_data, window_size, step_size)
    
    # Take the last frame of each window as the target (prediction target)
    y = joint_angle_windows[:, -1, :]
    
    # Extract features from EMG channels
    emg_features = extract_features_from_all_channels(emg_data, window_size, step_size)
    
    # Split features by channel
    n_channels = emg_data.shape[1]
    n_features_per_channel = emg_features.shape[1] // n_channels
    
    X_channels = []
    for i in range(n_channels):
        start_idx = i * n_features_per_channel
        end_idx = start_idx + n_features_per_channel
        channel_features = emg_features[:, start_idx:end_idx]
        X_channels.append(channel_features)
    
    return X_channels, y, n_features_per_channel

def visualize_joint_angles(true_angles, pred_angles, joint_names):
    """Visualize the true vs predicted joint angles"""
    n_joints = true_angles.shape[1]
    
    fig, axes = plt.subplots(n_joints, 1, figsize=(15, 3*n_joints))
    
    if n_joints == 1:
        axes = [axes]
    
    time = np.arange(true_angles.shape[0])
    
    for i in range(n_joints):
        axes[i].plot(time, true_angles[:, i], 'b-', label='True')
        axes[i].plot(time, pred_angles[:, i], 'r-', label='Predicted')
        axes[i].set_title(f'Joint Angle: {joint_names[i]}')
        axes[i].set_ylabel('Angle (degrees)')
        axes[i].legend()
    
    axes[-1].set_xlabel('Time (samples)')
    plt.tight_layout()
    return fig

def visualize_feature_importance(model, X_test_combined, y_test, feature_names, n_channels):
    """Visualize the importance of each feature for each channel using permutation importance"""
    import tensorflow as tf
    from sklearn.inspection import permutation_importance

    # Define a function to get predictions from the model
    def get_predictions(X):
        return model.predict(X)

    # Calculate base performance
    evaluation_result = model.evaluate(X_test_combined, y_test, verbose=0)
    base_score = evaluation_result[0] if isinstance(evaluation_result, list) else evaluation_result

    # Initialize importance scores
    n_features = len(feature_names)
    importance_scores = np.zeros((n_channels, n_features))

    # Perform permutation importance for each feature in each channel
    for channel_idx in range(n_channels):
        # Get feature indices for this channel
        start_idx = channel_idx * n_features
        end_idx = start_idx + n_features

        # Permute each feature and calculate importance
        for feature_idx in range(n_features):
            # Create a copy of the test data
            X_permuted = X_test_combined.copy()

            # Permute the feature
            np.random.shuffle(X_permuted[:, start_idx + feature_idx])

            # Calculate score with permuted feature
            permuted_score = model.evaluate(X_permuted, y_test, verbose=0)
            permuted_score = permuted_score[0] if isinstance(permuted_score, list) else permuted_score

            # Importance is the increase in error after permutation
            importance_scores[channel_idx, feature_idx] = permuted_score - base_score

    # Normalize importance scores
    importance_scores = (importance_scores - np.min(importance_scores)) / (np.max(importance_scores) - np.min(importance_scores) + 1e-10)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(importance_scores, cmap='viridis')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Normalized Feature Importance', rotation=-90, va="bottom")

    # Set labels
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(n_channels))
    ax.set_xticklabels(feature_names, rotation=90)

    # Map channel indices to muscle names
    muscle_names = ['APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR']
    ax.set_yticklabels(muscle_names[:n_channels])

    ax.set_title('Feature Importance by EMG Channel')
    ax.set_xlabel('Features')
    ax.set_ylabel('EMG Channels')

    plt.tight_layout()
    return fig

def visualize_hand_kinematics_3d(joint_positions, joint_angles=None, time_step=0):
    """Visualize hand kinematics in 3D at a specific time step"""
    # Extract positions for the given time step
    positions = joint_positions[time_step]
    
    # Reshape to (23, 3) - 23 markers with x, y, z coordinates
    positions = positions.reshape(23, 3)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all marker positions
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', marker='o')
    
    # Define connections between markers to create skeleton
    connections = [
        # Thumb
        (20, 17), (17, 18), (18, 19),
        # Index
        (20, 1), (1, 5), (5, 6), (6, 7),
        # Middle
        (20, 2), (2, 8), (8, 9), (9, 10),
        # Ring
        (20, 3), (3, 11), (11, 12), (12, 13),
        # Little
        (20, 4), (4, 14), (14, 15), (15, 16)
    ]
    
    # Plot connections
    for start, end in connections:
        ax.plot([positions[start-1, 0], positions[end-1, 0]],
                [positions[start-1, 1], positions[end-1, 1]],
                [positions[start-1, 2], positions[end-1, 2]], 'k-')
    
    # Add joint angle information if provided
    if joint_angles is not None:
        angles = joint_angles[time_step]
        angle_names = [
            "Thumb 1", "Thumb 2", 
            "Index 1", "Index 2", "Index 3",
            "Middle 1", "Middle 2", "Middle 3",
            "Ring 1", "Ring 2", "Ring 3",
            "Little 1", "Little 2", "Little 3"
        ]
        title = f"Hand Kinematics - Time Step: {time_step}"
        for i, angle in enumerate(angles):
            title += f"\n{angle_names[i]}: {angle:.1f}°"
        ax.set_title(title)
    else:
        ax.set_title(f"Hand Kinematics - Time Step: {time_step}")
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set equal aspect ratio
    max_range = np.max([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ])
    
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) / 2
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) / 2
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) / 2
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    return fig

def visualize_model_architecture(model, save_path='model_architecture.png'):
    """Create a visual diagram of the model architecture"""
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Define the muscle names and feature types for labeling
    muscle_names = ['APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR']
    feature_types = ['Time Domain', 'Frequency Domain', 'Wavelet']
    
    # Define colors
    colors = {
        'input': '#ADD8E6',     # Light blue
        'channel': '#98FB98',   # Light green
        'shared': '#FFFF99',    # Light yellow
        'output': '#FFA07A',    # Light salmon
        'text': '#000000',      # Black
        'line': '#808080'       # Gray
    }
    
    # Get the number of inputs (channels)
    n_channels = len([layer for layer in model.layers if 'input' in layer.name])
    
    # Set up the grid
    ax = plt.gca()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Draw input layer
    input_width = 15
    input_height = 60
    input_spacing = 2
    input_total_width = n_channels * (input_width + input_spacing) - input_spacing
    input_start_x = (100 - input_total_width) / 2
    
    for i in range(n_channels):
        # Input box
        x = input_start_x + i * (input_width + input_spacing)
        y = 10
        plt.gca().add_patch(plt.Rectangle((x, y), input_width, input_height, 
                                         facecolor=colors['input'], edgecolor=colors['line'], alpha=0.8))
        
        # Channel labels
        if i < len(muscle_names):
            plt.text(x + input_width/2, y + input_height - 5, muscle_names[i], 
                    ha='center', va='center', fontsize=9, fontweight='bold', color=colors['text'])
        else:
            plt.text(x + input_width/2, y + input_height - 5, f'EMG {i+1}', 
                    ha='center', va='center', fontsize=9, fontweight='bold', color=colors['text'])
        
        # Feature type labels
        feature_height = input_height / 3
        for j, feature_type in enumerate(feature_types):
            plt.text(x + input_width/2, y + j*feature_height + feature_height/2, feature_type, 
                    ha='center', va='center', fontsize=7, color=colors['text'])
    
    # Draw channel-specific layers
    cs_layer_x = 40
    cs_width = 12
    cs_height = 40
    cs_spacing = 2
    cs_total_height = n_channels * (cs_height + cs_spacing) - cs_spacing
    cs_start_y = (100 - cs_total_height) / 2
    
    for i in range(n_channels):
        # Channel layer box
        x = cs_layer_x
        y = cs_start_y + i * (cs_height + cs_spacing)
        plt.gca().add_patch(plt.Rectangle((x, y), cs_width, cs_height, 
                                         facecolor=colors['channel'], edgecolor=colors['line'], alpha=0.8))
        
        # Layer labels
        plt.text(x + cs_width/2, y + cs_height/2, f'Channel\nProcessing\n64 → 32 → 16', 
                ha='center', va='center', fontsize=7, color=colors['text'])
        
        # Connect input to channel processing
        input_x = input_start_x + i * (input_width + input_spacing) + input_width
        input_y = 10 + input_height/2
        plt.plot([input_x, cs_layer_x], [input_y, y + cs_height/2], 
                color=colors['line'], linestyle='-', linewidth=1, alpha=0.6)
    
    # Draw shared layers
    shared_layer_x = 65
    shared_width = 15
    shared_height = 50
    
    plt.gca().add_patch(plt.Rectangle((shared_layer_x, 25), shared_width, shared_height, 
                                     facecolor=colors['shared'], edgecolor=colors['line'], alpha=0.8))
    
    # Shared layer labels
    plt.text(shared_layer_x + shared_width/2, 50, 'Shared\nProcessing\n256 → 128 → 64 → 32', 
            ha='center', va='center', fontsize=8, color=colors['text'])
    
    # Connect channel processing to shared layer
    for i in range(n_channels):
        cs_x = cs_layer_x + cs_width
        cs_y = cs_start_y + i * (cs_height + cs_spacing) + cs_height/2
        plt.plot([cs_x, shared_layer_x], [cs_y, 50], 
                color=colors['line'], linestyle='-', linewidth=1, alpha=0.6)
    
    # Draw output layer
    output_x = 90
    output_width = 8
    output_height = 25
    
    plt.gca().add_patch(plt.Rectangle((output_x, 37.5), output_width, output_height, 
                                     facecolor=colors['output'], edgecolor=colors['line'], alpha=0.8))
    
    # Output layer labels
    plt.text(output_x + output_width/2, 50, 'Joint\nAngles\n14', 
            ha='center', va='center', fontsize=8, color=colors['text'])
    
    # Connect shared layer to output
    plt.plot([shared_layer_x + shared_width, output_x], [50, 50], 
            color=colors['line'], linestyle='-', linewidth=1.5)
    
    # Add title
    plt.title('EMG to Joint Angle Neural Network Architecture', fontsize=14, fontweight='bold', pad=20)
    
    # Add description
    description = (
        "Model flow: Each EMG channel extracts time, frequency, and wavelet features → "
        "Channel-specific processing → Shared processing → Joint angle estimation"
    )
    plt.figtext(0.5, 0.02, description, ha='center', fontsize=9, wrap=True)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path