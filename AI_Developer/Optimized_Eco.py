# Upgraded Python Code for Real-Time EMG-to-Angle Prediction (CNN Approach)
# Includes R2 score, Scatter Plot, Error Histogram visualizations

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score # Import R2 score calculation
import os
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Configuration ---
MAT_FILE_NAME = 's4_full.mat' # <-- Make sure this is your actual .MAT file name
NUM_TRIALS = 5
NUM_TASKS = 7
NUM_EMG_CHANNELS = 8
NUM_TOTAL_ANGLES = 14
FS = 200 # Sampling Frequency in Hz

# --- Real-Time Approach Parameters ---
WINDOW_SIZE_MS = 250  # Window duration in milliseconds
STEP_SIZE_MS = 10     # Step/stride for sliding window in milliseconds

# Convert ms to samples
WINDOW_SAMPLES = math.ceil(WINDOW_SIZE_MS * FS / 1000)
STEP_SAMPLES = math.ceil(STEP_SIZE_MS * FS / 1000)
logging.info(f"Real-time params: Window={WINDOW_SAMPLES} samples ({WINDOW_SIZE_MS}ms), Step={STEP_SAMPLES} samples ({STEP_SIZE_MS}ms)")

# Model Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2

# --- 2. Data Loading and Preprocessing (Windowing - No Changes Here) ---

def create_windows(data, window_size, step_size):
    """Creates overlapping windows from a sequence."""
    num_steps = data.shape[0]
    num_features = data.shape[1]
    windows = []
    indices = []
    for i in range(0, num_steps - window_size + 1, step_size):
        windows.append(data[i : i + window_size, :])
        indices.append(i + window_size - 1)
    if not windows: return np.empty((0, window_size, num_features)), []
    return np.array(windows), indices

def load_and_preprocess_windowed_data(mat_file, num_trials, num_tasks,
                                     window_size, step_size,
                                     num_emg, num_angles):
    """Loads data, creates windows, normalizes, and splits."""
    logging.info(f"Loading data from {mat_file}...")
    if not os.path.exists(mat_file): raise FileNotFoundError(f"MAT file not found: {mat_file}")
    try:
        data = scipy.io.loadmat(mat_file)
        emg_cell = data['dsfilt_emg']
        angle_cell = data['joint_angles']
    except KeyError as e: raise KeyError(f"Missing key: {e}")
    except Exception as e: raise IOError(f"Error loading MAT: {e}")

    all_emg_windows, all_target_angles = [], []
    logging.info("Creating sliding windows...")
    total_original_samples, original_seq_len = 0, -1

    for i in range(num_trials):
        for j in range(num_tasks):
            try:
                emg_seq, angle_seq = emg_cell[i, j], angle_cell[i, j]
                if original_seq_len == -1: original_seq_len = emg_seq.shape[0]
                total_original_samples += original_seq_len
                emg_windows, window_end_indices = create_windows(emg_seq, window_size, step_size)
                if emg_windows.size == 0: continue
                target_angles = angle_seq[window_end_indices, :]
                all_emg_windows.append(emg_windows)
                all_target_angles.append(target_angles)
            except Exception as e: logging.warning(f"Win Err T{i+1},T{j+1}: {e}")

    if not all_emg_windows: raise ValueError("No valid windows created.")

    X = np.concatenate(all_emg_windows, axis=0)
    y = np.concatenate(all_target_angles, axis=0)
    logging.info(f"Windowing OK. X:{X.shape}, y:{y.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SPLIT, random_state=42)
    logging.info(f"Split OK. Train X:{X_train.shape},y:{y_train.shape}. Val X:{X_val.shape},y:{y_val.shape}")

    num_train_win, _, num_emg_train = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_emg_train)
    emg_scaler = StandardScaler().fit(X_train_reshaped)
    X_train_scaled = emg_scaler.transform(X_train_reshaped).reshape(X_train.shape)

    angle_scaler = MinMaxScaler(feature_range=(0, 1)).fit(y_train)
    y_train_scaled = angle_scaler.transform(y_train)

    num_val_win, _, num_emg_val = X_val.shape
    X_val_reshaped = X_val.reshape(-1, num_emg_val)
    X_val_scaled = emg_scaler.transform(X_val_reshaped).reshape(X_val.shape)
    y_val_scaled = angle_scaler.transform(y_val)
    logging.info("Normalization OK.")
    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, emg_scaler, angle_scaler

# --- 3. Model Building (CNN - No Changes Here) ---

def build_realtime_cnn_model(window_size, num_emg, num_angles):
    """Builds a 1D CNN model structure."""
    logging.info("Building the CNN model structure...")
    input_shape = (window_size, num_emg)
    inputs = keras.Input(shape=input_shape, name='input_emg_window')
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x); x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_angles, activation='linear', name='output_angles')(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="RealTime_CNN_Model")
    logging.info("CNN model structure built successfully.")
    return model

# --- 4. Model Visualization (No Changes Here) ---

def visualize_model(model, filename="cnn_model_architecture.png"):
    """Saves plot of model architecture with enhanced error handling."""
    logging.info(f"Attempting to save model plot to {filename}...")
    # ...(Previous error handling code for pydot/graphviz remains the same)...
    try:
        keras.utils.plot_model( model, to_file=filename, show_shapes=True, show_layer_names=True, dpi=96 )
        logging.info(f"Model architecture plot saved successfully to {filename}")
    except ImportError: logging.warning( "Cannot plot model: Missing 'pydot' or 'graphviz'. Install libraries and check PATH." )
    except AttributeError as ae: logging.warning( f"Cannot plot model due to AttributeError: '{ae}'. Check pydot/Graphviz install/version." )
    except Exception as e:
        if "failed to execute" in str(e).lower() or "'dot'" in str(e) : logging.error(f"Error executing Graphviz 'dot': {e}. Check install/PATH.", exc_info=False)
        else: logging.error( f"Unexpected error plotting model: {e}", exc_info=True )


# --- 5. Model Training (No Changes Here) ---

def train_cnn_model(model, X_train, y_train, X_val, y_val, learning_rate, epochs, batch_size):
    """Compiles and trains the CNN model."""
    logging.info("Compiling the CNN model...")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    logging.info(f"Starting CNN training for {epochs} epochs...")
    history = model.fit( X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True )
    logging.info("CNN training finished.")
    return history

# --- 6. Training Visualization (No Changes Here) ---

def plot_training_history(history, filename_prefix="cnn_training_plot"):
    """Plots loss and metrics from training history."""
    logging.info("Plotting training history...")
    if not history or not history.history: return
    plt.figure(figsize=(12, 5)); metric = 'mae' # Assuming MAE was used
    # Plot Loss
    plt.subplot(1, 2, 1); plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history: plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss'); plt.ylabel('Loss (MSE)'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True)
    # Plot Metric (MAE)
    if metric in history.history:
        plt.subplot(1, 2, 2); plt.plot(history.history[metric], label=f'Training {metric.upper()}')
        if f'val_{metric}' in history.history: plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.upper()}')
        plt.title(f'Model {metric.upper()}'); plt.ylabel(metric.upper()); plt.xlabel('Epoch'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plot_filename = f"{filename_prefix}_loss_metrics.png"
    try: plt.savefig(plot_filename); logging.info(f"Training plot saved to {plot_filename}")
    except Exception as e: logging.error(f"Failed to save training plot: {e}")
    plt.close()

# --- 7. Prediction and Evaluation (Upgraded with R2, Scatter, Histogram) ---

def predict_and_evaluate_cnn(model, X_data_windowed, y_data_actual_scaled, # Pass scaled actual y
                             angle_scaler, step_samples, window_samples, num_angles_total): # Added num_angles_total
    """Makes predictions, inverse transforms, evaluates (RMSE, R2), and plots."""
    if X_data_windowed is None or not len(X_data_windowed):
        logging.warning("No data provided for prediction and evaluation.")
        return

    logging.info(f"Making predictions on {X_data_windowed.shape[0]} windows...")
    try:
        predictions_scaled = model.predict(X_data_windowed) # Shape: (NumWindows, NumAngles)
    except Exception as e: logging.error(f"Error during prediction: {e}", exc_info=True); return

    # --- Inverse Transform ---
    try:
        predictions_denormalized = angle_scaler.inverse_transform(predictions_scaled)
        y_actual_denormalized = angle_scaler.inverse_transform(y_data_actual_scaled) # Use passed scaled actual y
        logging.info("Predictions inverse transformed.")
    except Exception as e: logging.error(f"Error during inverse transform: {e}", exc_info=True); return

    num_predictions = predictions_denormalized.shape[0]
    if num_predictions == 0: logging.warning("No predictions generated."); return

    # --- Evaluation Metrics ---
    try:
        # Overall RMSE
        rmse = np.sqrt(np.mean((predictions_denormalized - y_actual_denormalized)**2))
        logging.info(f"Overall Root Mean Squared Error (RMSE): {rmse:.4f}")

        # R2 Score (per angle)
        # multioutput='raw_values' gives one score per output (angle)
        r2_scores = r2_score(y_actual_denormalized, predictions_denormalized, multioutput='raw_values')
        logging.info(f"R2 Scores (per angle): {np.round(r2_scores, 3)}")

        # Overall R2 Score (average variance explained)
        r2_overall = r2_score(y_actual_denormalized, predictions_denormalized, multioutput='variance_weighted') # or 'uniform_average'
        logging.info(f"Overall R2 Score (variance_weighted): {r2_overall:.4f}")

    except Exception as e: logging.error(f"Error calculating metrics: {e}")

    # --- Visualizations ---

    # 1. R2 Score Bar Plot
    try:
        plt.figure(figsize=(10, 5))
        angle_indices = np.arange(num_angles_total)
        plt.bar(angle_indices, r2_scores)
        plt.xlabel("Joint Angle Index")
        plt.ylabel("R² Score")
        plt.title("R² Score per Joint Angle")
        plt.xticks(angle_indices)
        plt.grid(axis='y', linestyle='--')
        # Add text labels for scores
        for index, score in enumerate(r2_scores):
             plt.text(index, score + (0.02 if score >= 0 else -0.05), f'{score:.2f}', ha='center', va='bottom', fontsize=9)
        plt.ylim(min(r2_scores.min() - 0.1, 0), 1.05) # Adjust y-lim slightly
        r2_plot_filename = "cnn_r2_score_per_angle.png"
        plt.savefig(r2_plot_filename)
        logging.info(f"R2 score plot saved to {r2_plot_filename}")
        plt.close()
    except Exception as e: logging.error(f"Error plotting R2 scores: {e}", exc_info=True)

    # 2. Time Series Comparison Plot (existing, slight refinement)
    try:
        prediction_indices = np.arange(num_predictions) * step_samples + (window_samples - 1)
        plot_limit = min(num_predictions, 4000 // step_samples) # Approx first 4 seconds (adjust as needed)
        angle_to_plot = 3 # Example: Index finger angle 1 (0-based index 2)

        if num_angles_total > angle_to_plot and plot_limit > 1:
            plt.figure(figsize=(15, 6))
            plt.plot(prediction_indices[:plot_limit], y_actual_denormalized[:plot_limit, angle_to_plot],
                     'b-', label='Actual Angle (at prediction steps)') # Solid line for actual
            plt.plot(prediction_indices[:plot_limit], predictions_denormalized[:plot_limit, angle_to_plot],
                     'r--', label='Predicted Angle') # Dashed line for predicted
            plt.title(f'Angle {angle_to_plot+1} Comparison (Windowed Predictions)')
            plt.xlabel('Time Step (Original Sequence Scale)'); plt.ylabel('Joint Angle (Original Units)')
            plt.legend(); plt.grid(True)
            ts_plot_filename = f"cnn_prediction_comparison_angle{angle_to_plot+1}.png"
            plt.savefig(ts_plot_filename); logging.info(f"Time series plot saved to {ts_plot_filename}")
            plt.close()
        else: logging.warning("Cannot plot time series: Not enough predictions or angles.")
    except Exception as e: logging.error(f"Error plotting time series: {e}", exc_info=True)

    # 3. Predicted vs. Actual Scatter Plot
    try:
        angle_to_plot_scatter = 4 # Example: Index angle 2 (0-based index 4)
        if num_angles_total > angle_to_plot_scatter and num_predictions > 1:
            plt.figure(figsize=(7, 7))
            actual_vals = y_actual_denormalized[:, angle_to_plot_scatter]
            pred_vals = predictions_denormalized[:, angle_to_plot_scatter]
            plt.scatter(actual_vals, pred_vals, alpha=0.3, s=10) # Use alpha for dense data
            # Add y=x line
            lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
            plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Ideal (y=x)')
            plt.xlabel("Actual Angle Value")
            plt.ylabel("Predicted Angle Value")
            plt.title(f"Predicted vs. Actual Scatter Plot (Angle {angle_to_plot_scatter+1})")
            plt.grid(True)
            plt.axis('equal') # Ensure aspect ratio is equal
            plt.legend()
            scatter_plot_filename = f"cnn_scatter_pred_vs_actual_angle{angle_to_plot_scatter+1}.png"
            plt.savefig(scatter_plot_filename)
            logging.info(f"Scatter plot saved to {scatter_plot_filename}")
            plt.close()
        else: logging.warning("Cannot plot scatter: Not enough predictions or angles.")
    except Exception as e: logging.error(f"Error plotting scatter plot: {e}", exc_info=True)

    # 4. Error Histogram
    try:
        angle_to_plot_hist = 6 # Example: Middle angle 2 (0-based index 6)
        if num_angles_total > angle_to_plot_hist and num_predictions > 1:
            errors = y_actual_denormalized[:, angle_to_plot_hist] - predictions_denormalized[:, angle_to_plot_hist]
            plt.figure(figsize=(8, 5))
            plt.hist(errors, bins=50, density=True, alpha=0.7) # Use density=True for normalized histogram
            plt.xlabel("Prediction Error (Actual - Predicted)")
            plt.ylabel("Density")
            plt.title(f"Histogram of Prediction Errors (Angle {angle_to_plot_hist+1})")
            plt.grid(axis='y', linestyle='--')
            # Add mean error line
            mean_error = np.mean(errors)
            plt.axvline(mean_error, color='r', linestyle='dashed', linewidth=1, label=f'Mean Error: {mean_error:.3f}')
            plt.legend()
            hist_plot_filename = f"cnn_error_histogram_angle{angle_to_plot_hist+1}.png"
            plt.savefig(hist_plot_filename)
            logging.info(f"Error histogram saved to {hist_plot_filename}")
            plt.close()
        else: logging.warning("Cannot plot histogram: Not enough predictions or angles.")
    except Exception as e: logging.error(f"Error plotting error histogram: {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    try:
        # 1. Load and Preprocess (Windowed Data)
        X_train, y_train_scaled, X_val, y_val_scaled, emg_scaler, angle_scaler = load_and_preprocess_windowed_data(
            MAT_FILE_NAME, NUM_TRIALS, NUM_TASKS,
            WINDOW_SAMPLES, STEP_SAMPLES,
            NUM_EMG_CHANNELS, NUM_TOTAL_ANGLES
        )

        # 2. Build CNN Model
        model = build_realtime_cnn_model(WINDOW_SAMPLES, NUM_EMG_CHANNELS, NUM_TOTAL_ANGLES)
        model.summary(line_length=100)

        # 3. Visualize Model Architecture (Optional - may fail if env not fixed)
        visualize_model(model)

        # 4. Train CNN Model
        history = train_cnn_model(model, X_train, y_train_scaled, X_val, y_val_scaled, LEARNING_RATE, EPOCHS, BATCH_SIZE)

        # 5. Visualize Training History
        plot_training_history(history)

        # 6. Evaluate on Validation Set (With Upgraded Visualizations)
        predict_and_evaluate_cnn(
            model,
            X_val,                 # Validation features (windowed, scaled)
            y_val_scaled,          # Validation targets (windowed, scaled) - NEEDED FOR INVERSE TRANSFORM & METRICS
            angle_scaler,          # Fitted angle scaler
            STEP_SAMPLES,          # Step size used for windowing
            WINDOW_SAMPLES,        # Window size used
            NUM_TOTAL_ANGLES       # Pass total number of angles
        )

        # Optional: Save Model
        # model.save("emg_to_angle_cnn_model_upgraded.keras")
        # logging.info("Trained CNN model saved.")

    except FileNotFoundError as e: logging.error(e)
    except ValueError as e: logging.error(f"Data processing/value error: {e}", exc_info=True)
    except Exception as e: logging.error(f"Unexpected error in main script: {e}", exc_info=True)]