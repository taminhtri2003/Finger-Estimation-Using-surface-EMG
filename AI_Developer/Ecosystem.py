# Full Corrected Python Code for EMG-to-Angle Prediction (v3)

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Configuration ---
MAT_FILE_NAME = 's4_full.mat' # <-- *** REPLACE WITH YOUR ACTUAL .MAT FILE NAME ***
NUM_TRIALS = 5
NUM_TASKS = 7
SEQUENCE_LENGTH = 4000 # Time steps per sequence
NUM_EMG_CHANNELS = 8
NUM_TOTAL_ANGLES = 14

# Angle Indices (based on MATLAB description, 0-based for Python)
THUMB_ANGLE_IDX = slice(0, 2)   # Angles 1-2
INDEX_ANGLE_IDX = slice(2, 5)   # Angles 3-5
MIDDLE_ANGLE_IDX = slice(5, 8)  # Angles 6-8
RING_ANGLE_IDX = slice(8, 11)  # Angles 9-11
PINKY_ANGLE_IDX = slice(11, 14) # Angles 12-14

# Model Hyperparameters (Examples - Tune these)
ENCODER_UNITS = 128
LEARNING_RATE = 0.002
EPOCHS = 50 # Increase for real training
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2 # Use 20% of data for validation

# --- 2. Data Loading and Preprocessing ---

def load_and_preprocess_data(mat_file, num_trials, num_tasks, seq_len, num_emg, num_angles):
    """Loads data from .mat file, reshapes, normalizes, and splits."""
    logging.info(f"Loading data from {mat_file}...")
    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"MAT file not found: {mat_file}")
    try:
        data = scipy.io.loadmat(mat_file)
        emg_cell = data['dsfilt_emg']
        angle_cell = data['joint_angles']
    except KeyError as e:
        raise KeyError(f"Missing expected key in MAT file: {e}")
    except Exception as e:
        raise IOError(f"Error loading MAT file: {e}")

    all_emg = []
    all_angles = []

    logging.info("Extracting and stacking sequences...")
    for i in range(num_trials):
        for j in range(num_tasks):
            try:
                emg_seq = emg_cell[i, j]
                angle_seq = angle_cell[i, j]

                # Basic shape validation
                if emg_seq.shape == (seq_len, num_emg) and angle_seq.shape == (seq_len, num_angles):
                    all_emg.append(emg_seq)
                    all_angles.append(angle_seq)
                else:
                    logging.warning(f"Skipping sequence Trial {i+1}, Task {j+1} due to unexpected shape. "
                                    f"EMG: {emg_seq.shape}, Angles: {angle_seq.shape}")
            except IndexError:
                 logging.warning(f"Index out of bounds when accessing Trial {i+1}, Task {j+1}. Check NUM_TRIALS/NUM_TASKS.")
            except Exception as e:
                 logging.warning(f"Error processing sequence Trial {i+1}, Task {j+1}: {e}")


    if not all_emg:
        raise ValueError("No valid sequences found in the data. Check shapes and file content.")

    # Stack into numpy arrays (NumSequences, TimeSteps, Features)
    X = np.array(all_emg)
    Y = np.array(all_angles)
    logging.info(f"Data stacked. X shape: {X.shape}, Y shape: {Y.shape}")

    # --- Train/Validation Split (before scaling) ---
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=VALIDATION_SPLIT, random_state=42 # for reproducibility
    )
    logging.info(f"Data split. Train shapes: X={X_train.shape}, Y={Y_train.shape}. "
                 f"Validation shapes: X={X_val.shape}, Y={Y_val.shape}")

    # --- Normalization ---
    # Reshape for Scaler: (Samples * TimeSteps, Features)
    nsamples_train, nx_train, nfeatures_train = X_train.shape
    X_train_reshaped = X_train.reshape((nsamples_train * nx_train, nfeatures_train))

    nsamples_train_y, nx_train_y, nfeatures_train_y = Y_train.shape
    Y_train_reshaped = Y_train.reshape((nsamples_train_y * nx_train_y, nfeatures_train_y))

    # EMG Scaler (Standardization)
    emg_scaler = StandardScaler()
    emg_scaler.fit(X_train_reshaped) # Fit ONLY on training data
    X_train_scaled_reshaped = emg_scaler.transform(X_train_reshaped)
    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape) # Reshape back

    # Angle Scaler (MinMax to [0, 1] - adjust range if needed)
    angle_scaler = MinMaxScaler(feature_range=(0, 1))
    angle_scaler.fit(Y_train_reshaped) # Fit ONLY on training data
    Y_train_scaled_reshaped = angle_scaler.transform(Y_train_reshaped)
    Y_train_scaled = Y_train_scaled_reshaped.reshape(Y_train.shape) # Reshape back

    # Apply same scaling to validation set
    nsamples_val, nx_val, nfeatures_val = X_val.shape
    X_val_reshaped = X_val.reshape((nsamples_val * nx_val, nfeatures_val))
    X_val_scaled_reshaped = emg_scaler.transform(X_val_reshaped) # Use fitted scaler
    X_val_scaled = X_val_scaled_reshaped.reshape(X_val.shape)

    nsamples_val_y, nx_val_y, nfeatures_val_y = Y_val.shape
    Y_val_reshaped = Y_val.reshape((nsamples_val_y * nx_val_y, nfeatures_val_y))
    Y_val_scaled_reshaped = angle_scaler.transform(Y_val_reshaped) # Use fitted scaler
    Y_val_scaled = Y_val_scaled_reshaped.reshape(Y_val.shape)

    logging.info("Normalization applied.")

    # --- Split Scaled Targets ---
    Y_train_split = {
        'thumb': Y_train_scaled[:, :, THUMB_ANGLE_IDX],
        'index': Y_train_scaled[:, :, INDEX_ANGLE_IDX],
        'middle': Y_train_scaled[:, :, MIDDLE_ANGLE_IDX],
        'ring': Y_train_scaled[:, :, RING_ANGLE_IDX],
        'pinky': Y_train_scaled[:, :, PINKY_ANGLE_IDX]
    }
    Y_val_split = {
        'thumb': Y_val_scaled[:, :, THUMB_ANGLE_IDX],
        'index': Y_val_scaled[:, :, INDEX_ANGLE_IDX],
        'middle': Y_val_scaled[:, :, MIDDLE_ANGLE_IDX],
        'ring': Y_val_scaled[:, :, RING_ANGLE_IDX],
        'pinky': Y_val_scaled[:, :, PINKY_ANGLE_IDX]
    }

    # Convert dict to list for model.fit(), matching model output order
    Y_train_list = [Y_train_split['thumb'], Y_train_split['index'], Y_train_split['middle'], Y_train_split['ring'], Y_train_split['pinky']]
    Y_val_list = [Y_val_split['thumb'], Y_val_split['index'], Y_val_split['middle'], Y_val_split['ring'], Y_val_split['pinky']]


    return X_train_scaled, Y_train_list, X_val_scaled, Y_val_list, emg_scaler, angle_scaler

# --- 3. Model Building ---

def build_emg_to_angle_model(seq_len, num_emg, encoder_units):
    """Builds the Keras GRU model with 5 output heads."""
    logging.info("Building the model...")

    input_emg = keras.Input(shape=(seq_len, num_emg), name='input_emg')

    # Common Encoder
    gru_out = layers.GRU(encoder_units, return_sequences=True, name='encoder_gru')(input_emg)
    # gru_out = layers.Dropout(0.2)(gru_out) # Optional regularization

    # Decoder Heads
    out_thumb = layers.TimeDistributed(layers.Dense(2, name='dense_thumb'), name='output_thumb')(gru_out)
    out_index = layers.TimeDistributed(layers.Dense(3, name='dense_index'), name='output_index')(gru_out)
    out_middle = layers.TimeDistributed(layers.Dense(3, name='dense_middle'), name='output_middle')(gru_out)
    out_ring = layers.TimeDistributed(layers.Dense(3, name='dense_ring'), name='output_ring')(gru_out)
    out_pinky = layers.TimeDistributed(layers.Dense(3, name='dense_pinky'), name='output_pinky')(gru_out)

    model = keras.Model(
        inputs=input_emg,
        outputs=[out_thumb, out_index, out_middle, out_ring, out_pinky],
        name="EMG_to_MultiAngle_Model"
    )
    logging.info("Model built successfully.")
    return model

# --- 4. Model Visualization ---

def visualize_model(model, filename="model_architecture.png"):
    """Saves plot of model architecture with enhanced error handling."""
    try:
        keras.utils.plot_model( model, to_file=filename, show_shapes=True, show_layer_names=True, dpi=96 )
        logging.info(f"Model architecture saved successfully to {filename}")
    except ImportError:
        logging.warning( "Cannot plot model: Missing 'pydot' or 'graphviz'. Install pydot (`pip install pydot`) and ensure Graphviz (https://graphviz.org/download/) is installed and in PATH." )
    except AttributeError as ae:
        logging.warning( f"Cannot plot model due to AttributeError: '{ae}'. Often a pydot version conflict. Try: 'pip uninstall pydot pydotplus -y && pip install pydot'. Check Graphviz install." )
    except Exception as e:
        logging.error( f"An unexpected error occurred while plotting model: {e}", exc_info=True )

# --- 5. Model Training ---

# ############################################################################ #
# CORRECTED FUNCTION (v3) with fix for multiple output metrics (using dict)
# ############################################################################ #
def train_model(model, X_train, Y_train_list, X_val, Y_val_list, learning_rate, epochs, batch_size):
    """Compiles and trains the Keras model."""
    logging.info("Compiling the model...")

    # Define losses for each output using a dictionary
    losses = {
        'output_thumb': 'mse',
        'output_index': 'mse',
        'output_middle': 'mse',
        'output_ring': 'mse',
        'output_pinky': 'mse'
    }

    # --- FIX APPLIED HERE (v3) ---
    # Use a dictionary for metrics, mapping output names to the desired metric.
    # This satisfies the requirement for a list/tuple/dict for multi-output models.
    metrics_dict = {
        'output_thumb': 'mae',
        'output_index': 'mae',
        'output_middle': 'mae',
        'output_ring': 'mae',
        'output_pinky': 'mae'
    }
    # --- End of Fix ---

    # Optional loss weights
    # loss_weights = {'output_thumb': 1.0, 'output_index': 1.0, ...}

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=losses,           # Use the dictionary for losses
        metrics=metrics_dict,  # Use the dictionary for metrics
        # loss_weights=loss_weights # Uncomment if using weights
    )

    logging.info(f"Starting training for {epochs} epochs...")
    history = model.fit(
        X_train,
        Y_train_list,
        validation_data=(X_val, Y_val_list),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
        # callbacks=[...] # Optional
    )
    logging.info("Training finished.")
    return history
# ############################################################################ #


# --- 6. Training Visualization ---

def plot_training_history(history, filename_prefix="training_plot"):
    """Plots loss and metrics from training history."""
    logging.info("Plotting training history...")
    if not history or not history.history:
         logging.warning("No training history found to plot.")
         return

    # Determine keys for overall loss and specific output metrics
    overall_loss_keys = ['loss', 'val_loss']
    # Keras names metrics for multi-output models like 'output_thumb_mae', 'val_output_thumb_mae'
    output_metric_keys = sorted([k for k in history.history.keys() if k.startswith('output_') and not k.startswith('val_')])
    output_val_metric_keys = sorted([k for k in history.history.keys() if k.startswith('val_output_')])

    num_plots = 1 + len(output_metric_keys) # Plot overall loss + one per output metric
    plt.figure(figsize=(max(12, 6 * num_plots / 2), 5 * (num_plots//2 + num_plots%2))) # Adjust figure size dynamically

    # --- Plot Overall Loss ---
    plt.subplot((num_plots + 1) // 2, 2, 1) # Arrange plots in ~2 columns
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Overall Model Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    # --- Plot Metrics for each Output ---
    for i, metric_key in enumerate(output_metric_keys):
         val_metric_key = 'val_' + metric_key
         # Extract info from metric key like 'output_thumb_mae'
         parts = metric_key.split('_')
         output_name = parts[1] # 'thumb'
         metric_name = parts[-1] # 'mae'

         plt.subplot((num_plots + 1) // 2, 2, i + 2) # Arrange plots
         plt.plot(history.history[metric_key], label=f'Train {output_name.capitalize()} {metric_name.upper()}')
         if val_metric_key in history.history:
             plt.plot(history.history[val_metric_key], label=f'Val {output_name.capitalize()} {metric_name.upper()}')
         plt.title(f'{output_name.capitalize()} Output {metric_name.upper()}')
         plt.ylabel(metric_name.upper())
         plt.xlabel('Epoch')
         plt.legend()
         plt.grid(True)

    plt.tight_layout()
    plot_filename = f"{filename_prefix}_loss_metrics.png"
    try:
        plt.savefig(plot_filename)
        logging.info(f"Training history plot saved to {plot_filename}")
    except Exception as e:
        logging.error(f"Failed to save training plot: {e}")
    plt.close()

# --- 7. Prediction and Evaluation ---

def predict_and_evaluate(model, X_data, Y_data_list, emg_scaler, angle_scaler, sequence_length):
    """Makes predictions, inverse transforms, evaluates, and plots."""
    if X_data is None or not len(X_data):
        logging.warning("No data provided for prediction and evaluation.")
        return

    logging.info(f"Making predictions on {X_data.shape[0]} sequences...")
    try:
        predictions_scaled_list = model.predict(X_data)
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return

    try:
        num_samples = X_data.shape[0]
        num_angles_total = sum(out.shape[-1] for out in Y_data_list)

        predictions_scaled_combined = np.concatenate(predictions_scaled_list, axis=-1)
        Y_data_combined = np.concatenate(Y_data_list, axis=-1)

        predictions_scaled_reshaped = predictions_scaled_combined.reshape(-1, num_angles_total)
        Y_data_reshaped = Y_data_combined.reshape(-1, num_angles_total)

        predictions_denormalized = angle_scaler.inverse_transform(predictions_scaled_reshaped)
        Y_data_denormalized = angle_scaler.inverse_transform(Y_data_reshaped)

        predictions_denormalized = predictions_denormalized.reshape(num_samples, sequence_length, num_angles_total)
        Y_data_denormalized = Y_data_denormalized.reshape(num_samples, sequence_length, num_angles_total)
        logging.info("Predictions inverse transformed.")

    except Exception as e:
        logging.error(f"Error during inverse transform: {e}", exc_info=True)
        return

    try:
        rmse = np.sqrt(np.mean((predictions_denormalized - Y_data_denormalized)**2))
        logging.info(f"Overall Root Mean Squared Error (RMSE) on provided data: {rmse:.4f}")
    except Exception as e:
        logging.error(f"Error calculating RMSE: {e}")

    try:
        angle_to_plot = 3
        sequence_to_plot = 0
        if num_samples > sequence_to_plot and num_angles_total > angle_to_plot:
            time_axis = np.arange(sequence_length)
            plt.figure(figsize=(15, 6))
            plt.plot(time_axis, Y_data_denormalized[sequence_to_plot, :, angle_to_plot], 'b-', label='Actual Angle')
            plt.plot(time_axis, predictions_denormalized[sequence_to_plot, :, angle_to_plot], 'r--', label='Predicted Angle')
            plt.title(f'Angle {angle_to_plot+1} Comparison (Sequence {sequence_to_plot+1})')
            plt.xlabel('Time Step')
            plt.ylabel('Joint Angle (Original Units)')
            plt.legend()
            plt.grid(True)
            plot_filename = f"prediction_comparison_seq{sequence_to_plot+1}_angle{angle_to_plot+1}.png"
            plt.savefig(plot_filename)
            logging.info(f"Prediction comparison plot saved to {plot_filename}")
            plt.close()
        else:
            logging.warning("Cannot plot prediction comparison: Not enough samples or angles.")
    except Exception as e:
        logging.error(f"Error plotting prediction comparison: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    try:
        # 1. Load and Preprocess
        X_train, Y_train, X_val, Y_val, emg_scaler, angle_scaler = load_and_preprocess_data(
            MAT_FILE_NAME, NUM_TRIALS, NUM_TASKS, SEQUENCE_LENGTH, NUM_EMG_CHANNELS, NUM_TOTAL_ANGLES
        )

        # 2. Build Model
        model = build_emg_to_angle_model(SEQUENCE_LENGTH, NUM_EMG_CHANNELS, ENCODER_UNITS)
        model.summary(line_length=120)

        # 3. Visualize Model Architecture
        visualize_model(model)

        # 4. Train Model (using the v3 corrected train_model function)
        history = train_model(model, X_train, Y_train, X_val, Y_val, LEARNING_RATE, EPOCHS, BATCH_SIZE)

        # 5. Visualize Training
        plot_training_history(history)

        # 6. Evaluate on Validation Set (as an example)
        predict_and_evaluate(model, X_val, Y_val, emg_scaler, angle_scaler, SEQUENCE_LENGTH)

        # Optional: Save the trained model
        # model_save_path = "emg_to_angle_model.keras"
        # model.save(model_save_path)
        # logging.info(f"Trained model saved to {model_save_path}")

    except FileNotFoundError as e:
        logging.error(e)
    except ValueError as e:
        logging.error(f"Data processing or value error: {e}", exc_info=True) # Add traceback for value errors
    except Exception as e:
        logging.error(f"An unexpected error occurred in the main script: {e}", exc_info=True)