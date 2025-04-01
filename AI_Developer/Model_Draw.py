# Code to Define and Visualize the Real-Time CNN Model Architecture ONLY

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
import os

# --- Minimal Configuration for Model Definition ---
# These values are needed just to define the model's layers.
# Use the same values intended for your real-time CNN model.
# Example values (replace with your actual ones if different):
FS = 200 # Hz
WINDOW_SIZE_MS = 250 # ms
WINDOW_SAMPLES = int(WINDOW_SIZE_MS * FS / 1000) # Calculated samples
NUM_EMG_CHANNELS = 8
NUM_TOTAL_ANGLES = 14

# Configure basic logging (optional, but helps see messages from functions)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model Building Function (CNN Approach) ---
def build_realtime_cnn_model(window_size, num_emg, num_angles):
    """Builds a 1D CNN model structure."""
    logging.info("Building the CNN model structure for visualization...")

    input_shape = (window_size, num_emg)
    inputs = keras.Input(shape=input_shape, name='input_emg_window')

    # Convolutional Block 1
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Convolutional Block 2
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Flatten or Pool before Dense layers
    x = layers.GlobalAveragePooling1D()(x)

    # Dense Layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)

    # Output Layer
    outputs = layers.Dense(num_angles, activation='linear', name='output_angles')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="RealTime_CNN_Model")
    logging.info("CNN model structure built successfully.")
    return model

# --- Model Visualization Function ---
def visualize_model(model, filename="cnn_model_architecture.png"):
    """Saves a plot of the model architecture if possible."""
    logging.info(f"Attempting to save model plot to {filename}...")
    print(f"\nAttempting to save model plot to {filename}...") # Console feedback
    try:
        keras.utils.plot_model(
            model,
            to_file=filename,
            show_shapes=True,        # Show tensor shapes
            show_layer_names=True,   # Show layer names
            dpi=96                   # Image resolution
        )
        logging.info(f"Model architecture plot saved successfully to {filename}")
        print(f"--- SUCCESS: Model plot saved to {filename} ---")

    except ImportError:
        logging.warning( "Cannot plot model: Missing 'pydot' or 'graphviz'. Install pydot (`pip install pydot`) and ensure Graphviz (https://graphviz.org/download/) is installed and in PATH." )
        print("--- FAILED: Missing required libraries (pydot or graphviz). Please install them. ---")

    except AttributeError as ae:
        # Catches the specific 'InvocationException' error if pydot version is incompatible
        logging.warning( f"Cannot plot model due to AttributeError: '{ae}'. Often pydot version conflict. Try: 'pip uninstall pydot pydotplus -y && pip install pydot'. Check Graphviz install/PATH." )
        print(f"--- FAILED: AttributeError ('{ae}'). This often means a pydot version conflict or Graphviz issue. See previous advice. ---")

    except Exception as e:
        # Catches other errors, like Graphviz 'dot' command not found or failing
        if "failed to execute" in str(e).lower() or "'dot'" in str(e) :
             logging.error(f"Error executing Graphviz 'dot' command: {e}. Ensure Graphviz is installed and its 'bin' directory is in the system PATH.", exc_info=False)
             print(f"--- FAILED: Could not execute Graphviz 'dot' command. Is Graphviz installed and in your system PATH? Error: {e} ---")
        else:
             logging.error( f"An unexpected error occurred while plotting model: {e}", exc_info=True )
             print(f"--- FAILED: An unexpected error occurred during plotting: {e} ---")

# --- Main Execution Block ---
if __name__ == "__main__":
    print("1. Building the CNN model structure...")
    # Build the CNN model with the specified dimensions
    cnn_model = build_realtime_cnn_model(
        window_size=WINDOW_SAMPLES,
        num_emg=NUM_EMG_CHANNELS,
        num_angles=NUM_TOTAL_ANGLES
    )

    print("\n2. Displaying model summary:")
    cnn_model.summary(line_length=100)

    # 3. Attempt to visualize the model
    visualize_model(cnn_model, filename="cnn_realtime_model_architecture.png")

    print("\nScript finished. Check console output and generated PNG file (if successful).")