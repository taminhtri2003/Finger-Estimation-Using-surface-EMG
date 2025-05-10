# Import necessary libraries
import os
import fnmatch
import numpy as np
import scipy.io
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# --- Configuration ---
# !!! IMPORTANT: Path to your .mat file (containing joint_angles, dsfilt_emg) !!!
MAT_FILE_PATH = "s4_full.mat" # Same .mat file used by the previous script
# !!! IMPORTANT: Directory where HHT spectrum images were saved by the previous script !!!
HHT_IMAGE_DIR = "hht_spectrum_plots"
# !!! IMPORTANT: EMG Muscle names, must match the ones used for naming HHT images !!!
EMG_MUSCLES = ['APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR']

# Image preprocessing parameters
IMAGE_HEIGHT = 128  # Target height for resizing images
IMAGE_WIDTH = 128   # Target width for resizing images
IMAGE_CHANNELS = 3  # Assuming RGB images from matplotlib plots

# Model and training parameters
NUM_JOINT_ANGLES = 14 # As per your data description
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2 # Proportion of data to use for validation

# Output files
MODEL_SAVE_PATH = "joint_angle_cnn_model.h5"
HISTORY_PLOT_PATH = "training_history.png"

def load_and_preprocess_image(image_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    """
    Loads an image, resizes it, and normalizes pixel values.
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Desired (width, height) for the image.
    Returns:
        np.ndarray: Preprocessed image as a NumPy array, or None if error.
    """
    try:
        img = Image.open(image_path).convert('RGB') # Ensure 3 channels
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        return img_array
    except FileNotFoundError:
        print(f"Warning: Image file not found: {image_path}")
        return None
    except Exception as e:
        print(f"Error loading or preprocessing image {image_path}: {e}")
        return None

def create_dataset(mat_data_path, hht_image_dir, emg_muscles_list, target_img_size):
    """
    Creates a dataset of (image_data, mean_joint_angles).
    Args:
        mat_data_path (str): Path to the .mat file.
        hht_image_dir (str): Directory containing HHT images.
        emg_muscles_list (list): List of EMG muscle names.
        target_img_size (tuple): Desired (width, height) for images.
    Returns:
        tuple: (X_data, y_data) where X_data is a list of preprocessed images
               and y_data is a list of corresponding mean joint angle vectors.
               Returns (None, None) if essential data is missing.
    """
    print("Creating dataset...")
    try:
        mat_data = scipy.io.loadmat(mat_data_path)
        print(f"Loaded .mat file: {mat_data_path}")
    except FileNotFoundError:
        print(f"Error: .mat file not found at {mat_data_path}")
        return None, None
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return None, None

    if 'joint_angles' not in mat_data or 'dsfilt_emg' not in mat_data:
        print("Error: 'joint_angles' or 'dsfilt_emg' not found in .mat file.")
        return None, None

    joint_angles_all = mat_data['joint_angles'] # Expected shape: (num_trials, num_tasks) cell array
    dsfilt_emg_all = mat_data['dsfilt_emg']     # Expected shape: (num_trials, num_tasks) cell array

    num_trials = dsfilt_emg_all.shape[0]
    num_tasks = dsfilt_emg_all.shape[1]

    print(f"Found {num_trials} trials and {num_tasks} tasks.")

    all_images_data = []
    all_target_angles = []

    for trial_idx in range(num_trials):
        for task_idx in range(num_tasks):
            print(f"  Processing Trial {trial_idx+1}, Task {task_idx+1}...")
            
            # Get joint angles for this trial/task and calculate mean
            # Each cell joint_angles_all[trial_idx, task_idx] is (4000, 14)
            try:
                current_joint_angles = joint_angles_all[trial_idx, task_idx]
                if current_joint_angles.shape[1] != NUM_JOINT_ANGLES:
                    print(f"    Warning: Joint angles for T{trial_idx+1},TK{task_idx+1} have {current_joint_angles.shape[1]} columns, expected {NUM_JOINT_ANGLES}. Skipping.")
                    continue
                mean_target_angles = np.mean(current_joint_angles, axis=0) # Shape (14,)
            except Exception as e:
                print(f"    Error processing joint angles for T{trial_idx+1},TK{task_idx+1}: {e}. Skipping.")
                continue

            # Find all HHT images for this trial and task
            for channel_idx, channel_name in enumerate(emg_muscles_list):
                # Pattern for images: hht_T<trial>_TK<task>_CH<channel_name>_IMF<imf_idx>.png
                # Example: hht_T1_TK1_CHAPL_IMF1.png
                image_file_pattern = f"hht_T{trial_idx+1}_TK{task_idx+1}_CH{channel_name}_IMF*.png"
                
                try:
                    matching_files = [f for f in os.listdir(hht_image_dir) if fnmatch.fnmatch(f, image_file_pattern)]
                except FileNotFoundError:
                    print(f"    Error: HHT image directory not found: {hht_image_dir}")
                    # If the directory itself is missing for one trial/task (unlikely),
                    # we might want to break or return None, None earlier.
                    # For now, assume it's per-file issues.
                    continue 
                
                if not matching_files:
                    print(f"    No images found for pattern: {image_file_pattern} in {hht_image_dir}")

                for image_filename in matching_files:
                    full_image_path = os.path.join(hht_image_dir, image_filename)
                    preprocessed_image = load_and_preprocess_image(full_image_path, target_img_size)
                    
                    if preprocessed_image is not None:
                        all_images_data.append(preprocessed_image)
                        all_target_angles.append(mean_target_angles)
                    else:
                        print(f"    Skipped image {image_filename} due to loading error.")
    
    if not all_images_data:
        print("No image data was successfully loaded. Cannot proceed.")
        return None, None

    print(f"Dataset creation complete. Loaded {len(all_images_data)} images.")
    return np.array(all_images_data), np.array(all_target_angles)


def build_cnn_model(input_shape, num_outputs):
    """
    Builds a CNN model for joint angle regression.
    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_outputs (int): Number of output units (number of joint angles).
    Returns:
        tensorflow.keras.models.Sequential: Compiled CNN model.
    """
    print("Building CNN model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'), # Added one more conv layer
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'), # Increased dense layer size
        Dropout(0.3), # Added dropout for regularization
        Dense(128, activation='relu'),
        Dropout(0.3), # Added dropout
        Dense(num_outputs, activation='linear')  # Linear activation for regression
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error',  # Common for regression
                  metrics=['mae', 'mse'])     # Mean Absolute Error, Mean Squared Error
    
    print("Model built and compiled.")
    model.summary()
    return model

def plot_training_history(history, save_path):
    """
    Plots training and validation loss and MAE.
    Args:
        history (tensorflow.keras.callbacks.History): History object from model.fit().
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(12, 6))

    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss (MSE)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    # Plot training & validation MAE values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model Mean Absolute Error (MAE)')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    try:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving training history plot: {e}")
    plt.show()


if __name__ == "__main__":
    # 1. Create the dataset
    # Ensure HHT_IMAGE_DIR exists
    if not os.path.isdir(HHT_IMAGE_DIR):
        print(f"Error: HHT image directory '{HHT_IMAGE_DIR}' not found.")
        print("Please ensure you have run the previous script to generate HHT images,")
        print("or update HHT_IMAGE_DIR to the correct path.")
        exit()
        
    X_data, y_data = create_dataset(MAT_FILE_PATH, HHT_IMAGE_DIR, EMG_MUSCLES, 
                                    target_img_size=(IMAGE_WIDTH, IMAGE_HEIGHT))

    if X_data is None or y_data is None or len(X_data) == 0:
        print("Failed to create dataset. Exiting.")
        exit()

    print(f"Shape of X_data (images): {X_data.shape}") # (num_images, height, width, channels)
    print(f"Shape of y_data (angles): {y_data.shape}") # (num_images, num_joint_angles)

    # 2. Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_data, test_size=VALIDATION_SPLIT, random_state=42
    )
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # 3. Build the CNN model
    model = build_cnn_model(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
                            num_outputs=NUM_JOINT_ANGLES)

    # 4. Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True)

    # 5. Train the model
    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint]
    )
    print("Model training complete.")
    print(f"Best model saved to {MODEL_SAVE_PATH}")

    # 6. Plot training history
    plot_training_history(history, HISTORY_PLOT_PATH)

    # 7. (Optional) Evaluate the best model on the validation set
    print("\nEvaluating the best model on the validation set:")
    # Load the best model saved by ModelCheckpoint
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    val_loss, val_mae, val_mse = best_model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss (MSE): {val_loss:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")
    print(f"Validation MSE (from evaluate): {val_mse:.4f}")

    # 8. (Optional) Make predictions on a few validation samples
    print("\nExample predictions on validation data:")
    num_predictions_to_show = 3
    for i in range(min(num_predictions_to_show, len(X_val))):
        sample_image = np.expand_dims(X_val[i], axis=0) # Model expects batch dimension
        predicted_angles = best_model.predict(sample_image)[0]
        actual_angles = y_val[i]
        print(f"  Sample {i+1}:")
        print(f"    Predicted Angles: {np.round(predicted_angles, 2)}")
        print(f"    Actual Angles:    {np.round(actual_angles, 2)}")
        print(f"    Difference (MAE): {np.mean(np.abs(predicted_angles - actual_angles)):.2f}")

    print("\n--- Script Finished ---")

