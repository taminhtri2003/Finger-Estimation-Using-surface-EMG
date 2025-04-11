import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import shap
import os
import scipy.io as sio
from tensorflow.keras.models import load_model
import matplotlib.cm as cm

# Create output directories
os.makedirs('/home/ubuntu/emg_project/visualizations/xai', exist_ok=True)

# Load the trained model
# Since there was an error saving the model, we'll need to retrain it with the correct extension
# or load it from the checkpoint if available
model_path = '/home/ubuntu/emg_project/models/multi_head_attention_model.h5'

try:
    model = load_model(model_path)
    print("Model loaded successfully from checkpoint.")
except:
    print("Could not load model from checkpoint. Please ensure the model is trained and saved correctly.")
    exit()

# Load test data
X_test = np.load('/home/ubuntu/emg_project/data/processed/X_test.npy')
y_test = np.load('/home/ubuntu/emg_project/data/processed/y_test.npy')

print(f"Test data shape: {X_test.shape}, {y_test.shape}")

# Define finger groups for visualization
finger_groups = [
    ('Thumb', 0, 2),
    ('Index', 2, 5),
    ('Middle', 5, 8),
    ('Ring', 8, 11),
    ('Little', 11, 14)
]

# 1. Implement SHAP for feature importance analysis
# Create a background dataset for SHAP (using a subset of training data)
X_train = np.load('/home/ubuntu/emg_project/data/processed/X_train.npy')
background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

# Create a SHAP explainer
print("Creating SHAP explainer...")
explainer = shap.DeepExplainer(model, background)

# Calculate SHAP values for a subset of test data
print("Calculating SHAP values...")
n_samples = 50  # Use a smaller subset for faster computation
shap_values = explainer.shap_values(X_test[:n_samples])

# Plot SHAP summary plots for each finger group
print("Creating SHAP summary plots...")
for i, (finger_name, start_idx, end_idx) in enumerate(finger_groups):
    plt.figure(figsize=(12, 8))
    
    # For multi-output models, shap_values is a list of arrays, one per output
    # We'll use the first joint angle for each finger
    joint_idx = start_idx
    
    # Create SHAP summary plot
    shap.summary_plot(
        shap_values[joint_idx], 
        X_test[:n_samples],
        feature_names=[f"EMG_{i}" for i in range(X_test.shape[2])],
        show=False
    )
    
    plt.title(f'SHAP Feature Importance for {finger_name} Finger')
    plt.tight_layout()
    plt.savefig(f'/home/ubuntu/emg_project/visualizations/xai/shap_summary_{finger_name}.png')
    plt.close()

# 2. Implement attention visualization techniques
# Create a model that outputs attention weights
attention_layer_name = None
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.MultiHeadAttention):
        attention_layer_name = layer.name
        break

if attention_layer_name:
    print(f"Found attention layer: {attention_layer_name}")
    
    # Create a model that outputs attention weights
    attention_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.output, model.get_layer(attention_layer_name).output]
    )
    
    # Get attention weights for a sample
    sample_idx = 0
    sample = X_test[sample_idx:sample_idx+1]
    _, attention_weights = attention_model.predict(sample)
    
    # Visualize attention weights
    plt.figure(figsize=(15, 10))
    
    # Plot attention heatmap
    plt.subplot(2, 1, 1)
    plt.imshow(attention_weights[0], cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.title('Multi-Head Attention Weights')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    # Plot input EMG signals
    plt.subplot(2, 1, 2)
    for i in range(sample.shape[2]):
        plt.plot(sample[0, :, i], label=f'EMG {i}')
    plt.title('Input EMG Signals')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/emg_project/visualizations/xai/attention_visualization.png')
    plt.close()
else:
    print("Could not find MultiHeadAttention layer in the model.")

# 3. Implement gradient-based attribution methods
# Create a GradientTape model for computing gradients
@tf.function
def get_gradients(inputs, target_idx):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs)
        # Get output for specific joint angle
        output = predictions[:, target_idx]
    return tape.gradient(output, inputs)

# Visualize gradients for each finger
print("Creating gradient attribution visualizations...")
for i, (finger_name, start_idx, end_idx) in enumerate(finger_groups):
    plt.figure(figsize=(15, 10))
    
    # Get sample
    sample_idx = 0
    sample = X_test[sample_idx:sample_idx+1]
    
    # Compute gradients for the first joint of this finger
    joint_idx = start_idx
    gradients = get_gradients(tf.convert_to_tensor(sample, dtype=tf.float32), joint_idx)
    gradients = gradients.numpy()[0]  # Convert to numpy and get first sample
    
    # Normalize gradients for visualization
    abs_gradients = np.abs(gradients)
    max_val = np.max(abs_gradients)
    if max_val > 0:
        normalized_gradients = abs_gradients / max_val
    else:
        normalized_gradients = abs_gradients
    
    # Plot heatmap of gradients
    plt.subplot(2, 1, 1)
    plt.imshow(normalized_gradients, aspect='auto', cmap='hot')
    plt.colorbar(label='Normalized Gradient Magnitude')
    plt.title(f'Gradient Attribution for {finger_name} Finger')
    plt.xlabel('EMG Channel')
    plt.ylabel('Time Step')
    
    # Plot input EMG signals
    plt.subplot(2, 1, 2)
    for j in range(sample.shape[2]):
        plt.plot(sample[0, :, j], label=f'EMG {j}')
    plt.title('Input EMG Signals')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'/home/ubuntu/emg_project/visualizations/xai/gradient_attribution_{finger_name}.png')
    plt.close()

# 4. Create correlation analysis between EMG features and predicted angles
print("Creating correlation analysis visualizations...")

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate correlation between EMG features and predicted angles
correlations = np.zeros((X_test.shape[2], y_pred.shape[1]))  # (n_emg_channels, n_angles)

# For each EMG channel and joint angle, calculate correlation across all samples
for i in range(X_test.shape[2]):  # For each EMG channel
    for j in range(y_pred.shape[1]):  # For each joint angle
        # Extract the mean EMG value for each sample
        emg_means = np.mean(X_test[:, :, i], axis=1)
        # Calculate correlation
        correlations[i, j] = np.corrcoef(emg_means, y_pred[:, j])[0, 1]

# Plot correlation heatmap
plt.figure(figsize=(15, 10))
plt.imshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.title('Correlation between EMG Channels and Predicted Joint Angles')
plt.xlabel('Joint Angle Index')
plt.ylabel('EMG Channel Index')

# Add finger group labels
for i, (finger_name, start_idx, end_idx) in enumerate(finger_groups):
    plt.text(start_idx + (end_idx - start_idx) / 2, -0.5, finger_name, 
             ha='center', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/ubuntu/emg_project/visualizations/xai/emg_angle_correlation.png')
plt.close()

# 5. Create feature importance visualization across different fingers
print("Creating feature importance visualization...")

# Calculate feature importance based on SHAP values
feature_importance = np.zeros((len(finger_groups), X_test.shape[2]))  # (n_fingers, n_emg_channels)

for i, (finger_name, start_idx, end_idx) in enumerate(finger_groups):
    # Use the first joint angle for each finger
    joint_idx = start_idx
    
    # Calculate mean absolute SHAP value for each feature
    feature_importance[i, :] = np.mean(np.abs(shap_values[joint_idx]), axis=0)

# Plot feature importance heatmap
plt.figure(figsize=(12, 8))
plt.imshow(feature_importance, cmap='YlOrRd')
plt.colorbar(label='Mean |SHAP Value|')
plt.title('EMG Channel Importance for Each Finger')
plt.xlabel('EMG Channel Index')
plt.ylabel('Finger')
plt.yticks(np.arange(len(finger_groups)), [fg[0] for fg in finger_groups])
plt.tight_layout()
plt.savefig('/home/ubuntu/emg_project/visualizations/xai/feature_importance_by_finger.png')
plt.close()

print("\nXAI methods implementation completed. Visualizations saved to /home/ubuntu/emg_project/visualizations/xai/")
