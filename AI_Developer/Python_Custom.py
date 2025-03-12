import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import matplotlib.pyplot as plt

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# 1. Load and Preprocess Data
# Load the .mat file (replace 'your_data_file.mat' with your actual file path)
data = scipy.io.loadmat('s1_full.mat')

# Extract EMG data (5x7 cell, each cell is 4000x8), but we'll use only 5 tasks
dsfilt_emg = data['dsfilt_emg'][:, :5]  # Take first 5 tasks: thumb, index, middle, ring, little

# Normalize EMG data and extract additional features
normalized_emg = np.zeros_like(dsfilt_emg, dtype=object)
raw_features = np.zeros_like(dsfilt_emg, dtype=object)
for i in range(5):  # 5 trials
    for j in range(5):  # 5 tasks
        emg = dsfilt_emg[i, j]  # 4000x8 matrix
        normalized_emg[i, j] = (emg - np.mean(emg, axis=0)) / np.std(emg, axis=0)
        # Compute raw features: mean, variance, RMS for each channel
        mean_vals = np.mean(emg, axis=0)
        var_vals = np.var(emg, axis=0)
        rms_vals = np.sqrt(np.mean(emg**2, axis=0))
        raw_features[i, j] = np.stack([mean_vals, var_vals, rms_vals], axis=0)  # Shape: (3, 8)

# Extract joint angles as labels (5x5 cell, each cell is 4000x14)
joint_angles = data['joint_angles'][:, :5]

# 2. Prepare Training and Testing Data
# Training: trials 1, 2, 3 (indices 0, 1, 2)
# Testing: trials 4, 5 (indices 3, 4)
X_train, X_test = [], []
raw_train, raw_test = [], []
Y_train, Y_test = [], []

for i in range(5):  # Trials
    for j in range(5):  # Tasks
        emg = np.expand_dims(normalized_emg[i, j], axis=-1)  # Shape: (4000, 8, 1)
        raw = raw_features[i, j]  # Shape: (3, 8)
        joint = joint_angles[i, j]  # Shape: (4000, 14)
        if i < 3:  # Trials 1, 2, 3
            X_train.append(emg)
            raw_train.append(raw)
            Y_train.append(joint)
        else:  # Trials 4, 5
            X_test.append(emg)
            raw_test.append(raw)
            Y_test.append(joint)

# Convert to numpy arrays
X_train = np.array(X_train)  # Shape: (15, 4000, 8, 1) -> 3 trials * 5 tasks
raw_train = np.array(raw_train)  # Shape: (15, 3, 8)
Y_train = np.array(Y_train)  # Shape: (15, 4000, 14)
X_test = np.array(X_test)  # Shape: (10, 4000, 8, 1) -> 2 trials * 5 tasks
raw_test = np.array(raw_test)  # Shape: (10, 3, 8)
Y_test = np.array(Y_test)  # Shape: (10, 4000, 14)

# 3. Define Advanced Model Architecture
# EMG input
emg_input = Input(shape=(4000, 8, 1), name='emg_input')

# Advanced CNN feature extraction
x = layers.Conv2D(32, (5, 1), padding='same', activation='relu', name='conv1')(emg_input)
x = layers.BatchNormalization(name='bn1')(x)
x = layers.MaxPooling2D((2, 1), name='pool1')(x)

x = layers.Conv2D(64, (5, 1), padding='same', activation='relu', name='conv2')(x)
x = layers.BatchNormalization(name='bn2')(x)
x = layers.MaxPooling2D((2, 1), name='pool2')(x)

x = layers.Conv2D(128, (5, 1), padding='same', activation='relu', name='conv3')(x)
x = layers.BatchNormalization(name='bn3')(x)
x = layers.MaxPooling2D((2, 1), name='pool3')(x)  # Shape: (500, 8, 128)

# Reshape for temporal modeling
x = layers.Reshape((500, 8 * 128))(x)  # Shape: (batch, 500, 1024)

# Bidirectional LSTM for temporal dependencies
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name='bilstm')(x)  # Shape: (batch, 500, 256)

# Multi-Head Self-Attention
attention = layers.MultiHeadAttention(num_heads=4, key_dim=64, name='multihead_attention')(x, x)  # Shape: (batch, 500, 256)
attention = layers.LayerNormalization(epsilon=1e-6, name='ln_attention')(attention)

# Global average pooling to reduce dimensionality
attention_pooled = layers.GlobalAveragePooling1D(name='attention_pool')(attention)  # Shape: (batch, 256)

# Raw features input
raw_input = Input(shape=(3, 8), name='raw_input')
raw_flat = layers.Flatten(name='flatten_raw')(raw_input)  # Shape: (batch, 24)

# Concatenate attention output and raw features
concat = layers.Concatenate(name='concat')([attention_pooled, raw_flat])  # Shape: (batch, 280)

# Dense layers with dropout for regression
x = layers.Dense(512, activation='relu', name='fc1')(concat)
x = layers.Dropout(0.3, name='dropout1')(x)
x = layers.Dense(256, activation='relu', name='fc2')(x)
x = layers.Dropout(0.3, name='dropout2')(x)
output = layers.Dense(4000 * 14, name='fc3')(x)
output = layers.Reshape((4000, 14), name='output')(output)  # Shape: (batch, 4000, 14)

# Define the model
model = Model(inputs=[emg_input, raw_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Print model summary
model.summary()

# 4. Train the Model
history = model.fit(
    [X_train, raw_train], Y_train,
    epochs=20,
    batch_size=4,
    validation_data=([X_test, raw_test], Y_test),
    verbose=1
)

# 5. Explainability: Visualize Attention Weights
# Extract attention weights for a test sample
sample_idx = 0
sample_emg = X_test[sample_idx:sample_idx+1]  # Shape: (1, 4000, 8, 1)
sample_raw = raw_test[sample_idx:sample_idx+1]  # Shape: (1, 3, 8)

# Sub-model to get attention output
attention_model = Model(inputs=model.input, outputs=model.get_layer('multihead_attention').output)
attention_output = attention_model.predict([sample_emg, sample_raw])[0]  # Shape: (500, 256)

# Average attention across heads and features for visualization
attention_weights = np.mean(attention_output, axis=-1)  # Shape: (500,)
attention_weights = np.mean(attention_weights, axis=0)  # Scalar (simplified)

# For simplicity, visualize channel-wise contribution from raw features
raw_contribution = model.get_layer('fc1').get_weights()[0][256:, :]  # Weights for raw features
raw_contribution = np.mean(np.abs(raw_contribution), axis=1)  # Shape: (24,)
raw_contribution = raw_contribution.reshape(3, 8)  # Shape: (3, 8) for mean, var, RMS

plt.figure(figsize=(10, 6))
for i, feature in enumerate(['Mean', 'Variance', 'RMS']):
    plt.bar(np.arange(8) + i*0.2, raw_contribution[i], width=0.2, label=feature)
plt.xticks(np.arange(8) + 0.3, ['APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'])
plt.xlabel('Muscles')
plt.ylabel('Feature Contribution')
plt.title('Raw Feature Contributions to Prediction')
plt.legend()
plt.show()

# 6. Visualize Predictions vs Actual
predictions = model.predict([sample_emg, sample_raw])[0]  # Shape: (4000, 14)

plt.figure(figsize=(12, 6))
plt.plot(predictions, label='Predicted', linewidth=1.5)
plt.plot(Y_test[sample_idx], linestyle='--', label='Actual', linewidth=1)
plt.xlabel('Time')
plt.ylabel('Joint Angles')
plt.title('Predicted vs Actual Joint Angles (Test Sample)')
plt.legend()
plt.show()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Training and Validation MAE')
plt.legend()
plt.tight_layout()
plt.show()