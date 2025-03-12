import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# --- Data Preparation Function ---
def create_sequences(data, seq_length=100, stride=1):
    """Create overlapping sequences from EMG data."""
    X, y = [], []
    for i in range(0, len(data) - seq_length + 1, stride):
        X.append(data[i:i + seq_length, :-14])  # EMG (8 columns)
        y.append(data[i + seq_length - 1, -14:])  # Last angles in window (14 columns)
    return np.array(X), np.array(y)

# --- Load and Preprocess Data ---
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

emg_cols = ['EMG_APL', 'EMG_FCR', 'EMG_FDS', 'EMG_FDP', 'EMG_ED', 'EMG_EI', 'EMG_ECU', 'EMG_ECR']
angle_cols = ['Thumb1', 'Thumb2', 'Index1', 'Index2', 'Index3', 'Middle1', 'Middle2', 'Middle3',
              'Ring1', 'Ring2', 'Ring3', 'Little1', 'Little2', 'Little3']
all_cols = emg_cols + angle_cols

# Combine EMG and angles into a single array for sequence creation
train_data = train_df[all_cols].values  # Shape: (84000, 22)
test_data = test_df[all_cols].values    # Shape: (56000, 22)

# Create sequences
seq_length = 200
X_train, y_train = create_sequences(train_data, seq_length=seq_length)  # X: (n_samples, 100, 8), y: (n_samples, 14)
X_test, y_test = create_sequences(test_data, seq_length=seq_length)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Scale EMG data (not angles, to keep them interpretable)
scaler = StandardScaler()
X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])  # (n_samples * 100, 8)
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)

# --- Attention Layer Definition ---
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
    
    def call(self, inputs):
        # Alignment scores: e = tanh(W * inputs + b)
        e = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)  # Shape: (batch, timesteps, 1)
        # Attention weights: softmax(e)
        alpha = tf.nn.softmax(e, axis=1)  # Shape: (batch, timesteps, 1)
        # Context vector: sum(inputs * alpha)
        context = inputs * alpha  # Shape: (batch, timesteps, features)
        context = tf.reduce_sum(context, axis=1)  # Shape: (batch, features)
        return context

# --- Build Deep Attention Model ---
def build_model(input_shape=(200 k,l, 8), output_dim=14):
    inputs = layers.Input(shape=input_shape)
    
    # CNN layers for feature extraction
    x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Attention mechanism
    x = AttentionLayer()(x)  # Shape: (batch, 32)
    
    # Dense layers for regression
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(output_dim)(x)  # 14 joint angles
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# --- Train Model ---
model = build_model()
model.summary()

history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=64, 
                    validation_split=0.2, verbose=1)

# --- Evaluate Model ---
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
r2 = r2_score(y_test, y_pred, multioutput='raw_values')

print("\nEvaluation Metrics:")
for i, angle in enumerate(angle_cols):
    print(f"{angle}: MSE = {mse[i]:.4f}, R² = {r2[i]:.4f}")
print(f"\nAverage MSE: {np.mean(mse):.4f}")
print(f"Average R²: {np.mean(r2):.4f}")

# --- Plot Training History ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.show()

# --- Visualize Predictions (Example: First 1000 Test Samples) ---
plt.figure(figsize=(12, 6))
time = np.arange(min(1000, len(y_test)))
for i, angle in enumerate(['Thumb1', 'Index1', 'Middle1']):
    plt.subplot(3, 1, i+1)
    plt.plot(time, y_test[:1000, angle_cols.index(angle)], label='True', color='blue')
    plt.plot(time, y_pred[:1000, angle_cols.index(angle)], label='Predicted', color='red', linestyle='--')
    plt.title(f'{angle} (Test Data)')
    plt.xlabel('Time (samples)')
    plt.ylabel('Angle (degrees)')
    plt.legend()
plt.tight_layout()
plt.show()