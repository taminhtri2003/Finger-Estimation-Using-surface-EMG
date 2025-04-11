import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# Create output directories
os.makedirs('/home/ubuntu/emg_project/models', exist_ok=True)
os.makedirs('/home/ubuntu/emg_project/visualizations/model_training', exist_ok=True)

# Load preprocessed data
X_train = np.load('/home/ubuntu/emg_project/data/processed/X_train.npy')
y_train = np.load('/home/ubuntu/emg_project/data/processed/y_train.npy')
X_val = np.load('/home/ubuntu/emg_project/data/processed/X_val.npy')
y_val = np.load('/home/ubuntu/emg_project/data/processed/y_val.npy')
X_test = np.load('/home/ubuntu/emg_project/data/processed/X_test.npy')
y_test = np.load('/home/ubuntu/emg_project/data/processed/y_test.npy')

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")

# Define model parameters
input_shape = X_train.shape[1:]  # (time_steps, features)
output_shape = y_train.shape[1]  # number of joint angles
num_heads = 5  # One head for each finger
embedding_dim = 32  # Dimension of the embedding
dropout_rate = 0.2

# Define the multi-head attention model with 5 heads (one for each finger)
def create_multi_head_attention_model(input_shape, output_shape, num_heads, embedding_dim, dropout_rate):
    """
    Create a multi-head attention model for EMG to joint angle prediction
    
    Args:
        input_shape: Shape of input data (time_steps, features)
        output_shape: Number of output angles
        num_heads: Number of attention heads (5 for 5 fingers)
        embedding_dim: Dimension of the embedding
        dropout_rate: Dropout rate for regularization
        
    Returns:
        model: Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Project input to higher dimension for attention
    x = Dense(embedding_dim, activation='relu')(inputs)
    
    # Create positional encoding manually
    seq_length = input_shape[0]
    pos_encoding = np.zeros((1, seq_length, embedding_dim))
    for pos in range(seq_length):
        for i in range(0, embedding_dim, 2):
            pos_encoding[0, pos, i] = np.sin(pos / (10000 ** (i / embedding_dim)))
            if i + 1 < embedding_dim:
                pos_encoding[0, pos, i + 1] = np.cos(pos / (10000 ** (i / embedding_dim)))
    
    # Add positional encoding as a constant tensor
    pos_encoding_tensor = tf.constant(pos_encoding, dtype=tf.float32)
    x = x + pos_encoding_tensor
    
    # Multi-head attention layer
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embedding_dim // num_heads,
        dropout=dropout_rate
    )(x, x)
    
    # Skip connection and layer normalization
    x = LayerNormalization(epsilon=1e-6)(x + attention_output)
    
    # Feed-forward network
    ffn = Dense(embedding_dim * 2, activation='relu')(x)
    ffn = Dropout(dropout_rate)(ffn)
    ffn = Dense(embedding_dim, activation='relu')(ffn)
    
    # Skip connection and layer normalization
    x = LayerNormalization(epsilon=1e-6)(x + ffn)
    
    # Global average pooling
    x = GlobalAveragePooling1D()(x)
    
    # Output layers - one for each finger group
    # We'll create 5 separate output heads, one for each finger
    
    # Thumb angles (2 angles)
    thumb_output = Dense(32, activation='relu')(x)
    thumb_output = Dense(2, name='thumb_output')(thumb_output)
    
    # Index finger angles (3 angles)
    index_output = Dense(32, activation='relu')(x)
    index_output = Dense(3, name='index_output')(index_output)
    
    # Middle finger angles (3 angles)
    middle_output = Dense(32, activation='relu')(x)
    middle_output = Dense(3, name='middle_output')(middle_output)
    
    # Ring finger angles (3 angles)
    ring_output = Dense(32, activation='relu')(x)
    ring_output = Dense(3, name='ring_output')(ring_output)
    
    # Little finger angles (3 angles)
    little_output = Dense(32, activation='relu')(x)
    little_output = Dense(3, name='little_output')(little_output)
    
    # Concatenate all outputs
    outputs = Concatenate()(
        [thumb_output, index_output, middle_output, ring_output, little_output]
    )
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# Create and compile the model
model = create_multi_head_attention_model(
    input_shape=input_shape,
    output_shape=output_shape,
    num_heads=num_heads,
    embedding_dim=embedding_dim,
    dropout_rate=dropout_rate
)

# Print model summary
model.summary()

# Define callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(
        filepath='/home/ubuntu/emg_project/models/multi_head_attention_model.keras',
        monitor='val_loss',
        save_best_only=True
    ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Save the model with correct extension
model.save('/home/ubuntu/emg_project/models/multi_head_attention_model.keras')

# Plot training history
plt.figure(figsize=(12, 5))

# Plot training & validation loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.grid(True)

# Plot training & validation mean absolute error
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model Mean Absolute Error')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.savefig('/home/ubuntu/emg_project/visualizations/model_training/training_history.png')
plt.close()

# Evaluate the model on test data
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Make predictions on test data
y_pred = model.predict(X_test)

# Plot predictions vs actual for each finger group
finger_groups = [
    ('Thumb', 0, 2),
    ('Index', 2, 5),
    ('Middle', 5, 8),
    ('Ring', 8, 11),
    ('Little', 11, 14)
]

plt.figure(figsize=(15, 12))

for i, (finger_name, start_idx, end_idx) in enumerate(finger_groups):
    plt.subplot(len(finger_groups), 1, i+1)
    
    # Plot actual vs predicted for the first joint of each finger
    joint_idx = start_idx
    plt.plot(y_test[:100, joint_idx], 'b-', label=f'Actual {finger_name}')
    plt.plot(y_pred[:100, joint_idx], 'r-', label=f'Predicted {finger_name}')
    
    plt.title(f'{finger_name} Finger - Joint Angle Prediction')
    plt.ylabel('Normalized Angle')
    plt.xlabel('Sample')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig('/home/ubuntu/emg_project/visualizations/model_training/prediction_results.png')
plt.close()

print("\nModel training completed. Results saved to /home/ubuntu/emg_project/visualizations/model_training/")
