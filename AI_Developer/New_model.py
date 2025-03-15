import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import scipy.io
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. Data Loading and Preprocessing ---

def load_mat_data(file_path):
    """Loads data from the .mat file and preprocesses it.

    Args:
        file_path (str): Path to the .mat file.

    Returns:
        tuple: (emg_data, joint_angles_data), both are lists of numpy arrays,
               where each element in the list represents a trial, and each trial
               contains combined task data.
    """
    mat_data = scipy.io.loadmat(file_path)
    emg_data = []
    joint_angles_data = []

    for trial in range(5):  # Iterate through trials (rows)
        trial_emg = []
        trial_angles = []
        for task in range(7):  # Iterate through tasks (columns)
            trial_emg.append(mat_data['dsfilt_emg'][trial, task])
            trial_angles.append(mat_data['joint_angles'][trial, task])
        # Concatenate task data for each trial
        emg_data.append(np.concatenate(trial_emg, axis=0))
        joint_angles_data.append(np.concatenate(trial_angles, axis=0))
    return emg_data, joint_angles_data

class EMGDataset(Dataset):
    def __init__(self, emg_data, joint_angles_data, sequence_length):
        self.emg_data = emg_data
        self.joint_angles_data = joint_angles_data
        self.sequence_length = sequence_length
        self.scalers_emg = []
        self.scalers_angles = []


        # Scale and store scalers
        self.emg_data_scaled = []
        self.joint_angles_data_scaled = []
        for i in range(len(self.emg_data)):
            scaler_emg = StandardScaler()
            scaler_angles = StandardScaler()
            # Check for empty arrays before scaling
            if self.emg_data[i].size > 0 and self.joint_angles_data[i].size > 0:
                emg_scaled = scaler_emg.fit_transform(self.emg_data[i])
                angles_scaled = scaler_angles.fit_transform(self.joint_angles_data[i])
                self.emg_data_scaled.append(emg_scaled)
                self.joint_angles_data_scaled.append(angles_scaled)
                self.scalers_emg.append(scaler_emg)
                self.scalers_angles.append(scaler_angles)
            else:
                # Handle empty arrays (e.g., skip, pad, or remove)
                print(f"Warning: Empty array found in trial {i}. Skipping scaling.")
                # Here, we'll append empty arrays.  You might want to handle differently.
                self.emg_data_scaled.append(np.array([]).reshape(0, self.emg_data[i].shape[1] if self.emg_data[i].size > 0 else 8)) #keep consistent number of features
                self.joint_angles_data_scaled.append(np.array([]).reshape(0, self.joint_angles_data[i].shape[1] if self.joint_angles_data[i].size>0 else 14)) # keep consitent number of features.
                self.scalers_emg.append(None)  # No scaler for empty data
                self.scalers_angles.append(None)


    def __len__(self):
        #This is tricky. We have multiple trials. We return combined length and handle indexing.
        total_len = 0
        for trial_emg in self.emg_data_scaled:
            total_len += max(0, len(trial_emg) - self.sequence_length + 1)  # Ensure non-negative
        return total_len

    def __getitem__(self, idx):
        # Determine which trial this index falls into.
        trial_idx = 0
        cumulative_len = 0
        for i, trial_emg in enumerate(self.emg_data_scaled):
            trial_len = max(0, len(trial_emg) - self.sequence_length + 1) # Ensure non-negative length.
            if idx < cumulative_len + trial_len:
                trial_idx = i
                idx_within_trial = idx - cumulative_len  # Adjust idx to be relative to the trial
                break
            cumulative_len += trial_len

        start_idx = idx_within_trial
        end_idx = start_idx + self.sequence_length

        #Handle cases where trial length might be shorter than sequence length.
        if len(self.emg_data_scaled[trial_idx]) == 0: # If the whole trial is empty
            emg_seq = np.zeros((self.sequence_length, 8))  # Or appropriate shape
            angles_seq = np.zeros((self.sequence_length, 14))
        elif end_idx > len(self.emg_data_scaled[trial_idx]):
            #Padding.
            emg_seq = np.pad(self.emg_data_scaled[trial_idx][start_idx:],((0,end_idx- len(self.emg_data_scaled[trial_idx])),(0,0)),'constant')
            angles_seq = np.pad(self.joint_angles_data_scaled[trial_idx][start_idx:],((0,end_idx- len(self.emg_data_scaled[trial_idx])),(0,0)),'constant')
        else:
            emg_seq = self.emg_data_scaled[trial_idx][start_idx:end_idx]
            angles_seq = self.joint_angles_data_scaled[trial_idx][start_idx:end_idx]

        return torch.tensor(emg_seq, dtype=torch.float32), torch.tensor(angles_seq, dtype=torch.float32)


    def get_scalers(self):
         return self.scalers_emg, self.scalers_angles


# --- 2. Model Definition (Transformer-LSTM) ---

class TransformerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_heads, dropout):
        super(TransformerLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=hidden_size*2, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)


        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):

        # Transformer Encoder
        transformer_out = self.transformer_encoder(x)  # (batch_size, seq_len, input_size)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(transformer_out)

        # Fully Connected
        output = self.fc(lstm_out)
        return output


# --- 3. Training Setup ---
def create_data_loaders(emg_data, joint_angles_data, sequence_length, batch_size, train_ratio=0.8):
    # Combine all trials for splitting, then separate
    all_emg = np.concatenate(emg_data, axis=0)
    all_angles = np.concatenate(joint_angles_data, axis=0)

    emg_train, emg_test, angles_train, angles_test = train_test_split(
        all_emg, all_angles, test_size=1 - train_ratio, random_state=42
    )
    # Split back into trials (this is a simplification; a more robust approach would track original trial indices)
    train_emg_trials, test_emg_trials = [], []
    train_angles_trials, test_angles_trials = [], []

    trial_lengths = [len(trial) for trial in emg_data]  # Original trial lengths.
    current_idx = 0
    for length in trial_lengths:
      train_emg_trials.append(emg_train[current_idx : current_idx + int(length*train_ratio)])
      #Correct the test set indexing to get the remaining part of each trial.
      test_emg_trials.append(emg_test[current_idx - int(length*train_ratio) : current_idx - int(length * train_ratio) + (length - int(length*train_ratio))])

      train_angles_trials.append(angles_train[current_idx : current_idx+int(length*train_ratio)])
      test_angles_trials.append(angles_test[current_idx - int(length*train_ratio) : current_idx - int(length*train_ratio) + (length-int(length*train_ratio))])

      current_idx += int(length*train_ratio)  #Increment by the *train* portion


    train_dataset = EMGDataset(train_emg_trials, train_angles_trials, sequence_length)
    test_dataset = EMGDataset(test_emg_trials, test_angles_trials, sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader, train_dataset.get_scalers(), test_dataset.get_scalers()


# --- 4. Training Loop ---

def train(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for emg_seq, angles_seq in train_loader:
            emg_seq, angles_seq = emg_seq.to(device), angles_seq.to(device)

            optimizer.zero_grad()
            outputs = model(emg_seq)
            loss = criterion(outputs, angles_seq)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * emg_seq.size(0)  # Accumulate loss for the entire batch

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)


        # Evaluate on the test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for emg_seq, angles_seq in test_loader:
                emg_seq, angles_seq = emg_seq.to(device), angles_seq.to(device)
                outputs = model(emg_seq)
                loss = criterion(outputs, angles_seq)
                test_loss += loss.item() * emg_seq.size(0)

        test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(test_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    return train_losses, test_losses


# --- 5. Main Execution ---

if __name__ == '__main__':
    file_path = 's1_full.mat'  # Replace with the actual path to your .mat file
    emg_data, joint_angles_data = load_mat_data(file_path)

    # Hyperparameters
    input_size = 8  # Number of EMG channels
    hidden_size = 64
    num_layers = 2
    output_size = 14  # Number of joint angles
    num_heads = 4 # Number of heads for the Transformer
    dropout = 0.1
    sequence_length = 50 # Window size for creating sequences
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, test_loader, train_scalers, test_scalers = create_data_loaders(emg_data, joint_angles_data, sequence_length, batch_size)

    # Initialize the model
    model = TransformerLSTM(input_size, hidden_size, num_layers, output_size, num_heads, dropout).to(device)


    # Loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_losses, test_losses = train(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)

    # Plot training and testing loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Testing Loss')
    plt.show()

    # --- 6.  Visualization and Inverse Transform ---
    # Choose a trial and task for demonstration
    trial_idx = 0  # Example: first trial
    scalers_emg, scalers_angles = train_scalers
    scaler_emg = scalers_emg[trial_idx]
    scaler_angles = scalers_angles[trial_idx]


    model.eval()
    with torch.no_grad():
      # Get *one* batch from the test loader, and then pick the trial we want
      emg_seq, angles_seq = next(iter(test_loader))  # Get a single batch
      emg_seq = emg_seq.to(device)
      predicted_angles_seq = model(emg_seq)


      # Inverse transform *one* sequence from the batch.
      predicted_angles = predicted_angles_seq[0].cpu().numpy()  # First sequence in the batch
      true_angles = angles_seq[0].cpu().numpy()

      if scaler_angles is not None: # Handle cases where scaler might be None (empty trial)
          predicted_angles = scaler_angles.inverse_transform(predicted_angles)
          true_angles = scaler_angles.inverse_transform(true_angles)
      else:
        print("Scaler is None. Cannot inverse transform. Plotting scaled data.")



    # Plotting a few joint angles for visualization
    plt.figure(figsize=(12, 6))
    for i in range(3):  # Plot the first 3 joint angles
        plt.subplot(3, 1, i+1)
        plt.plot(true_angles[:, i], label='True Angle')
        plt.plot(predicted_angles[:, i], label='Predicted Angle')
        plt.ylabel(f'Angle {i+1}')
        plt.legend()
    plt.xlabel('Time Steps')
    plt.suptitle(f'True vs. Predicted Joint Angles (Trial {trial_idx+1})')
    plt.tight_layout()
    plt.show()



# --- 7. Model Architecture Visualization (using torchviz) ---

    from torchviz import make_dot
    # Create a dummy input tensor
    dummy_input = torch.randn(1, sequence_length, input_size).to(device)  # Batch size 1

    # Get the output from the model
    output = model(dummy_input)
    # Create the graph
    graph = make_dot(output, params=dict(model.named_parameters()))
    graph.render("transformer_lstm_architecture", format="png", cleanup=True)  # Saves a PNG image
    print("Model architecture visualization saved as transformer_lstm_architecture.png")