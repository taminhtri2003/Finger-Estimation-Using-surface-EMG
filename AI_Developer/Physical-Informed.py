# main_script.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Configuration ---
CONFIG = {
    "mat_file_path": "s1_full.mat",  # Replace with your .mat file path
    "window_size": 100,  # Input window size for EMG data
    "batch_size": 64,
    "learning_rate": 0.001, 
    "num_epochs": 50,    # Adjust as needed
    "num_emg_channels": 8,
    "num_joint_angles": 14,
    "num_muscle_forces": 0, # MODIFIED: No muscle forces to be predicted
    "cnn_out_channels": 128,
    "fc_hidden_nodes": 128,
    "dropout_rate": 0.3,
    "include_time_feature": True, 
    "lambda_L_theta": 1.0,  # Weight for joint angle loss
    "lambda_L_F": 0.0,      # MODIFIED: No force loss
    "lambda_L_P": 0.0,      # MODIFIED: No physics loss as it depends on forces
    "test_size": 0.2,
    "random_seed": 42
}

# --- 1. Data Loading and Preprocessing ---

def load_and_prepare_data(mat_file_path, num_emg, num_angles, include_time_feature=False):
    """
    Loads data from the .mat file, concatenates trials and tasks,
    and returns raw EMG, angles.
    """
    try:
        mat_data = loadmat(mat_file_path)
    except FileNotFoundError:
        print(f"Error: MAT file not found at {mat_file_path}")
        print("Please ensure the file path is correct and the file exists.")
        print("For now, returning dummy data to allow script structure to run.")
        dummy_emg = np.random.rand(1000, num_emg + (1 if include_time_feature else 0))
        dummy_angles = np.random.rand(1000, num_angles)
        return dummy_emg, dummy_angles

    emg_cells = mat_data.get('dsfilt_emg')
    angle_cells = mat_data.get('joint_angles')

    if emg_cells is None or angle_cells is None:
        raise ValueError("Variables 'dsfilt_emg' or 'joint_angles' not found in MAT file.")

    all_emg_data = []
    all_angle_data = []

    for i in range(emg_cells.shape[0]):  # Trials
        for j in range(emg_cells.shape[1]):  # Tasks
            trial_task_emg = emg_cells[i, j]
            trial_task_angles = angle_cells[i, j]

            if isinstance(trial_task_emg, np.ndarray) and trial_task_emg.shape[0] > 0 \
               and isinstance(trial_task_angles, np.ndarray) and trial_task_angles.shape[0] > 0 \
               and trial_task_emg.shape[0] == trial_task_angles.shape[0]:
                
                if trial_task_emg.shape[1] != num_emg:
                    print(f"Warning: EMG data in cell ({i},{j}) has {trial_task_emg.shape[1]} channels, expected {num_emg}. Skipping.")
                    continue
                if trial_task_angles.shape[1] != num_angles:
                    print(f"Warning: Angle data in cell ({i},{j}) has {trial_task_angles.shape[1]} angles, expected {num_angles}. Skipping.")
                    continue
                
                all_emg_data.append(trial_task_emg)
                all_angle_data.append(trial_task_angles)
            else:
                print(f"Skipping cell ({i},{j}) due to empty data or mismatched lengths.")


    if not all_emg_data or not all_angle_data:
        raise ValueError("No valid data extracted from MAT file cells.")

    emg_full = np.concatenate(all_emg_data, axis=0)
    angles_full = np.concatenate(all_angle_data, axis=0)
    
    if include_time_feature:
        time_feature_vec = np.linspace(0, 1, emg_full.shape[0]).reshape(-1, 1)
        emg_full = np.concatenate((emg_full, time_feature_vec), axis=1)

    return emg_full, angles_full

class EMGDataset(Dataset):
    def __init__(self, emg_data_2d, angle_data_2d, window_size): # MODIFIED: removed force_data_2d
        """
        Initializes the dataset.
        Args:
            emg_data_2d (np.ndarray): 2D numpy array of EMG data (samples, features).
            angle_data_2d (np.ndarray): 2D numpy array of angle data (samples, num_angles).
            window_size (int): The size of the input window.
        """
        self.emg_data = emg_data_2d
        self.angle_data = angle_data_2d
        self.window_size = window_size

        self.X = [] 
        self.y_angles = []

        for i in range(len(self.emg_data) - self.window_size + 1): 
            self.X.append(self.emg_data[i:i + self.window_size, :])
            self.y_angles.append(self.angle_data[i + self.window_size - 1, :]) 


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample_X = torch.FloatTensor(self.X[idx]) 
        target_angles = torch.FloatTensor(self.y_angles[idx])
        # MODIFIED: Only return EMG window and target angles
        return sample_X.permute(1, 0), target_angles


# --- 2. Model Architecture (CNN) ---
class MusculoskeletalCNN(nn.Module):
    # MODIFIED: num_forces is effectively 0 if not used for output
    def __init__(self, input_channels, num_angles, cnn_out_channels, fc_hidden_nodes, dropout_rate):
        super(MusculoskeletalCNN, self).__init__()
        self.num_angles = num_angles

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, 
                      out_channels=cnn_out_channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_out_channels),
            nn.Dropout(dropout_rate)
        )
        
        self.pool = nn.AdaptiveMaxPool1d(1) 
        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(cnn_out_channels, fc_hidden_nodes), 
            nn.ReLU(),
            nn.BatchNorm1d(fc_hidden_nodes),
            nn.Dropout(dropout_rate)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(fc_hidden_nodes, fc_hidden_nodes),
            nn.ReLU(),
            nn.BatchNorm1d(fc_hidden_nodes),
            nn.Dropout(dropout_rate)
        )

        # MODIFIED: Regression head only outputs num_angles
        self.regression_head = nn.Linear(fc_hidden_nodes, num_angles)

    def forward(self, x):
        x = self.conv_block(x)   
        x = self.pool(x)         
        x = self.flatten(x)      
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        # MODIFIED: Output is only predicted angles
        pred_angles = self.regression_head(x) 
        
        return pred_angles # MODIFIED: Only return angles

# --- 3. Loss Functions ---
# MSE Loss is standard: nn.MSELoss()

# physics_based_loss and get_physics_parameters are kept for potential future use
# but will not be called if CONFIG["lambda_L_P"] is 0.
def calculate_derivatives(theta_predicted_sequence, dt=1.0/4000.0):
    if theta_predicted_sequence.dim() < 2 or theta_predicted_sequence.shape[1] < 2 :
        return torch.zeros_like(theta_predicted_sequence), torch.zeros_like(theta_predicted_sequence)
    theta_dot_list = torch.gradient(theta_predicted_sequence, spacing=(dt,), dim=1)
    theta_dot = theta_dot_list[0] 
    if theta_predicted_sequence.shape[1] < 3: 
        return theta_dot, torch.zeros_like(theta_dot)
    theta_ddot_list = torch.gradient(theta_dot, spacing=(dt,), dim=1)
    theta_ddot = theta_ddot_list[0]
    return theta_dot, theta_ddot

def get_physics_parameters(theta_current_step, device='cpu'):
    batch_size, num_angles = theta_current_step.shape
    M = torch.stack([torch.eye(num_angles) for _ in range(batch_size)]).to(device) 
    C_vec = torch.zeros(batch_size, num_angles).to(device) 
    G_vec = torch.zeros(batch_size, num_angles).to(device) 
    # Moment arms are irrelevant if not predicting forces for physics loss
    moment_arms_matrix = torch.zeros(num_angles, CONFIG.get("num_muscle_forces", 0)).to(device)
    return M, C_vec, G_vec, moment_arms_matrix

def physics_based_loss(pred_angles_current_step, pred_forces_current_step, device='cpu'):
    # This function will not be called if lambda_L_P is 0
    if CONFIG["num_muscle_forces"] == 0 or pred_forces_current_step is None:
        return torch.tensor(0.0).to(device) # No physics loss if no forces

    dot_theta_pred = torch.zeros_like(pred_angles_current_step)
    ddot_theta_pred = torch.zeros_like(pred_angles_current_step)
    M, C_vec, G_vec, moment_arms = get_physics_parameters(pred_angles_current_step, device=device)
    term_M_ddot_theta = torch.bmm(M, ddot_theta_pred.unsqueeze(-1)).squeeze(-1) 
    dynamics_lhs = term_M_ddot_theta + C_vec + G_vec
    tau_muscles = torch.matmul(pred_forces_current_step, moment_arms.T) 
    physics_residual = dynamics_lhs - tau_muscles
    loss_p = torch.mean(physics_residual**2)
    return loss_p

# --- 4. Training Loop ---
def train_model(model, train_loader, optimizer, criterion_mse, device):
    model.train()
    epoch_loss = 0
    epoch_loss_theta = 0
    # MODIFIED: Removed epoch_loss_F and epoch_loss_P as they are not used

    for batch_idx, data_pack in enumerate(train_loader):
        # MODIFIED: Unpack only EMG and angles
        emg_batch, angle_batch_true = data_pack
            
        emg_batch = emg_batch.to(device)
        angle_batch_true = angle_batch_true.to(device)

        optimizer.zero_grad()
        
        # MODIFIED: Model only returns pred_angles
        pred_angles = model(emg_batch) 
        
        loss_theta = criterion_mse(pred_angles, angle_batch_true)
        
        # MODIFIED: Only theta loss is used
        total_loss = loss_theta 
        
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item()
        epoch_loss_theta += loss_theta.item()

    num_batches = len(train_loader)
    avg_loss = epoch_loss / num_batches
    avg_loss_theta = epoch_loss_theta / num_batches
    
    # MODIFIED: Return only total and theta loss
    return avg_loss, avg_loss_theta

# --- 5. Evaluation ---
def evaluate_model(model, test_loader, criterion_mse, device):
    model.eval()
    total_loss_theta = 0
    all_pred_angles = []
    all_true_angles = []

    with torch.no_grad():
        for data_pack in test_loader:
            # MODIFIED: Unpack only EMG and angles
            emg_batch, angle_batch_true = data_pack
            
            emg_batch = emg_batch.to(device)
            angle_batch_true = angle_batch_true.to(device)
            
            # MODIFIED: Model only returns pred_angles
            pred_angles = model(emg_batch) 
            
            loss_theta = criterion_mse(pred_angles, angle_batch_true)
            total_loss_theta += loss_theta.item()
            
            all_pred_angles.append(pred_angles.cpu().numpy())
            all_true_angles.append(angle_batch_true.cpu().numpy())

    avg_loss_theta = total_loss_theta / len(test_loader)
    
    all_pred_angles = np.concatenate(all_pred_angles, axis=0)
    all_true_angles = np.concatenate(all_true_angles, axis=0)
    
    num_angles_eval = all_pred_angles.shape[1]
    rmses = []
    ccs = []
    for i in range(num_angles_eval):
        pred_col = all_pred_angles[:, i]
        true_col = all_true_angles[:, i]
        
        if np.any(np.isnan(pred_col)) or np.any(np.isinf(pred_col)) or \
           np.any(np.isnan(true_col)) or np.any(np.isinf(true_col)) or \
           np.std(pred_col) == 0 or np.std(true_col) == 0: 
            rmse = np.sqrt(np.mean((pred_col - true_col)**2)) if not (np.any(np.isnan(pred_col)) or np.any(np.isnan(true_col))) else float('nan')
            cc = float('nan')
        else:
            rmse = np.sqrt(np.mean((pred_col - true_col)**2))
            cc = np.corrcoef(pred_col, true_col)[0, 1]
        
        rmses.append(rmse)
        ccs.append(cc)
        
    return avg_loss_theta, rmses, ccs


# --- Main Script Execution ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading and preparing data...")
    
    try:
        raw_emg, raw_angles = load_and_prepare_data(
            CONFIG["mat_file_path"],
            CONFIG["num_emg_channels"],
            CONFIG["num_joint_angles"],
            CONFIG["include_time_feature"]
        )
    except Exception as e:
        print(f"Failed to load or prepare data: {e}")
        exit()

    actual_emg_channels_in_data = raw_emg.shape[1] 

    scaler_emg = StandardScaler()
    scaled_emg_2d = scaler_emg.fit_transform(raw_emg)

    scaler_angles = StandardScaler()
    scaled_angles_2d = scaler_angles.fit_transform(raw_angles)
    
    # MODIFIED: Removed force data handling
    emg_train_2d, emg_test_2d, angles_train_2d, angles_test_2d = train_test_split(
        scaled_emg_2d, scaled_angles_2d, test_size=CONFIG["test_size"], 
        random_state=CONFIG["random_seed"], shuffle=False 
    )

    # MODIFIED: Pass only emg and angle data to dataset
    train_dataset = EMGDataset(emg_train_2d, angles_train_2d, CONFIG["window_size"])
    test_dataset = EMGDataset(emg_test_2d, angles_test_2d, CONFIG["window_size"])

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("Error: Training or testing dataset is empty. Check data and window size.")
        exit()

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # MODIFIED: Initialize model without num_forces for output
    model = MusculoskeletalCNN(
        input_channels=actual_emg_channels_in_data, 
        num_angles=CONFIG["num_joint_angles"],
        # num_forces argument removed from constructor call if not used for output
        cnn_out_channels=CONFIG["cnn_out_channels"],
        fc_hidden_nodes=CONFIG["fc_hidden_nodes"],
        dropout_rate=CONFIG["dropout_rate"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion_mse = nn.MSELoss()

    print("Starting training...")
    for epoch in range(CONFIG["num_epochs"]):
        # MODIFIED: train_model returns only total and theta loss
        avg_loss, avg_loss_theta = train_model(
            model, train_loader, optimizer, criterion_mse, device
        )
        # MODIFIED: Print only relevant losses
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}, Total Loss (Angle MSE): {avg_loss:.4f}")


    print("Training finished.")
    print("Evaluating model...")
    test_loss_theta, rmses, ccs = evaluate_model(model, test_loader, criterion_mse, device)
    print(f"Test MSE (Angles): {test_loss_theta:.4f}")
    for i in range(len(rmses)):
        print(f"Angle {i+1} - RMSE: {rmses[i]:.4f}, CC: {ccs[i]:.4f}")

    print("Script finished.")
    if CONFIG["lambda_L_P"] > 0 or CONFIG["lambda_L_F"] > 0 :
         print("\nWARNING: lambda_L_P or lambda_L_F are > 0 in CONFIG, but this script version does not use them.")
    print("This version predicts JOINT ANGLES ONLY, without muscle force prediction or physics-based loss.")

