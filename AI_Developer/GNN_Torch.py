import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    mat_data = scipy.io.loadmat(file_path)
    emg_data = mat_data['dsfilt_emg']  # [5, 7] cell, each [4000, 8]
    joint_angles = mat_data['joint_angles']  # [5, 7] cell, each [4000, 14]
    
    # Convert cell arrays to numpy arrays
    X = np.array([np.vstack(emg_data[i, j]) for i in range(5) for j in range(7)])  # [35, 4000, 8]
    y = np.array([np.vstack(joint_angles[i, j]) for i in range(5) for j in range(7)])  # [35, 4000, 14]
    
    # Normalize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_normalized = scaler_X.fit_transform(X.reshape(-1, 8)).reshape(X.shape)  # Normalize across all samples
    y_normalized = scaler_y.fit_transform(y.reshape(-1, 14)).reshape(y.shape)
    
    return X_normalized, y_normalized, scaler_X, scaler_y

# 2. Create Graph Structure
def create_graph_data(X, y):
    num_nodes = 22  # 8 EMG nodes + 14 joint angle nodes
    # Define edges: each EMG node connects to all joint angle nodes
    edge_index = []
    for i in range(8):  # EMG nodes 0-7
        for j in range(8, 22):  # Joint angle nodes 8-21
            edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, 112]
    
    data_list = []
    for i in range(X.shape[0]):  # 35 trial/tasks
        for t in range(X[i].shape[0]):  # 4000 time steps
            x = torch.zeros(num_nodes, 1, dtype=torch.float)  # [22, 1]
            x[0:8, 0] = torch.tensor(X[i][t, :], dtype=torch.float)  # EMG features
            y_tensor = torch.tensor(y[i][t, :], dtype=torch.float)  # [14]
            data = Data(x=x, edge_index=edge_index, y=y_tensor)
            data_list.append(data)
    return data_list

# 3. GNN Model Definition
class EMG_GNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super(EMG_GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x[8:].squeeze(-1)  # Output for nodes 8-21, shape [14] or [batch_size, 14]

# 4. Training Function
def train_model(model, train_loader, optimizer, criterion, epochs=50):
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)  # [batch_size, 14]
            loss = criterion(out, batch.y)  # batch.y is [batch_size, 14]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
    
    return losses

# 5. Evaluation Function
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch)
            y_true.append(batch.y.numpy())
            y_pred.append(pred.numpy())
    y_true = np.concatenate(y_true)  # [num_test_samples, 14]
    y_pred = np.concatenate(y_pred)  # [num_test_samples, 14]
    return y_true, y_pred

# 6. Visualization Functions
def plot_training_loss(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predicted_vs_actual(y_true, y_pred, joint_idx=0):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true[:, joint_idx], name='Actual', mode='lines'))
    fig.add_trace(go.Scatter(y=y_pred[:, joint_idx], name='Predicted', mode='lines'))
    fig.update_layout(
        title=f'Joint Angle {joint_idx+1} Prediction (e.g., Thumb 1)',
        xaxis_title='Time Step',
        yaxis_title='Angle (normalized)',
        template='plotly_dark'
    )
    fig.show()

def plot_3d_trajectory(y_true, y_pred, joint_indices=[0, 2, 4]):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=y_true[:, joint_indices[0]], 
        y=y_true[:, joint_indices[1]], 
        z=y_true[:, joint_indices[2]], 
        name='Actual',
        mode='lines'
    ))
    fig.add_trace(go.Scatter3d(
        x=y_pred[:, joint_indices[0]], 
        y=y_pred[:, joint_indices[1]], 
        z=y_pred[:, joint_indices[2]], 
        name='Predicted',
        mode='lines'
    ))
    fig.update_layout(
        title='3D Joint Angle Trajectory (Thumb 1, Index 1, Middle 1)',
        scene=dict(
            xaxis_title='Thumb 1',
            yaxis_title='Index 1',
            zaxis_title='Middle 1'
        ),
        template='plotly_dark'
    )
    fig.show()

# 7. Main Execution
def main():
    # Load and preprocess data
    file_path = 's4_full.mat'  # Replace with actual path
    X, y, scaler_X, scaler_y = load_and_preprocess_data(file_path)
    
    # Create graph data
    data_list = create_graph_data(X, y)  # 140,000 Data objects (35 * 4000)
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    
    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Initialize model
    model = EMG_GNN(input_dim=1, hidden_dim=64, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Train model
    losses = train_model(model, train_loader, optimizer, criterion, epochs=50)
    
    # Evaluate model
    y_true, y_pred = evaluate_model(model, test_loader)
    
    # Visualizations
    plot_training_loss(losses)
    plot_predicted_vs_actual(y_true, y_pred, joint_idx=0)  # Thumb 1 angle
    plot_3d_trajectory(y_true, y_pred, joint_indices=[0, 2, 4])  # Thumb 1, Index 1, Middle 1

if __name__ == "__main__":
    main()