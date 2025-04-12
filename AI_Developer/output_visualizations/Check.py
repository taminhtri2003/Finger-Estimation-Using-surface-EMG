import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
# Import this if you need to load .mat files
# from scipy.io import loadmat

# --- Simulation Parameters (Replace with your actual data dimensions) ---
NUM_TIME_STEPS = 200  # Number of time steps to animate (reduced for speed)
NUM_MUSCLES = 8
NUM_JOINTS = 14
MAX_SAMPLES_PER_CELL = 4000 # Max samples in the original data

# --- Data Loading Placeholder (Replace with your .mat loading) ---
# Example:
# data = loadmat('your_data_file.mat')
# # Assuming you select one specific trial and task cell, e.g., trial 0, task 0
# trial_idx, task_idx = 0, 0
# sEMG_data = data['dsfilt_emg'][trial_idx, task_idx][:NUM_TIME_STEPS, :] # Shape: (NUM_TIME_STEPS, NUM_MUSCLES)
# angle_data = data['joint_angles'][trial_idx, task_idx][:NUM_TIME_STEPS, :] # Shape: (NUM_TIME_STEPS, NUM_JOINTS)
# --- End Data Loading Placeholder ---

# --- Simulate Data (Remove this section when using real data) ---
print("Simulating data...")
# Simulate sEMG data (random walk style)
sEMG_data = np.cumsum(np.random.randn(NUM_TIME_STEPS, NUM_MUSCLES) * 0.1, axis=0)
sEMG_data = (sEMG_data - np.mean(sEMG_data, axis=0)) / np.std(sEMG_data, axis=0) # Normalize

# Simulate angle data (random walk style, loosely related to sEMG for demo)
angle_data = np.cumsum(np.random.randn(NUM_TIME_STEPS, NUM_JOINTS) * 0.05, axis=0)
# --- End Simulate Data ---

# --- AI Model Prediction Placeholder ---
# Here you would typically feed sEMG_data into your trained model
# predicted_angles = your_ai_model.predict(sEMG_data)
# For this example, we'll just use the simulated angle_data as the "prediction"
predicted_angles = angle_data
# --- End AI Model Prediction Placeholder ---


# --- XAI Contribution Calculation Placeholder ---
# This is the CRITICAL part you need to implement.
# You need an XAI method (SHAP, LIME, Integrated Gradients, Attention, etc.)
# applied to your model to get the contribution of each muscle `m`
# to each joint angle `j` at each time step `t`.
# The result should be a numpy array of shape: (NUM_TIME_STEPS, NUM_MUSCLES, NUM_JOINTS)
# Positive values might mean excitation, negative inhibition.

print("Simulating XAI contributions...")
# Simulate contributions: random fluctuations, some muscles influence some joints more
np.random.seed(42) # for reproducibility
xai_contributions = np.random.randn(NUM_TIME_STEPS, NUM_MUSCLES, NUM_JOINTS) * 0.5

# Make contributions sparser and more structured (example)
mask = np.random.rand(NUM_MUSCLES, NUM_JOINTS) > 0.7 # Only ~30% connections active
xai_contributions *= mask[np.newaxis, :, :]
# Add some temporal structure (e.g., sine wave modulation)
time_modulation = np.sin(np.linspace(0, 4 * np.pi, NUM_TIME_STEPS))[:, np.newaxis, np.newaxis]
xai_contributions *= (1 + time_modulation * 0.5)
print(f"Simulated XAI contributions shape: {xai_contributions.shape}")
# --- End XAI Contribution Calculation Placeholder ---


# --- Visualization Setup ---
print("Setting up visualization...")

# Muscle and Joint Names (Customize as needed)
muscle_names = [f'M{i}' for i in range(NUM_MUSCLES)] # e.g., ['APL', 'FCR', ...]
joint_names = [f'J{i}' for i in range(NUM_JOINTS)]   # e.g., ['Thumb1', 'Thumb2', ...]

# Create Graph
G = nx.Graph()
nodes = muscle_names + joint_names
G.add_nodes_from(nodes)

# Define Node Positions (Customize layout)
pos = {}
muscle_x = 0
joint_x = 1
for i, name in enumerate(muscle_names):
    pos[name] = (muscle_x, i)
for i, name in enumerate(joint_names):
    pos[name] = (joint_x, i * (NUM_MUSCLES - 1) / (NUM_JOINTS - 1)) # Spread joints vertically

# Add edges for all possible connections (initially invisible/thin)
edges = []
for m_idx, m_name in enumerate(muscle_names):
    for j_idx, j_name in enumerate(joint_names):
        G.add_edge(m_name, j_name, weight=0) # Add edge with initial weight
        edges.append((m_name, j_name))

# --- Matplotlib Figure Setup ---
fig, ax = plt.subplots(figsize=(10, 8))
plt.title("Dynamic Muscle Contribution Flow (Time: 0)")

# Draw nodes
node_colors = ['skyblue'] * NUM_MUSCLES + ['lightgreen'] * NUM_JOINTS
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

# Prepare edge drawing (store edge collection for updates)
edge_collection = nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', alpha=0.1, ax=ax)

# --- Animation Function ---
def update(frame):
    """Updates the plot for each animation frame (time step)."""
    contributions_at_t = xai_contributions[frame, :, :] # Shape: (NUM_MUSCLES, NUM_JOINTS)

    # Normalize contributions for visualization (e.g., absolute value for width)
    # You might want different scaling (linear, log) or color mapping
    abs_contributions = np.abs(contributions_at_t)
    max_abs_contribution = np.max(abs_contributions) if np.max(abs_contributions) > 0 else 1
    scaled_widths = (abs_contributions / max_abs_contribution) * 5.0 + 0.1 # Min width 0.1, Max width 5.1

    # Map contributions to colors (e.g., blue for negative/inhibitory, red for positive/excitatory)
    # Normalize contributions between -1 and 1 for colormap
    max_val = np.max(np.abs(contributions_at_t)) if np.max(np.abs(contributions_at_t)) > 0 else 1
    norm_contributions = contributions_at_t / max_val
    colors = plt.cm.coolwarm(norm_contributions.flatten() * 0.5 + 0.5) # Map [-1, 1] -> [0, 1] for colormap

    edge_widths = []
    edge_colors = []
    edge_alphas = [] # Control visibility based on contribution strength

    edge_idx = 0
    for m_idx in range(NUM_MUSCLES):
        for j_idx in range(NUM_JOINTS):
            width = scaled_widths[m_idx, j_idx]
            color = colors[edge_idx, :]
            alpha = min(max(abs_contributions[m_idx, j_idx] / max_abs_contribution * 2.0, 0.1), 1.0) # Make stronger contributions more opaque

            edge_widths.append(width)
            edge_colors.append(color)
            edge_alphas.append(alpha)
            edge_idx += 1

    # Update edge properties directly
    edge_collection.set_linewidths(edge_widths)
    edge_collection.set_edgecolors(edge_colors)
    # Note: set_alpha doesn't work directly on LineCollection, need to update colors with alpha
    edge_colors_with_alpha = [(r, g, b, a) for (r, g, b, _), a in zip(edge_colors, edge_alphas)]
    edge_collection.set_edgecolors(edge_colors_with_alpha)


    # Update title with current time step
    ax.set_title(f"Dynamic Muscle Contribution Flow (Time: {frame}/{NUM_TIME_STEPS-1})")

    # Return the modified artists (important for blitting)
    return edge_collection,

# --- Create and Run Animation ---
print("Creating animation...")
# Note: interval is delay between frames in ms. blit=True speeds up rendering.
ani = animation.FuncAnimation(fig, update, frames=NUM_TIME_STEPS,
                              interval=50, blit=True, repeat=False)

# To save the animation (requires ffmpeg or other writer)
# print("Saving animation...")
# ani.save('muscle_contribution_flow.mp4', writer='ffmpeg', fps=15)
# print("Animation saved.")

# To display the animation
plt.show()
print("Visualization finished.")

