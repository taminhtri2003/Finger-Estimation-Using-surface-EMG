import numpy as np
import scipy.io # To load .mat files
import scipy.signal # For filtering EMG

# --- Configuration - YOU MUST EDIT THESE ---

# 1. Path to your MATLAB data file
mat_file_path = 's4_full.mat' # <-- Replace with the actual path to your .mat file

# 2. Select Trial and Task (Indices start from 0 in Python)
# Example: Trial 1, Task 1 corresponds to row 0, column 0
trial_index = 0
task_index = 0

# 3. Sampling Frequency (Hz) - !! IMPORTANT !!
sampling_frequency = 200.0 # <-- Replace with your actual sampling frequency

# 4. Output file names
kinematics_sto_path = f'kinematics_trial{trial_index+1}_task{task_index+1}.sto'
activations_sto_path = f'activations_trial{trial_index+1}_task{task_index+1}.sto'

# 5. OpenSim Model Names - !! CRITICAL !!
# These MUST EXACTLY match the coordinate and muscle names in your .osim model
# Example names - replace with your actual model names
joint_names = [
    'thumb_angle_1', 'thumb_angle_2', 'index_angle_1', 'index_angle_2',
    'index_angle_3', 'middle_angle_1', 'middle_angle_2', 'middle_angle_3',
    'ring_angle_1', 'ring_angle_2', 'ring_angle_3', 'little_angle_1',
    'little_angle_2', 'little_angle_3'
] # <-- Replace with names from your .osim file

muscle_names = [
    '/forceset/APL', '/forceset/FCR', '/forceset/FDS', '/forceset/FDP',
    '/forceset/ED', '/forceset/EI', '/forceset/ECU', '/forceset/ECR'
] # <-- Replace with names from your .osim file (check the <ForceSet> section)

# 6. EMG Processing Parameters
emg_filter_cutoff = 6.0 # Low-pass filter cutoff frequency (Hz) for envelope
emg_filter_order = 4 # Filter order
normalize_emg = True # True to normalize EMG, False otherwise

# 7. Activation Dynamics Parameters (if used)
use_activation_dynamics = True # True to apply dynamics, False to use normalized EMG directly
tau_activation = 0.050 # Activation time constant (seconds) - Example value
tau_deactivation = 0.080 # Deactivation time constant (seconds) - Example value

# 8. Joint Angle Units
# Are your joint_angles in the .mat file in Degrees or Radians?
# OpenSim typically uses Degrees for joint coordinates in .sto files.
# Set 'inDegrees' header accordingly.
angles_are_in_degrees = True # <-- Set to True or False based on your data

# --- End of Configuration ---

# --- Helper Functions ---

def write_sto_file(filepath, data, header_info, column_labels):
    """Writes data to an OpenSim .sto file."""
    num_rows, num_cols = data.shape

    with open(filepath, 'w') as f:
        # Write header
        f.write(f"{header_info['name']}\n")
        f.write(f"version=1\n")
        f.write(f"nRows={num_rows}\n")
        f.write(f"nColumns={num_cols}\n")
        if 'inDegrees' in header_info:
            f.write(f"inDegrees={'yes' if header_info['inDegrees'] else 'no'}\n")
        # Optional: Add DataRate if needed, but often omitted if time column is present
        # f.write(f"DataRate={header_info['data_rate']}\n")
        f.write("DataType=double\n")
        f.write("endheader\n")

        # Write column labels
        f.write("time\t" + "\t".join(column_labels) + "\n")

        # Write data
        # Use a format string for consistent spacing (optional but nice)
        data_format = "{:.8f}" # Format for time and data values
        for i in range(num_rows):
            row_data = [data_format.format(val) for val in data[i, :]]
            f.write("\t".join(row_data) + "\n")
    print(f"Successfully wrote {filepath}")

def process_emg(raw_emg, fs, cutoff, order, normalize, use_dynamics, tau_act, tau_deact):
    """Processes raw EMG to activations."""
    num_samples, num_muscles = raw_emg.shape
    dt = 1.0 / fs
    activations = np.zeros_like(raw_emg)

    # Design filter
    b, a = scipy.signal.butter(order, cutoff / (fs / 2.0), btype='low')

    for i in range(num_muscles):
        # 1. Rectify
        rectified_emg = np.abs(raw_emg[:, i])

        # 2. Low-pass filter
        filtered_emg = scipy.signal.filtfilt(b, a, rectified_emg)
        # Ensure non-negative after filtering
        filtered_emg[filtered_emg < 0] = 0

        # 3. Normalize (optional, based on peak of this specific trial/task)
        if normalize:
            max_val = np.max(filtered_emg)
            if max_val > 1e-6: # Avoid division by zero
                 neural_command = filtered_emg / max_val
            else:
                 neural_command = np.zeros(num_samples)
        else:
             neural_command = filtered_emg # Use unnormalized if desired

        # 4. Activation Dynamics (optional)
        if use_dynamics:
            act = np.zeros(num_samples)
            act[0] = neural_command[0] # Initialize activation
            for k in range(1, num_samples):
                 # Determine time constant
                 if neural_command[k] > act[k-1]:
                     tau = tau_act
                 else:
                     tau = tau_deact

                 # Solve ODE using forward Euler (simple method)
                 dadt = (neural_command[k] - act[k-1]) / tau
                 act[k] = act[k-1] + dadt * dt

                 # Ensure activation stays within [0, 1] bounds (or slightly above 1 if normalized > 1)
                 act[k] = np.clip(act[k], 0, np.max([1.0, np.max(neural_command)]))
            activations[:, i] = act
        else:
            activations[:, i] = neural_command # Use normalized EMG directly

    return activations

# --- Main Script ---

# 1. Load Data
try:
    mat_data = scipy.io.loadmat(mat_file_path)
    print(f"Loaded data from {mat_file_path}")
except FileNotFoundError:
    print(f"Error: MATLAB file not found at {mat_file_path}")
    exit()
except Exception as e:
    print(f"Error loading .mat file: {e}")
    exit()

# Access data for the selected trial and task
try:
    # Remember MATLAB uses 1-based indexing, Python uses 0-based.
    # The loaded data might be nested; inspect mat_data if needed.
    emg_data_raw = mat_data['dsfilt_emg'][trial_index, task_index]
    joint_angle_data = mat_data['joint_angles'][trial_index, task_index]
    print(f"Extracted data for Trial {trial_index+1}, Task {task_index+1}")
except IndexError:
    print(f"Error: Trial index {trial_index} or Task index {task_index} out of bounds.")
    print(f"Available trials/tasks shape: {mat_data['dsfilt_emg'].shape}")
    exit()
except KeyError as e:
    print(f"Error: Variable name {e} not found in the .mat file.")
    print(f"Available variables: {list(mat_data.keys())}")
    exit()

# Verify data shapes
num_samples_emg, num_emg_channels = emg_data_raw.shape
num_samples_kin, num_kin_channels = joint_angle_data.shape

if num_samples_emg != num_samples_kin:
    print("Warning: EMG and Kinematics have different numbers of samples!")
    # Decide how to handle: truncate, error out, etc.
    # For now, we'll use the minimum number of samples
    num_samples = min(num_samples_emg, num_samples_kin)
    emg_data_raw = emg_data_raw[:num_samples, :]
    joint_angle_data = joint_angle_data[:num_samples, :]
    print(f"Using {num_samples} samples.")
else:
    num_samples = num_samples_emg

print(f"Data samples: {num_samples}, EMG channels: {num_emg_channels}, Kinematic DOFs: {num_kin_channels}")

# Check if number of channels matches configuration
if num_kin_channels != len(joint_names):
    print(f"Error: Number of kinematic channels in data ({num_kin_channels}) does not match number of joint_names provided ({len(joint_names)}).")
    exit()
if num_emg_channels != len(muscle_names):
     print(f"Error: Number of EMG channels in data ({num_emg_channels}) does not match number of muscle_names provided ({len(muscle_names)}).")
     exit()

# 2. Create Time Vector
time_vector = np.linspace(0, (num_samples - 1) / sampling_frequency, num_samples)

# 3. Prepare Kinematics Data for .sto
# Add time vector as the first column
kinematics_output_data = np.column_stack((time_vector, joint_angle_data))

# 4. Process EMG Data
muscle_activations = process_emg(
    emg_data_raw,
    fs=sampling_frequency,
    cutoff=emg_filter_cutoff,
    order=emg_filter_order,
    normalize=normalize_emg,
    use_dynamics=use_activation_dynamics,
    tau_act=tau_activation,
    tau_deact=tau_deactivation
)

# 5. Prepare Activations Data for .sto
# Add time vector as the first column
activations_output_data = np.column_stack((time_vector, muscle_activations))

# 6. Write .sto Files

# Write Kinematics .sto
kin_header = {
    'name': f'Kinematics_Trial{trial_index+1}_Task{task_index+1}',
    'inDegrees': angles_are_in_degrees,
    'data_rate': sampling_frequency # Optional header info
}
write_sto_file(kinematics_sto_path, kinematics_output_data, kin_header, joint_names)

# Write Activations .sto
act_header = {
    'name': f'Activations_Trial{trial_index+1}_Task{task_index+1}',
     'data_rate': sampling_frequency # Optional header info
}
write_sto_file(activations_sto_path, activations_output_data, act_header, muscle_names)

print("\n--- Script Finished ---")
print(f"Make sure the column labels in the generated .sto files:")
print(f"  - {kinematics_sto_path}")
print(f"  - {activations_sto_path}")
print(f"EXACTLY match the coordinate and muscle names in your OpenSim model (.osim file).")
print(f"Also verify the 'inDegrees' setting in the kinematics file header.")

