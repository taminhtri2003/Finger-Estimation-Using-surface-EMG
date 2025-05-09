# Import necessary libraries
import scipy.io
import numpy as np
from PyEMD import EMD # Requires: pip install EMD-signal
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# !!! IMPORTANT: Replace with the actual path to your .mat file !!!
MAT_FILE_PATH = "s4_full.mat"
# !!! IMPORTANT: Set the actual sampling rate of your EMG data in Hz !!!
# If unknown, frequencies will be normalized (cycles/sample).
# For 4000 samples, if it's e.g. 4 seconds of data, SAMPLING_RATE = 1000.0 (Hz)
SAMPLING_RATE = 200.0  # Example: 1000 Hz
# Directory to save HHT plots
OUTPUT_PLOT_DIR = "hht_spectrum_plots"

# --- Data Column Definitions (from your description) ---
EMG_MUSCLES = ['APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR']
# finger_kinematics: 69 columns (23 markers * 3D coordinates <x,y,z>)
# joint_angles: 14 columns, specific definitions provided by user.
# The Marker_Position.png file helps interpret the marker data for kinematics and joint angles.

JOINT_ANGLE_DEFINITIONS = [
    "Thumb 1 (20-17 to 17-18)", "Thumb 2 (17-18 to 18-19)",
    "Index 1 (20-1 to 1-5)", "Index 2 (1-5 to 5-6)", "Index 3 (5-6 to 6-7)",
    "Middle 1 (20-2 to 2-8)", "Middle 2 (2-8 to 8-9)", "Middle 3 (8-9 to 9-10)",
    "Ring 1 (20-3 to 3-11)", "Ring 2 (3-11 to 11-12)", "Ring 3 (11-12 to 12-13)",
    "Little 1 (20-4 to 4-14)", "Little 2 (4-14 to 14-15)", "Little 3 (14-15 to 15-16)"
]


def load_mat_file(filepath):
    """
    Loads data from a .mat file.
    Args:
        filepath (str): Path to the .mat file.
    Returns:
        dict: Dictionary containing the loaded data.
              Returns None if file not found or error occurs.
    """
    try:
        print(f"Loading data from: {filepath}")
        data = scipy.io.loadmat(filepath)
        print("Data loaded successfully.")
        # Print names of variables loaded
        print("Variables in .mat file:", list(data.keys()))
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        print("Please ensure MAT_FILE_PATH is set correctly.")
        return None
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return None

def extract_imf_features(imfs, num_features_to_extract=3):
    """
    Extracts example features from IMFs.
    (Placeholder for "IMFs Feature 1, IMFs Feature 2, IMFs Feature 3")
    Args:
        imfs (np.ndarray): Array of IMFs (rows are IMFs, columns are time samples).
        num_features_to_extract (int): Number of IMFs to extract features from.
    Returns:
        dict: Dictionary of extracted features.
    """
    features = {}
    if imfs is None or imfs.shape[0] == 0:
        return features

    num_imfs_available = imfs.shape[0]
    for i in range(min(num_features_to_extract, num_imfs_available)):
        imf = imfs[i, :]
        features[f'imf_{i+1}_energy'] = np.sum(imf**2)
        features[f'imf_{i+1}_mean_abs_amp'] = np.mean(np.abs(imf))
        features[f'imf_{i+1}_std_amp'] = np.std(imf)
    return features

def perform_hht(imfs, sampling_rate):
    """
    Performs Hilbert-Huang Transform on IMFs.
    Args:
        imfs (np.ndarray): Array of IMFs.
        sampling_rate (float): Sampling rate of the signal.
    Returns:
        tuple: (instantaneous_frequencies, instantaneous_amplitudes)
               Returns (None, None) if no IMFs.
    """
    if imfs is None or imfs.shape[0] == 0:
        return None, None

    n_imfs = imfs.shape[0]
    n_samples = imfs.shape[1]
    
    instantaneous_frequencies_list = []
    instantaneous_amplitudes_list = []

    for i in range(n_imfs):
        analytic_signal = hilbert(imfs[i, :])
        instantaneous_amplitude = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        
        # Calculate instantaneous frequency
        # (d(phase)/dt) / (2*pi)
        # dt = 1.0 / sampling_rate
        inst_freq = (np.diff(instantaneous_phase) / (2.0 * np.pi)) * sampling_rate
        
        # Pad to match original length (simple forward fill for the last point)
        inst_freq = np.concatenate((inst_freq, [inst_freq[-1]])) if len(inst_freq) > 0 else np.zeros(n_samples)


        instantaneous_frequencies_list.append(inst_freq)
        instantaneous_amplitudes_list.append(instantaneous_amplitude)

    if not instantaneous_frequencies_list: # Should not happen if imfs is not empty
        return np.array([]), np.array([])

    return np.array(instantaneous_frequencies_list), np.array(instantaneous_amplitudes_list)


def plot_hht_spectrum(time_vector, imf_idx, inst_freq_imf, inst_amp_imf,
                        trial, task, channel_name, output_dir):
    """
    Generates and saves a HHT spectrum representation for a single IMF.
    Args:
        time_vector (np.ndarray): Time vector for the x-axis.
        imf_idx (int): Index of the IMF.
        inst_freq_imf (np.ndarray): Instantaneous frequency of the IMF.
        inst_amp_imf (np.ndarray): Instantaneous amplitude of the IMF.
        trial (int): Trial number (0-indexed).
        task (int): Task number (0-indexed).
        channel_name (str): Name of the EMG channel.
        output_dir (str): Directory to save the plot.
    """
    if inst_freq_imf.size == 0 or inst_amp_imf.size == 0:
        print(f"      Skipping HHT plot for IMF {imf_idx+1}: No data.")
        return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            return

    plt.figure(figsize=(12, 6))
    # Using scatter plot: x=time, y=frequency, color=amplitude
    sc = plt.scatter(time_vector, inst_freq_imf, s=5, c=inst_amp_imf, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='Amplitude')
    plt.ylabel(f'Frequency (Hz if SAMPLING_RATE is in Hz)')
    plt.xlabel(f'Time (seconds if SAMPLING_RATE is in Hz)')
    
    # Set a reasonable y-limit, avoiding excessively high frequencies if any
    valid_freqs = inst_freq_imf[np.isfinite(inst_freq_imf) & (inst_freq_imf >= 0)]
    if len(valid_freqs) > 0:
        upper_freq_limit = np.percentile(valid_freqs, 99) if len(valid_freqs) > 100 else np.max(valid_freqs)
        plt.ylim(0, max(1, upper_freq_limit * 1.1)) # Ensure ylim is at least 0 to 1
    else:
        plt.ylim(0,1)


    plt.title(f'HHT Spectrum: IMF {imf_idx+1}\nTrial {trial+1}, Task {task+1}, Channel {channel_name}')
    plt.tight_layout()
    
    filename = f"hht_T{trial+1}_TK{task+1}_CH{channel_name}_IMF{imf_idx+1}.png"
    filepath = os.path.join(output_dir, filename)
    try:
        plt.savefig(filepath)
        print(f"      Saved HHT plot: {filepath}")
    except Exception as e:
        print(f"      Error saving plot {filepath}: {e}")
    plt.close()


def process_emg_data(mat_data, sampling_rate, output_plot_dir):
    """
    Main function to process EMG data from the loaded .mat file.
    Args:
        mat_data (dict): Data loaded from .mat file.
        sampling_rate (float): Sampling rate of EMG data.
        output_plot_dir (str): Directory to save plots.
    """
    if 'dsfilt_emg' not in mat_data:
        print("Error: 'dsfilt_emg' not found in the .mat file.")
        return
    if 'finger_kinematics' not in mat_data:
        print("Warning: 'finger_kinematics' not found in the .mat file.")
    if 'joint_angles' not in mat_data:
        print("Warning: 'joint_angles' not found in the .mat file.")

    dsfilt_emg = mat_data['dsfilt_emg']
    finger_kinematics = mat_data.get('finger_kinematics') # Use .get for optional keys
    joint_angles = mat_data.get('joint_angles')

    # Data structure: dsfilt_emg is <5x7 cell>, each cell <4000x8>
    num_trials = dsfilt_emg.shape[0]
    num_tasks = dsfilt_emg.shape[1]

    print(f"\nFound {num_trials} trials and {num_tasks} tasks.")
    print(f"EMG muscle order: {EMG_MUSCLES}")
    if joint_angles is not None:
         print(f"Joint angle definitions correspond to the {len(JOINT_ANGLE_DEFINITIONS)} columns.")


    # Initialize EMD object
    emd_analyzer = EMD()
    # Common EMD parameters you might adjust:
    # emd_analyzer.FIXE = 10  # Example: Set a fixed number of sifting iterations
    # emd_analyzer.FIXE_S = 4 # Example: Number of sifts for FIXE mode

    for trial_idx in range(num_trials):
        for task_idx in range(num_tasks):
            print(f"\n--- Processing Trial {trial_idx+1}, Task {task_idx+1} ---")
            
            emg_trial_task_data = dsfilt_emg[trial_idx, task_idx] # Shape: (4000, 8)
            
            if finger_kinematics is not None:
                kin_trial_task_data = finger_kinematics[trial_idx, task_idx] # Shape: (4000, 69)
                print(f"  Finger Kinematics data shape: {kin_trial_task_data.shape}")
            if joint_angles is not None:
                ja_trial_task_data = joint_angles[trial_idx, task_idx] # Shape: (4000, 14)
                print(f"  Joint Angles data shape: {ja_trial_task_data.shape}")


            if emg_trial_task_data.shape[1] != len(EMG_MUSCLES):
                print(f"  Warning: EMG data has {emg_trial_task_data.shape[1]} channels, "
                      f"but {len(EMG_MUSCLES)} muscle names defined.")
            
            num_samples = emg_trial_task_data.shape[0]
            time_vector = np.arange(num_samples) / sampling_rate # Time in seconds

            for channel_idx in range(emg_trial_task_data.shape[1]):
                channel_name = EMG_MUSCLES[channel_idx] if channel_idx < len(EMG_MUSCLES) else f"UnknownCH{channel_idx+1}"
                print(f"  Processing EMG Channel {channel_idx+1}: {channel_name}")
                
                single_emg_channel_signal = emg_trial_task_data[:, channel_idx]

                # 1. Empirical Mode Decomposition (EMD)
                imfs = emd_analyzer.emd(single_emg_channel_signal, max_imf=7) # Limit max_imf if needed
                print(f"    EMD: Found {imfs.shape[0] if imfs is not None and imfs.ndim == 2 else 0} IMFs.")

                if imfs is None or imfs.shape[0] == 0:
                    print("    Skipping feature extraction and HHT due to no IMFs.")
                    continue

                # 2. Extract features from IMFs
                imf_features = extract_imf_features(imfs)
                print(f"    IMF Features: {imf_features}")

                # 3. Hilbert-Huang Analysis (HHT)
                inst_freqs, inst_amps = perform_hht(imfs, sampling_rate)
                
                if inst_freqs is not None and inst_amps is not None:
                    print(f"    HHT: Inst. Frequencies shape: {inst_freqs.shape}, Inst. Amplitudes shape: {inst_amps.shape}")
                    
                    # 4. Generate Hilbert-Huang Spectrum Image (plot for each IMF)
                    for imf_i in range(inst_freqs.shape[0]):
                        plot_hht_spectrum(time_vector, imf_i, inst_freqs[imf_i, :], inst_amps[imf_i, :],
                                          trial_idx, task_idx, channel_name, output_plot_dir)
                else:
                    print("    HHT: No data generated.")
    print("\n--- Processing Complete ---")

if __name__ == "__main__":
    # Load data from .mat file
    matlab_data = load_mat_file(MAT_FILE_PATH)

    if matlab_data:
        # Process the EMG data
        process_emg_data(matlab_data, SAMPLING_RATE, OUTPUT_PLOT_DIR)
        print(f"\nCheck the '{OUTPUT_PLOT_DIR}' directory for HHT spectrum plots (if any were generated).")
        print("Further analysis can be done using the extracted IMFs, their features, and HHT results,")
        print("potentially in conjunction with 'finger_kinematics' and 'joint_angles' data.")
    else:
        print("Could not load data. Exiting.")

