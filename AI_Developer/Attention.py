#import the functions from the Attention_Function.py file
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns

from Attention_Function import (
    load_mat_data, preprocess_data, build_finger_mha_lstm, train_model, get_mha_attention_weights,
    plot_mha_attention_heatmap, calculate_performance_scores, plot_performance_scores,
    plot_regression_correlation, get_integrated_gradients, explain_with_shap, visualize_attributions,
    visualize_shap_summary, plot_emg_angle_relationship, summarize_mha_attention,
    plot_attention_weighted_emg_snapshot, plot_trial_dynamics_simulation,
    simulate_activation_prediction_comparison)

# --- Configuration ---
MAT_FILE_PATH = 's4_full.mat' # <<<--- Example path, UPDATE FOR YOUR ENVIRONMENT
SEQUENCE_LENGTH = 200
PREDICTION_HORIZON = 1 # Keep as 1 for current model structure
BATCH_SIZE = 128
EPOCHS = 10 # Adjust as needed (increase for better performance)
LSTM_UNITS = 128
VALIDATION_SPLIT = 0.2
MAX_SAMPLES_TO_VIZ_MHA = 1 # Limit detailed MHA heatmap visualization output
NUM_TRIALS_TO_PLOT_DYNAMICS = 1 # Limit dynamic plot output
NUM_SAMPLES_FOR_XAI = 2 # How many samples to run XAI (IG & SHAP) on
MAX_SAMPLES_TO_VIZ = 2 # Limit detailed visualization output (used by MHA, IG, SHAP, EMG-Angle plots)
NUM_SHAP_BACKGROUND_SAMPLES = 50 # Number of background samples for SHAP explainer
ATTENTION_HIGHLIGHT_THRESHOLD_PERCENTILE = 75 # Percentile for highlighting high attention
PLOT_ATTN_WEIGHTED_SNAPSHOT = True # Flag to enable the attention-weighted EMG plot
RUN_ACTIVATION_SIMULATION = True # Flag to enable the new activation simulation
MUSCLE_ACTIVATION_THRESHOLD = 0.8 # Threshold (in std dev) for considering a muscle group active in simulation

# MHA Configuration
NUM_ATTENTION_HEADS = 5
D_MODEL = LSTM_UNITS
if D_MODEL % NUM_ATTENTION_HEADS != 0:
     D_MODEL = math.ceil(D_MODEL / NUM_ATTENTION_HEADS) * NUM_ATTENTION_HEADS
     print(f"Adjusting D_MODEL to {D_MODEL} to be divisible by {NUM_ATTENTION_HEADS} heads.")

# Integrated Gradients Config
IG_STEPS = 50 # Number of steps for approximation

# EMG Channel Names (Ensure this matches your .mat file)
EMG_CHANNEL_NAMES = ['APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR']
# Joint Angle Names (Ensure this matches your .mat file)
JOINT_ANGLE_NAMES = [
    'Thumb 1', 'Thumb 2', 'Index 1', 'Index 2', 'Index 3', 'Middle 1', 'Middle 2', 'Middle 3',
    'Ring 1', 'Ring 2', 'Ring 3', 'Little 1', 'Little 2', 'Little 3'
]
# Finger to Joint Mapping (Ensure this matches your angle setup)
FINGER_JOINT_INDICES = {
    'Thumb': list(range(0, 2)), 'Index': list(range(2, 5)), 'Middle': list(range(5, 8)),
    'Ring': list(range(8, 11)), 'Little': list(range(11, 14))
}
FINGER_NAMES = list(FINGER_JOINT_INDICES.keys())

# --- Muscle Group Definitions ---
MUSCLE_GROUPS = {
    'Finger Flexors': [EMG_CHANNEL_NAMES.index(name) for name in ['FDS', 'FDP'] if name in EMG_CHANNEL_NAMES],
    'Finger Extensors': [EMG_CHANNEL_NAMES.index(name) for name in ['ED', 'EI'] if name in EMG_CHANNEL_NAMES],
    'Wrist Flex/Dev': [EMG_CHANNEL_NAMES.index(name) for name in ['FCR'] if name in EMG_CHANNEL_NAMES],
    'Wrist Extend/Dev': [EMG_CHANNEL_NAMES.index(name) for name in ['ECU', 'ECR'] if name in EMG_CHANNEL_NAMES],
    'Thumb': [EMG_CHANNEL_NAMES.index(name) for name in ['APL'] if name in EMG_CHANNEL_NAMES]
}
# Assign colors
try: cmap = plt.colormaps.get_cmap('tab10')
except AttributeError: cmap = plt.cm.get_cmap('tab10')
GROUP_COLORS = cmap(np.linspace(0, 1, len(MUSCLE_GROUPS)))
MUSCLE_GROUP_VIS_INFO = {name: {'indices': indices, 'color': GROUP_COLORS[i]}
                         for i, (name, indices) in enumerate(MUSCLE_GROUPS.items()) if indices} # Only include groups with found channels
print("Defined Muscle Groups (Indices):", MUSCLE_GROUPS)

# --- Heuristic Mapping for Simulation (NEW) ---
# Simplified mapping from active muscle group to likely activated finger(s)
# This is a very basic heuristic for comparison purposes.
MUSCLE_TO_FINGER_HEURISTIC = {
    'Finger Flexors': ['Index', 'Middle', 'Ring', 'Little'],
    'Finger Extensors': ['Index', 'Middle', 'Ring', 'Little'],
    'Thumb': ['Thumb'],
    'Wrist Flex/Dev': [], # Primarily wrist, ignore for direct finger heuristic
    'Wrist Extend/Dev': [], # Primarily wrist, ignore for direct finger heuristic
}

if __name__ == "__main__":
    # 1. Load Data
    emg_data_cells, joint_angles_cells, _ = load_mat_data(MAT_FILE_PATH)

    if emg_data_cells is not None and joint_angles_cells is not None:
        # 2. Preprocess Data
        # Now returns metadata as well
        X_train, X_val, y_train, y_val, emg_scaler, angle_scaler, original_scaled_trials, train_metadata, val_metadata = preprocess_data(
            emg_data_cells, joint_angles_cells, SEQUENCE_LENGTH, PREDICTION_HORIZON)

        if X_train is not None and len(X_val) > 0: # Check if validation data exists
            num_emg_features = X_train.shape[2]
            num_angle_features = y_train.shape[1]

            # Sanity check dimensions
            if num_emg_features != len(EMG_CHANNEL_NAMES):
                exit(f"FATAL Error: Number of EMG features in data ({num_emg_features}) does not match EMG_CHANNEL_NAMES length ({len(EMG_CHANNEL_NAMES)}). Please check data loading and channel names.")
            if num_angle_features != len(JOINT_ANGLE_NAMES):
                exit(f"FATAL Error: Number of angle features in data ({num_angle_features}) does not match JOINT_ANGLE_NAMES length ({len(JOINT_ANGLE_NAMES)}). Please check data loading and angle names.")
            if NUM_ATTENTION_HEADS != len(FINGER_NAMES):
                 exit(f"FATAL Error: NUM_ATTENTION_HEADS ({NUM_ATTENTION_HEADS}) must match the number of fingers defined in FINGER_JOINT_INDICES ({len(FINGER_NAMES)}).")


            # 3. Build MHA Models
            model, attention_model = build_finger_mha_lstm(
                SEQUENCE_LENGTH, num_emg_features, num_angle_features,
                LSTM_UNITS, D_MODEL, NUM_ATTENTION_HEADS, FINGER_JOINT_INDICES
            )
            print("\n--- Training Model Summary ---")
            model.summary(line_length=150)
            # Optional: Plot model architecture
            # try:
            #     tf.keras.utils.plot_model(model, to_file='mha_model_plot.png', show_shapes=True, show_layer_names=True)
            #     print("\nModel architecture plot saved to mha_model_plot.png")
            # except Exception as e:
            #     print(f"\nCould not plot model architecture: {e}")

            # 4. Train Model
            history = train_model(model, X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE)

            # Plot training history
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1); plt.plot(history.history['loss'], label='Train Loss'); plt.plot(history.history['val_loss'], label='Val Loss'); plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend(); plt.grid(True)
            plt.subplot(1, 2, 2); plt.plot(history.history['mae'], label='Train MAE'); plt.plot(history.history['val_mae'], label='Val MAE'); plt.title('MAE'); plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.grid(True)
            plt.suptitle("Training History")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

            # 5. Evaluate & Plot Performance Scores
            print("\n--- Performance Evaluation on Validation Set ---")
            y_pred_scaled_all = model.predict(X_val)
            cc_scores = None # Initialize
            if hasattr(angle_scaler, 'mean_') and angle_scaler.mean_ is not None and hasattr(angle_scaler, 'scale_') and angle_scaler.scale_ is not None:
                 try:
                     y_val_rescaled = angle_scaler.inverse_transform(y_val)
                     y_pred_rescaled_all = angle_scaler.inverse_transform(y_pred_scaled_all)
                     r2_scores, cc_scores, _ = calculate_performance_scores(y_val_rescaled, y_pred_rescaled_all)
                     print(f"Average R2 Score (Rescaled): {np.mean(r2_scores):.3f}")
                     print(f"Average CC Score (Rescaled): {np.mean(cc_scores):.3f}")
                     plot_performance_scores(r2_scores, cc_scores, JOINT_ANGLE_NAMES)
                     plot_regression_correlation(y_val_rescaled, y_pred_rescaled_all, JOINT_ANGLE_NAMES, cc_scores)
                 except ValueError as e:
                     print(f"Error during rescaling or performance calculation: {e}")
                     print("Evaluating on scaled data instead.")
                     loss, mae = model.evaluate(X_val, y_val, verbose=0)
                     print(f"Scaled Validation Loss (MSE): {loss:.4f}, Scaled Validation MAE: {mae:.4f}")
                     # Still plot correlation on scaled data
                     plot_regression_correlation(y_val, y_pred_scaled_all, [f"{name} (Scaled)" for name in JOINT_ANGLE_NAMES])

            else:
                 print("Warning: Angle scaler not fitted properly. Cannot evaluate/plot performance on original scale.")
                 loss, mae = model.evaluate(X_val, y_val, verbose=0)
                 print(f"Scaled Validation Loss (MSE): {loss:.4f}, Scaled Validation MAE: {mae:.4f}")
                 # Plot correlation on scaled data
                 plot_regression_correlation(y_val, y_pred_scaled_all, [f"{name} (Scaled)" for name in JOINT_ANGLE_NAMES])


            # --- 6. MHA Attention Visualization (Heatmaps per sample window) ---
            print("\n--- Detailed MHA Attention Visualization (Sample Windows) ---")
            num_samples_to_viz_mha = min(MAX_SAMPLES_TO_VIZ, X_val.shape[0])
            mha_attention_weights_all_val = None # Initialize weights for all validation samples
            if num_samples_to_viz_mha > 0:
                print(f"Visualizing MHA attention heatmaps for the first {num_samples_to_viz_mha} validation sample windows...")
                X_val_subset_mha = X_val[:num_samples_to_viz_mha]
                # Get attention weights for the subset
                mha_attention_weights_subset = get_mha_attention_weights(X_val_subset_mha, attention_model)

                if mha_attention_weights_subset is not None and mha_attention_weights_subset.shape[0] == num_samples_to_viz_mha:
                    for i in range(num_samples_to_viz_mha):
                        print(f"\n--- Visualizing MHA Attention Heatmap for Validation Sample Index {i} ---")
                        plot_mha_attention_heatmap(
                            X_val_subset_mha[i], # Single EMG sequence (seq_len, num_emg)
                            mha_attention_weights_subset[i], # Corresponding attention weights (num_heads, seq_len)
                            EMG_CHANNEL_NAMES, MUSCLE_GROUP_VIS_INFO, FINGER_NAMES,
                            title=f"MHA Attention Heatmap Analysis for Validation Sample Window {i}"
                        )
                    # Get weights for all validation samples for summary table later
                    print("Getting attention weights for all validation samples (for summary)...")
                    mha_attention_weights_all_val = get_mha_attention_weights(X_val, attention_model)

                else: print("\nCould not extract MHA attention weights for the specified samples. Skipping heatmaps and summary.")
            else: print("Skipping MHA Heatmap visualization.")


            # --- 7. Dynamic Trial Simulation Visualization ---
            print("\n--- Dynamic Trial Simulation Visualization (EMG/Angle/Attention) ---")
            num_trials_to_plot = min(NUM_TRIALS_TO_PLOT_DYNAMICS, len(original_scaled_trials))
            # Store data needed for snapshot plot
            snapshot_data = {'trial_idx': None, 'peak_window_idx': None, 'head_idx': None, 'X_windows': None, 'mha_weights': None}

            if num_trials_to_plot > 0 and 'attention_model' in locals() and 'model' in locals() and original_scaled_trials:
                 print(f"Visualizing dynamics simulation for the first {num_trials_to_plot} original trials...")
                 for i in range(num_trials_to_plot):
                     trial_data = original_scaled_trials[i]
                     # Call the MODIFIED function - it now returns peak info
                     peak_idx, head_idx, trial_X_windows, trial_mha_weights = plot_trial_dynamics_simulation(
                         trial_emg_scaled=trial_data['emg'], trial_angles_scaled=trial_data['angle'],
                         model=model, attention_model=attention_model,
                         emg_channels=EMG_CHANNEL_NAMES, joint_angle_names=JOINT_ANGLE_NAMES,
                         muscle_group_info=MUSCLE_GROUP_VIS_INFO, finger_names=FINGER_NAMES,
                         finger_joint_indices=FINGER_JOINT_INDICES, sequence_length=SEQUENCE_LENGTH,
                         prediction_horizon=PREDICTION_HORIZON,
                         trial_id=trial_data['id']
                     )
                     # Store info from the first plotted trial for the snapshot plot
                     if i == 0 and peak_idx is not None:
                         snapshot_data['trial_idx'] = i
                         snapshot_data['peak_window_idx'] = peak_idx
                         snapshot_data['head_idx'] = head_idx
                         snapshot_data['X_windows'] = trial_X_windows # Store all windows for this trial
                         snapshot_data['mha_weights'] = trial_mha_weights # Store all weights for this trial

            else: print("Skipping dynamic trial plotting (no trials found, or models not available).")


            # --- 8. Integrated Gradients XAI Visualization ---
            print("\n--- Integrated Gradients XAI Visualization ---")
            num_samples_to_explain_ig = min(MAX_SAMPLES_TO_VIZ, X_val.shape[0]) # Use MAX_SAMPLES_TO_VIZ
            if num_samples_to_explain_ig > 0:
                print(f"Calculating Integrated Gradients for the first {num_samples_to_explain_ig} validation sample(s)...")
                # Use zero baseline (common practice)
                baseline = np.zeros((1, SEQUENCE_LENGTH, num_emg_features)).astype(np.float32)
                X_explain_ig = X_val[:num_samples_to_explain_ig]

                for i in range(num_samples_to_explain_ig):
                    print(f"\n--- Explaining Validation Sample Index {i} with Integrated Gradients ---")
                    input_sample = X_explain_ig[i:i+1] # Keep batch dimension

                    # Select a target angle to explain (e.g., 'Index 2')
                    target_angle_name_ig = 'Index 2'
                    try: target_angle_index_ig = JOINT_ANGLE_NAMES.index(target_angle_name_ig)
                    except ValueError: target_angle_index_ig = 0; target_angle_name_ig = JOINT_ANGLE_NAMES[0] # Fallback
                    print(f"Explaining prediction for: {target_angle_name_ig} (Index {target_angle_index_ig})")

                    # Calculate IG attributions
                    ig_attributions = get_integrated_gradients(
                        model=model, baseline=baseline, input_sample=input_sample,
                        target_angle_index=target_angle_index_ig, num_steps=IG_STEPS
                    )

                    # Visualize IG attributions (if calculation succeeded)
                    # ig_attributions shape should be (seq_len, num_emg) if batch size was 1
                    if ig_attributions is not None:
                        # If input_sample had batch > 1, ig_attributions might have batch dim. Select first sample.
                        if ig_attributions.ndim == 3 and ig_attributions.shape[0] > 1:
                             ig_attributions_sample = ig_attributions[0]
                        else:
                             ig_attributions_sample = ig_attributions

                        visualize_attributions(
                            attributions=ig_attributions_sample, # Pass single sample's attributions
                            emg_channels=EMG_CHANNEL_NAMES,
                            muscle_group_info=MUSCLE_GROUP_VIS_INFO,
                            title=f"Integrated Gradients for {target_angle_name_ig} (Validation Sample {i})"
                        )
                    else: print(f"Skipping IG visualization for sample {i} due to calculation error.")
            else: print("Skipping Integrated Gradients visualization.")


            # --- 9. SHAP XAI Visualization ---
            print("\n--- SHAP XAI Visualization ---")
            num_samples_to_explain_shap = min(MAX_SAMPLES_TO_VIZ, X_val.shape[0]) # Use MAX_SAMPLES_TO_VIZ
            num_background = min(NUM_SHAP_BACKGROUND_SAMPLES, X_train.shape[0])

            if num_samples_to_explain_shap > 0 and num_background > 0:
                print(f"Calculating SHAP values using {num_background} background samples from X_train...")
                print(f"Explaining the first {num_samples_to_explain_shap} validation samples...")

                # Prepare background and explanation data
                background_data = X_train[:num_background].astype(np.float32)
                X_explain_shap = X_val[:num_samples_to_explain_shap].astype(np.float32)

                # Select a target angle to explain (can be the same or different from IG)
                target_angle_name_shap = 'Index 2'
                try: target_angle_index_shap = JOINT_ANGLE_NAMES.index(target_angle_name_shap)
                except ValueError: target_angle_index_shap = 0; target_angle_name_shap = JOINT_ANGLE_NAMES[0] # Fallback
                print(f"Explaining prediction for: {target_angle_name_shap} (Index {target_angle_index_shap})")

                # Calculate SHAP values
                shap_values = explain_with_shap(
                    model, background_data, X_explain_shap, target_angle_index_shap, EMG_CHANNEL_NAMES
                )

                # Visualize SHAP summary plot (feature importance across samples)
                if shap_values is not None:
                    visualize_shap_summary(
                        shap_values, X_explain_shap, EMG_CHANNEL_NAMES,
                        title=f"SHAP Summary for {target_angle_name_shap}"
                    )
                    # Optional: Add other SHAP plots like dependence plots or force plots for individual samples if needed
                    # e.g., shap.force_plot(explainer.expected_value, shap_values[0], X_explain_shap[0], feature_names=EMG_CHANNEL_NAMES)
                else: print("Skipping SHAP visualization due to calculation error.")

            else: print(f"Skipping SHAP visualization (Need >0 samples to explain and >0 background samples).")


            # --- 10. EMG Activation vs. Angle Relationship Plot ---
            print("\n--- EMG Activation vs. Angle Relationship Visualization ---")
            num_samples_for_relation = X_val.shape[0] # Use all validation samples for scatter plot
            if num_samples_for_relation > 0:
                # Example 1: Single EMG channel vs. Joint Angle
                plot_emg_angle_relationship(
                    X_val, y_pred_scaled_all, y_val, # Use all validation data
                    emg_scaler, angle_scaler,
                    emg_channel_name='FDS', # Example EMG channel
                    joint_angle_name='Index 2', # Example Joint Angle
                    use_muscle_group=False,
                    title_suffix="Channel: FDS"
                )

                # Example 2: Muscle Group vs. Joint Angle
                plot_emg_angle_relationship(
                    X_val, y_pred_scaled_all, y_val, # Use all validation data
                    emg_scaler, angle_scaler,
                    emg_channel_name='Finger Flexors', # Example Muscle Group
                    joint_angle_name='Index 2', # Example Joint Angle
                    use_muscle_group=True,
                    muscle_group_info=MUSCLE_GROUP_VIS_INFO,
                    title_suffix="Group: Finger Flexors"
                )
            else: print("Skipping EMG vs. Angle relationship plot (No validation data).")


            # --- 11. MHA Attention Summary Table ---
            # Uses mha_attention_weights_all_val calculated in section 6
            if mha_attention_weights_all_val is not None:
                 summarize_mha_attention(mha_attention_weights_all_val, EMG_CHANNEL_NAMES, FINGER_NAMES)
            else:
                 print("\nSkipping MHA Attention Summary Table (Attention weights not available).")


            # --- 12. Attention-Weighted EMG Snapshot Plot ---
            if PLOT_ATTN_WEIGHTED_SNAPSHOT:
                print("\n--- Attention-Weighted EMG Snapshot Visualization ---")
                # Check if we have the necessary data from the dynamic plot
                if snapshot_data['peak_window_idx'] is not None and \
                   snapshot_data['head_idx'] is not None and \
                   snapshot_data['X_windows'] is not None and \
                   snapshot_data['mha_weights'] is not None:

                    peak_window_idx = snapshot_data['peak_window_idx']
                    head_idx = snapshot_data['head_idx']
                    trial_X_windows = snapshot_data['X_windows']
                    trial_mha_weights = snapshot_data['mha_weights']
                    trial_idx = snapshot_data['trial_idx']
                    trial_id_str = original_scaled_trials[trial_idx]['id'] if trial_idx < len(original_scaled_trials) else f"Trial {trial_idx}"
                    head_name = FINGER_NAMES[head_idx] if head_idx < len(FINGER_NAMES) else f"Head {head_idx}"

                    # Ensure indices are valid
                    if peak_window_idx < trial_X_windows.shape[0] and head_idx < trial_mha_weights.shape[1]:
                        # Extract the specific EMG window
                        emg_snapshot = trial_X_windows[peak_window_idx] # Shape: (seq_len, num_emg)
                        # Extract the attention weights for the specific head and window
                        attention_snapshot = trial_mha_weights[peak_window_idx, head_idx, :] # Shape: (seq_len,)

                        plot_attention_weighted_emg_snapshot(
                            emg_window=emg_snapshot,
                            attention_weights_head=attention_snapshot,
                            muscle_group_info=MUSCLE_GROUP_VIS_INFO,
                            sequence_length=SEQUENCE_LENGTH,
                            head_name=head_name,
                            window_idx=peak_window_idx,
                            trial_id=trial_id_str
                        )
                    else:
                        print(f"Error: Invalid indices for snapshot plot. Peak window: {peak_window_idx}, Head: {head_idx}. Skipping.")

                else:
                    print("Skipping Attention-Weighted EMG Snapshot plot (required data not collected from dynamic simulation).")
            else:
                print("Skipping Attention-Weighted EMG Snapshot plot (disabled by flag).")


            # --- 13. Activation Simulation (NEW) ---
            if RUN_ACTIVATION_SIMULATION:
                 print("\n--- Activation Simulation ---")
                 num_simulations = min(4, X_val.shape[0]) # Simulate for the first validation sample
                 if num_simulations > 0:
                     for i in range(num_simulations):
                         simulate_activation_prediction(
                             emg_input_sequence=X_val[i], # Use i-th validation sequence
                             model=model,
                             angle_scaler=angle_scaler,
                             muscle_group_info=MUSCLE_GROUP_VIS_INFO,
                             muscle_to_finger_heuristic=MUSCLE_TO_FINGER_HEURISTIC,
                             finger_joint_indices=FINGER_JOINT_INDICES,
                             joint_angle_names=JOINT_ANGLE_NAMES,
                             activation_threshold=MUSCLE_ACTIVATION_THRESHOLD,
                             sequence_length=SEQUENCE_LENGTH,
                             sample_idx=i # Pass sample index for title
                         )
                 else:
                     print("Skipping Activation Simulation (no validation samples).")
            else:
                 print("Skipping Activation Simulation (disabled by flag).")


        else:
            print("Preprocessing failed or no validation data. Exiting.")
    else:
        print("Data loading failed. Exiting.")

    print("\n--- Script Execution Finished ---")

    print("Note: Check for any warnings or errors in the output.")