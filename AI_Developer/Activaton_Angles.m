% MATLAB Code for Explainable EMG-to-Joint Angle Prediction using Lasso
% Includes example visualization of EMG patterns for a specific angle peak
% Updated to use Time (seconds) on the x-axis based on Fs = 200 Hz
% Removed 'UseParallel' option for compatibility.

%% --- Configuration ---
% Specify the path to your .mat data file
dataFilePath = 's4_full.mat'; % !!! REPLACE with your actual file path !!!

% Define names for EMG channels and Joint Angles for clarity in results
emgChannelNames = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};
jointAngleNames = { ...
    'Thumb 1 (20-17-18)', 'Thumb 2 (17-18-19)', ...
    'Index 1 (20-1-5)', 'Index 2 (1-5-6)', 'Index 3 (5-6-7)', ...
    'Middle 1 (20-2-8)', 'Middle 2 (2-8-9)', 'Middle 3 (8-9-10)', ...
    'Ring 1 (20-3-11)', 'Ring 2 (3-11-12)', 'Ring 3 (11-12-13)', ...
    'Little 1 (20-4-14)', 'Little 2 (4-14-15)', 'Little 3 (14-15-16)'};

numEMGChannels = length(emgChannelNames);
numJointAngles = length(jointAngleNames);
numTrials = 5;
numTasks = 7;

% --- Define Sampling Frequency ---
Fs = 200; % Sampling frequency in Hz (as provided by user)

%% --- 1. Load Data ---
fprintf('Loading data from %s...\n', dataFilePath);
try
    load(dataFilePath, 'dsfilt_emg', 'joint_angles');
    fprintf('Data loaded successfully.\n');
catch ME
    fprintf('Error loading data file: %s\n', ME.message);
    fprintf('Please ensure the file path is correct and the file contains the variables "dsfilt_emg" and "joint_angles".\n');
    return; % Stop execution if file loading fails
end

% --- Verify Data Structure ---
if ~exist('dsfilt_emg', 'var') || ~exist('joint_angles', 'var')
    fprintf('Error: Required variables "dsfilt_emg" or "joint_angles" not found in the loaded file.\n');
    return;
end
if ~iscell(dsfilt_emg) || ~iscell(joint_angles) || ...
   ~isequal(size(dsfilt_emg), [numTrials, numTasks]) || ~isequal(size(joint_angles), [numTrials, numTasks])
    fprintf('Error: Data variables are not in the expected 5x7 cell format.\n');
    return;
end

%% --- 2. Preprocess Data: Concatenate Trials and Tasks ---
fprintf('Preprocessing data (concatenating trials and tasks)...\n');
all_emg_data = [];
all_joint_angle_data = [];
sample_count_offset = 0; % Keep track of sample indices relative to the start of concatenated data
original_indices = []; % Store original indices for time calculation

for trial = 1:numTrials
    for task = 1:numTasks
        if ~isempty(dsfilt_emg{trial, task}) && size(dsfilt_emg{trial, task}, 2) == numEMGChannels && ...
           ~isempty(joint_angles{trial, task}) && size(joint_angles{trial, task}, 2) == numJointAngles && ...
           size(dsfilt_emg{trial, task}, 1) == size(joint_angles{trial, task}, 1)

            num_samples_in_cell = size(dsfilt_emg{trial, task}, 1);
            all_emg_data = [all_emg_data; dsfilt_emg{trial, task}];
            all_joint_angle_data = [all_joint_angle_data; joint_angles{trial, task}];

            % Store the original sample index within the concatenated array
            original_indices = [original_indices; (sample_count_offset + 1 : sample_count_offset + num_samples_in_cell)'];
            sample_count_offset = sample_count_offset + num_samples_in_cell;

        else
             fprintf('Warning: Skipping cell (%d, %d) due to inconsistent data or dimensions.\n', trial, task);
             % (Error details as before)
             if isempty(dsfilt_emg{trial, task}) || isempty(joint_angles{trial, task})
                 fprintf('   Reason: Cell is empty.\n');
             elseif size(dsfilt_emg{trial, task}, 2) ~= numEMGChannels
                  fprintf('   Reason: EMG data has %d columns, expected %d.\n', size(dsfilt_emg{trial, task}, 2), numEMGChannels);
             elseif size(joint_angles{trial, task}, 2) ~= numJointAngles
                  fprintf('   Reason: Joint angle data has %d columns, expected %d.\n', size(joint_angles{trial, task}, 2), numJointAngles);
             elseif size(dsfilt_emg{trial, task}, 1) ~= size(joint_angles{trial, task}, 1)
                  fprintf('   Reason: Mismatch in number of time samples between EMG (%d) and angles (%d).\n', size(dsfilt_emg{trial, task}, 1), size(joint_angles{trial, task}, 1));
             end
        end
    end
end
fprintf('Data concatenated. Total samples: %d\n', size(all_emg_data, 1));
if isempty(all_emg_data) || isempty(all_joint_angle_data)
    fprintf('Error: No valid data found after attempting concatenation. Check data file content.\n');
    return;
end

% --- Store original (unstandardized) data for visualization ---
X_raw = all_emg_data;
Y_raw = all_joint_angle_data;

% --- Data Standardization for Model Training ---
X_mean = mean(X_raw);
X_std = std(X_raw);
Y_mean = mean(Y_raw);
Y_std = std(Y_raw);
X_std(X_std == 0) = 1; % Avoid division by zero
Y_std(Y_std == 0) = 1;
X = (X_raw - X_mean) ./ X_std; % Standardized EMG data
Y = (Y_raw - Y_mean) ./ Y_std; % Standardized Joint Angle data

%% --- 3. Train Explainable Model (Lasso Regression) ---
fprintf('Training Lasso models for each joint angle...\n');
lasso_coefficients = zeros(numEMGChannels, numJointAngles);
lasso_intercepts = zeros(1, numJointAngles);
lasso_fitinfo = cell(1, numJointAngles);
for i = 1:numJointAngles
    fprintf('  Training model for: %s\n', jointAngleNames{i});
    current_Y = Y(:, i);
    % --- Removed 'UseParallel', true for compatibility ---
    [B, FitInfo] = lasso(X, current_Y, 'CV', 10, 'Alpha', 1, 'MaxIter', 1e4);
    lasso_fitinfo{i} = FitInfo;
    idxLambdaMinMSE = FitInfo.IndexMinMSE;
    lasso_coefficients(:, i) = B(:, idxLambdaMinMSE);
    lasso_intercepts(i) = FitInfo.Intercept(idxLambdaMinMSE);
    fprintf('    Model training complete for %s.\n', jointAngleNames{i});
end
fprintf('All Lasso models trained.\n');

%% --- 4. Explainability: Analyze Coefficients ---
fprintf('\n--- Explainability Analysis (Lasso Coefficients) ---\n');
figure;
h = heatmap(jointAngleNames, emgChannelNames, lasso_coefficients, 'Colormap', cool);
h.Title = 'Lasso Coefficients: EMG Channel Importance for Each Joint Angle';
h.XLabel = 'Joint Angles';
h.YLabel = 'EMG Channels';
h.CellLabelFormat = '%.2f';
colorbar;
fprintf('Displayed heatmap of Lasso coefficients.\n');

% (Detailed text output remains the same)
for i = 1:numJointAngles
    fprintf('\n--- Joint Angle: %s ---\n', jointAngleNames{i});
    fprintf('  Intercept (standardized): %.4f\n', lasso_intercepts(i));
    fprintf('  Coefficients (standardized EMG -> standardized Angle):\n');
    [sorted_coeffs, sort_idx] = sort(abs(lasso_coefficients(:, i)), 'descend');
    sorted_names = emgChannelNames(sort_idx);
    has_contribution = false;
    for j = 1:numEMGChannels
        coeff_val = lasso_coefficients(sort_idx(j), i);
        if abs(coeff_val) > 1e-6
            fprintf('    - %s: %.4f\n', sorted_names{j}, coeff_val);
            has_contribution = true;
        end
    end
    if ~has_contribution
        fprintf('    (No significant EMG contributions found by Lasso for this angle)\n');
    end
end
fprintf('\n--- Interpretation Notes ---\n');
fprintf('* Coefficients are based on STANDARDIZED EMG and Angle data.\n');
fprintf('* Larger absolute coefficient magnitude implies stronger influence.\n');
fprintf('* Positive coefficient: Increased EMG -> Increased Angle.\n');
fprintf('* Negative coefficient: Increased EMG -> Decreased Angle.\n');
fprintf('* Zero coefficient: Lasso deemed this EMG channel less important.\n');
fprintf('* To get predictions in original units: Predicted_Angle = (X * Coeffs + Intercept) * Y_std + Y_mean;\n');
fprintf('--- End of Analysis ---\n');


%% --- 5. Visualization: Example EMG Pattern for Specific Angle ---
fprintf('\n--- Visualization: EMG Pattern for Specific Angle Peak ---\n');

% --- Configuration for Visualization ---
angle_to_visualize_idx = 4; % Example: Index 2 (1-5-6). Change this index (1-14) to visualize other angles.
window_size_half = 100; % Number of samples before and after the peak (100 samples = 0.5 seconds at 200 Hz)

% Ensure the chosen index is valid
if angle_to_visualize_idx < 1 || angle_to_visualize_idx > numJointAngles
    fprintf('Error: Invalid angle_to_visualize_idx (%d). Must be between 1 and %d.\n', angle_to_visualize_idx, numJointAngles);
    return;
end

targetAngleName = jointAngleNames{angle_to_visualize_idx};
fprintf('Visualizing data around the peak value of: %s (Fs = %d Hz)\n', targetAngleName, Fs);

% Find the time index where the selected angle (raw data) is maximum
[maxAngleValue, t_peak_idx] = max(Y_raw(:, angle_to_visualize_idx)); % Index within concatenated array
fprintf('  Peak angle value %.2f found at sample index %d.\n', maxAngleValue, t_peak_idx);

% Define the sample index window for plotting
t_start_idx = max(1, t_peak_idx - window_size_half);
t_end_idx = min(size(X_raw, 1), t_peak_idx + window_size_half);
sample_indices_window = t_start_idx:t_end_idx;

% --- Calculate time vector in seconds ---
% Use the original_indices to get the correct time relative to the start of recording
% Assuming the first sample corresponds to time 0 or 1/Fs
time_vector_sec = (original_indices(sample_indices_window) - 1) / Fs; % Time in seconds, starting near 0

% Extract data for the window using sample indices
angle_window = Y_raw(sample_indices_window, angle_to_visualize_idx);
emg_window = X_raw(sample_indices_window, :);

% Get the time of the peak in seconds
t_peak_sec = (original_indices(t_peak_idx) - 1) / Fs;

% Create the plot
figure;

% Subplot 1: Joint Angle vs Time
subplot(2, 1, 1); % 2 rows, 1 column, first plot
plot(time_vector_sec, angle_window, 'LineWidth', 1.5, 'Color', 'k');
hold on;
plot(t_peak_sec, Y_raw(t_peak_idx, angle_to_visualize_idx), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8); % Mark the peak
hold off;
title(['Joint Angle: ', targetAngleName]);
ylabel('Angle Value (Original Units)');
xlabel('Time (seconds)'); % Updated X-axis label
xlim([time_vector_sec(1) time_vector_sec(end)]); % Set x-axis limits
grid on;

% Subplot 2: EMG Signals vs Time
subplot(2, 1, 2); % 2 rows, 1 column, second plot
plot(time_vector_sec, emg_window, 'LineWidth', 1); % Plot all 8 EMG channels vs time
title('Corresponding EMG Signals (8 Channels)');
ylabel('EMG Value (Original Units)');
xlabel('Time (seconds)'); % Updated X-axis label
xlim([time_vector_sec(1) time_vector_sec(end)]); % Set x-axis limits
legend(emgChannelNames, 'Location', 'eastoutside'); % Add legend for EMG channels
grid on;

fprintf('Displayed line charts showing angle and corresponding EMG pattern around the peak vs Time (seconds).\n');
fprintf('--- End of Visualization ---\n');

