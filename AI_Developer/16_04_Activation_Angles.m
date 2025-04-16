% MATLAB Code for Explainable EMG-to-Joint Angle Prediction using Lasso

%% --- Configuration ---
% Specify the path to your .mat data file
dataFilePath = 'your_data_file.mat'; % !!! REPLACE with your actual file path !!!

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
% We will combine data from all trials and tasks into single matrices.
% This provides more data for training the general relationship.
% Alternatively, you could train models for specific tasks if needed.

fprintf('Preprocessing data (concatenating trials and tasks)...\n');

all_emg_data = [];
all_joint_angle_data = [];

for trial = 1:numTrials
    for task = 1:numTasks
        % Check if cell is empty or has incorrect dimensions before concatenating
        if ~isempty(dsfilt_emg{trial, task}) && size(dsfilt_emg{trial, task}, 2) == numEMGChannels && ...
           ~isempty(joint_angles{trial, task}) && size(joint_angles{trial, task}, 2) == numJointAngles && ...
           size(dsfilt_emg{trial, task}, 1) == size(joint_angles{trial, task}, 1)

            all_emg_data = [all_emg_data; dsfilt_emg{trial, task}];
            all_joint_angle_data = [all_joint_angle_data; joint_angles{trial, task}];
        else
             fprintf('Warning: Skipping cell (%d, %d) due to inconsistent data or dimensions.\n', trial, task);
             % Optional: Add more detailed checks here if needed
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

% --- Optional: Data Standardization ---
% Standardizing features (EMG) and targets (angles) can improve Lasso performance.
% z-score normalization: (value - mean) / std_dev
X_raw = all_emg_data;
Y_raw = all_joint_angle_data;

X_mean = mean(X_raw);
X_std = std(X_raw);
Y_mean = mean(Y_raw);
Y_std = std(Y_raw);

% Avoid division by zero if std dev is zero for any column
X_std(X_std == 0) = 1;
Y_std(Y_std == 0) = 1;

X = (X_raw - X_mean) ./ X_std; % Standardized EMG data
Y = (Y_raw - Y_mean) ./ Y_std; % Standardized Joint Angle data

%% --- 3. Train Explainable Model (Lasso Regression) ---
fprintf('Training Lasso models for each joint angle...\n');

% Store results (coefficients and intercepts)
lasso_coefficients = zeros(numEMGChannels, numJointAngles);
lasso_intercepts = zeros(1, numJointAngles);
lasso_fitinfo = cell(1, numJointAngles); % Store FitInfo for more details

% Loop through each joint angle and train a separate model
for i = 1:numJointAngles
    fprintf('  Training model for: %s\n', jointAngleNames{i});

    % Select the target joint angle (Y column)
    current_Y = Y(:, i);

    % Perform Lasso regression with 10-fold cross-validation
    % 'CV', 10 tells lasso to find the best regularization parameter (Lambda)
    % automatically using cross-validation.
    % 'Alpha', 1 specifies Lasso (default). Alpha=0 would be Ridge.
    [B, FitInfo] = lasso(X, current_Y, 'CV', 10, 'Alpha', 1, 'MaxIter', 1e4);

    % Store FitInfo
    lasso_fitinfo{i} = FitInfo;

    % Select the coefficients corresponding to the Lambda that minimizes MSE
    % Alternatively, use 'Index1SE' for a potentially sparser model within
    % one standard error of the minimum MSE.
    idxLambdaMinMSE = FitInfo.IndexMinMSE;
    % idxLambda1SE = FitInfo.Index1SE; % Use this for a sparser model

    lasso_coefficients(:, i) = B(:, idxLambdaMinMSE);
    lasso_intercepts(i) = FitInfo.Intercept(idxLambdaMinMSE);

    fprintf('    Model training complete for %s.\n', jointAngleNames{i});
end

fprintf('All Lasso models trained.\n');

%% --- 4. Explainability: Analyze Coefficients ---
% The magnitude of the coefficients indicates the importance of each EMG channel
% for predicting that specific joint angle (after standardization).
% Zero coefficients mean Lasso excluded that EMG channel as less important.
% The sign indicates the direction of the correlation (positive or negative).

fprintf('\n--- Explainability Analysis (Lasso Coefficients) ---\n');

% --- Visualization: Heatmap of Coefficients ---
% This provides a good overview of muscle contributions across all angles.
figure;
h = heatmap(jointAngleNames, emgChannelNames, lasso_coefficients, 'Colormap', cool); % Use a diverging colormap
h.Title = 'Lasso Coefficients: EMG Channel Importance for Each Joint Angle';
h.XLabel = 'Joint Angles';
h.YLabel = 'EMG Channels';
h.CellLabelFormat = '%.2f'; % Format cell labels to 2 decimal places
colorbar; % Show color scale
fprintf('Displayed heatmap of Lasso coefficients.\n');


% --- Detailed Output per Joint Angle ---
for i = 1:numJointAngles
    fprintf('\n--- Joint Angle: %s ---\n', jointAngleNames{i});
    fprintf('  Intercept (standardized): %.4f\n', lasso_intercepts(i));
    fprintf('  Coefficients (standardized EMG -> standardized Angle):\n');

    % Sort coefficients by magnitude for easier interpretation
    [sorted_coeffs, sort_idx] = sort(abs(lasso_coefficients(:, i)), 'descend');
    sorted_names = emgChannelNames(sort_idx);

    has_contribution = false;
    for j = 1:numEMGChannels
        coeff_val = lasso_coefficients(sort_idx(j), i);
        if abs(coeff_val) > 1e-6 % Check if coefficient is effectively non-zero
            fprintf('    - %s: %.4f\n', sorted_names{j}, coeff_val);
            has_contribution = true;
        end
    end

    if ~has_contribution
        fprintf('    (No significant EMG contributions found by Lasso for this angle)\n');
    end

    % Optional: Display cross-validation plot for this angle
    % figure;
    % lassoPlot(lasso_coefficients_all{i}, lasso_fitinfo{i}, 'PlotType', 'CV');
    % legend('show'); % Show legend
    % title(['Lasso Cross-Validation for ', jointAngleNames{i}]);
    % xlabel('Lambda (Regularization Parameter)');
    % ylabel('Cross-Validated Mean Squared Error (MSE)');
end

fprintf('\n--- Interpretation Notes ---\n');
fprintf('* Coefficients are based on STANDARDIZED EMG and Angle data.\n');
fprintf('* Larger absolute coefficient magnitude implies stronger influence of that EMG channel on the angle.\n');
fprintf('* Positive coefficient: Increased EMG activity correlates with increased angle value (e.g., flexion).\n');
fprintf('* Negative coefficient: Increased EMG activity correlates with decreased angle value (e.g., extension).\n');
fprintf('* Zero coefficient: Lasso deemed this EMG channel less important for predicting this specific angle.\n');
fprintf('* To get predictions in original angle units, you would need to un-standardize:\n');
fprintf('  Predicted_Angle_Original = (X * lasso_coefficients(:, i) + lasso_intercepts(i)) * Y_std(i) + Y_mean(i);\n');
fprintf('--- End of Analysis ---\n');

% Example: Get prediction for the first joint angle in original units
% predicted_angle1_std = X * lasso_coefficients(:, 1) + lasso_intercepts(1);
% predicted_angle1_original = predicted_angle1_std * Y_std(1) + Y_mean(1);
% You can compare 'predicted_angle1_original' with 'Y_raw(:, 1)' to evaluate performance.

