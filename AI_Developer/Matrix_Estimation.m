% MATLAB Script for Estimating Joint Angles from sEMG using LSTM
% Uses data from ONE specified subject.
% Trains on Trials 1-4, Tests on Trial 5.
% Includes Early Stopping using a validation set.
% Calculates RMSE, Correlation Coefficient (CC), and R-squared (R2).
% Assumes Deep Learning Toolbox is installed.
% The calculateR2 function is included at the end of this script.

clear; clc; close all; % Start fresh

% --- 0. Configuration ---
% !!! SELECT THE SUBJECT FILE TO USE HERE !!!
selected_subject_file = 's4_full.mat'; % Options: 's1_full.mat', 's2_full.mat', 's4_full.mat'

fprintf('Using data from selected subject: %s\n', selected_subject_file);

% Define Training and Testing Trials
train_trials = 1:4;
test_trials = 5;

% --- Data Structure Assumptions ---
num_trials_per_subj = 5; % Total trials available per subject file
num_tasks_per_subj = 7;

% --- 1. LSTM Network Parameters (Define Once) ---
window_size = 200; 
num_features_emg = 8; 
num_responses_angles = 14; 
num_hidden_units = 150; % (tune this)

% --- 2. Training Parameters (Define Once) ---
solver_name = 'adam'; 
max_epochs = 20; 
mini_batch_size = 64; 
initial_learn_rate = 0.005; 
gradient_threshold = 1;
validation_frequency = 200; 
validation_split_ratio = 0.15; % Use 15% of training data for validation
shuffle_opt = 'every-epoch';
plots_opt = 'training-progress'; 
verbose_opt = 1; 
execution_env = 'gpu'; % Uncomment for GPU training

% --- 3. Load and Aggregate Data for the Selected Subject ---
fprintf('Loading and aggregating data for %s...\n', selected_subject_file);

% --- 3.1 Load Data for Selected Subject ---
try
    subject_data = load(selected_subject_file);
catch ME
    error('Failed to load data file: %s. Error: %s', selected_subject_file, ME.message);
end

% Basic validation 
if ~isfield(subject_data, 'dsfilt_emg') || ~isfield(subject_data, 'joint_angles') || ...
   ~iscell(subject_data.dsfilt_emg) || ~iscell(subject_data.joint_angles) || ...
   size(subject_data.dsfilt_emg, 1) ~= num_trials_per_subj || size(subject_data.dsfilt_emg, 2) ~= num_tasks_per_subj || ...
   size(subject_data.joint_angles, 1) ~= num_trials_per_subj || size(subject_data.joint_angles, 2) ~= num_tasks_per_subj
     error('Data structure validation failed for %s.', selected_subject_file);
end

% --- 3.2 Aggregate Training Data (Trials 1-4) ---
emg_train_list = {}; 
angle_train_list = {};
for trial_idx = 1:length(train_trials)
    trial = train_trials(trial_idx);
    for task = 1:num_tasks_per_subj
        if ~isempty(subject_data.dsfilt_emg{trial, task}) && ~isempty(subject_data.joint_angles{trial, task})
             if size(subject_data.dsfilt_emg{trial, task}, 2) ~= num_features_emg || size(subject_data.joint_angles{trial, task}, 2) ~= num_responses_angles
                 warning('Skipping Trial %d, Task %d (Train) due to inconsistent column count.', trial, task); continue;
             end
             if size(subject_data.dsfilt_emg{trial, task}, 1) ~= size(subject_data.joint_angles{trial, task}, 1)
                 warning('Skipping Trial %d, Task %d (Train) due to inconsistent time steps.', trial, task); continue;
             end
            emg_train_list{end+1} = subject_data.dsfilt_emg{trial, task};
            angle_train_list{end+1} = subject_data.joint_angles{trial, task};
        end
    end
end
if isempty(emg_train_list)
    error('No training data collected for trials %s. Check subject file and trial/task structure.', mat2str(train_trials));
end
emg_data_train_full = vertcat(emg_train_list{:});
angle_data_train_full = vertcat(angle_train_list{:});
fprintf('Training data aggregated (%d total time steps from trials %s).\n', ...
        size(emg_data_train_full, 1), mat2str(train_trials));
clear emg_train_list angle_train_list; 

% --- 3.3 Aggregate Test Data (Trial 5) ---
emg_test_list = {}; 
angle_test_list = {};
for trial_idx = 1:length(test_trials)
    trial = test_trials(trial_idx);
     for task = 1:num_tasks_per_subj
        if ~isempty(subject_data.dsfilt_emg{trial, task}) && ~isempty(subject_data.joint_angles{trial, task})
             if size(subject_data.dsfilt_emg{trial, task}, 2) ~= num_features_emg || size(subject_data.joint_angles{trial, task}, 2) ~= num_responses_angles
                 warning('Skipping Trial %d, Task %d (Test) due to inconsistent column count.', trial, task); continue;
             end
             if size(subject_data.dsfilt_emg{trial, task}, 1) ~= size(subject_data.joint_angles{trial, task}, 1)
                 warning('Skipping Trial %d, Task %d (Test) due to inconsistent time steps.', trial, task); continue;
             end
            emg_test_list{end+1} = subject_data.dsfilt_emg{trial, task};
            angle_test_list{end+1} = subject_data.joint_angles{trial, task};
        end
    end
end
if isempty(emg_test_list)
    error('No testing data collected for trials %s. Check subject file and trial/task structure.', mat2str(test_trials));
end
emg_data_test_full = vertcat(emg_test_list{:});
angle_data_test_full = vertcat(angle_test_list{:});
fprintf('Testing data aggregated (%d total time steps from trials %s).\n', ...
        size(emg_data_test_full, 1), mat2str(test_trials));
clear emg_test_list angle_test_list subject_data; 

% --- 4. Normalization (Based on TRAINING data ONLY from the selected subject) ---
fprintf('Normalizing data...\n');
[emg_data_train_norm, mu_emg_train, sig_emg_train] = zscore(emg_data_train_full);
[angle_data_train_norm, mu_angle_train, sig_angle_train] = zscore(angle_data_train_full);
sig_emg_train(sig_emg_train == 0) = eps;
sig_angle_train(sig_angle_train == 0) = eps;
emg_data_train_norm = (emg_data_train_full - mu_emg_train) ./ sig_emg_train; 
angle_data_train_norm = (angle_data_train_full - mu_angle_train) ./ sig_angle_train;
emg_data_train_norm(isnan(emg_data_train_norm)) = 0; 
angle_data_train_norm(isnan(angle_data_train_norm)) = 0;

emg_data_test_norm = (emg_data_test_full - mu_emg_train) ./ sig_emg_train;
angle_data_test_norm = (angle_data_test_full - mu_angle_train) ./ sig_angle_train;
emg_data_test_norm(isnan(emg_data_test_norm)) = 0; 
angle_data_test_norm(isnan(angle_data_test_norm)) = 0;

clear emg_data_train_full angle_data_train_full emg_data_test_full angle_data_test_full;

% --- 5. Windowing and Splitting Training/Validation Data ---
fprintf('Windowing and splitting data...\n');
num_time_steps_train_all = size(emg_data_train_norm, 1);
num_samples_train_all = num_time_steps_train_all - window_size;
if num_samples_train_all < 2 
    error('Training data length (%d) is too short for window size (%d) to allow validation split.', num_time_steps_train_all, window_size);
end

X_train_all_sequences = zeros(num_features_emg, window_size, 1, num_samples_train_all);
Y_train_all_matrix_norm = zeros(num_samples_train_all, num_responses_angles);
for i = 1:num_samples_train_all
    X_train_all_sequences(:, :, 1, i) = emg_data_train_norm(i : i+window_size-1, :)';
    Y_train_all_matrix_norm(i, :) = angle_data_train_norm(i+window_size-1, :);
end
X_train_all_cell = squeeze(num2cell(X_train_all_sequences, [1 2]));
clear X_train_all_sequences; 

cv_train_val = cvpartition(num_samples_train_all, 'HoldOut', validation_split_ratio);
idx_train_final = training(cv_train_val);
idx_val = test(cv_train_val);

X_train = X_train_all_cell(idx_train_final);
Y_train = Y_train_all_matrix_norm(idx_train_final, :); 
X_val = X_train_all_cell(idx_val);
Y_val = Y_train_all_matrix_norm(idx_val, :);
clear X_train_all_cell Y_train_all_matrix_norm; 

fprintf('Training data split into %d training samples and %d validation samples.\n', length(X_train), length(X_val));

 if any(isnan(Y_train), 'all') || any(isnan(Y_val), 'all')
    error('NaNs found in Y_train or Y_val after windowing/splitting.');
end

% --- 6. Window Test Data ---
num_time_steps_test = size(emg_data_test_norm, 1);
num_samples_test = num_time_steps_test - window_size;
 if num_samples_test < 1
    warning('Test data length (%d) is less than window size (%d). No test samples generated.', num_time_steps_test, window_size);
    X_test = {}; 
    Y_test_matrix_norm = zeros(0, num_responses_angles); 
 else
    X_test_sequences = zeros(num_features_emg, window_size, 1, num_samples_test);
    Y_test_matrix_norm = zeros(num_samples_test, num_responses_angles); 
    for i = 1:num_samples_test
        X_test_sequences(:, :, 1, i) = emg_data_test_norm(i : i+window_size-1, :)';
        Y_test_matrix_norm(i, :) = angle_data_test_norm(i+window_size-1, :);
    end
    X_test = squeeze(num2cell(X_test_sequences, [1 2]));
    clear X_test_sequences; 
    fprintf('Test data windowed into %d samples.\n', num_samples_test);
 end
 clear emg_data_train_norm angle_data_train_norm emg_data_test_norm angle_data_test_norm; 

% --- 7. Define LSTM Network Architecture ---
layers = [ ...
    sequenceInputLayer(num_features_emg, 'Name', 'input', 'Normalization', 'none')
    lstmLayer(num_hidden_units, 'OutputMode', 'last', 'Name', 'lstm')
    fullyConnectedLayer(num_responses_angles, 'Name', 'fc')
    regressionLayer('Name', 'output')];

% --- 8. Set Training Options and Train ---
options = trainingOptions(solver_name, ... 
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch_size, ...
    'InitialLearnRate', initial_learn_rate, ...
    'GradientThreshold', gradient_threshold, ...
    'Shuffle', shuffle_opt, ...
    'Plots', plots_opt, ... 
    'Verbose', verbose_opt, ... 
    'ValidationData', {X_val, Y_val}, ... 
    'ValidationFrequency', validation_frequency); 
    % 'ExecutionEnvironment', execution_env, ... 
    
fprintf('Starting network training for subject %s...\n', selected_subject_file);
[net, trainInfo] = trainNetwork(X_train, Y_train, layers, options); 

final_state_message = 'Training state not available.';
if isfield(trainInfo, 'State')
    final_state_message = trainInfo.State;
end
fprintf('Training finished. Final State: %s\n', final_state_message);

clear X_train Y_train X_val Y_val; 

% --- 9. Evaluate the Network on Test Set ---
if isempty(X_test)
    fprintf('Skipping evaluation as no test samples were generated.\n');
    overall_rmse = NaN;
    overall_cc = NaN;
    overall_r2 = NaN; % Initialize R2 as NaN
    rmse_per_angle = nan(1, num_responses_angles);
    cc_per_angle = nan(1, num_responses_angles);
    r2_per_angle = nan(1, num_responses_angles); % Initialize R2 per angle as NaN
else
    fprintf('Evaluating performance on the test set (Subject: %s, Trials: %s)...\n', ...
            selected_subject_file, mat2str(test_trials));
            
    Y_pred_normalized = predict(net, X_test, 'MiniBatchSize', mini_batch_size);
    
    % Denormalize Predictions 
    Y_pred = Y_pred_normalized .* sig_angle_train + mu_angle_train;

    % Denormalize Actual Test Data 
    Y_actual = Y_test_matrix_norm .* sig_angle_train + mu_angle_train;
    
    % --- Calculate Performance Metrics ---
    % RMSE
    rmse_per_angle = sqrt(mean((Y_pred - Y_actual).^2));
    overall_rmse = mean(rmse_per_angle);
    
    % Correlation Coefficient (CC)
    cc_per_angle = zeros(1, num_responses_angles);
    for j = 1:num_responses_angles
        if std(Y_actual(:, j)) > 1e-6 && std(Y_pred(:, j)) > 1e-6
            cc_per_angle(j) = corr(Y_pred(:, j), Y_actual(:, j));
        else
            cc_per_angle(j) = NaN;
        end
    end
    overall_cc = mean(cc_per_angle, 'omitnan');

    % R-squared (R2) - Call the function defined below
    r2_per_angle = calculateR2(Y_actual, Y_pred); % Calculate R2 for each angle
    overall_r2 = mean(r2_per_angle, 'omitnan'); % Calculate average R2, ignoring NaNs
    
    fprintf('Overall Test RMSE = %.4f, Overall Test CC = %.4f, Overall Test R2 = %.4f\n', ...
            overall_rmse, overall_cc, overall_r2);

    % --- 10. Report and Visualize Results ---
    fprintf('\n--- Test Set Performance (Subject: %s, Trials: %s) ---\n', ...
            selected_subject_file, mat2str(test_trials));
    fprintf('Metric        | Value   \n');
    fprintf('--------------|---------\n');
    fprintf('Overall RMSE  | %7.4f \n', overall_rmse);
    fprintf('Overall CC    | %7.4f \n', overall_cc); 
    fprintf('Overall R2    | %7.4f \n', overall_r2); % Added R2 report
    fprintf('\nPer-Angle Test Performance:\n');
    fprintf('Angle |   RMSE  |   CC   |   R2   \n'); % Added R2 column header
    fprintf('------|---------|--------|--------\n');
    for j = 1:num_responses_angles
        fprintf('%5d | %7.4f | %6.4f | %6.4f\n', ... % Added R2 value
                j, rmse_per_angle(j), cc_per_angle(j), r2_per_angle(j));
    end

    % Visualize: Plot prediction vs actual for Angle 1 on Test Set
    figure('Name', sprintf('Test Set: Angle 1 Prediction (%s)', selected_subject_file), 'NumberTitle', 'off');
    plot(Y_actual(:, 1), 'b-', 'LineWidth', 1.5); hold on;
    plot(Y_pred(:, 1), 'r--', 'LineWidth', 1.5); hold off;
    title(sprintf('Prediction vs Actual (Angle 1) on Test Set (Subject: %s, Trials %s)', selected_subject_file, mat2str(test_trials)));
    xlabel('Time Step (in test set)'); ylabel('Angle Value'); legend('Actual Angle', 'Predicted Angle'); grid on;
    
    % Visualize: Scatter plot for Angle 1 on Test Set
    figure('Name', sprintf('Test Set: Angle 1 Scatter (%s)', selected_subject_file), 'NumberTitle', 'off');
    scatter(Y_actual(:, 1), Y_pred(:, 1), 10, 'filled'); hold on;
    min_val = min([Y_actual(:, 1); Y_pred(:, 1)]);
    max_val = max([Y_actual(:, 1); Y_pred(:, 1)]);
    plot([min_val, max_val], [min_val, max_val], 'k--'); hold off; 
    title(sprintf('Predicted vs Actual Scatter (Angle 1) on Test Set (Subject: %s, CC=%.3f, R2=%.3f)', ... % Added R2 to title
                  selected_subject_file, cc_per_angle(1), r2_per_angle(1))); 
    xlabel('Actual Angle Value'); ylabel('Predicted Angle Value'); axis equal; grid on;
    xlim([min_val, max_val]); ylim([min_val, max_val]); 
    
end % End of evaluation check

disp('Script finished.');


% --- Helper Function Definition ---
function r2 = calculateR2(y_actual, y_pred)
%calculateR2 Calculates the R-squared (Coefficient of Determination).
%
%   R2 = calculateR2(Y_ACTUAL, Y_PRED) calculates the R-squared value
%   comparing the predicted values Y_PRED to the actual values Y_ACTUAL.
%
%   Inputs:
%       y_actual - Vector or matrix of actual target values. If a matrix,
%                  R2 is calculated column-wise.
%       y_pred   - Vector or matrix of predicted values, same size as y_actual.
%
%   Output:
%       r2       - R-squared value(s). If inputs are matrices, R2 is a row
%                  vector containing the R2 value for each column. Returns
%                  NaN for columns where y_actual is constant.
%
%   Formula: R2 = 1 - (SS_res / SS_tot)
%       SS_res = sum((y_actual - y_pred).^2)  (Residual Sum of Squares)
%       SS_tot = sum((y_actual - mean(y_actual)).^2) (Total Sum of Squares)

    % Ensure inputs are numeric and have the same size
    if ~isnumeric(y_actual) || ~isnumeric(y_pred)
        error('Inputs must be numeric.');
    end
    if ~isequal(size(y_actual), size(y_pred))
        error('Inputs y_actual and y_pred must have the same size.');
    end

    % Calculate column-wise mean of actual values
    mean_y_actual = mean(y_actual, 1); 
    
    % Calculate Residual Sum of Squares (SS_res) column-wise
    ss_res = sum((y_actual - y_pred).^2, 1);
    
    % Calculate Total Sum of Squares (SS_tot) column-wise
    ss_tot = sum((y_actual - mean_y_actual).^2, 1);
    
    % Calculate R2, handle cases where SS_tot is close to zero (constant actual data)
    r2 = zeros(1, size(y_actual, 2)); % Preallocate result row vector
    for j = 1:size(y_actual, 2) % Iterate through columns (angles)
        if ss_tot(j) < eps % Check if SS_tot is effectively zero
            % R2 is undefined or meaningless if actual data doesn't vary
            r2(j) = NaN; 
            % Suppress warning for cleaner output during loops if needed
            % warning('R2 calculation: Column %d has near-zero total sum of squares (constant actual data). R2 set to NaN.', j);
        else
            r2(j) = 1 - (ss_res(j) / ss_tot(j));
        end
    end
    
end
