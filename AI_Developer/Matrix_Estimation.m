% MATLAB Script for Estimating Joint Angles from sEMG using a Physics-Informed MLP (PINN MLP)
% Compares PINN MLP (trained with custom physics loss) vs Standard MLP (trained with MSE loss).
% Uses Feature Extraction pipeline. Trains on Trials 1-4, Tests on Trial 5 (Single Subject).
% Implements custom training loop for PINN MLP.
% Includes Early Stopping (manual implementation in custom loop).
% Calculates RMSE, CC, R2.
% Adds Comparative Feature Importance visualization (XAI).
% Assumes Deep Learning Toolbox is installed.
% Helper functions included at the end.

clear; clc; close all; % Start fresh

% --- 0. Configuration ---
selected_subject_file = 's1_full.mat'; % Options: 's1_full.mat', 's2_full.mat', 's4_full.mat'
fprintf('Using data from selected subject: %s\n', selected_subject_file);

train_trials = 1:4;
test_trials = 5;

% --- Data Structure Assumptions ---
num_trials_per_subj = 5; 
num_tasks_per_subj = 7;

% --- 1. Model & Feature Parameters ---
window_size = 50; 
num_emg_channels = 8; 
num_responses_angles = 14; 

feature_list = {'RMS', 'MAV', 'WL'}; 
num_features_per_channel = length(feature_list);
num_combined_features = num_emg_channels * num_features_per_channel; 

% MLP Network Parameters
hidden_layer_1_size = 64;  
hidden_layer_2_size = 48;  
fc_output_layer_size = num_responses_angles;

% --- 2. Training Parameters ---
solver_name = 'adam'; % Used for standard MLP via trainNetwork
max_epochs = 100; % Reduced epochs for custom loop example, adjust as needed
mini_batch_size = 128; 
initial_learn_rate = 0.001; 
gradient_threshold = Inf; % Not used in standard adamupdate
validation_frequency_iters = 50; % Check validation every N iterations in custom loop
validation_split_ratio = 0.15; 
shuffle_opt = 'every-epoch'; % Manual shuffle needed in custom loop
plots_opt = 'training-progress'; % For standard MLP trainNetwork
verbose_opt = 1; 

% PINN Specific Parameters
lambda_physics = 0.01; % Weighting factor for the physics penalty term (TUNE THIS)
early_stopping_patience = 10; % Stop if validation loss doesn't improve for N checks

% --- ASSUMED Joint Angle Limits (Degrees) ---
% !!! IMPORTANT: Replace these with physiologically accurate values !!!
% Example: Assuming all angles roughly between -30 and 120 degrees
% These limits are applied AFTER denormalization within the loss function
angle_lower_limits_deg = -30 * ones(1, num_responses_angles); 
angle_upper_limits_deg = 120 * ones(1, num_responses_angles);
% Example: Make Thumb 1 & 2 limits different if known
% angle_lower_limits_deg(1:2) = -10; 
% angle_upper_limits_deg(1:2) = 90; 

fprintf('INFO: Using ASSUMED angle limits (deg): Min=%.1f, Max=%.1f (may vary per angle)\n', ...
        min(angle_lower_limits_deg), max(angle_upper_limits_deg));

% --- 3. Load and Aggregate Data ---
fprintf('Loading and aggregating data for %s...\n', selected_subject_file);
% (Loading and Aggregation logic remains the same)
% ... [Code from Section 3.1, 3.2, 3.3 remains unchanged] ...
% --- 3.1 Load Data for Selected Subject ---
try, subject_data = load(selected_subject_file); catch ME, error('Failed to load data file: %s. Error: %s', selected_subject_file, ME.message); end
if ~isfield(subject_data, 'dsfilt_emg') || ~isfield(subject_data, 'joint_angles') || ~iscell(subject_data.dsfilt_emg) || ~iscell(subject_data.joint_angles) || size(subject_data.dsfilt_emg, 1) ~= num_trials_per_subj || size(subject_data.dsfilt_emg, 2) ~= num_tasks_per_subj || size(subject_data.joint_angles, 1) ~= num_trials_per_subj || size(subject_data.joint_angles, 2) ~= num_tasks_per_subj, error('Data structure validation failed for %s.', selected_subject_file); end
% --- 3.2 Aggregate Training Data (Trials 1-4 - Historical Data) ---
emg_train_list = {}; angle_train_list = {};
fprintf('  Aggregating historical data for training (Trials %s)...\n', mat2str(train_trials));
for trial_idx = 1:length(train_trials), trial = train_trials(trial_idx); for task = 1:num_tasks_per_subj
if ~isempty(subject_data.dsfilt_emg{trial, task}) && ~isempty(subject_data.joint_angles{trial, task})
if size(subject_data.dsfilt_emg{trial, task}, 2) ~= num_emg_channels || size(subject_data.joint_angles{trial, task}, 2) ~= num_responses_angles, warning('Skipping Trial %d, Task %d (Train) due to inconsistent column count.', trial, task); continue; end
if size(subject_data.dsfilt_emg{trial, task}, 1) ~= size(subject_data.joint_angles{trial, task}, 1), warning('Skipping Trial %d, Task %d (Train) due to inconsistent time steps.', trial, task); continue; end
emg_train_list{end+1} = subject_data.dsfilt_emg{trial, task}; angle_train_list{end+1} = subject_data.joint_angles{trial, task};
end, end, end
if isempty(emg_train_list), error('No training data collected for trials %s.', mat2str(train_trials)); end
emg_data_train_full = vertcat(emg_train_list{:}); angle_data_train_full = vertcat(angle_train_list{:});
fprintf('Training data aggregated (%d total time steps from trials %s).\n', size(emg_data_train_full, 1), mat2str(train_trials));
clear emg_train_list angle_train_list; 
% --- 3.3 Aggregate Test Data (Trial 5 - Future/Unseen Data) ---
emg_test_list = {}; angle_test_list = {};
fprintf('  Aggregating future/unseen data for testing (Trials %s)...\n', mat2str(test_trials));
for trial_idx = 1:length(test_trials), trial = test_trials(trial_idx); for task = 1:num_tasks_per_subj
if ~isempty(subject_data.dsfilt_emg{trial, task}) && ~isempty(subject_data.joint_angles{trial, task})
if size(subject_data.dsfilt_emg{trial, task}, 2) ~= num_emg_channels || size(subject_data.joint_angles{trial, task}, 2) ~= num_responses_angles, warning('Skipping Trial %d, Task %d (Test) due to inconsistent column count.', trial, task); continue; end
if size(subject_data.dsfilt_emg{trial, task}, 1) ~= size(subject_data.joint_angles{trial, task}, 1), warning('Skipping Trial %d, Task %d (Test) due to inconsistent time steps.', trial, task); continue; end
emg_test_list{end+1} = subject_data.dsfilt_emg{trial, task}; angle_test_list{end+1} = subject_data.joint_angles{trial, task};
end, end, end
if isempty(emg_test_list), error('No testing data collected for trials %s.', mat2str(test_trials)); end
emg_data_test_full = vertcat(emg_test_list{:}); angle_data_test_full = vertcat(angle_test_list{:});
fprintf('Testing data aggregated (%d total time steps from trials %s).\n', size(emg_data_test_full, 1), mat2str(test_trials));
clear emg_test_list angle_test_list subject_data; 

% --- 4. Normalization ---
fprintf('Normalizing data...\n');
% (Normalization code remains the same)
% ... [Code from Section 4 remains unchanged] ...
[emg_data_train_norm, mu_emg_train, sig_emg_train] = zscore(emg_data_train_full);
[angle_data_train_norm, mu_angle_train, sig_angle_train] = zscore(angle_data_train_full);
sig_emg_train(sig_emg_train == 0) = eps; sig_angle_train(sig_angle_train == 0) = eps;
emg_data_train_norm = (emg_data_train_full - mu_emg_train) ./ sig_emg_train; 
angle_data_train_norm = (angle_data_train_full - mu_angle_train) ./ sig_angle_train;
emg_data_train_norm(isnan(emg_data_train_norm)) = 0; angle_data_train_norm(isnan(angle_data_train_norm)) = 0;
emg_data_test_norm = (emg_data_test_full - mu_emg_train) ./ sig_emg_train;
angle_data_test_norm = (angle_data_test_full - mu_angle_train) ./ sig_angle_train;
emg_data_test_norm(isnan(emg_data_test_norm)) = 0; angle_data_test_norm(isnan(angle_data_test_norm)) = 0;
clear emg_data_train_full angle_data_train_full emg_data_test_full angle_data_test_full;

% --- 5. Windowing, Feature Extraction, and Splitting Training/Validation Data ---
fprintf('Windowing, extracting features, and splitting data...\n');
% (Feature extraction logic remains the same)
% ... [Code from Section 5 remains unchanged] ...
num_time_steps_train_all = size(emg_data_train_norm, 1);
num_samples_train_all = num_time_steps_train_all - window_size;
if num_samples_train_all < 2 , error('Training data length (%d) is too short for window size (%d) to allow validation split.', num_time_steps_train_all, window_size); end
X_train_all_features = zeros(num_samples_train_all, num_combined_features, 'single');
Y_train_all_matrix_norm = zeros(num_samples_train_all, num_responses_angles, 'single');
for i = 1:num_samples_train_all
    emg_window = emg_data_train_norm(i : i+window_size-1, :); 
    X_train_all_features(i, :) = single(extractFeatures(emg_window, feature_list)); 
    Y_train_all_matrix_norm(i, :) = single(angle_data_train_norm(i+window_size-1, :));
end
cv_train_val = cvpartition(num_samples_train_all, 'HoldOut', validation_split_ratio);
idx_train_final = training(cv_train_val); idx_val = test(cv_train_val);
X_train = X_train_all_features(idx_train_final, :); Y_train = Y_train_all_matrix_norm(idx_train_final, :); 
X_val = X_train_all_features(idx_val, :); Y_val = Y_train_all_matrix_norm(idx_val, :);
clear X_train_all_features Y_train_all_matrix_norm; 
fprintf('Training data split into %d training samples and %d validation samples.\n', size(X_train, 1), size(X_val, 1));
 if any(isnan(Y_train), 'all') || any(isnan(Y_val), 'all'), error('NaNs found in Y_train or Y_val after feature extraction/splitting.'); end

% --- 6. Window and Extract Features for Test Data ---
% (Feature extraction logic remains the same)
% ... [Code from Section 6 remains unchanged] ...
num_time_steps_test = size(emg_data_test_norm, 1);
num_samples_test = num_time_steps_test - window_size;
 if num_samples_test < 1
    warning('Test data length (%d) is less than window size (%d). No test samples generated.', num_time_steps_test, window_size);
    X_test = zeros(0, num_combined_features, 'single'); Y_test_matrix_norm = zeros(0, num_responses_angles, 'single'); 
 else
    X_test_features = zeros(num_samples_test, num_combined_features, 'single'); Y_test_matrix_norm = zeros(num_samples_test, num_responses_angles, 'single'); 
    for i = 1:num_samples_test
        emg_window = emg_data_test_norm(i : i+window_size-1, :);
        X_test_features(i, :) = single(extractFeatures(emg_window, feature_list));
        Y_test_matrix_norm(i, :) = single(angle_data_test_norm(i+window_size-1, :));
    end
    X_test = X_test_features; clear X_test_features;
    fprintf('Test data windowed and features extracted into %d samples.\n', num_samples_test);
 end
 clear emg_data_train_norm angle_data_train_norm emg_data_test_norm angle_data_test_norm; 

% --- 7. Define MLP Network Architecture ---
% Create Layer Graph for use with dlnetwork and trainNetwork
layers = [ ...
    featureInputLayer(num_combined_features, 'Name', 'input', 'Normalization', 'none') 
    fullyConnectedLayer(hidden_layer_1_size, 'Name', 'fc1')
    reluLayer('Name', 'relu1') 
    fullyConnectedLayer(hidden_layer_2_size, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(fc_output_layer_size, 'Name', 'fc_output') 
    regressionLayer('Name', 'output')]; 
lgraph = layerGraph(layers);

% --- 8. Train Standard MLP (using trainNetwork for comparison) ---
fprintf('\n--- Training Standard MLP (MSE Loss) ---\n');
options_standard = trainingOptions(solver_name, ... 
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch_size, ...
    'InitialLearnRate', initial_learn_rate, ...
    'Shuffle', shuffle_opt, ...
    'Plots', plots_opt, ... 
    'Verbose', verbose_opt, ... 
    'ValidationData', {X_val, Y_val}, ... 
    'ValidationFrequency', validation_frequency * floor(size(X_train,1)/mini_batch_size), ... % Adjust freq based on iters/epoch
    'ValidationPatience', early_stopping_patience); 
    % 'ExecutionEnvironment', execution_env, ... 
    
[net_standard, trainInfo_standard] = trainNetwork(X_train, Y_train, lgraph, options_standard); 
final_state_message_standard = 'Training state not available.';
if isfield(trainInfo_standard, 'State'), final_state_message_standard = trainInfo_standard.State; end
fprintf('Standard MLP Training finished. Final State: %s\n', final_state_message_standard);

% --- 9. Train PINN MLP (using Custom Training Loop) ---
fprintf('\n--- Training PINN MLP (Physics-Informed Loss) ---\n');

% Convert layer graph to dlnetwork
dlnet_pinn = dlnetwork(lgraph);

% Prepare data for custom loop (using dlarray and minibatchqueue)
dsTrain = arrayDatastore([X_train, Y_train], 'OutputType', "same-type", 'ReadSize', mini_batch_size);
mbqTrain = minibatchqueue(dsTrain,...
    'MiniBatchSize', mini_batch_size,...
    'MiniBatchFormat', {'BC', 'BC'},... % Batch, Channel (Features/Angles)
    'OutputEnvironment', 'auto'); % Use GPU if available

% Validation data as dlarray
dlX_val = dlarray(X_val', 'CB'); % Transpose X_val to [Features x Samples] -> CB format
dlY_val = dlarray(Y_val', 'CB'); % Transpose Y_val to [Angles x Samples] -> CB format

% ADAM optimizer state
trailingAvg = [];
trailingAvgSq = [];

% Custom Training Loop Variables
iteration = 0;
start = tic;
best_val_loss = inf;
epochs_without_improvement = 0;
plotter = trainingProgressMonitor(Metrics=["TrainingLoss","ValidationLoss"], XLabel="Iteration"); % Monitor progress

% Convert angle limits to normalized scale (Z-score) for use in loss function
angle_lower_limits_norm = (angle_lower_limits_deg - mu_angle_train) ./ sig_angle_train;
angle_upper_limits_norm = (angle_upper_limits_deg - mu_angle_train) ./ sig_angle_train;

% Custom Training Loop
for epoch = 1:max_epochs
    
    % Shuffle training data queue
    shuffle(mbqTrain);
    
    % Loop over mini-batches
    while hasdata(mbqTrain)
        iteration = iteration + 1;
        
        % Read mini-batch and convert to dlarray
        [dlX_batch, dlY_batch] = next(mbqTrain);
        
        % Evaluate model gradients and loss using dlfeval and modelGradients helper
        [gradients, loss, state] = dlfeval(@modelGradients, dlnet_pinn, dlX_batch, dlY_batch, ...
                                           lambda_physics, angle_lower_limits_norm, angle_upper_limits_norm);
        dlnet_pinn.State = state; % Update batch norm state if applicable
        
        % Update network parameters using ADAM optimizer
        [dlnet_pinn.Learnables, trailingAvg, trailingAvgSq] = adamupdate(dlnet_pinn.Learnables, gradients, ...
            trailingAvg, trailingAvgSq, iteration, initial_learn_rate);
        
        % Log training loss
        current_loss = double(gather(extractdata(loss)));
        recordMetrics(plotter, iteration, TrainingLoss=current_loss);

        % --- Validation Check ---
        if mod(iteration, validation_frequency_iters) == 0
            % Predict on validation set
            dlY_val_pred = predict(dlnet_pinn, dlX_val);
            
            % Calculate validation loss (using only MSE part for fair comparison)
            val_loss_mse = mse(dlY_val_pred, dlY_val);
            val_loss = double(gather(extractdata(val_loss_mse)));
            recordMetrics(plotter, iteration, ValidationLoss=val_loss);
            updateInfo(plotter, Epoch=epoch + iteration/floor(size(X_train,1)/mini_batch_size)); % Approximate epoch
            
            % --- Early Stopping Logic ---
            if val_loss < best_val_loss
                best_val_loss = val_loss;
                epochs_without_improvement = 0;
                % Optional: Save the best network state here
                % net_pinn_best = dlnet_pinn; 
            else
                epochs_without_improvement = epochs_without_improvement + 1;
            end
            
            if epochs_without_improvement >= early_stopping_patience
                fprintf('Early stopping triggered at iteration %d (Epoch ~%d) due to validation loss.\n', iteration, epoch);
                break; % Break inner while loop
            end
        end % End validation check
        
    end % End while hasdata
    
    % Check early stopping condition again to break outer epoch loop
    if epochs_without_improvement >= early_stopping_patience
        break; 
    end
    
end % End epoch loop
fprintf('PINN MLP Training finished. Total Iterations: %d\n', iteration);
net_pinn = dlnet_pinn; % Assign the final network

clear dlX_val dlY_val mbqTrain dsTrain dlX_batch dlY_batch trailingAvg trailingAvgSq plotter; % Clean up

% --- 10. Evaluate Both Networks on Test Set ---
if isempty(X_test) || size(X_test,1)==0 
    fprintf('Skipping evaluation as no test samples were generated.\n');
else
    fprintf('\n--- Evaluating Standard MLP ---\n');
    Y_pred_norm_standard = predict(net_standard, X_test, 'MiniBatchSize', mini_batch_size);
    Y_pred_standard = double(single(Y_pred_norm_standard) .* sig_angle_train + mu_angle_train);
    Y_actual = double(Y_test_matrix_norm .* sig_angle_train + mu_angle_train); % Calculate once
    
    rmse_standard = sqrt(mean((Y_pred_standard - Y_actual).^2));
    cc_standard = calculateCorrelation(Y_actual, Y_pred_standard);
    r2_standard = calculateR2(Y_actual, Y_pred_standard);
    fprintf('Standard MLP: Overall RMSE=%.4f, CC=%.4f, R2=%.4f\n', mean(rmse_standard), mean(cc_standard,'omitnan'), mean(r2_standard,'omitnan'));

    fprintf('\n--- Evaluating PINN MLP ---\n');
    % Predict with PINN (requires dlarray input)
    dlX_test = dlarray(X_test', 'CB'); % Transpose X_test to [Features x Samples] -> CB format
    dlY_pred_norm_pinn = predict(net_pinn, dlX_test);
    Y_pred_norm_pinn = extractdata(gather(dlY_pred_norm_pinn))'; % Convert back to matrix [Samples x Angles]
    Y_pred_pinn = double(single(Y_pred_norm_pinn) .* sig_angle_train + mu_angle_train);
    
    rmse_pinn = sqrt(mean((Y_pred_pinn - Y_actual).^2));
    cc_pinn = calculateCorrelation(Y_actual, Y_pred_pinn);
    r2_pinn = calculateR2(Y_actual, Y_pred_pinn);
    fprintf('PINN MLP:     Overall RMSE=%.4f, CC=%.4f, R2=%.4f\n', mean(rmse_pinn), mean(cc_pinn,'omitnan'), mean(r2_pinn,'omitnan'));
    
    clear dlX_test dlY_pred_norm_pinn; % Clean up

    % --- 11. Report and Visualize Results ---
    fprintf('\n--- Test Set Performance Comparison (Subject: %s, Trials: %s) ---\n', ...
            selected_subject_file, mat2str(test_trials));
    fprintf('Metric        | Standard MLP |   PINN MLP   \n');
    fprintf('--------------|--------------|--------------\n');
    fprintf('Overall RMSE  |   %7.4f    |   %7.4f    \n', mean(rmse_standard), mean(rmse_pinn));
    fprintf('Overall CC    |   %7.4f    |   %7.4f    \n', mean(cc_standard, 'omitnan'), mean(cc_pinn, 'omitnan')); 
    fprintf('Overall R2    |   %7.4f    |   %7.4f    \n', mean(r2_standard, 'omitnan'), mean(r2_pinn, 'omitnan')); 
    
    % Visualize: Time series comparison (Angle 1)
    figure('Name', sprintf('Test Set: Angle 1 Prediction Comparison (%s)', selected_subject_file), 'NumberTitle', 'off');
    plot(Y_actual(:, 1), 'k-', 'LineWidth', 1.5); hold on;
    plot(Y_pred_standard(:, 1), 'b--', 'LineWidth', 1); 
    plot(Y_pred_pinn(:, 1), 'r:', 'LineWidth', 1); hold off;
    title(sprintf('Prediction vs Actual (Angle 1) on Test Set (Subject: %s)', selected_subject_file));
    xlabel('Time Step (in test set)'); ylabel('Angle Value'); 
    legend('Actual', 'Standard MLP', 'PINN MLP'); grid on;
    
    % --- 12. XAI: Comparative Feature Importance ---
    fprintf('\nCalculating Comparative Feature Importance for a sample...\n');
    feature_importance_standard = []; feature_importance_pinn = []; feature_labels = {};
    xai_data_saved = false; 
    
    try 
        dlnet_standard = dlnetwork(layerGraph(net_standard)); % Convert standard net
        dlnet_pinn = net_pinn; % Already a dlnetwork

        sample_idx_for_xai = 1; 
        target_angle_idx_for_xai = 1; 

        if sample_idx_for_xai > size(X_test, 1)
            warning('Selected sample index for XAI is out of bounds.');
        else
            X_sample_features = X_test(sample_idx_for_xai, :); 
            dlX_sample = dlarray(single(X_sample_features), 'CB'); 

            % --- Gradients for Standard MLP ---
            dlPredictFcn_Std = @(dlX) modelPredictFeature(dlnet_standard, dlX, target_angle_idx_for_xai);
            [gradients_std] = dlgradient(dlfeval(dlPredictFcn_Std, dlX_sample), dlX_sample);
            feature_importance_standard = abs(extractdata(gradients_std)); 
            feature_importance_standard = squeeze(feature_importance_standard); 

            % --- Gradients for PINN MLP ---
            dlPredictFcn_Pinn = @(dlX) modelPredictFeature(dlnet_pinn, dlX, target_angle_idx_for_xai);
            [gradients_pinn] = dlgradient(dlfeval(dlPredictFcn_Pinn, dlX_sample), dlX_sample);
            feature_importance_pinn = abs(extractdata(gradients_pinn)); 
            feature_importance_pinn = squeeze(feature_importance_pinn); 

            if isvector(feature_importance_standard) && isvector(feature_importance_pinn) && ...
               length(feature_importance_standard) == num_combined_features && length(feature_importance_pinn) == num_combined_features
                 
                 % Create labels
                 feature_labels = cell(1, num_combined_features);
                 idx = 1;
                 for ch = 1:num_emg_channels, for f = 1:num_features_per_channel
                    feature_labels{idx} = sprintf('Ch%d-%s', ch, feature_list{f}); idx = idx + 1;
                 end, end

                 % --- SAVE DATA FOR PYTHON ---
                 xai_output_filename = 'xai_compare_data.mat';
                 save(xai_output_filename, 'feature_importance_standard', 'feature_importance_pinn', 'feature_labels', ...
                      'sample_idx_for_xai', 'target_angle_idx_for_xai', 'selected_subject_file');
                 fprintf('Comparative feature importance data saved to %s\n', xai_output_filename);
                 xai_data_saved = true;
                 % ----------------------------

                 % Visualize Comparison (Grouped Bar Chart)
                figure('Name', sprintf('Comparative Feature Importance (Sample %d, Angle %d) (%s)', ...
                       sample_idx_for_xai, target_angle_idx_for_xai, selected_subject_file), ...
                       'NumberTitle', 'off', 'Position', [100, 100, 1200, 600]); % Wider figure
                
                bar_data = [feature_importance_standard(:), feature_importance_pinn(:)]; % Ensure column vectors
                b = bar(bar_data, 'grouped');
                
                title(sprintf('Comparative Feature Importance (Gradient Mag) for Angle %d Prediction', target_angle_idx_for_xai));
                ylabel('Absolute Gradient Magnitude');
                xlabel('Extracted Features');
                xticks(1:num_combined_features);
                xticklabels(feature_labels);
                xtickangle(90); 
                grid on;
                legend('Standard MLP', 'PINN MLP');
                
                fprintf('Comparative feature importance map generated.\n');
            else
                warning('Feature importance calculation resulted in unexpected dimensions.');
            end
        end
    catch ME_xai
        
    end
     if ~xai_data_saved, fprintf('NOTE: XAI data was not saved.\n'); end

end % End of evaluation check

disp('Script finished.');


% =========================================================================
%                       HELPER FUNCTIONS
% =========================================================================

% --- Physics-Informed Loss Function ---
function [loss, loss_mse, loss_physics] = physicsLoss(Y_pred_norm, Y_actual_norm, lambda, lower_limits_norm, upper_limits_norm, mu_angle, sig_angle)
% Calculates physics-informed loss: MSE + Penalty for violating angle limits
% Inputs are NORMALIZED predictions and targets. Limits are also NORMALIZED.

    % 1. MSE Loss (Standard Data Fidelity)
    loss_mse = mse(Y_pred_norm, Y_actual_norm);
    
    % 2. Physics Penalty (Angle Limits)
    % Denormalize predictions to check against original degree limits
    % Y_pred_denorm = Y_pred_norm .* sig_angle + mu_angle; % This requires mu/sig inside dlfeval - complex
    % Alternative: Check violation using NORMALIZED limits
    
    % Penalty for lower bound violation: max(0, lower_limit - prediction)
    lower_violation = relu(lower_limits_norm - Y_pred_norm); 
    % Penalty for upper bound violation: max(0, prediction - upper_limit)
    upper_violation = relu(Y_pred_norm - upper_limits_norm);
    
    % Calculate mean penalty across batch and angles
    penalty = mean(lower_violation.^2 + upper_violation.^2, 'all'); % Mean squared violation
    loss_physics = lambda * penalty;
    
    % 3. Total Loss
    loss = loss_mse + loss_physics;

end

% --- Model Gradients Function (for dlfeval) ---
function [gradients, loss, state] = modelGradients(dlnet, dlX, dlY, lambda, lower_limits_norm, upper_limits_norm, mu_angle, sig_angle)
% Calculates gradients of the physics-informed loss w.r.t learnable parameters

    % Predict using the current network state
    [dlY_pred, state] = forward(dlnet, dlX); % Use forward for state update (e.g., batch norm)
    
    % Calculate physics-informed loss
    [loss, ~, ~] = physicsLoss(dlY_pred, dlY, lambda, lower_limits_norm, upper_limits_norm, mu_angle, sig_angle);
    
    % Calculate gradients
    gradients = dlgradient(loss, dlnet.Learnables);
    
end


% --- Feature Extraction Function ---
function features = extractFeatures(window_data, feature_list)
%extractFeatures Calculates specified features for each channel in a window.
% ... (function code remains the same) ...
    [win_len, num_channels] = size(window_data); num_features = length(feature_list);
    features_per_channel = zeros(num_features, num_channels);
    for ch = 1:num_channels, channel_data = window_data(:, ch);
        for f = 1:num_features, feature_name = feature_list{f};
            switch upper(feature_name)
                case 'RMS', features_per_channel(f, ch) = sqrt(mean(channel_data.^2));
                case 'MAV', features_per_channel(f, ch) = mean(abs(channel_data));
                case 'WL', features_per_channel(f, ch) = sum(abs(diff(channel_data)));
                otherwise, warning('Unknown feature requested: %s. Skipping.', feature_name); features_per_channel(f, ch) = NaN;
            end
        end
    end, features = reshape(features_per_channel, 1, []); 
end

% --- R2 Calculation Function ---
function r2 = calculateR2(y_actual, y_pred)
%calculateR2 Calculates the R-squared (Coefficient of Determination).
% ... (function code remains the same) ...
    if ~isnumeric(y_actual) || ~isnumeric(y_pred), error('Inputs must be numeric.'); end
    if ~isequal(size(y_actual), size(y_pred)), error('Inputs y_actual and y_pred must have the same size.'); end
    mean_y_actual = mean(y_actual, 1); ss_res = sum((y_actual - y_pred).^2, 1); ss_tot = sum((y_actual - mean_y_actual).^2, 1);
    r2 = zeros(1, size(y_actual, 2)); 
    for j = 1:size(y_actual, 2) , if ss_tot(j) < eps, r2(j) = NaN; else r2(j) = 1 - (ss_res(j) / ss_tot(j)); end, end
end

% --- Correlation Calculation Helper ---
function cc = calculateCorrelation(y_actual, y_pred)
    num_outputs = size(y_actual, 2);
    cc = zeros(1, num_outputs);
    for j = 1:num_outputs
        if std(y_actual(:, j)) > 1e-6 && std(y_pred(:, j)) > 1e-6
            cc(j) = corr(y_pred(:, j), y_actual(:, j));
        else
            cc(j) = NaN;
        end
    end
end


% --- Helper function for XAI gradient calculation (MLP input) ---
function output_scalar = modelPredictFeature(dlnet, dlX_feature_vector, target_idx)
% Predict using the dlnetwork (input is feature vector)
% ... (function code remains the same) ...
    dlY = predict(dlnet, dlX_feature_vector); 
    output_scalar = dlY(target_idx, :); 
end

