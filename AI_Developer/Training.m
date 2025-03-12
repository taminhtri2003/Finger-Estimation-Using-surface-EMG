% EMG to Joint Angle Prediction with Feature Extraction and Visualization
% This script takes EMG data and predicts joint angles using ML models

%% Load Data
% Replace with your actual file path
load('s1_full.mat');

%% Data Preparation
% Initialize cell arrays to store processed data
emg_features = cell(5,5);
all_joint_angles = cell(5,5);

% Define EMG feature extraction parameters
window_size = 100;  % 200 samples window (50ms at 4kHz)
window_overlap = 50;  % 50% overlap
num_muscles = 8;

% Define which features to extract from EMG
feature_names = {'MAV', 'RMS', 'WL', 'ZC', 'SSC', 'WA', 'MF'};
num_features = length(feature_names);

%% Feature Extraction Function
% Define threshold for ZC and SSC
threshold = 0.01;

% Process all trials and tasks
for trial = 1:5
    for task = 1:5
        % Get EMG and joint angle data for current trial and task
        current_emg = dsfilt_emg{trial, task};
        current_joint_angles = joint_angles{trial, task};
        
        % Number of windows
        num_samples = size(current_emg, 1);
        num_windows = floor((num_samples - window_size) / (window_size - window_overlap)) + 1;
        
        % Initialize feature matrix for current trial and task
        features_matrix = zeros(num_windows, num_muscles * num_features);
        
        % Initialize downsampled joint angles to match feature windows
        ds_joint_angles = zeros(num_windows, 14);  % 14 joint angles
        
        % Process each window
        for w = 1:num_windows
            % Window indices
            start_idx = (w-1) * (window_size - window_overlap) + 1;
            end_idx = start_idx + window_size - 1;
            
            % Get center index for joint angles (to match with features)
            center_idx = round((start_idx + end_idx) / 2);
            
            % Store joint angles at center of window
            if center_idx <= size(current_joint_angles, 1)
                ds_joint_angles(w, :) = current_joint_angles(center_idx, :);
            end
            
            % Extract features for each muscle
            for m = 1:num_muscles
                % Get window data for current muscle
                window_data = current_emg(start_idx:end_idx, m);
                feat_idx = (m-1) * num_features;
                
                % Feature 1: Mean Absolute Value (MAV)
                features_matrix(w, feat_idx + 1) = mean(abs(window_data));
                
                % Feature 2: Root Mean Square (RMS)
                features_matrix(w, feat_idx + 2) = sqrt(mean(window_data.^2));
                
                % Feature 3: Waveform Length (WL)
                features_matrix(w, feat_idx + 3) = sum(abs(diff(window_data)));
                
                % Feature 4: Zero Crossings (ZC)
                zc = 0;
                for i = 1:length(window_data)-1
                    if ((window_data(i) > 0 && window_data(i+1) < 0) || ...
                        (window_data(i) < 0 && window_data(i+1) > 0)) && ...
                       (abs(window_data(i) - window_data(i+1)) >= threshold)
                        zc = zc + 1;
                    end
                end
                features_matrix(w, feat_idx + 4) = zc;
                
                % Feature 5: Slope Sign Changes (SSC)
                ssc = 0;
                for i = 2:length(window_data)-1
                    if ((window_data(i) > window_data(i-1) && window_data(i) > window_data(i+1)) || ...
                        (window_data(i) < window_data(i-1) && window_data(i) < window_data(i+1))) && ...
                       (abs(window_data(i) - window_data(i+1)) >= threshold || ...
                        abs(window_data(i) - window_data(i-1)) >= threshold)
                        ssc = ssc + 1;
                    end
                end
                features_matrix(w, feat_idx + 5) = ssc;
                
                % Feature 6: Willison Amplitude (WA)
                wa = 0;
                for i = 1:length(window_data)-1
                    if abs(window_data(i) - window_data(i+1)) >= threshold
                        wa = wa + 1;
                    end
                end
                features_matrix(w, feat_idx + 6) = wa;
                
                % Feature 7: Median Frequency (MF)
                % Calculate power spectral density
                if length(window_data) > 1  % Ensure we have enough data
                    [pxx, f] = pwelch(window_data, hamming(min(length(window_data), 128)), [], [], 4000);
                    % Find median frequency
                    total_power = sum(pxx);
                    cumulative_power = cumsum(pxx);
                    median_freq_idx = find(cumulative_power >= total_power/2, 1, 'first');
                    if ~isempty(median_freq_idx) && median_freq_idx <= length(f)
                        features_matrix(w, feat_idx + 7) = f(median_freq_idx);
                    else
                        features_matrix(w, feat_idx + 7) = 0;
                    end
                else
                    features_matrix(w, feat_idx + 7) = 0;
                end
            end
        end
        
        % Store the features and downsampled joint angles
        emg_features{trial, task} = features_matrix;
        all_joint_angles{trial, task} = ds_joint_angles;
    end
end

%% Prepare Data for Machine Learning
% Combine all data across trials and tasks
X_all = [];
Y_all = [];

for trial = 1:5
    for task = 1:2
        X_all = [X_all; emg_features{trial, task}];
        Y_all = [Y_all; all_joint_angles{trial, task}];
    end
end

% Remove rows with NaN or Inf values
valid_rows = all(isfinite(X_all), 2) & all(isfinite(Y_all), 2);
X_all = X_all(valid_rows, :);
Y_all = Y_all(valid_rows, :);

% Create feature names for plots
feature_col_names = cell(1, num_muscles * num_features);
for m = 1:num_muscles
    muscle_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};
    for f = 1:num_features
        feature_col_names{(m-1)*num_features + f} = [muscle_names{m} '_' feature_names{f}];
    end
end

% Joint angle names
joint_names = {'Thumb_1', 'Thumb_2', 'Index_1', 'Index_2', 'Index_3', ...
               'Middle_1', 'Middle_2', 'Middle_3', 'Ring_1', 'Ring_2', ...
               'Ring_3', 'Little_1', 'Little_2', 'Little_3'};

%% Feature Selection and Normalization
% Normalize features
[X_normalized, mu, sigma] = zscore(X_all);

% Feature selection using correlation
corr_threshold = 0.3;
selected_features = [];

% Calculate correlation between each feature and each joint angle
feature_importance = zeros(size(X_normalized, 2), size(Y_all, 2));
for j = 1:size(Y_all, 2)
    for f = 1:size(X_normalized, 2)
        r = corrcoef(X_normalized(:, f), Y_all(:, j));
        feature_importance(f, j) = abs(r(1, 2));
    end
    
    % Select features that have correlation above threshold for this joint angle
    selected_features_j = find(feature_importance(:, j) > corr_threshold);
    selected_features = union(selected_features, selected_features_j);
end

% Ensure we have at least 10 features 
if length(selected_features) < 10
    [~, sorted_idx] = sort(sum(feature_importance, 2), 'descend');
    selected_features = sorted_idx(1:min(10, size(X_normalized, 2)));
end

% Use selected features
X_selected = X_normalized(:, selected_features);

% Create labels for selected features
selected_feature_names = feature_col_names(selected_features);

%% Split Data
% 80% training, 20% testing
cv = cvpartition(size(X_selected, 1), 'HoldOut', 0.2);
X_train = X_selected(cv.training, :);
Y_train = Y_all(cv.training, :);
X_test = X_selected(cv.test, :);
Y_test = Y_all(cv.test, :);

%% Train Machine Learning Models
% We'll train separate models for each joint angle
num_joints = size(Y_all, 2);
models = cell(1, num_joints);
predictions = zeros(size(X_test, 1), num_joints);
test_rmse = zeros(1, num_joints);

for j = 1:num_joints
    % Train a model for each joint angle
    fprintf('Training model for %s joint angle...\n', joint_names{j});
    
    % Create and train model (Random Forest regression)
    model = TreeBagger(50, X_train, Y_train(:, j), 'Method', 'regression');
    
    % Store model
    models{j} = model;
    
    % Make predictions
    predictions(:, j) = predict(model, X_test);
    
    % Calculate RMSE
    test_rmse(j) = sqrt(mean((predictions(:, j) - Y_test(:, j)).^2));
    fprintf('RMSE for %s: %.4f degrees\n', joint_names{j}, test_rmse(j));
end

% Calculate overall RMSE
overall_rmse = sqrt(mean(mean((predictions - Y_test).^2)));
fprintf('Overall RMSE: %.4f degrees\n', overall_rmse);

%% Visualizations

figure('Name', 'Feature Importance for Joint Angle Prediction', 'Position', [100, 100, 1200, 600]);

% Get only the feature importance values for selected features
feature_importance_selected = feature_importance(selected_features, :);

% Ensure dimensions match
if length(selected_feature_names) ~= size(feature_importance_selected, 1)
    fprintf('Warning: Feature names and importance matrix dimensions dont match.\n');
    fprintf('Number of selected features: %d\n', length(selected_features));
    fprintf('Number of selected feature names: %d\n', length(selected_feature_names));
    fprintf('Size of feature importance matrix: %d x %d\n', size(feature_importance_selected));
    
    % Use indices as feature names if there's a mismatch
    selected_feature_names = cellstr(string(1:size(feature_importance_selected, 1)));
end

% Ensure joint names match the second dimension
if length(joint_names) ~= size(feature_importance_selected, 2)
    fprintf('Warning: Joint names and importance matrix dimensions don match.\n');
    % Use indices as joint names if there's a mismatch
    joint_names = cellstr(string(1:size(feature_importance_selected, 2)));
end

% Now create the heatmap with correct dimensions
h = heatmap(joint_names, selected_feature_names, feature_importance_selected, 'ColorMap', jet, 'ColorLimits', [0 1]);
title('Feature Importance Heatmap (Correlation)');
xlabel('Joint Angles');
ylabel('EMG Features');

% Alternative visualization using imagesc if heatmap still doesn't work
figure('Name', 'Feature Importance (Alternative)', 'Position', [100, 100, 1200, 600]);
imagesc(feature_importance_selected);
colormap(jet);
colorbar;
title('Feature Importance Matrix (Correlation)');
xlabel('Joint Angles');
ylabel('EMG Features');
xticks(1:length(joint_names));
xticklabels(joint_names);
xtickangle(45);
yticks(1:length(selected_feature_names));
yticklabels(selected_feature_names);

% 2. Plot actual vs predicted for each joint angle
figure('Name', 'Actual vs Predicted Joint Angles', 'Position', [100, 100, 1200, 800]);
num_plots = min(6, num_joints);  % Show first 6 joints
for j = 1:num_plots
    subplot(2, 3, j);
    
    % Get time indices
    time_indices = 1:length(Y_test(:, j));
    
    % Plot actual vs predicted
    plot(time_indices, Y_test(:, j), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(time_indices, predictions(:, j), 'r-', 'LineWidth', 1.5);
    hold off;
    
    title(sprintf('%s (RMSE: %.2f°)', joint_names{j}, test_rmse(j)));
    xlabel('Time Window');
    ylabel('Joint Angle (degrees)');
    legend('Actual', 'Predicted');
    grid on;
end

% 3. Scatter plots of actual vs predicted for a few joints
figure('Name', 'Correlation Between Actual and Predicted Joint Angles', 'Position', [100, 100, 1200, 800]);
num_plots = min(6, num_joints);
for j = 1:num_plots
    subplot(2, 3, j);
    
    scatter(Y_test(:, j), predictions(:, j), 20, 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    
    % Add regression line
    min_val = min(min(Y_test(:, j)), min(predictions(:, j)));
    max_val = max(max(Y_test(:, j)), max(predictions(:, j)));
    line([min_val, max_val], [min_val, max_val], 'Color', 'red', 'LineWidth', 2);
    
    % Calculate R^2
    r = corrcoef(Y_test(:, j), predictions(:, j));
    r_squared = r(1,2)^2;
    
    title(sprintf('%s (R^2: %.2f)', joint_names{j}, r_squared));
    xlabel('Actual Angle (degrees)');
    ylabel('Predicted Angle (degrees)');
    grid on;
end

% 4. Test RMSE Visualization
figure('Name', 'RMSE for Each Joint Angle', 'Position', [100, 100, 900, 500]);
bar(test_rmse);
xticks(1:num_joints);
xticklabels(joint_names);
xtickangle(45);
ylabel('RMSE (degrees)');
title('Prediction Error for Each Joint Angle');
grid on;

% 5. 3D Animation Visualization (if you have finger kinematic data)
% This would require more complex visualization code to animate the hand
% For a simple visualization, let's create a single frame showing how the joints
% are positioned

% 6. Temporal correlation between EMG and Joint Angles
figure('Name', 'Temporal Relationship Between EMG Features and Joint Angles', 'Position', [100, 100, 1200, 600]);

% Choose a representative time window and joint
joint_idx = 3;  % Index finger proximal joint
time_window = 1:min(200, size(Y_test, 1));

% Get the normalized EMG feature values for this time window
emg_feature_idx = find(contains(selected_feature_names, 'FDS_MAV') | contains(selected_feature_names, 'FDP_MAV'));
if ~isempty(emg_feature_idx)
    emg_feature_idx = emg_feature_idx(1);
    
    % Plot EMG feature and joint angle over time
    yyaxis left;
    plot(time_window, X_test(time_window, emg_feature_idx), 'b-', 'LineWidth', 1.5);
    ylabel('Normalized EMG Feature');
    
    yyaxis right;
    plot(time_window, Y_test(time_window, joint_idx), 'r-', 'LineWidth', 1.5);
    hold on;
    plot(time_window, predictions(time_window, joint_idx), 'g--', 'LineWidth', 1.5);
    hold off;
    
    ylabel('Joint Angle (degrees)');
    title(sprintf('Relationship Between %s and %s Movement', selected_feature_names{emg_feature_idx}, joint_names{joint_idx}));
    legend('EMG Feature', 'Actual Angle', 'Predicted Angle');
    xlabel('Time Window');
    grid on;
end

%% Save Training Results
% Save the models, feature selection, and normalization parameters for future use
save('emg_joint_model.mat', 'models', 'selected_features', 'mu', 'sigma', 'test_rmse', 'overall_rmse');
fprintf('Models and parameters saved to emg_joint_model.mat\n');

