%% EMG to Kinematics Regression Model with Visualizations
% This script extracts features from EMG signals and uses machine learning
% regression models to estimate 69 kinematics variables
% Trials 1-3 are used for training, trials 4-5 for testing
% Only tasks 1-5 (individual finger movements) are analyzed

% Load the data file
load('s1.mat');

% Define parameters
num_trials = 5;
num_tasks = 5;  % Only using tasks 1-5 (individual finger movements)
train_trials = 1:3;
test_trials = 4:5;

% Create folder for saving visualizations
visualizationFolder = 'Kinematics_Prediction_Visualizations';
if ~exist(visualizationFolder, 'dir')
    mkdir(visualizationFolder);
end

% Channel names for plots
channel_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};

%% Feature Extraction Section
disp('========== Feature Extraction ==========');

% Define the time-domain features to extract
feature_names = {'RMS', 'MAV', 'VAR', 'WL', 'ZC', 'SSC', 'WAMP', 'AR4'};
num_features_per_channel = length(feature_names);

% Initialize arrays to store features and labels
X_train = [];
Y_train = [];
X_test = [];
Y_test = [];

% Window parameters for feature extraction
window_size = 200;  % 1 second at 200 Hz
overlap = 100;      % 50% overlap

%% Extract Time-Domain Features Function
function features = extractTimedomainFeatures(emg_data)
    % Get dimensions
    [num_samples, num_channels] = size(emg_data);
    
    % Initialize feature matrix
    num_features_per_channel = 8;  % We're extracting 8 features per channel
    features = zeros(1, num_channels * num_features_per_channel);
    
    % Extract features for each channel
    for ch = 1:num_channels
        channel_data = emg_data(:, ch);
        feature_idx = (ch-1) * num_features_per_channel + 1;
        
        % 1. Root Mean Square (RMS) - measure of signal power
        rms_value = sqrt(mean(channel_data.^2));
        
        % 2. Mean Absolute Value (MAV) - average of absolute values
        mav = mean(abs(channel_data));
        
        % 3. Variance (VAR) - measure of signal variability
        var_value = var(channel_data);
        
        % 4. Waveform Length (WL) - cumulative length of waveform
        wl = sum(abs(diff(channel_data)));
        
        % 5. Zero Crossings (ZC) - number of times signal crosses zero
        zc = sum(abs(diff(sign(channel_data))) > 0) / (2 * (num_samples - 1));
        
        % 6. Slope Sign Changes (SSC) - number of times slope changes sign
        diff_data = diff(channel_data);
        ssc = sum(diff(sign(diff_data)) ~= 0) / (num_samples - 2);
        
        % 7. Willison Amplitude (WAMP) - number of times difference between consecutive samples exceeds threshold
        threshold = 0.1 * std(channel_data);
        wamp = sum(abs(diff(channel_data)) > threshold) / (num_samples - 1);
        
        % 8. 4th order Autoregressive coefficients (simplified - using only first coefficient)
        if length(channel_data) > 5  % Make sure we have enough data points
            ar_coeffs = arburg(channel_data, 4);
            ar4 = ar_coeffs(1);  % Using only first coefficient to simplify
        else
            ar4 = 0;
        end
        
        % Store all features for this channel
        features(1, feature_idx:(feature_idx + num_features_per_channel - 1)) = [rms_value, mav, var_value, wl, zc, ssc, wamp, ar4];
    end
end

%% Feature Extraction with Sliding Window
function [X, Y, window_indices] = extractFeaturesWithWindow(emg_data, kinematics_data, window_size, overlap)
    % Get dimensions
    [num_samples, num_channels] = size(emg_data);
    
    % Calculate step size
    step_size = window_size - overlap;
    
    % Calculate number of windows
    num_windows = floor((num_samples - window_size) / step_size) + 1;
    
    % Initialize feature and label matrices
    num_features_per_channel = 8;  % Match with extractTimedomainFeatures function
    X = zeros(num_windows, num_channels * num_features_per_channel);
    Y = zeros(num_windows, size(kinematics_data, 2));
    
    % Keep track of window indices for visualization
    window_indices = zeros(num_windows, 2);  % [start_idx, end_idx]
    
    % Extract features for each window
    for w = 1:num_windows
        % Window indices
        start_idx = (w-1) * step_size + 1;
        end_idx = start_idx + window_size - 1;
        
        window_indices(w, :) = [start_idx, end_idx];
        
        % Extract EMG segment
        emg_segment = emg_data(start_idx:end_idx, :);
        
        % Extract features from this segment
        X(w, :) = extractTimedomainFeatures(emg_segment);
        
        % Find the corresponding kinematics sample
        % (accounting for different sampling rates)
        kinematics_idx = round(start_idx / 10);  % Original downsampling factor from EMG to kinematics
        if kinematics_idx < 1
            kinematics_idx = 1;
        elseif kinematics_idx > size(kinematics_data, 1)
            kinematics_idx = size(kinematics_data, 1);
        end
        
        % Get corresponding kinematics
        Y(w, :) = kinematics_data(kinematics_idx, :);
    end
end

%% Visualize Raw Data for a Sample Trial/Task
sample_trial = 1;
sample_task = 1;

if ~isempty(preprocessed_emg{sample_trial, sample_task}) && ~isempty(finger_kinematics{sample_trial, sample_task})
    % Get data
    emg_data = preprocessed_emg{sample_trial, sample_task};
    kinematics_data = finger_kinematics{sample_trial, sample_task};
    
    % Create figure for raw data visualization
    figure('Position', [100, 100, 1200, 800]);
    
    % Plot EMG channels
    subplot(2, 1, 1);
    time_emg = (0:size(emg_data, 1)-1) / motion_data_fs;
    plot(time_emg, emg_data);
    title('Raw Preprocessed EMG Signals');
    xlabel('Time (s)');
    ylabel('Amplitude');
    legend(channel_names, 'Location', 'eastoutside');
    grid on;
    
    % Plot a subset of kinematics channels (first 8 for visibility)
    subplot(2, 1, 2);
    time_kin = (0:size(kinematics_data, 1)-1) / motion_data_fs;
    plot(time_kin, kinematics_data(:, 1:8));
    title('First 8 Kinematic Variables');
    xlabel('Time (s)');
    ylabel('Position/Angle');
    grid on;
    
    % Save figure
    saveas(gcf, fullfile(visualizationFolder, 'Raw_Data_Sample.png'));
    disp('Saved raw data visualization.');
end

%% Extract features from all trials and tasks
disp('Extracting features from all trials and tasks...');

% Create a figure to visualize the windowing process
figure('Position', [100, 100, 1200, 600]);
sample_windows_to_plot = 5;  % Number of windows to visualize

% Keep track of trial/task info for each sample
train_trial_task = [];
test_trial_task = [];

% Progress tracking
total_combinations = length(train_trials) * num_tasks + length(test_trials) * num_tasks;
current_progress = 0;

% Process each trial and task
for trial = 1:num_trials
    for task = 1:num_tasks
        % Check if data exists for this trial and task
        if ~isempty(preprocessed_emg{trial, task}) && ~isempty(finger_kinematics{trial, task})
            % Update progress
            current_progress = current_progress + 1;
            progress_percent = (current_progress / total_combinations) * 100;
            disp(['Processing Trial ', num2str(trial), ', Task ', num2str(task), ...
                  ' (', num2str(progress_percent, '%.1f'), '% complete)']);
            
            % Get EMG and kinematics data
            emg_data = preprocessed_emg{trial, task};
            kinematics_data = finger_kinematics{trial, task};
            
            % Extract features with sliding window
            [X_windows, Y_windows, window_indices] = extractFeaturesWithWindow(emg_data, kinematics_data, window_size, overlap);
            
            % Store trial/task info
            trial_task_info = repmat([trial, task], size(X_windows, 1), 1);
            
            % Assign to train or test set based on trial number
            if ismember(trial, train_trials)
                X_train = [X_train; X_windows];
                Y_train = [Y_train; Y_windows];
                train_trial_task = [train_trial_task; trial_task_info];
                
                % Visualize windowing process for the first training trial/task
                if trial == train_trials(1) && task == 1 && sample_windows_to_plot > 0
                    % Clear the plot
                    clf;
                    
                    % Plot first channel of EMG data
                    time_emg = (0:size(emg_data, 1)-1) / motion_data_fs;
                    plot(time_emg, emg_data(:, 1), 'k-', 'LineWidth', 1);
                    hold on;
                    
                    % Plot the first few windows
                    num_windows_to_show = min(sample_windows_to_plot, size(window_indices, 1));
                    colors = jet(num_windows_to_show);
                    
                    legend_entries = {'EMG Signal'};
                    for w = 1:num_windows_to_show
                        window_start = window_indices(w, 1);
                        window_end = window_indices(w, 2);
                        
                        % Highlight the window
                        window_time = time_emg(window_start:window_end);
                        window_data = emg_data(window_start:window_end, 1);
                        
                        plot(window_time, window_data, 'LineWidth', 2, 'Color', colors(w, :));
                        legend_entries{end+1} = ['Window ', num2str(w)];
                    end
                    
                    title('Feature Extraction Windowing Process');
                    xlabel('Time (s)');
                    ylabel('EMG Amplitude');
                    legend(legend_entries, 'Location', 'eastoutside');
                    grid on;
                    
                    % Save figure
                    saveas(gcf, fullfile(visualizationFolder, 'Windowing_Process.png'));
                    disp('Saved windowing process visualization.');
                    
                    sample_windows_to_plot = 0;  % Only do this once
                end
                
            elseif ismember(trial, test_trials)
                X_test = [X_test; X_windows];
                Y_test = [Y_test; Y_windows];
                test_trial_task = [test_trial_task; trial_task_info];
            end
        else
            disp(['No data for Trial ', num2str(trial), ', Task ', num2str(task)]);
        end
    end
end

disp(['Feature extraction complete. Training samples: ', num2str(size(X_train, 1)), ...
      ', Testing samples: ', num2str(size(X_test, 1))]);

%% Visualize feature distributions
disp('Visualizing feature distributions...');

% Calculate number of features per channel and total features
num_features_per_channel = length(feature_names);
total_features = size(X_train, 2);
num_channels = total_features / num_features_per_channel;

% Create figure for feature distribution
figure('Position', [100, 100, 1200, 800]);

% Plot histograms for each feature type (averaging across channels)
for f = 1:num_features_per_channel
    subplot(2, 4, f);
    
    % Get this feature for all channels
    feature_values = [];
    for ch = 1:num_channels
        feature_idx = (ch-1) * num_features_per_channel + f;
        feature_values = [feature_values; X_train(:, feature_idx)];
    end
    
    % Plot histogram
    histogram(feature_values, 30, 'Normalization', 'probability');
    title(['Distribution of ', feature_names{f}]);
    xlabel('Value');
    ylabel('Probability');
    grid on;
end

% Save figure
saveas(gcf, fullfile(visualizationFolder, 'Feature_Distributions.png'));
disp('Saved feature distribution visualization.');

%% Visualize correlation between features
disp('Visualizing feature correlations...');

% Sample some features (first feature of each type for first 4 channels)
num_features_to_sample = 4; % Number of features to sample per channel
num_channels_to_sample = 4; % Number of channels to sample
sample_feature_indices = zeros(1, num_channels_to_sample * num_features_to_sample);  % Pre-allocate

for ch = 1:num_channels_to_sample
    for f = 1:num_features_to_sample
        idx = (ch-1) * num_features_per_channel + f;
        sample_feature_indices((ch-1)*num_features_to_sample + f) = idx;
    end
end
% Create feature correlation matrix
X_corr = X_train(:, sample_feature_indices);
correlation_matrix = corrcoef(X_corr);

% Create feature names for correlation plot
corr_feature_names = cell(1, length(sample_feature_indices));
for i = 1:length(sample_feature_indices)
    idx = sample_feature_indices(i);
    ch = floor((idx - 1) / num_features_per_channel) + 1;
    f = mod(idx - 1, num_features_per_channel) + 1;
    corr_feature_names{i} = [channel_names{ch}, '-', feature_names{f}];
end

% Create correlation heatmap
figure('Position', [100, 100, 1000, 800]);
imagesc(correlation_matrix);
colormap('jet');
colorbar;
title('Feature Correlation Matrix');
xticks(1:length(corr_feature_names));
yticks(1:length(corr_feature_names));
xticklabels(corr_feature_names);
yticklabels(corr_feature_names);
xtickangle(45);
set(gca, 'FontSize', 8);

% Save figure
saveas(gcf, fullfile(visualizationFolder, 'Feature_Correlation.png'));
disp('Saved feature correlation visualization.');

%% Feature Importance Analysis using PCA
disp('Performing PCA for feature importance analysis...');

% Normalize data for PCA
X_mean = mean(X_train);
X_std = std(X_train);
X_std(X_std == 0) = 1;  % Avoid division by zero
X_train_norm = (X_train - X_mean) ./ X_std;

% Perform PCA
[coeff, score, latent, ~, explained] = pca(X_train_norm);

% Visualize PCA explained variance
figure('Position', [100, 100, 1000, 500]);

% Plot explained variance
subplot(1, 2, 1);
plot(cumsum(explained), 'bo-', 'LineWidth', 2);
title('Cumulative Explained Variance');
xlabel('Number of Principal Components');
ylabel('Explained Variance (%)');
grid on;

% Find number of PCs needed for 95% variance
pc_95 = find(cumsum(explained) >= 95, 1);
line([pc_95, pc_95], [0, 100], 'Color', 'r', 'LineStyle', '--');
text(pc_95+1, 50, [num2str(pc_95), ' PCs needed for 95% variance'], 'Color', 'r');

% Plot feature loadings for first 2 PCs
subplot(1, 2, 2);
biplot(coeff(:, 1:2), 'scores', score(:, 1:2), 'varlabels', {});
title('Feature Loadings for First 2 PCs');
xlabel(['PC1 (', num2str(explained(1), '%.1f'), '%)']);
ylabel(['PC2 (', num2str(explained(2), '%.1f'), '%)']);
grid on;

% Save figure
saveas(gcf, fullfile(visualizationFolder, 'PCA_Analysis.png'));
disp('Saved PCA analysis visualization.');

%% Normalize features for ML models
disp('Normalizing features...');

% Normalize training and testing data
X_train_norm = (X_train - X_mean) ./ X_std;
X_test_norm = (X_test - X_mean) ./ X_std;

%% Visualize data by trial and task
disp('Visualizing data distribution by trial and task...');

% Perform t-SNE on a subset of the training data for visualization
% (t-SNE is computationally intensive, so limit sample size)
max_samples_for_tsne = min(5000, size(X_train_norm, 1));
random_indices = randperm(size(X_train_norm, 1), max_samples_for_tsne);
X_train_subset = X_train_norm(random_indices, :);
train_trial_task_subset = train_trial_task(random_indices, :);

% Perform t-SNE
X_tsne = tsne(X_train_subset, 'Perplexity', 30, 'NumDimensions', 2);

% Create figure
figure('Position', [100, 100, 1500, 600]);

% Create color scheme for trials and tasks
trial_colors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250];
task_markers = {'o', 's', 'd', '^', 'v'};

% Plot by trial
subplot(1, 2, 1);
hold on;
for t = 1:length(train_trials)
    trial = train_trials(t);
    % Find samples from this trial
    mask = train_trial_task_subset(:, 1) == trial;
    scatter(X_tsne(mask, 1), X_tsne(mask, 2), 20, ones(sum(mask), 1) * t, 'filled');
end
title('t-SNE Visualization by Trial');
xlabel('t-SNE Component 1');
ylabel('t-SNE Component 2');
colormap(jet(length(train_trials)));
colorbar('Ticks', 1:length(train_trials), 'TickLabels', arrayfun(@num2str, train_trials, 'UniformOutput', false));
grid on;

% Plot by task
subplot(1, 2, 2);
hold on;
for t = 1:num_tasks
    % Find samples from this task
    mask = train_trial_task_subset(:, 2) == t;
    scatter(X_tsne(mask, 1), X_tsne(mask, 2), 20, ones(sum(mask), 1) * t, 'filled');
end
title('t-SNE Visualization by Task');
xlabel('t-SNE Component 1');
ylabel('t-SNE Component 2');
colormap(jet(num_tasks));
colorbar('Ticks', 1:num_tasks, 'TickLabels', arrayfun(@num2str, 1:num_tasks, 'UniformOutput', false));
grid on;

% Save figure
saveas(gcf, fullfile(visualizationFolder, 'tSNE_Data_Distribution.png'));
disp('Saved t-SNE data distribution visualization.');

%% Train regression models
disp('========== Model Training ==========');

% Group kinematics into functional groups to reduce number of models
% This is based on the assumption that finger joints often move together
disp('Grouping kinematics for more efficient modeling...');

% Determine number of kinematic groups (using 10 groups for efficiency)
num_groups = 10;
kinematics_per_group = ceil(size(Y_train, 2) / num_groups);

% Visualize correlation between kinematic variables
kin_corr = corrcoef(Y_train);
figure('Position', [100, 100, 1000, 800]);
imagesc(kin_corr);
colormap('jet');
colorbar;
title('Correlation Between Kinematic Variables');
xlabel('Kinematic Variable Index');
ylabel('Kinematic Variable Index');
axis square;

% Save figure
saveas(gcf, fullfile(visualizationFolder, 'Kinematics_Correlation.png'));

% Store models for each kinematic group
regression_models = cell(1, num_groups);
group_indices = cell(1, num_groups);
prediction_errors = zeros(1, num_groups);
r_squared_values = zeros(1, num_groups);

% Try different regression methods for comparison
regression_methods = {'Linear', 'SVM', 'RandomForest'};
method_rmse = zeros(length(regression_methods), num_groups);

% Create waitbar for progress tracking
h = waitbar(0, 'Training regression models...');

for g = 1:num_groups
    % Determine indices for this group
    start_idx = (g-1) * kinematics_per_group + 1;
    end_idx = min(start_idx + kinematics_per_group - 1, size(Y_train, 2));
    group_indices{g} = start_idx:end_idx;
    
    % Display progress
    waitbar(g/num_groups, h, sprintf('Training model for kinematic group %d/%d', g, num_groups));
    
    % Get training data for this group
    Y_train_group = Y_train(:, group_indices{g});
    
    % Test different regression methods
    for m = 1:length(regression_methods)
        method = regression_methods{m};
        
        % Train model based on selected method
        switch method
            case 'Linear'
                
                Y_pred_group = zeros(size(X_test_norm, 1), length(group_indices{g}));
    
                for dim = 1:size(Y_train_group, 2)
                    % Train a linear regression model for this dimension
                    mdl = fitlm(X_train_norm, Y_train_group(:, dim));
                    
                    % Make predictions for this dimension
                    Y_pred_group(:, dim) = predict(mdl, X_test_norm);
                end
                
            case 'SVM'
                % Train SVM regression model for each dimension in the group
                Y_pred_group = zeros(size(X_test_norm, 1), length(group_indices{g}));
                
                for dim = 1:size(Y_train_group, 2)
                    mdl = fitrsvm(X_train_norm, Y_train_group(:, dim), 'KernelFunction', 'rbf', ...
                                  'Standardize', false);  % Already normalized
                    Y_pred_group(:, dim) = predict(mdl, X_test_norm);
                end
                
            case 'RandomForest'
                % Train random forest regression model
                Y_pred_group = zeros(size(X_test_norm, 1), length(group_indices{g}));
                
                for dim = 1:size(Y_train_group, 2)
                    mdl = TreeBagger(30, X_train_norm, Y_train_group(:, dim), ...
                                     'Method', 'regression', 'MinLeafSize', 5);
                    Y_pred_group(:, dim) = predict(mdl, X_test_norm);
                end
        end
        
        % Calculate RMSE for this method
        rmse = sqrt(mean((Y_test(:, group_indices{g}) - Y_pred_group).^2, 'all'));
        method_rmse(m, g) = rmse;
        
        % Store the best model (will be overwritten with the best one)
        if m == 1 || rmse < min(method_rmse(1:m-1, g))
            regression_models{g} = mdl;
            best_method = method;
        end
    end
    
    % Train the best model for this group
    disp(['Best method for group ', num2str(g), ': ', best_method]);
    
    % For visualization, store predictions from the best model
    switch best_method
        case 'Linear'
            Y_pred_group = predict(regression_models{g}, X_test_norm);
        case 'SVM'
            % Predictions were already made above
        case 'RandomForest'
            % Predictions were already made above
    end
    
    % Calculate RMSE and R² for this group
    group_rmse = sqrt(mean((Y_test(:, group_indices{g}) - Y_pred_group).^2, 'all'));
    prediction_errors(g) = group_rmse;
    
    % Calculate R-squared for this group
    sse = sum((Y_test(:, group_indices{g}) - Y_pred_group).^2, 'all');
    sst = sum((Y_test(:, group_indices{g}) - mean(Y_test(:, group_indices{g}), 'all')).^2, 'all');
    r_squared_values(g) = 1 - (sse/sst);
    
    disp(['Group ', num2str(g), ' RMSE: ', num2str(group_rmse), ', R²: ', num2str(r_squared_values(g))]);
end

% Close waitbar
close(h);

%% Compare regression methods
disp('Comparing regression methods...');

% Create figure to compare methods
figure('Position', [100, 100, 1000, 400]);
bar(method_rmse');
title('Comparison of Regression Methods by Kinematic Group');
xlabel('Kinematic Group');
ylabel('RMSE');
legend(regression_methods);
grid on;

% Save figure
saveas(gcf, fullfile(visualizationFolder, 'Regression_Method_Comparison.png'));

%% Make predictions for all test data
disp('Making predictions on test data...');

% Initialize matrix for all predictions
Y_pred = zeros(size(Y_test));

% Make predictions for each group
for g = 1:num_groups
    % Get indices for this group
    indices = group_indices{g};
    
    % Get model
    mdl = regression_models{g};
    
    % Check model type and make predictions accordingly
    if isa(mdl, 'LinearModel')
        Y_pred(:, indices) = predict(mdl, X_test_norm);
    elseif isa(mdl, 'RegressionSVM')
        for dim = 1:length(indices)
            Y_pred(:, indices(dim)) = predict(mdl, X_test_norm);
        end
    elseif isa(mdl, 'TreeBagger')
        for dim = 1:length(indices)
            Y_pred(:, indices(dim)) = predict(mdl, X_test_norm);
        end
    end
end

%% Evaluate overall model performance
disp('Evaluating model performance...');

% Calculate overall RMSE
overall_rmse = sqrt(mean((Y_test - Y_pred).^2, 'all'));
disp(['Overall RMSE: ', num2str(overall_rmse)]);

% Calculate R-squared for each kinematic dimension
r_squared = zeros(1, size(Y_test, 2));
for k = 1:size(Y_test, 2)
    % Calculate TSS (Total Sum of Squares)
    tss = sum((Y_test(:, k) - mean(Y_test(:, k))).^2);
    
    % Calculate RSS (Residual Sum of Squares)
    rss = sum((Y_test(:, k) - Y_pred(:, k)).^2);
    
    % Calculate R-squared
    r_squared(k) = 1 - (rss / tss);
end

% Display average R-squared
avg_r_squared = mean(r_squared);
disp(['Average R-squared: ', num2str(avg_r_squared)]);

%% Visualize overall model performance
disp('Creating performance visualizations...');

% Create figure for overall performance
figure('Position', [100, 100, 1200, 700]);

% Plot RMSE and R² by kinematic group
subplot(2, 2, 1);
bar(prediction_errors);
title('RMSE by Kinematic Group');
xlabel('Group');
ylabel('RMSE');
grid on;

subplot(2, 2, 2);
bar(r_squared_values);
title('R-squared by Kinematic Group');
xlabel('Group');
ylabel('R-squared');
ylim([0, 1]);
grid on;

% Plot R-squared distribution for all kinematic variables
subplot(2, 2, 3);
histogram(r_squared, 20, 'Normalization', 'probability');
title('R-squared Distribution Across All Kinematics');
xlabel('R-squared');
ylabel('Probability');
grid on;

% Plot overall RMSE and R² metrics
subplot(2, 2, 4);
bar([overall_rmse, avg_r_squared]);
title('Overall Performance Metrics');
xticklabels({'RMSE', 'Avg R²'});
grid on;

% Save figure
saveas(gcf, fullfile(visualizationFolder, 'Overall_Performance.png'));

%% Visualize predictions for selected kinematic variables
disp('Visualizing predictions for selected variables...');

% Select a few representative dimensions to visualize
num_to_visualize = 5;
best_indices = find(r_squared > 0.7);
if length(best_indices) >= num_to_visualize
    vis_indices = best_indices(1:num_to_visualize);
else
    % If we don't have enough good models, just pick the best ones
    [~, sorted_indices] = sort(r_squared, 'descend');
    vis_indices = sorted_indices(1:num_to_visualize);
end

% Create a figure
figure('Position', [100, 100, 1200, 800]);

% Plot predictions vs. actual values for selected dimensions
for i = 1:num_to_visualize
    dim = vis_indices(i);
    
    subplot(2, 3, i);
    
    % Get a sample of test data points to plot (to avoid overcrowding)
    sample_size = min(500, size(Y_test, 1));
    sample_indices = round(linspace(1, size(Y_test, 1), sample_size));
    
    % Plot actual vs. predicted values
    plot(Y_test(sample_indices, dim), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(Y_test(sample_indices, dim), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(Y_pred(sample_indices, dim), 'r--', 'LineWidth', 1.5);
    
    % Add title and labels
    title(['Kinematic Variable ', num2str(dim), ' (R² = ', num2str(r_squared(dim), '%.2f'), ')']);
    xlabel('Sample Index');
    ylabel('Value');
    legend('Actual', 'Predicted', 'Location', 'best');
    grid on;
end

% Save figure
saveas(gcf, fullfile(visualizationFolder, 'Prediction_Visualization.png'));

%% Visualize predictions by task
disp('Visualizing predictions by task...');

% Select representative dimensions (one per task)
task_vis_indices = zeros(1, num_tasks);
for t = 1:num_tasks
    % Find test samples for this task
    task_mask = test_trial_task(:, 2) == t;
    
    % Find kinematic variable with best prediction for this task
    task_r_squared = zeros(1, size(Y_test, 2));
    for k = 1:size(Y_test, 2)
        % Calculate R-squared for this variable within this task
        task_y_test = Y_test(task_mask, k);
        task_y_pred = Y_pred(task_mask, k);
        
        % Calculate TSS and RSS
        tss = sum((task_y_test - mean(task_y_test)).^2);
        rss = sum((task_y_test - task_y_pred).^2);
        
        % Calculate R-squared (handle division by zero)
        if tss > 0
            task_r_squared(k) = 1 - (rss / tss);
        else
            task_r_squared(k) = 0;
        end
    end
    
    % Get kinematic with best R-squared
    [~, best_k] = max(task_r_squared);
    task_vis_indices(t) = best_k;
end

% Create figure for task-specific visualization
figure('Position', [100, 100, 1500, 800]);

% Plot predictions for each task
for t = 1:num_tasks
    subplot(2, 3, t);
    
    % Find test samples for this task
    task_mask = test_trial_task(:, 2) == t;
    task_indices = find(task_mask);
    
    % Get kinematic dimension
    dim = task_vis_indices(t);
    
    % Get sample of test data points for this task
    sample_size = min(500, sum(task_mask));
    if sum(task_mask) > sample_size
        sample_indices = task_indices(round(linspace(1, sum(task_mask), sample_size)));
    else
        sample_indices = task_indices;
    end
    
    % Calculate task-specific R-squared
    task_y_test = Y_test(task_mask, dim);
    task_y_pred = Y_pred(task_mask, dim);
    tss = sum((task_y_test - mean(task_y_test)).^2);
    rss = sum((task_y_test - task_y_pred).^2);
    task_r2 = 1 - (rss / tss);
    
    % Plot actual vs. predicted values
    plot(Y_test(sample_indices, dim), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(Y_pred(sample_indices, dim), 'r--', 'LineWidth', 1.5);
    
    % Add title and labels
    title(['Task ', num2str(t), ' - Kinematic Variable ', num2str(dim), ' (R² = ', num2str(task_r2, '%.2f'), ')']);
    xlabel('Sample Index');
    ylabel('Value');
    legend('Actual', 'Predicted', 'Location', 'best');
    grid on;
end

% Save figure
saveas(gcf, fullfile(visualizationFolder, 'Task_Specific_Predictions.png'));

%% Visualize Feature Importance
disp('Analyzing feature importance...');

% For simplicity, we'll analyze feature importance for a few representative kinematics
importance_vis_indices = vis_indices(1:3);  % Use the first 3 from previous visualization

% Create figure for feature importance
figure('Position', [100, 100, 1200, 800]);

% For each selected kinematic dimension
for i = 1:length(importance_vis_indices)
    dim = importance_vis_indices(i);
    
    % Find which group this dimension belongs to
    for g = 1:num_groups
        if ismember(dim, group_indices{g})
            group_idx = g;
            local_dim_idx = find(group_indices{g} == dim);
            break;
        end
    end
    
    % Get the model for this group
    mdl = regression_models{group_idx};
    
    % Extract feature importance based on model type
    if isa(mdl, 'LinearModel')
        % For linear model, use absolute coefficient values
        coef = abs(mdl.Coefficients.Estimate(2:end, local_dim_idx));
        
        % Normalize importance
        importance = coef / sum(coef);
        
    elseif isa(mdl, 'TreeBagger')
        % For random forest, use Out-of-Bag feature importance
        importance = mdl.OOBPermutedPredictorDeltaError;
        
        % Normalize importance
        importance = importance / sum(importance);
    else
        % For other models, use a correlation-based approach
        importance = zeros(size(X_train_norm, 2), 1);
        for f = 1:size(X_train_norm, 2)
            importance(f) = abs(corr(X_train_norm(:, f), Y_train(:, dim)));
        end
        
        % Normalize importance
        importance = importance / sum(importance);
    end
    
    % Reorganize importance by channel and feature type
    channel_feature_importance = zeros(num_channels, num_features_per_channel);
    for ch = 1:num_channels
        for f = 1:num_features_per_channel
            feature_idx = (ch-1) * num_features_per_channel + f;
            channel_feature_importance(ch, f) = importance(feature_idx);
        end
    end
    
    % Plot feature importance heatmap
    subplot(1, 3, i);
    imagesc(channel_feature_importance);
    colormap('jet');
    colorbar;
    title(['Feature Importance - Kinematic ', num2str(dim)]);
    xlabel('Feature Type');
    ylabel('EMG Channel');
    xticks(1:num_features_per_channel);
    yticks(1:num_channels);
    xticklabels(feature_names);
    yticklabels(channel_names);
    xtickangle(45);
end

% Save figure
saveas(gcf, fullfile(visualizationFolder, 'Feature_Importance.png'));

%% Create Real-Time Prediction Visualization
disp('Creating real-time prediction simulation...');

% Select a test trial and task for visualization
vis_trial = test_trials(1);
vis_task = 1;

% Find the samples in the test set from this trial and task
vis_mask = test_trial_task(:, 1) == vis_trial & test_trial_task(:, 2) == vis_task;
vis_indices = find(vis_mask);

% If no samples found, select any test samples
if isempty(vis_indices)
    vis_indices = 1:min(1000, size(Y_test, 1));
end

% Select a subset of samples for cleaner visualization
num_vis_samples = min(500, length(vis_indices));
vis_indices = vis_indices(1:num_vis_samples);

% Select kinematics variables to visualize (first 5 with good R²)
[sorted_r2, sorted_k_indices] = sort(r_squared, 'descend');
vis_k_indices = sorted_k_indices(1:5);

% Create a figure for real-time visualization
figure('Position', [100, 100, 1200, 800]);

% Plot actual and predicted values for these kinematics
for k = 1:length(vis_k_indices)
    subplot(2, 3, k);
    
    % Get kinematic index
    k_idx = vis_k_indices(k);
    
    % Plot actual and predicted values
    plot(Y_test(vis_indices, k_idx), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(Y_pred(vis_indices, k_idx), 'r--', 'LineWidth', 1.5);
    
    % Add title and labels
    title(['Kinematic ', num2str(k_idx), ' (R² = ', num2str(sorted_r2(k), '%.2f'), ')']);
    xlabel('Time Step');
    ylabel('Value');
    legend('Actual', 'Predicted', 'Location', 'best');
    grid on;
end

% Add overall performance subplot
subplot(2, 3, 6);
bar(sorted_r2(1:10));
title('R² for Top 10 Kinematic Variables');
xlabel('Variable Rank');
ylabel('R-squared');
ylim([0, 1]);
grid on;

% Save figure
saveas(gcf, fullfile(visualizationFolder, 'Real_Time_Prediction.png'));

%% Save Results
disp('Saving results...');

% Save models and results
results = struct();
results.regression_models = regression_models;
results.group_indices = group_indices;
results.feature_names = feature_names;
results.channel_names = channel_names;
results.X_mean = X_mean;
results.X_std = X_std;
results.prediction_errors = prediction_errors;
results.r_squared = r_squared;
results.overall_rmse = overall_rmse;
results.avg_r_squared = avg_r_squared;

% Save results
save(fullfile(visualizationFolder, 'EMG_Kinematics_Results.mat'), 'results');

%% Final Summary
disp('========== Results Summary ==========');
disp(['Number of training samples: ', num2str(size(X_train, 1))]);
disp(['Number of testing samples: ', num2str(size(X_test, 1))]);
disp(['Number of EMG channels: ', num2str(num_channels)]);
disp(['Number of features per channel: ', num2str(num_features_per_channel)]);
disp(['Total number of features: ', num2str(size(X_train, 2))]);
disp(['Number of kinematic variables: ', num2str(size(Y_train, 2))]);
disp(['Overall RMSE: ', num2str(overall_rmse)]);
disp(['Average R-squared: ', num2str(avg_r_squared)]);
disp(['Maximum R-squared: ', num2str(max(r_squared))]);
disp(['Percentage of kinematics with R² > 0.7: ', num2str(100 * sum(r_squared > 0.7) / length(r_squared)), '%']);

disp(['Results and visualizations saved to: ', visualizationFolder]);
disp('Analysis complete.');

%% Cross-Validation Analysis (optional extension)
perform_cross_validation = false;  % Set to true to run cross-validation

if perform_cross_validation
    disp('========== Cross-Validation Analysis ==========');
    
    % Combine training and testing data for cross-validation
    X_all = [X_train_norm; X_test_norm];
    Y_all = [Y_train; Y_test];
    
    % Number of folds
    k_folds = 5;
    
    % Initialize array to store cross-validation results
    cv_r_squared = zeros(k_folds, size(Y_all, 2));
    
    % Create fold indices
    cv = cvpartition(size(X_all, 1), 'KFold', k_folds);
    
    % For each fold
    for fold = 1:k_folds
        disp(['Processing fold ', num2str(fold), '/', num2str(k_folds)]);
        
        % Get training and validation indices for this fold
        train_idx = cv.training(fold);
        val_idx = cv.test(fold);
        
        % Split data
        X_cv_train = X_all(train_idx, :);
        Y_cv_train = Y_all(train_idx, :);
        X_cv_val = X_all(val_idx, :);
        Y_cv_val = Y_all(val_idx, :);
        
        % For each kinematic group
        for g = 1:num_groups
            % Get indices for this group
            indices = group_indices{g};
            
            % Get training data for this group
            Y_train_group = Y_cv_train(:, indices);
            
            % Train a model (using linear regression for speed)
            mdl = fitlm(X_cv_train, Y_train_group);
            
            % Make predictions
            Y_pred_group = predict(mdl, X_cv_val);
            
            % Calculate R-squared for each dimension in this group
            for dim = 1:length(indices)
                k_idx = indices(dim);
                
                % Calculate TSS (Total Sum of Squares)
                tss = sum((Y_cv_val(:, k_idx) - mean(Y_cv_val(:, k_idx))).^2);
                
                % Calculate RSS (Residual Sum of Squares)
                rss = sum((Y_cv_val(:, k_idx) - Y_pred_group(:, dim)).^2);
                
                % Calculate R-squared
                if tss > 0
                    cv_r_squared(fold, k_idx) = 1 - (rss / tss);
                else
                    cv_r_squared(fold, k_idx) = 0;
                end
            end
        end
    end
    
    % Calculate average R-squared across folds
    avg_cv_r_squared = mean(cv_r_squared, 1);
    
    % Visualize cross-validation results
    figure('Position', [100, 100, 1000, 400]);
    
    % Plot average R-squared and compare with test R-squared
    plot(r_squared, 'b-', 'LineWidth', 1.5);
    hold on;
    plot(avg_cv_r_squared, 'r--', 'LineWidth', 1.5);
    
    % Add title and labels
    title('Comparison of Test and Cross-Validation Performance');
    xlabel('Kinematic Variable Index');
    ylabel('R-squared');
    legend('Test Set', 'Cross-Validation', 'Location', 'best');
    grid on;
    
    % Save figure
    saveas(gcf, fullfile(visualizationFolder, 'Cross_Validation_Results.png'));
    
    % Display summary
    disp(['Average R-squared across all folds: ', num2str(mean(avg_cv_r_squared))]);
end

%% Deep Learning Model (optional extension)
% This section implements a simple deep learning model for comparison
try_deep_learning = false;  % Set to true to run deep learning model

if try_deep_learning && exist('trainNetwork', 'file')
    disp('========== Deep Learning Model ==========');
    
    % Reshape input features as a time series for LSTM
    % We need to recover the windowed time series structure from our feature extraction
    
    % Select a subset of data to speed up training
    max_samples = 10000;
    if size(X_train_norm, 1) > max_samples
        train_indices = randperm(size(X_train_norm, 1), max_samples);
        X_dl_train = X_train_norm(train_indices, :);
        Y_dl_train = Y_train(train_indices, :);
    else
        X_dl_train = X_train_norm;
        Y_dl_train = Y_train;
    end
    
    if size(X_test_norm, 1) > max_samples
        test_indices = randperm(size(X_test_norm, 1), max_samples);
        X_dl_test = X_test_norm(test_indices, :);
        Y_dl_test = Y_test(test_indices, :);
    else
        X_dl_test = X_test_norm;
        Y_dl_test = Y_test;
    end
    
    % For simplicity, we'll predict only a subset of kinematic variables
    num_outputs = 10;  % Predict 10 kinematics with highest R-squared from traditional models
    [~, best_k_indices] = sort(r_squared, 'descend');
    output_indices = best_k_indices(1:num_outputs);
    
    Y_dl_train = Y_dl_train(:, output_indices);
    Y_dl_test = Y_dl_test(:, output_indices);
    
    % Create deep learning layers
    input_size = size(X_dl_train, 2);
    hidden_size = 100;
    
    layers = [
        featureInputLayer(input_size)
        fullyConnectedLayer(hidden_size)
        reluLayer
        dropoutLayer(0.2)
        fullyConnectedLayer(hidden_size/2)
        reluLayer
        fullyConnectedLayer(num_outputs)
        regressionLayer
    ];
    
    % Set training options
    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 128, ...
        'InitialLearnRate', 0.001, ...
        'GradientThreshold', 1, ...
        'Verbose', true, ...
        'Plots', 'training-progress', ...
        'ValidationData', {X_dl_test, Y_dl_test}, ...
        'ValidationFrequency', 30);
    
    % Train the network
    disp('Training deep learning model...');
    net = trainNetwork(X_dl_train, Y_dl_train, layers, options);
    
    % Make predictions
    Y_dl_pred = predict(net, X_dl_test);
    
    % Evaluate performance
    dl_rmse = sqrt(mean((Y_dl_test - Y_dl_pred).^2, 'all'));
    
    % Calculate R-squared for each output
    dl_r_squared = zeros(1, num_outputs);
    for k = 1:num_outputs
        % Calculate TSS (Total Sum of Squares)
        tss = sum((Y_dl_test(:, k) - mean(Y_dl_test(:, k))).^2);
        
        % Calculate RSS (Residual Sum of Squares)
        rss = sum((Y_dl_test(:, k) - Y_dl_pred(:, k)).^2);
        
        % Calculate R-squared
        dl_r_squared(k) = 1 - (rss / tss);
    end
    
    % Display results
    disp(['Deep Learning RMSE: ', num2str(dl_rmse)]);
    disp(['Deep Learning Average R-squared: ', num2str(mean(dl_r_squared))]);
    
    % Visualize deep learning results
    figure('Position', [100, 100, 1200, 800]);
    
    % Compare traditional and deep learning models
    subplot(2, 2, 1);
    bar([mean(r_squared(output_indices)), mean(dl_r_squared)]);
    title('Average R-squared Comparison');
    xticklabels({'Traditional', 'Deep Learning'});
    ylim([0, 1]);
    grid on;
    
    % Plot R-squared for each output variable
    subplot(2, 2, 2);
    bar([r_squared(output_indices)', dl_r_squared']);
    title('R-squared by Kinematic Variable');
    xlabel('Variable Index');
    ylabel('R-squared');
    legend({'Traditional', 'Deep Learning'});
    grid on;
    
    % Plot predictions for a sample output
    subplot(2, 2, [3, 4]);
    sample_output = 1;  % First output (highest R-squared from traditional)
    
    % Get sample of points to plot
    sample_size = min(200, size(Y_dl_test, 1));
    sample_indices = round(linspace(1, size(Y_dl_test, 1), sample_size));
    
    % Plot actual vs. predictions
    plot(Y_dl_test(sample_indices, sample_output), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(Y_dl_pred(sample_indices, sample_output), 'r--', 'LineWidth', 1.5);
    
    title(['Kinematic Variable ', num2str(output_indices(sample_output)), ' Predictions']);
    xlabel('Sample Index');
    ylabel('Value');
    legend('Actual', 'Deep Learning Prediction', 'Location', 'best');
    grid on;
    
    % Save figure
    saveas(gcf, fullfile(visualizationFolder, 'Deep_Learning_Results.png'));
    
    % Save deep learning model
    save(fullfile(visualizationFolder, 'DL_Model.mat'), 'net', 'output_indices', 'X_mean', 'X_std');
end

disp('End of script.');