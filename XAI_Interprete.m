% Random Forest Regressor for EMG to Kinematics mapping with XAI
% This script trains a random forest model to predict finger kinematics from EMG signals
% and performs explainable AI analysis on the model

%% Load data
load('s1.mat');  % Load the data file with EMG and kinematics

fprintf('Data loaded successfully.\n');
fprintf('EMG data: %d trials, %d tasks\n', size(dsfilt_emg, 1), size(dsfilt_emg, 2));
fprintf('Finger kinematics: %d trials, %d tasks\n', size(finger_kinematics, 1), size(finger_kinematics, 2));

% Define channel names for visualization and XAI
channel_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};

% Create a directory for saving results
results_dir = 'RF_EMG_Results';
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

%% Prepare data for machine learning

% Initialize arrays to store features (X) and targets (Y)
X_all = [];
Y_all = [];
trial_task_indices = []; % Keep track of which trial and task each sample comes from

% Loop through all trials and tasks to collect data
for trial = 1:5 % 5 trials
    for task = 1:7 % 7 tasks
        if ~isempty(dsfilt_emg{trial, task}) && ~isempty(finger_kinematics{trial, task})
            % Get EMG data (features)
            emg_data = dsfilt_emg{trial, task};
            
            % Get kinematics data (targets)
            kin_data = finger_kinematics{trial, task};
            
            % Downsample EMG to match kinematics sampling rate (if needed)
            % EMG: 40000x8, Kinematics: 4000x69 → 10:1 ratio
            if size(emg_data, 1) > size(kin_data, 1)
                downsample_factor = size(emg_data, 1) / size(kin_data, 1);
                if downsample_factor == 10  % Expected ratio
                    emg_data = emg_data(1:downsample_factor:end, :);
                else
                    % If ratio is not exactly 10, use interpolation for more accurate alignment
                    emg_time = linspace(0, 1, size(emg_data, 1));
                    kin_time = linspace(0, 1, size(kin_data, 1));
                    emg_data_resampled = zeros(size(kin_data, 1), size(emg_data, 2));
                    
                    for ch = 1:size(emg_data, 2)
                        emg_data_resampled(:, ch) = interp1(emg_time, emg_data(:, ch), kin_time);
                    end
                    emg_data = emg_data_resampled;
                end
            end
            
            % Ensure same length (in case there are any remaining differences)
            min_len = min(size(emg_data, 1), size(kin_data, 1));
            emg_data = emg_data(1:min_len, :);
            kin_data = kin_data(1:min_len, :);
            
            % Append to the full dataset
            X_all = [X_all; emg_data];
            Y_all = [Y_all; kin_data];
            
            % Record trial and task for each sample
            trial_task_indices = [trial_task_indices; repmat([trial, task], min_len, 1)];
            
            fprintf('Trial %d, Task %d: Added %d samples\n', trial, task, min_len);
        else
            fprintf('Warning: Missing data for trial %d, task %d\n', trial, task);
        end
    end
end

fprintf('\nTotal samples collected: %d\n', size(X_all, 1));
fprintf('Feature dimensions (EMG channels): %d\n', size(X_all, 2));
fprintf('Target dimensions (kinematics variables): %d\n', size(Y_all, 2));

%% Split data into training and testing sets using leave-one-trial-out cross-validation
% This means we'll train on 4 trials and test on the remaining trial

% Initialize arrays to store performance metrics
r2_scores = zeros(5, size(Y_all, 2)); % R² for each trial and kinematic variable
rmse_scores = zeros(5, size(Y_all, 2)); % RMSE for each trial and kinematic variable
feature_importance_all = zeros(size(X_all, 2), size(Y_all, 2), 5); % Feature importance for each channel, kinematic, and trial

% Structure to store predictions for later plotting
predictions = cell(5, 1);

% Define RF parameters
n_trees = 100;        % Number of trees in the forest
min_leaf_size = 5;    % Minimum number of observations per leaf
max_depth = 10;       % Maximum tree depth

% Create a file to log results
log_file = fullfile(results_dir, 'training_log.txt');
fileID = fopen(log_file, 'w');
fprintf(fileID, 'Random Forest Training for EMG to Kinematics Mapping\n');
fprintf(fileID, '=================================================\n\n');
fprintf(fileID, 'Data summary:\n');
fprintf(fileID, '- Total samples: %d\n', size(X_all, 1));
fprintf(fileID, '- EMG channels: %d\n', size(X_all, 2));
fprintf(fileID, '- Kinematic variables: %d\n', size(Y_all, 2));
fprintf(fileID, '- Random Forest parameters: Trees=%d, MaxDepth=%d, MinLeafSize=%d\n\n', n_trees, max_depth, min_leaf_size);

for test_trial = 1:5
    fprintf('\nTraining with leave-one-trial-out: testing on trial %d\n', test_trial);
    fprintf(fileID, '\n==== Testing on Trial %d ====\n', test_trial);
    
    % Create training and testing indices
    test_indices = trial_task_indices(:, 1) == test_trial;
    train_indices = ~test_indices;
    
    % Split data
    X_train = X_all(train_indices, :);
    Y_train = Y_all(train_indices, :);
    X_test = X_all(test_indices, :);
    Y_test = Y_all(test_indices, :);
    
    fprintf('Training samples: %d, Testing samples: %d\n', size(X_train, 1), size(X_test, 1));
    fprintf(fileID, 'Training samples: %d, Testing samples: %d\n', size(X_train, 1), size(X_test, 1));
    
    % Train a Random Forest model for each kinematic variable
    models = cell(size(Y_all, 2), 1);
    Y_pred = zeros(size(X_test, 1), size(Y_all, 2));
    
    % Create a progress bar
    fprintf('Training models for 69 kinematic variables: ');
    progress_step = ceil(size(Y_all, 2) / 10);
    
    for k = 1:size(Y_all, 2)
        % Create and train Random Forest model
        model = TreeBagger(n_trees, X_train, Y_train(:, k), 'Method', 'regression', ...
                           'OOBPrediction', 'on', 'MinLeafSize', min_leaf_size, ...
                           'MaxDepth', max_depth);
        
        % Store the model
        models{k} = model;
        
        % Make predictions
        Y_pred(:, k) = predict(model, X_test);
        
        % Calculate performance metrics
        % R² (coefficient of determination)
        SS_tot = sum((Y_test(:, k) - mean(Y_test(:, k))).^2);
        SS_res = sum((Y_test(:, k) - Y_pred(:, k)).^2);
        r2 = 1 - SS_res/SS_tot;
        r2_scores(test_trial, k) = r2;
        
        % RMSE (root mean squared error)
        rmse = sqrt(mean((Y_test(:, k) - Y_pred(:, k)).^2));
        rmse_scores(test_trial, k) = rmse;
        
        % Extract feature importance for XAI
        feature_importance_all(:, k, test_trial) = model.OOBPermutedPredictorDeltaError;
        
        % Update progress indicator
        if mod(k, progress_step) == 0
            fprintf('.');
        end
    end
    fprintf(' Done!\n');
    
    % Store predictions for this test trial
    predictions{test_trial} = struct('Y_test', Y_test, 'Y_pred', Y_pred, 'test_indices', find(test_indices));
    
    % Log trial results
    fprintf(fileID, 'Average R² for trial %d: %.4f\n', test_trial, mean(r2_scores(test_trial, :)));
    fprintf(fileID, 'Average RMSE for trial %d: %.4f\n', test_trial, mean(rmse_scores(test_trial, :)));
end

%% Calculate and display overall performance metrics

% Average R² and RMSE across all trials
mean_r2 = mean(r2_scores, 1);
mean_rmse = mean(rmse_scores, 1);
std_r2 = std(r2_scores, 0, 1);
std_rmse = std(rmse_scores, 0, 1);

% Display overall results
fprintf('\n===== OVERALL PERFORMANCE =====\n');
fprintf('Average R² across all kinematics: %.4f (±%.4f)\n', mean(mean_r2), std(mean(mean_r2)));
fprintf('Average RMSE across all kinematics: %.4f (±%.4f)\n', mean(mean_rmse), std(mean(mean_rmse)));

% Save metrics to the log file
fprintf(fileID, '\n===== OVERALL PERFORMANCE =====\n');
fprintf(fileID, 'Average R² across all kinematics: %.4f (±%.4f)\n', mean(mean_r2), std(mean(mean_r2)));
fprintf(fileID, 'Average RMSE across all kinematics: %.4f (±%.4f)\n', mean(mean_rmse), std(mean(mean_rmse)));

% Display detailed metrics for each kinematic variable
fprintf(fileID, '\n===== DETAILED METRICS BY KINEMATIC VARIABLE =====\n');
fprintf(fileID, 'Variable\tR² (mean±std)\tRMSE (mean±std)\n');
for k = 1:size(Y_all, 2)
    fprintf(fileID, '%d\t%.4f±%.4f\t%.4f±%.4f\n', k, mean_r2(k), std_r2(k), mean_rmse(k), std_rmse(k));
end

% Display top 5 and worst 5 predicted kinematics based on R²
[sorted_r2, sorted_indices] = sort(mean_r2, 'descend');
fprintf('\n----- Top 5 Best Predicted Kinematics -----\n');
fprintf(fileID, '\n----- Top 5 Best Predicted Kinematics -----\n');
for i = 1:5
    idx = sorted_indices(i);
    fprintf('Kinematic #%d: R² = %.4f (±%.4f), RMSE = %.4f (±%.4f)\n', idx, mean_r2(idx), std_r2(idx), mean_rmse(idx), std_rmse(idx));
    fprintf(fileID, 'Kinematic #%d: R² = %.4f (±%.4f), RMSE = %.4f (±%.4f)\n', idx, mean_r2(idx), std_r2(idx), mean_rmse(idx), std_rmse(idx));
end

fprintf('\n----- 5 Worst Predicted Kinematics -----\n');
fprintf(fileID, '\n----- 5 Worst Predicted Kinematics -----\n');
for i = 1:5
    idx = sorted_indices(end-i+1);
    fprintf('Kinematic #%d: R² = %.4f (±%.4f), RMSE = %.4f (±%.4f)\n', idx, mean_r2(idx), std_r2(idx), mean_rmse(idx), std_rmse(idx));
    fprintf(fileID, 'Kinematic #%d: R² = %.4f (±%.4f), RMSE = %.4f (±%.4f)\n', idx, mean_r2(idx), std_r2(idx), mean_rmse(idx), std_rmse(idx));
end

% Calculate and display average R² and RMSE by task
task_r2 = zeros(7, 1);
task_rmse = zeros(7, 1);
counts = zeros(7, 1);

% Create a boxplot of R² values by task
r2_by_task = cell(7, 1);

for i = 1:length(trial_task_indices)
    trial = trial_task_indices(i, 1);
    task = trial_task_indices(i, 2);
    pred_idx = -1;
    
    % Find the corresponding prediction index
    for t = 1:5
        if ismember(i, predictions{t}.test_indices)
            pred_idx = find(predictions{t}.test_indices == i);
            break;
        end
    end
    
    if pred_idx > 0
        counts(task) = counts(task) + 1;
        
        % Add R² values to the respective task array for boxplot
        if isempty(r2_by_task{task})
            r2_by_task{task} = r2_scores(trial, :);
        end
    end
end

% Plot R² distribution by task
figure('Position', [100, 100, 1000, 600]);
task_labels = {'Thumb', 'Index', 'Middle', 'Ring', 'Little', 'All', 'Random'};
r2_data = [];
group_data = [];

for task = 1:7
    if ~isempty(r2_by_task{task})
        r2_data = [r2_data, r2_by_task{task}];
        group_data = [group_data, repmat(task, 1, length(r2_by_task{task}))];
    end
end

boxplot(r2_data, group_data, 'Labels', task_labels);
title('Distribution of R² Values by Task');
ylabel('R² Value');
xlabel('Task');
grid on;
saveas(gcf, fullfile(results_dir, 'r2_by_task.png'));

%% XAI: Explainable AI Analysis

% Calculate average feature importance across all trials
avg_feature_importance = mean(feature_importance_all, 3);

% Normalize feature importance for better visualization
normalized_importance = zeros(size(avg_feature_importance));
for k = 1:size(avg_feature_importance, 2)
    if max(avg_feature_importance(:, k)) > 0
        normalized_importance(:, k) = avg_feature_importance(:, k) / max(avg_feature_importance(:, k));
    end
end

% Create a heatmap of feature importance
figure('Position', [100, 100, 1200, 800]);
imagesc(normalized_importance');
colormap('hot');
colorbar;
title('Feature Importance Heatmap: EMG Channels vs Kinematics');
xlabel('EMG Channels');
ylabel('Kinematic Variables');
xticks(1:8);
xticklabels(channel_names);
yticks(1:5:size(Y_all, 2));
yticklabels(1:5:size(Y_all, 2));
set(gca, 'FontSize', 12);
saveas(gcf, fullfile(results_dir, 'feature_importance_heatmap.png'));

% Calculate and plot the average importance of each channel across all kinematics
avg_channel_importance = mean(normalized_importance, 2);
figure('Position', [100, 100, 800, 600]);
bar(avg_channel_importance);
title('Average EMG Channel Importance Across All Kinematics');
xlabel('EMG Channels');
ylabel('Average Normalized Importance');
xticks(1:8);
xticklabels(channel_names);
grid on;
saveas(gcf, fullfile(results_dir, 'average_channel_importance.png'));

% Log the average importance of each channel
fprintf(fileID, '\n===== AVERAGE EMG CHANNEL IMPORTANCE =====\n');
for c = 1:8
    fprintf(fileID, '%s: %.4f\n', channel_names{c}, avg_channel_importance(c));
end

% Analyze the most important EMG channel for each kinematic variable
[max_imp_values, max_imp_channels] = max(normalized_importance, [], 1);
fprintf(fileID, '\n===== MOST IMPORTANT EMG CHANNEL FOR EACH KINEMATIC =====\n');
for k = 1:size(Y_all, 2)
    fprintf(fileID, 'Kinematic #%d: %s (Importance: %.4f)\n', k, channel_names{max_imp_channels(k)}, max_imp_values(k));
end

% Analyze which kinematic variables are most influenced by each EMG channel
[~, max_kin_per_channel] = max(normalized_importance, [], 2);
fprintf(fileID, '\n===== MOST INFLUENCED KINEMATIC FOR EACH EMG CHANNEL =====\n');
for c = 1:8
    fprintf(fileID, '%s: Kinematic #%d (Importance: %.4f)\n', channel_names{c}, max_kin_per_channel(c), ...
            normalized_importance(c, max_kin_per_channel(c)));
end

% Find the top 3 most important EMG channels for the best predicted kinematics
fprintf(fileID, '\n===== TOP 3 EMG CHANNELS FOR BEST PREDICTED KINEMATICS =====\n');
for i = 1:5
    k = sorted_indices(i);
    [sorted_imp, sorted_ch_idx] = sort(normalized_importance(:, k), 'descend');
    fprintf(fileID, 'Kinematic #%d (R² = %.4f):\n', k, mean_r2(k));
    for j = 1:3
        fprintf(fileID, '  #%d: %s (Importance: %.4f)\n', j, channel_names{sorted_ch_idx(j)}, sorted_imp(j));
    end
end

%% Plot sample predictions for visual comparison
% Select the top 3 best predicted kinematics
top_kinematics = sorted_indices(1:3);

% Find a test segment with at least 300 consecutive samples from the first test trial
test_trial_to_plot = 1;
segment_length = min(300, size(predictions{test_trial_to_plot}.Y_test, 1));
segment_start = 1;

figure('Position', [100, 100, 1200, 800]);
for i = 1:3
    k = top_kinematics(i);
    
    subplot(3, 1, i);
    plot(1:segment_length, predictions{test_trial_to_plot}.Y_test(segment_start:segment_start+segment_length-1, k), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(1:segment_length, predictions{test_trial_to_plot}.Y_pred(segment_start:segment_start+segment_length-1, k), 'r--', 'LineWidth', 1.5);
    hold off;
    
    title(sprintf('Kinematic #%d (R² = %.4f, RMSE = %.4f)', k, mean_r2(k), mean_rmse(k)));
    xlabel('Time (samples)');
    ylabel('Value');
    legend('Actual', 'Predicted');
    grid on;
end
sgtitle('Top 3 Best Predicted Kinematics - Actual vs Predicted');
saveas(gcf, fullfile(results_dir, 'top_predictions_visualization.png'));

%% XAI: Partial Dependence Plots for top 3 kinematics and their most important EMG channels
% Find the most important channel for each of the top 3 kinematics
top_3_kin = sorted_indices(1:3);
for k_idx = 1:3
    k = top_3_kin(k_idx);
    [~, most_imp_channel] = max(normalized_importance(:, k));
    
    fprintf('\nCreating partial dependence plot for Kinematic #%d and Channel %s\n', k, channel_names{most_imp_channel});
    fprintf(fileID, '\nPartial dependence analysis for Kinematic #%d and Channel %s\n', k, channel_names{most_imp_channel});
    
    % Create a range of values for the chosen EMG channel
    X_range = linspace(min(X_all(:, most_imp_channel)), max(X_all(:, most_imp_channel)), 50);
    
    % Create a partial dependence plot
    figure('Position', [100, 100, 800, 600]);
    
    % Use one of the trained models for this kinematic
    model = models{k};
    
    % For each value in X_range, predict using all test samples and average
    pd_values = zeros(size(X_range));
    X_temp = X_test;
    
    for i = 1:length(X_range)
        X_temp(:, most_imp_channel) = X_range(i);
        pd_values(i) = mean(predict(model, X_temp));
    end
    
    % Plot the partial dependence
    plot(X_range, pd_values, 'b-', 'LineWidth', 2);
    title(sprintf('Partial Dependence Plot: Kinematic #%d vs %s', k, channel_names{most_imp_channel}));
    xlabel(sprintf('%s EMG Activity', channel_names{most_imp_channel}));
    ylabel(sprintf('Predicted Kinematic #%d Value', k));
    grid on;
    
    % Add a histogram of the actual data distribution
    yyaxis right;
    histogram(X_all(:, most_imp_channel), 30, 'Normalization', 'probability', 'FaceAlpha', 0.3);
    ylabel('Data Distribution');
    
    saveas(gcf, fullfile(results_dir, sprintf('partial_dependence_kin%d_channel%s.png', k, channel_names{most_imp_channel})));
    
    % Log the relationship pattern
    % Calculate correlation between EMG values and predictions
    corr_val = corr(X_range', pd_values');
    if corr_val > 0.7
        relationship = 'strong positive';
    elseif corr_val > 0.3
        relationship = 'moderate positive';
    elseif corr_val > 0
        relationship = 'weak positive';
    elseif corr_val > -0.3
        relationship = 'weak negative';
    elseif corr_val > -0.7
        relationship = 'moderate negative';
    else
        relationship = 'strong negative';
    end
    fprintf(fileID, '- Correlation: %.4f (%s relationship)\n', corr_val, relationship);
    
    % Check for non-linearity
    % Fit linear and quadratic models to the partial dependence curve
    p1 = polyfit(X_range, pd_values, 1);
    p2 = polyfit(X_range, pd_values, 2);
    y1 = polyval(p1, X_range);
    y2 = polyval(p2, X_range);
    
    % Compare errors of linear vs quadratic fit
    err1 = mean((pd_values - y1).^2);
    err2 = mean((pd_values - y2).^2);
    
    if err2 < 0.8 * err1
        fprintf(fileID, '- Relationship appears to be non-linear (quadratic fit improves error by %.1f%%)\n', 100 * (1 - err2/err1));
    else
        fprintf(fileID, '- Relationship appears to be approximately linear\n');
    end
end

%% Feature interaction analysis for the top kinematic variable
% This section examines how pairs of EMG channels interact to predict the top kinematic variable
top_kin = sorted_indices(1);
[sorted_imp, sorted_ch_idx] = sort(normalized_importance(:, top_kin), 'descend');
top_2_channels = sorted_ch_idx(1:2);

fprintf(fileID, '\n===== FEATURE INTERACTION ANALYSIS =====\n');
fprintf(fileID, 'Analyzing interaction between %s and %s for Kinematic #%d\n', ...
        channel_names{top_2_channels(1)}, channel_names{top_2_channels(2)}, top_kin);

% Create a grid of values for the two most important channels
X_range1 = linspace(min(X_all(:, top_2_channels(1))), max(X_all(:, top_2_channels(1))), 15);
X_range2 = linspace(min(X_all(:, top_2_channels(2))), max(X_all(:, top_2_channels(2))), 15);
[X1, X2] = meshgrid(X_range1, X_range2);

% Create a 2D partial dependence plot
pd_values_2d = zeros(size(X1));
X_temp = X_test;

model = models{top_kin};

for i = 1:length(X_range1)
    for j = 1:length(X_range2)
        X_temp(:, top_2_channels(1)) = X_range1(i);
        X_temp(:, top_2_channels(2)) = X_range2(j);
        pd_values_2d(j, i) = mean(predict(model, X_temp));
    end
end

% Plot the 2D partial dependence
figure('Position', [100, 100, 900, 700]);
surf(X1, X2, pd_values_2d);
title(sprintf('2D Partial Dependence: Kinematic #%d vs %s and %s', ...
      top_kin, channel_names{top_2_channels(1)}, channel_names{top_2_channels(2)}));
xlabel(channel_names{top_2_channels(1)});
ylabel(channel_names{top_2_channels(2)});
zlabel(sprintf('Predicted Kinematic #%d', top_kin));
colormap('jet');
colorbar;
view(45, 30);
saveas(gcf, fullfile(results_dir, sprintf('2d_partial_dependence_kin%d.png', top_kin)));

% Calculate interaction strength
% First, get 1D partial dependence for each feature
pd_values_1d_1 = mean(pd_values_2d, 1);
pd_values_1d_2 = mean(pd_values_2d, 2);

% Calculate the interaction strength
interaction_strength = pd_values_2d - pd_values_1d_1' - pd_values_1d_2 + mean(pd_values_2d(:));
interaction_magnitude = sum(abs(interaction_strength(:))) / numel(interaction_strength);

fprintf(fileID, 'Interaction strength between %s and %s: %.4f\n', ...
        channel_names{top_2_channels(1)}, channel_names{top_2_channels(2)}, interaction_magnitude);

if interaction_magnitude > 0.1
    fprintf(fileID, '- Strong interaction detected: Features work together non-additively\n');
elseif interaction_magnitude > 0.05
    fprintf(fileID, '- Moderate interaction detected\n');
else
    fprintf(fileID, '- Weak interaction: Features contribute mostly independently\n');
end

%% Generate a decision tree visualization for interpretability
% Create a simplified decision tree for the top kinematic variable
fprintf('Creating decision tree visualization for top kinematic variable...\n');
top_kin = sorted_indices(1);

% Train a simpler decision tree for visualization
tree_model = fitrtree(X_train, Y_train(:, top_kin), 'MaxDepth', 4);
view(tree_model, 'Mode', 'graph');
saveas(gcf, fullfile(results_dir, sprintf('decision_tree_kin%d.png', top_kin)));

% Extract and log the decision rules
fprintf(fileID, '\n===== DECISION TREE RULES FOR KINEMATIC #%d =====\n', top_kin);
rules = extractRules(tree_model);
for i = 1:length(rules)
    fprintf(fileID, 'Rule %d: %s\n', i, rules{i});
end

% Helper function to extract rules from decision tree
function rules = extractRules(tree)
    rules = {};
    nodes = tree.PruneList;
    for i = 1:length(nodes)
        if ~tree.IsBranchNode(i)
            rule = '';
            node = i;
            while node ~= 1
                parent = tree.Parent(node);
                if tree.Children(parent, 1) == node
                    condition = sprintf('%s < %.4f', tree.PredictorNames{tree.CutPredictor(parent)}, tree.CutPoint(parent));
                else
                    condition = sprintf('%s >= %.4f', tree.PredictorNames{tree.CutPredictor(parent)}, tree.CutPoint(parent));
                end
                
                if isempty(rule)
                    rule = condition;
                else
                    rule = [condition ' AND ' rule];
                end
                
                node = parent;
            end
            rule = [rule ' => ' num2str(tree.NodeMean(i))];
            rules{end+1} = rule;
        end
    end
end

%% Save the results
results = struct();
results.r2_scores = r2_scores;
results.rmse_scores = rmse_scores;
results.mean_r2 = mean_r2;
results.mean_rmse = mean_rmse;
results.feature_importance = avg_feature_importance;
results.normalized_importance = normalized_importance;
results.sorted_indices = sorted_indices;
results.predictions = predictions;

save(fullfile(results_dir, 'emg_kinematics_rf_results.mat'), 'results');

% Close the log file
fclose(fileID);

fprintf('\nAnalysis completed and results saved to %s\n', results_dir);
fprintf('Check %s for detailed log file\n', log_file);