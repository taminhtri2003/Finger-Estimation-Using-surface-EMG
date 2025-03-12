% Reduced EMG to Joint Angle Prediction with MAV, RMS, Entropy, and Wavelet
%% Load Data
load('s1_full.mat');
%% Data Preparation
emg_features = cell(5, 5);
all_joint_angles = cell(5, 5);
window_size = 200; % Increased window size
window_overlap = 100; % Increased overlap
num_muscles = 8;
feature_names = {'MAV', 'RMS', 'Entropy', 'Wavelet'}; % Reduced feature set
num_features = length(feature_names);
%% Feature Extraction
for trial = 1:5
    for task = 1:5
        current_emg = dsfilt_emg{trial, task};
        current_joint_angles = joint_angles{trial, task};
        num_samples = size(current_emg, 1);
        num_windows = floor((num_samples - window_size) / (window_size - window_overlap)) + 1;
        features_matrix = zeros(num_windows, num_muscles * num_features);
        ds_joint_angles = zeros(num_windows, 14);
        for w = 1:num_windows
            start_idx = (w - 1) * (window_size - window_overlap) + 1;
            end_idx = start_idx + window_size - 1;
            center_idx = round((start_idx + end_idx) / 2);
            if center_idx <= size(current_joint_angles, 1)
                ds_joint_angles(w, :) = current_joint_angles(center_idx, :);
            end
            for m = 1:num_muscles
                window_data = current_emg(start_idx:end_idx, m);
                feat_idx = (m - 1) * num_features;
                % Feature 1: Mean Absolute Value (MAV)
                features_matrix(w, feat_idx + 1) = mean(abs(window_data));
                % Feature 2: Root Mean Square (RMS)
                features_matrix(w, feat_idx + 2) = sqrt(mean(window_data.^2));
                % Feature 3: Entropy
                features_matrix(w, feat_idx + 3) = entropy(window_data);
                 % Feature 4: Wavelet Feature extraction example.
                [c, l] = wavedec(window_data, 3, 'db4');
                features_matrix(w, feat_idx + 4) = mean(abs(c));
            end
        end
        emg_features{trial, task} = features_matrix;
        all_joint_angles{trial, task} = ds_joint_angles;
    end
end
%% Prepare Data for Machine Learning
X_all = [];
Y_all = [];
for trial = 1:5
    for task = 1:5
        X_all = [X_all; emg_features{trial, task}];
        Y_all = [Y_all; all_joint_angles{trial, task}];
    end
end
valid_rows = all(isfinite(X_all), 2) & all(isfinite(Y_all), 2);
X_all = X_all(valid_rows, :);
Y_all = Y_all(valid_rows, :);
feature_col_names = cell(1, num_muscles * num_features);
muscle_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};
for m = 1:num_muscles
    for f = 1:num_features
        feature_col_names{(m - 1) * num_features + f} = [muscle_names{m} '_' feature_names{f}];
    end
end
joint_names = {'Thumb_1', 'Thumb_2', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Ring_1', 'Ring_2', 'Ring_3', 'Little_1', 'Little_2', 'Little_3'};
%% Feature Selection and Normalization
[X_normalized, mu, sigma] = zscore(X_all);
corr_threshold = 0.3;
selected_features = [];
feature_importance = zeros(size(X_normalized, 2), size(Y_all, 2));
for j = 1:size(Y_all, 2)
    for f = 1:size(X_normalized, 2)
        r = corrcoef(X_normalized(:, f), Y_all(:, j));
        feature_importance(f, j) = abs(r(1, 2));
    end
    selected_features_j = find(feature_importance(:, j) > corr_threshold);
    selected_features = union(selected_features, selected_features_j);
end
if length(selected_features) < 10
    [~, sorted_idx] = sort(sum(feature_importance, 2), 'descend');
    selected_features = sorted_idx(1:min(10, size(X_normalized, 2)));
end
X_selected = X_normalized(:, selected_features);
selected_feature_names = feature_col_names(selected_features);
%% Split Data and K-Fold Cross Validation
k = 5; % Number of folds
cv = cvpartition(size(X_selected, 1), 'KFold', k);
test_rmse_cv = zeros(k, size(Y_all, 2));
test_r2_cv = zeros(k, size(Y_all, 2));
for fold = 1:k
    X_train = X_selected(cv.training(fold), :);
    Y_train = Y_all(cv.training(fold), :);
    X_test = X_selected(cv.test(fold), :);
    Y_test = Y_all(cv.test(fold), :);
    %% Train Machine Learning Models (Random Forest)
    num_joints = size(Y_all, 2);
    models = cell(1, num_joints);
    predictions = zeros(size(X_test, 1), num_joints);
    for j = 1:num_joints
        fprintf('Training model for %s joint angle (Fold %d)...\n', joint_names{j}, fold);
        model = TreeBagger(100, X_train, Y_train(:, j), 'Method', 'regression'); % Increased trees
        models{j} = model;
        predictions(:, j) = predict(model, X_test);
        test_rmse_cv(fold, j) = sqrt(mean((predictions(:, j) - Y_test(:, j)).^2));
        y_mean = mean(Y_test(:,j));
        ss_total = sum((Y_test(:,j) - y_mean).^2);
        ss_residual = sum((Y_test(:,j) - predictions(:,j)).^2);
        test_r2_cv(fold, j) = 1 - (ss_residual / ss_total);
    end
end
% Calculate average RMSE and R^2 across folds
test_rmse = mean(test_rmse_cv, 1);
test_r2 = mean(test_r2_cv, 1);
overall_rmse = sqrt(mean(test_rmse.^2));
average_r2 = mean(test_r2);
fprintf('Overall RMSE: %.4f degrees\n', overall_rmse);
fprintf('Average R^2: %.4f\n', average_r2);
%% Visualizations
% 1. Feature Importance Heatmap
figure('Name', 'Feature Importance for Joint Angle Prediction', 'Position', [100, 100, 1200, 600]);
feature_importance_selected = feature_importance(selected_features, :);
if length(selected_feature_names) ~= size(feature_importance_selected, 1)
    selected_feature_names = cellstr(string(1:size(feature_importance_selected, 1)));
end
if length(joint_names) ~= size(feature_importance_selected, 2)
   joint_names = cellstr(string(1:size(feature_importance_selected, 2)));
end
h = heatmap(joint_names, selected_feature_names, feature_importance_selected, 'ColorMap', jet, 'ColorLimits', [0 1]);
title('Feature Importance Heatmap (Correlation)');
xlabel('Joint Angles');
ylabel('EMG Features');
% 2. Plot actual vs predicted for each joint angle
figure('Name', 'Actual vs Predicted Joint Angles', 'Position', [100, 100, 1200, 800]);
num_plots = min(6, num_joints);
for j = 1:num_plots
    subplot(2, 3, j);
    time_indices = 1:length(Y_test(:, j));
    plot(time_indices, Y_test(:, j), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(time_indices, predictions(:, j), 'r-', 'LineWidth', 1.5);
    hold off;
    title(sprintf('%s (RMSE: %.2f°, R^2: %.2f)', joint_names{j}, test_rmse(j), test_r2(j)));
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
    min_val = min(min(Y_test(:, j)), min(predictions(:, j)));
    max_val = max(max(Y_test(:, j)), max(predictions(:, j)));
    line([min_val, max_val], [min_val, max_val], 'Color', 'red', 'LineWidth', 2);
    title(sprintf('%s (R^2: %.2f)', joint_names{j}, test_r2(j)));
    xlabel('Actual Angle (degrees)');
    ylabel('Predicted Angle (degrees)');
    grid on;
end
% 4. Test RMSE and R^2 Visualization
figure('Name', 'RMSE and R^2 for Each Joint Angle', 'Position', [100, 100, 1200, 600]);
subplot(1, 2, 1);
bar(test_rmse);
xticks(1:num_joints);
xticklabels(joint_names);
xtickangle(45);
ylabel('RMSE (degrees)');
title('Prediction Error for Each Joint Angle');
grid on;
subplot(1, 2, 2);
bar(test_r2);
xticks(1:num_joints);
xticklabels(joint_names);
xtickangle(45);
ylabel('R^2');
title('R^2 for Each Joint Angle');
grid on;
% 5. Temporal correlation between EMG and Joint Angles (Example for one joint)
figure('Name', 'Temporal Relationship Between EMG Features and Joint Angles', 'Position', [100, 100, 1200, 600]);
joint_idx = 3;
time_window = 1:min(200, size(Y_test, 1));
emg_feature_idx = find(contains(selected_feature_names, 'FDS_MAV') | contains(selected_feature_names, 'FDP_MAV'));
if ~isempty(emg_feature_idx)
    emg_feature_idx = emg_feature_idx(1);
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
save('emg_joint_model_enhanced.mat', 'models', 'selected_features', 'mu', 'sigma', 'test_rmse', 'test_r2', 'overall_rmse', 'average_r2');
fprintf('Enhanced models and parameters saved to emg_joint_model_enhanced.mat\n');