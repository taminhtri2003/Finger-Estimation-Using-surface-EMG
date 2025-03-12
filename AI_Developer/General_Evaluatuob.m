% Enhanced EMG to Joint Angle Prediction with Feature Extraction and Visualization
%% Load Data
load('s1_full.mat');

%% Data Preparation
emg_features = cell(5, 5);
all_joint_angles = cell(5, 5);
window_size = 200; % Increased window size
window_overlap = 100; % Increased overlap
num_muscles = 8;
feature_names = {'MAV', 'RMS', 'WL', 'ZC', 'SSC', 'WA', 'MF', 'Entropy', 'Wavelet'}; % Added features
num_features = length(feature_names);
threshold = 0.01;

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

                features_matrix(w, feat_idx + 1) = mean(abs(window_data));
                features_matrix(w, feat_idx + 2) = sqrt(mean(window_data.^2));
                features_matrix(w, feat_idx + 3) = sum(abs(diff(window_data)));
                zc = sum(((window_data(1:end-1) > 0 & window_data(2:end) < 0) | (window_data(1:end-1) < 0 & window_data(2:end) > 0)) & abs(diff(window_data)) >= threshold);
                features_matrix(w, feat_idx + 4) = zc;
                ssc = sum(((window_data(2:end-1) > window_data(1:end-2) & window_data(2:end-1) > window_data(3:end)) | (window_data(2:end-1) < window_data(1:end-2) & window_data(2:end-1) < window_data(3:end))) & (abs(window_data(2:end-1) - window_data(3:end)) >= threshold | abs(window_data(2:end-1) - window_data(1:end-2)) >= threshold));
                features_matrix(w, feat_idx + 5) = ssc;
                wa = sum(abs(diff(window_data)) >= threshold);
                features_matrix(w, feat_idx + 6) = wa;

                if length(window_data) > 1
                    [pxx, f] = pwelch(window_data, hamming(min(length(window_data), 128)), [], [], 4000);
                    total_power = sum(pxx);
                    cumulative_power = cumsum(pxx);
                    median_freq_idx = find(cumulative_power >= total_power / 2, 1, 'first');
                    features_matrix(w, feat_idx + 7) = f(median_freq_idx);
                else
                    features_matrix(w, feat_idx + 7) = 0;
                end

                features_matrix(w, feat_idx + 8) = entropy(window_data);
                [c, l] = wavedec(window_data, 3, 'db4');
                features_matrix(w, feat_idx + 9) = mean(abs(c));
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

%% Feature Selection and Normalization (for the original combined model)
[X_normalized, mu, sigma] = zscore(X_all);
corr_threshold = 0.3;
selected_features_combined = [];  % Renamed to avoid confusion
feature_importance_combined = zeros(size(X_normalized, 2), size(Y_all, 2));

for j = 1:size(Y_all, 2)
    for f = 1:size(X_normalized, 2)
        r = corrcoef(X_normalized(:, f), Y_all(:, j));
        feature_importance_combined(f, j) = abs(r(1, 2));
    end
    selected_features_j = find(feature_importance_combined(:, j) > corr_threshold);
    selected_features_combined = union(selected_features_combined, selected_features_j);
end

if length(selected_features_combined) < 10
    [~, sorted_idx] = sort(sum(feature_importance_combined, 2), 'descend');
    selected_features_combined = sorted_idx(1:min(10, size(X_normalized, 2)));
end

X_selected_combined = X_normalized(:, selected_features_combined);
selected_feature_names_combined = feature_col_names(selected_features_combined);


%% Split Data and K-Fold Cross Validation (for all models)
k = 5; % Number of folds
cv = cvpartition(size(X_all, 1), 'KFold', k);  % Use size(X_all, 1)

%% Training: Combined Model (Original)
test_rmse_combined = zeros(k, size(Y_all, 2));
test_r2_combined = zeros(k, size(Y_all, 2));

for fold = 1:k
    X_train = X_selected_combined(cv.training(fold), :);
    Y_train = Y_all(cv.training(fold), :);
    X_test = X_selected_combined(cv.test(fold), :);
    Y_test = Y_all(cv.test(fold), :);

    models_combined = cell(1, size(Y_all, 2));
    predictions_combined = zeros(size(X_test, 1), size(Y_all, 2));

    for j = 1:size(Y_all, 2)
        model = TreeBagger(100, X_train, Y_train(:, j), 'Method', 'regression');
        models_combined{j} = model;
        predictions_combined(:, j) = predict(model, X_test);
        test_rmse_combined(fold, j) = sqrt(mean((predictions_combined(:, j) - Y_test(:, j)).^2));
        y_mean = mean(Y_test(:,j));
        ss_total = sum((Y_test(:,j) - y_mean).^2);
        ss_residual = sum((Y_test(:,j) - predictions_combined(:,j)).^2);
        test_r2_combined(fold, j) = 1 - (ss_residual / ss_total);
    end
end

rmse_combined = mean(test_rmse_combined, 1);
r2_combined = mean(test_r2_combined, 1);


%% Training: Individual Muscle Models
rmse_individual = zeros(num_muscles, size(Y_all, 2));
r2_individual = zeros(num_muscles, size(Y_all, 2));

for m = 1:num_muscles
     fprintf('Training models for muscle %s...\n', muscle_names{m});
    muscle_features_indices = (m - 1) * num_features + 1 : m * num_features;
    X_muscle = X_all(:, muscle_features_indices);
    [X_muscle_normalized, ~, ~] = zscore(X_muscle); % Normalize per muscle

        test_rmse_individual = zeros(k, size(Y_all, 2));
        test_r2_individual = zeros(k, size(Y_all, 2));

    for fold = 1:k
        X_train = X_muscle_normalized(cv.training(fold), :);
        Y_train = Y_all(cv.training(fold), :);
        X_test = X_muscle_normalized(cv.test(fold), :);
        Y_test = Y_all(cv.test(fold), :);

        models_individual = cell(1, size(Y_all, 2));
        predictions_individual = zeros(size(X_test, 1), size(Y_all, 2));


        for j = 1:size(Y_all, 2)
            model = TreeBagger(100, X_train, Y_train(:, j), 'Method', 'regression');
            models_individual{j} = model;
            predictions_individual(:,j) = predict(model, X_test);

            test_rmse_individual(fold, j) = sqrt(mean((predictions_individual(:, j) - Y_test(:, j)).^2));
            y_mean = mean(Y_test(:,j));
            ss_total = sum((Y_test(:,j) - y_mean).^2);
            ss_residual = sum((Y_test(:,j) - predictions_individual(:,j)).^2);
            test_r2_individual(fold, j) = 1 - (ss_residual / ss_total);

        end
    end
            rmse_individual(m,:) = mean(test_rmse_individual, 1);
            r2_individual(m,:) = mean(test_r2_individual, 1);
end

%% Training: Muscle Group Models
muscle_groups = {
    [1, 2], ...     % Wrist Flexors/Extensors (APL, FCR)
    [3, 4], ...     % Finger Flexors (FDS, FDP)
    [5, 6], ...     % Finger Extensors (ED, EI)
    [7, 8], ...     % Wrist Ulnar/Radial Deviators (ECU, ECR)
    [1, 2, 7, 8],... % All Wrist Muscles
    [3, 4, 5, 6],...  % All Finger Muscles
    [1,3,5],...      % Muscles for Thumb and Index finger (APL, FDS, ED) - Example
    [2,4,6], ...  % Muscles related by function (FCR, FDP, EI)
    [1,2,3,4] ... % Forearm Flexors
    [5,6,7,8] ...   % Forearm Extensors
};
group_names = {
    'Wrist Flex/Ext',
    'Finger Flex',
    'Finger Ext',
    'Wrist Dev',
    'All Wrist',
    'All Finger',
    'Thumb/Index',  % Descriptive name
    'Related Func',  % Descriptive name
    'Forearm Flex',
    'Forearm Ext'
    };
num_groups = length(muscle_groups);

rmse_group = zeros(num_groups, size(Y_all, 2));
r2_group = zeros(num_groups, size(Y_all, 2));


for g = 1:num_groups
    fprintf('Training models for muscle group %s...\n', group_names{g});
    group_indices = muscle_groups{g};
    group_features_indices = [];
    for muscle_index = group_indices
        group_features_indices = [group_features_indices, (muscle_index - 1) * num_features + 1 : muscle_index * num_features];
    end
    X_group = X_all(:, group_features_indices);
    [X_group_normalized, ~, ~] = zscore(X_group);

        test_rmse_group = zeros(k, size(Y_all, 2));
        test_r2_group = zeros(k, size(Y_all, 2));

     for fold = 1:k
        X_train = X_group_normalized(cv.training(fold), :);
        Y_train = Y_all(cv.training(fold), :);
        X_test = X_group_normalized(cv.test(fold), :);
        Y_test = Y_all(cv.test(fold), :);

        models_group = cell(1, size(Y_all, 2));
        predictions_group = zeros(size(X_test, 1), size(Y_all, 2));

        for j = 1:size(Y_all, 2)
            model = TreeBagger(100, X_train, Y_train(:, j), 'Method', 'regression');
            models_group{j} = model;
            predictions_group(:, j) = predict(model, X_test);
            test_rmse_group(fold, j) = sqrt(mean((predictions_group(:, j) - Y_test(:, j)).^2));
            y_mean = mean(Y_test(:,j));
            ss_total = sum((Y_test(:,j) - y_mean).^2);
            ss_residual = sum((Y_test(:,j) - predictions_group(:,j)).^2);
            test_r2_group(fold, j) = 1 - (ss_residual / ss_total);
        end
     end
        rmse_group(g,:) = mean(test_rmse_group, 1);
        r2_group(g,:) = mean(test_r2_group, 1);
end


%% Visualization: Comparison of RMSE and R^2
for j = 1:size(Y_all, 2)  % Iterate through each joint
    figure;

    % RMSE Comparison
    subplot(1, 2, 1);
    % Concatenate horizontally *within* the bar function
    bar([rmse_combined(j), rmse_individual(:, j)', rmse_group(:, j)']);
    xticks(1:(1 + num_muscles + num_groups));
    xticklabels({ 'Combined', muscle_names{:}, group_names{:} }); % FIX HERE
    xtickangle(45);
    ylabel('RMSE (degrees)');
    title(['RMSE Comparison for ' joint_names{j}]);
    grid on;
    set(gca, 'FontSize', 10);

    % R^2 Comparison
    subplot(1, 2, 2);
    % Concatenate horizontally *within* the bar function
    bar([r2_combined(j), r2_individual(:, j)', r2_group(:, j)']);
    xticks(1:(1 + num_muscles + num_groups));
    xticklabels({ 'Combined', muscle_names{:}, group_names{:} }); % FIX HERE
    xtickangle(45);
    ylabel('R^2');
    title(['R^2 Comparison for ' joint_names{j}]);
    grid on;
     set(gca, 'FontSize', 10);
end

%% Overall average performance across all joints.

overall_rmse_combined = sqrt(mean(rmse_combined.^2));
overall_r2_combined = mean(r2_combined);

overall_rmse_individual = squeeze(sqrt(mean(rmse_individual.^2, 2)));
overall_r2_individual = squeeze(mean(r2_individual,2));


overall_rmse_group = squeeze(sqrt(mean(rmse_group.^2, 2)));
overall_r2_group = squeeze(mean(r2_group, 2));


figure;
% RMSE Comparison
subplot(1, 2, 1);
% Concatenate horizontally *within* the bar function
bar([overall_rmse_combined, overall_rmse_individual', overall_rmse_group']);
xticks(1:(1 + num_muscles + num_groups));
xticklabels({ 'Combined', muscle_names{:}, group_names{:} }); % FIX HERE
xtickangle(45);
ylabel('Overall RMSE (degrees)');
title('Overall RMSE Comparison');
grid on;
set(gca, 'FontSize', 10);

% R^2 Comparison
subplot(1, 2, 2);
% Concatenate horizontally *within* the bar function
bar([overall_r2_combined, overall_r2_individual', overall_r2_group']);
xticks(1:(1 + num_muscles + num_groups));
xticklabels({ 'Combined', muscle_names{:}, group_names{:} }); % FIX HERE
xtickangle(45);
ylabel('Overall R^2');
title('Overall R^2 Comparison');
grid on;
set(gca, 'FontSize', 10);