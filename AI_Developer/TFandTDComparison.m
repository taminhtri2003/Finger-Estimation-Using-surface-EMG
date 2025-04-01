% MATLAB Code for sEMG-based Joint Angle Estimation using Random Forest
clear; clc; close all;

%% --- 1. Load Data ---
fprintf('Loading data...\n');
try
    % Ensure 's1_full.mat' contains 'dsfilt_emg' and 'joint_angles'
    data = load('s4_full.mat'); 
    dsfilt_emg = data.dsfilt_emg;
    joint_angles = data.joint_angles;
    fprintf('Data loaded successfully.\n');
catch ME
    error('Failed to load data file. Make sure "s1_full.mat" is in the MATLAB path. Error: %s', ME.message);
end

%% --- 2. Data Preparation ---
fprintf('Preparing data...\n');
% Constants
num_trials = size(dsfilt_emg, 1); % Should be 5
num_tasks = size(dsfilt_emg, 2); % Should be 7
num_emg_channels = 8;
num_angles = 14;
Fs = 200; % Sampling Frequency (Hz) - CRITICAL for window interpretation & CWT

% Pre-allocate cell arrays for temporary storage (more memory efficient for varying lengths)
emg_list = cell(num_trials * num_tasks, 1);
angle_list = cell(num_trials * num_tasks, 1);
list_idx = 0;

% Concatenate data from all trials and tasks
for i = 1:num_trials
    for j = 1:num_tasks
        if ~isempty(dsfilt_emg{i, j}) && ~isempty(joint_angles{i, j})
            minLength = min(size(dsfilt_emg{i, j}, 1), size(joint_angles{i, j}, 1));
            if minLength > 0
                list_idx = list_idx + 1;
                emg_list{list_idx} = dsfilt_emg{i, j}(1:minLength, :);
                angle_list{list_idx} = joint_angles{i, j}(1:minLength, :);
            else
                 fprintf('Warning: Zero length data found at Trial %d, Task %d after length correction. Skipping.\n', i, j);
            end
        else
             fprintf('Warning: Empty cell found at Trial %d, Task %d. Skipping.\n', i, j);
        end
    end
end

% Trim empty cells if any were skipped
emg_list = emg_list(1:list_idx);
angle_list = angle_list(1:list_idx);

% Perform final concatenation
all_emg = vertcat(emg_list{:});
all_angles = vertcat(angle_list{:});
clear emg_list angle_list; % Free memory

fprintf('Data concatenated: %d samples\n', size(all_emg, 1));

% --- Parameters for Feature Extraction ---
window_duration_sec = 0.5; % Window duration in seconds (e.g., 200ms)
overlap_percentage = 0.5;  % Overlap percentage (e.g., 0.5 for 50%)

window_length = round(window_duration_sec * Fs); % Window size in samples
overlap_length = round(window_length * overlap_percentage); % Overlap in samples
step_size = window_length - overlap_length;

fprintf('Feature window length: %d samples (%.2f s)\n', window_length, window_duration_sec);
fprintf('Feature step size: %d samples\n', step_size);


% ========================================================================
%% --- NEW: Preprocessing Steps (Apply before Feature Extraction) ---
% ========================================================================
fprintf('Applying preprocessing...\n');

apply_smoothing = true; % Set to true to apply smoothing, false to skip
smoothing_method = 'movmean'; % 'movmean' or potentially others like 'gaussian', 'sgolay'
smoothing_window_duration_sec = 0.5; % Smoothing window in seconds (e.g., 50ms)
smoothing_window_length = round(smoothing_window_duration_sec * Fs);

if apply_smoothing
    fprintf('Applying %s smoothing with window size %d samples (%.2f s)...\n', ...
            smoothing_method, smoothing_window_length, smoothing_window_duration_sec);

    % --- Note on Rectification ---
    % If your 'dsfilt_emg' is a bipolar AC signal (positive and negative values),
    % you might want to rectify it BEFORE smoothing to get an envelope.
    % Example: all_emg_processed = abs(all_emg);
    % However, since MAV calculation later uses abs(), and RMS squares values,
    % smoothing the potentially bipolar signal directly might also be valid.
    % CWT typically works best on the original (non-rectified) signal shape.
    % We will proceed by smoothing 'all_emg' directly here.

    all_emg_processed = zeros(size(all_emg)); % Pre-allocate
    for ch = 1:num_emg_channels
        switch smoothing_method
            case 'movmean'
                % Check if Signal Processing Toolbox is available for movmean
                 if license('test', 'Signal_Toolbox')
                    all_emg_processed(:, ch) = movmean(all_emg(:, ch), smoothing_window_length);
                 else
                     warning('Signal Processing Toolbox not found. Skipping movmean smoothing.');
                     all_emg_processed(:, ch) = all_emg(:, ch); % Use original if toolbox missing
                 end
            % Add other methods if needed:
            % case 'sgolay'
            %    order = 3; % Polynomial order
            %    framelen = smoothing_window_length;
            %    if mod(framelen,2)==0; framelen = framelen+1; end % Must be odd
            %    if order >= framelen; order = framelen-1; end % Order must be < framelen
            %    all_emg_processed(:, ch) = sgolayfilt(all_emg(:, ch), order, framelen);
            otherwise
                warning('Unknown smoothing method: %s. Skipping smoothing.', smoothing_method);
                all_emg_processed(:, ch) = all_emg(:, ch);
        end
    end
    fprintf('Smoothing applied.\n');
else
    fprintf('Skipping smoothing step.\n');
    all_emg_processed = all_emg; % Use original data if not smoothing
end

% --- Other Potential Preprocessing ---
% 1. Normalization (e.g., by MVC or max value per channel): Requires MVC data or calculation.
%    Example (max value normalization):
%    max_vals = max(abs(all_emg_processed), [], 1);
%    all_emg_processed = all_emg_processed ./ (max_vals + 1e-9); % Add epsilon for stability

% 2. Detrending (e.g., high-pass filter or polynomial fit): Might be redundant if 'dsfilt_emg' is already well-filtered.
%    Example (polynomial detrend):
%    for ch = 1:num_emg_channels
%        p = polyfit((1:size(all_emg_processed,1))', all_emg_processed(:,ch), 5); % Fit 5th order poly
%        trend = polyval(p, (1:size(all_emg_processed,1))');
%        all_emg_processed(:, ch) = all_emg_processed(:, ch) - trend;
%    end

% Use 'all_emg_processed' for feature extraction from now on
% ========================================================================


%% --- 3. Time-Domain (TD) Feature Extraction ---
fprintf('Extracting Time-Domain features...\n');
num_samples_processed = size(all_emg_processed, 1); % Use size of processed EMG
num_windows = floor((num_samples_processed - window_length) / step_size) + 1;

% Pre-allocate feature matrix
num_td_features = 3; % MAV, RMS, Entropy
td_features = zeros(num_windows, num_emg_channels * num_td_features);
td_target_angles = zeros(num_windows, num_angles);

last_valid_window_idx = 0; % Keep track of actual windows filled

for k = 1:num_windows
    start_idx = (k - 1) * step_size + 1;
    end_idx = start_idx + window_length - 1;

    % Ensure window does not exceed data bounds (important!)
    if end_idx > num_samples_processed
        fprintf('Window %d exceeds data bounds (%d > %d). Stopping feature extraction.\n', k, end_idx, num_samples_processed);
        break; % Stop if the calculated end index goes beyond the available data
    end

    % Extract window from potentially preprocessed EMG
    window_emg = all_emg_processed(start_idx:end_idx, :);

    feature_idx_offset = 0;
    for ch = 1:num_emg_channels
        emg_channel_win = window_emg(:, ch);

        % MAV (Mean Absolute Value) - applied to potentially bipolar or smoothed signal
        mav = mean(abs(emg_channel_win));

        % RMS (Root Mean Square)
        rms_val = rms(emg_channel_win); % rms function handles positive/negative

        % Entropy - check if Wavelet Toolbox is needed for wentropy
         if license('test', 'Wavelet_Toolbox')
            ent = wentropy(emg_channel_win, 'shannon');
         else
             warning('Wavelet Toolbox not found. Setting entropy feature to 0.');
             ent = 0;
         end

        td_features(k, feature_idx_offset + 1) = mav;
        td_features(k, feature_idx_offset + 2) = rms_val;
        td_features(k, feature_idx_offset + 3) = ent;
        feature_idx_offset = feature_idx_offset + num_td_features;
    end

    % Align target angles: Use the angle at the END of the window
    td_target_angles(k, :) = all_angles(end_idx, :);
    last_valid_window_idx = k; % Mark this window as successfully processed
end

% Trim any unused preallocated rows if extraction stopped early
if last_valid_window_idx < num_windows
    fprintf('Adjusting TD feature matrix size to %d windows.\n', last_valid_window_idx);
    td_features = td_features(1:last_valid_window_idx, :);
    td_target_angles = td_target_angles(1:last_valid_window_idx, :);
end

fprintf('TD Features extracted: %d windows, %d features per window\n', size(td_features, 1), size(td_features, 2));


%% --- 4. Time-Frequency (TF) Feature Extraction ---
fprintf('Extracting Time-Frequency features (CWT)...\n');
% CWT parameters
wavelet_name = 'morl'; % Morlet wavelet

% Define frequency bands (adjust based on expected EMG content and Fs)
min_freq = 20;  % Hz
max_freq = (Fs/2) * 0.9; % Go up to 90% of Nyquist frequency
num_freq_bands = 10;
frequencies = linspace(min_freq, max_freq, num_freq_bands);
scales = centfrq(wavelet_name) * Fs ./ frequencies;
scales = sort(scales, 'ascend'); % Ensure scales are ascending

num_tf_features_per_channel = length(scales); % Energy at each scale
tf_features = zeros(num_windows, num_emg_channels * num_tf_features_per_channel); % Use num_windows from TD
tf_target_angles = zeros(num_windows, num_angles);

last_valid_window_idx_tf = 0;

if ~license('test', 'Wavelet_Toolbox')
    warning('Wavelet Toolbox not found. Skipping CWT feature extraction.');
    tf_features = []; % Indicate no TF features extracted
else
    for k = 1:num_windows % Loop up to the number of windows determined in TD part
        start_idx = (k - 1) * step_size + 1;
        end_idx = start_idx + window_length - 1;

        % Bounds check (should be consistent with TD loop)
        if end_idx > num_samples_processed
             fprintf('TF Window %d exceeds data bounds (%d > %d). Stopping.\n', k, end_idx, num_samples_processed);
            break;
        end

        % Use preprocessed EMG for CWT as well
        window_emg = all_emg_processed(start_idx:end_idx, :);

        feature_idx_offset = 0;
        for ch = 1:num_emg_channels
            emg_channel_win = window_emg(:, ch);

            % Compute CWT coefficients
            cfs = cwt(emg_channel_win, scales, wavelet_name);

            % Calculate energy (mean power) at each scale within the window
            cwt_energy = mean(abs(cfs).^2, 2);

            tf_features(k, feature_idx_offset + 1 : feature_idx_offset + num_tf_features_per_channel) = cwt_energy';
            feature_idx_offset = feature_idx_offset + num_tf_features_per_channel;
        end

        tf_target_angles(k, :) = all_angles(end_idx, :); % Targets are the same
        last_valid_window_idx_tf = k;
    end

    % Trim unused rows if necessary (should match TD section)
    if last_valid_window_idx_tf < num_windows
        fprintf('Adjusting TF feature matrix size to %d windows.\n', last_valid_window_idx_tf);
        tf_features = tf_features(1:last_valid_window_idx_tf, :);
        tf_target_angles = tf_target_angles(1:last_valid_window_idx_tf, :);
    end
     fprintf('TF Features extracted: %d windows, %d features per window\n', size(tf_features, 1), size(tf_features, 2));
end % End check for Wavelet Toolbox


% Ensure target angles are consistent (use TD targets as reference)
target_angles = td_target_angles;
num_windows_final = size(target_angles, 1); % Final number of windows used

% Adjust TF features if window counts mismatched (e.g., due to toolbox checks)
if ~isempty(tf_features) && size(tf_features, 1) ~= num_windows_final
    warning('Mismatch in final window count for TF features (%d) vs TD (%d). Using %d windows.', size(tf_features, 1), num_windows_final, num_windows_final);
    tf_features = tf_features(1:num_windows_final, :);
end


%% --- 5. Data Splitting (Train/Test) ---
fprintf('Splitting data into training and testing sets (%d total windows)...\n', num_windows_final);
if num_windows_final == 0
    error('No feature windows were successfully generated. Cannot proceed.');
end

cv = cvpartition(num_windows_final, 'HoldOut', 0.2); % 80% train, 20% test
idxTrain = training(cv);
idxTest = test(cv);

% --- TD Data Split ---
XTrain_td = td_features(idxTrain, :);
YTrain_td = target_angles(idxTrain, :);
XTest_td = td_features(idxTest, :);
YTest_td = target_angles(idxTest, :);
fprintf('TD Split: %d Train samples, %d Test samples\n', size(XTrain_td,1), size(XTest_td,1));

% --- TF Data Split ---
% Only split if TF features were extracted
if ~isempty(tf_features)
    XTrain_tf = tf_features(idxTrain, :);
    YTrain_tf = target_angles(idxTrain, :); % Targets are the same
    XTest_tf = tf_features(idxTest, :);
    YTest_tf = target_angles(idxTest, :); % Targets are the same
    fprintf('TF Split: %d Train samples, %d Test samples\n', size(XTrain_tf,1), size(XTest_tf,1));
else
    XTrain_tf = []; XTest_tf = []; YTrain_tf = []; YTest_tf = []; % Ensure they exist but are empty
    fprintf('TF features were not extracted. Skipping TF model training/evaluation.\n');
end


%% --- 6. Feature Standardization ---
fprintf('Standardizing features...\n');

% --- TD Standardization ---
mu_td = mean(XTrain_td, 1);
sigma_td = std(XTrain_td, 0, 1);
sigma_td(sigma_td < 1e-9) = 1; % Avoid division by zero for constant features
XTrain_td_scaled = (XTrain_td - mu_td) ./ sigma_td;
XTest_td_scaled = (XTest_td - mu_td) ./ sigma_td;
% Handle potential NaN/Inf if scaling produced them (though unlikely with epsilon)
XTest_td_scaled(~isfinite(XTest_td_scaled)) = 0;

% --- TF Standardization ---
if ~isempty(XTrain_tf)
    mu_tf = mean(XTrain_tf, 1);
    sigma_tf = std(XTrain_tf, 0, 1);
    sigma_tf(sigma_tf < 1e-9) = 1;
    XTrain_tf_scaled = (XTrain_tf - mu_tf) ./ sigma_tf;
    XTest_tf_scaled = (XTest_tf - mu_tf) ./ sigma_tf;
    XTest_tf_scaled(~isfinite(XTest_tf_scaled)) = 0;
else
     XTrain_tf_scaled = []; XTest_tf_scaled = []; % Keep empty if no TF data
end

fprintf('Feature standardization complete.\n');


%% --- 7. Random Forest Model Training ---
fprintf('Training Random Forest models...\n');

% Check if Statistics and Machine Learning Toolbox is available
if ~license('test', 'Statistics_Toolbox')
    error('Statistics and Machine Learning Toolbox is required for fitrensemble.');
end

% Random Forest Parameters
nTrees = 100; % Reduced for potentially faster run, adjust as needed (e.g., 100-200)
UseParallel = true; % Set to true to use parallel processing if available

if UseParallel && license('test', 'Parallel_Computing_Toolbox') && isempty(gcp('nocreate'))
     try
        parpool; % Start parallel pool if not already running
     catch ME_parpool
        warning('Could not start parallel pool. Training sequentially. Error: %s', E_parpool.message);
        UseParallel = false; % Fallback to sequential
     end
 elseif ~license('test', 'Parallel_Computing_Toolbox')
     UseParallel = false; % Ensure UseParallel is false if toolbox missing
end
opts = statset('UseParallel', UseParallel);

% --- Train TD Model ---
fprintf('Training TD model...\n');
rf_models_td = cell(num_angles, 1);
for i = 1:num_angles
    fprintf('  Training model for angle %d/%d...\n', i, num_angles);
    rf_models_td{i} = fitrensemble(XTrain_td_scaled, YTrain_td(:, i), ...
                                  'Method', 'Bag', ...
                                  'NumLearningCycles', nTrees, ...
                                  'Learners', 'Tree', ... % Regression trees
                                  'Options', opts); % Apply parallel options
end
fprintf('TD Model training complete.\n');

% --- Train TF Model ---
if ~isempty(XTrain_tf_scaled)
    fprintf('Training TF model...\n');
    rf_models_tf = cell(num_angles, 1);
    for i = 1:num_angles
         fprintf('  Training model for angle %d/%d...\n', i, num_angles);
        rf_models_tf{i} = fitrensemble(XTrain_tf_scaled, YTrain_tf(:, i), ...
                                      'Method', 'Bag', ...
                                      'NumLearningCycles', nTrees, ...
                                      'Learners', 'Tree', ...
                                      'Options', opts);
    end
    fprintf('TF Model training complete.\n');
else
     rf_models_tf = {}; % Keep empty if no TF model trained
end


%% --- 8. Prediction on Test Set ---
fprintf('Predicting angles on the test set...\n');

YPred_td = zeros(size(YTest_td));
for i = 1:num_angles
    if ~isempty(rf_models_td{i}) % Check model exists
        YPred_td(:, i) = predict(rf_models_td{i}, XTest_td_scaled);
    end
end

if ~isempty(rf_models_tf) && ~isempty(XTest_tf_scaled)
    YPred_tf = zeros(size(YTest_tf));
    for i = 1:num_angles
         if ~isempty(rf_models_tf{i}) % Check model exists
            YPred_tf(:, i) = predict(rf_models_tf{i}, XTest_tf_scaled);
         end
    end
else
    YPred_tf = []; % No TF predictions
end
fprintf('Prediction complete.\n');


%% --- 9. Evaluation ---
fprintf('Evaluating model performance...\n');

rmse_td = nan(1, num_angles); % Use NaN for missing values
r2_td = nan(1, num_angles);
rmse_tf = nan(1, num_angles);
r2_tf = nan(1, num_angles);

for i = 1:num_angles
    % TD Model
    rmse_td(i) = sqrt(mean((YTest_td(:, i) - YPred_td(:, i)).^2));
    r2_td(i) = 1 - sum((YTest_td(:, i) - YPred_td(:, i)).^2) / sum((YTest_td(:, i) - mean(YTest_td(:, i))).^2);

    % TF Model (only if predictions exist)
    if ~isempty(YPred_tf)
        rmse_tf(i) = sqrt(mean((YTest_tf(:, i) - YPred_tf(:, i)).^2));
        r2_tf(i) = 1 - sum((YTest_tf(:, i) - YPred_tf(:, i)).^2) / sum((YTest_tf(:, i) - mean(YTest_tf(:, i))).^2);
    end
end

% Display overall average performance (ignoring NaNs)
fprintf('\n--- Performance Summary ---\n');
fprintf('Time-Domain Features Model:\n');
fprintf('  Average RMSE: %.4f degrees\n', mean(rmse_td, 'omitnan'));
fprintf('  Average R^2:  %.4f\n', mean(r2_td, 'omitnan'));

if ~isempty(YPred_tf)
    fprintf('Time-Frequency Features Model:\n');
    fprintf('  Average RMSE: %.4f degrees\n', mean(rmse_tf, 'omitnan'));
    fprintf('  Average R^2:  %.4f\n', mean(r2_tf, 'omitnan'));
else
     fprintf('Time-Frequency Features Model: Not evaluated (no features/predictions).\n');
end
fprintf('---------------------------\n');


%% --- 10. Visualization ---
fprintf('Generating visualizations...\n');
angle_names = {'Thumb 1', 'Thumb 2', 'Index 1', 'Index 2', 'Index 3', ...
               'Middle 1', 'Middle 2', 'Middle 3', 'Ring 1', 'Ring 2', 'Ring 3', ...
               'Little 1', 'Little 2', 'Little 3'};
angles_to_plot = [1, 4, 7, 11]; % Example angles
num_plot_samples = min(500, size(YTest_td, 1));

% Check if there are test samples to plot
if num_plot_samples == 0
    warning('No test samples available to plot.');
else
    % --- TD Plot ---
    figure('Name', 'Predicted vs Actual Angles (TD Model)', 'Position', [100, 100, 1200, 600]);
    for i = 1:length(angles_to_plot)
        idx = angles_to_plot(i);
        subplot(2, 2, i);
        plot(1:num_plot_samples, YTest_td(1:num_plot_samples, idx), 'b-', 'LineWidth', 1);
        hold on;
        plot(1:num_plot_samples, YPred_td(1:num_plot_samples, idx), 'r--', 'LineWidth', 1);
        hold off;
        title(sprintf('%s (TD) | RMSE: %.2f, R^2: %.2f', angle_names{idx}, rmse_td(idx), r2_td(idx)));
        xlabel('Sample Index (Test Set)'); ylabel('Angle (degrees)'); legend('Actual', 'Predicted','Location','best'); grid on;
    end
    sgtitle('Time-Domain Feature Model: Predicted vs Actual Angles');

    % --- TF Plot ---
    if ~isempty(YPred_tf)
        figure('Name', 'Predicted vs Actual Angles (TF Model)', 'Position', [150, 150, 1200, 600]);
        for i = 1:length(angles_to_plot)
            idx = angles_to_plot(i);
            subplot(2, 2, i);
            plot(1:num_plot_samples, YTest_tf(1:num_plot_samples, idx), 'b-', 'LineWidth', 1);
            hold on;
            plot(1:num_plot_samples, YPred_tf(1:num_plot_samples, idx), 'g--', 'LineWidth', 1);
            hold off;
            title(sprintf('%s (TF) | RMSE: %.2f, R^2: %.2f', angle_names{idx}, rmse_tf(idx), r2_tf(idx)));
            xlabel('Sample Index (Test Set)'); ylabel('Angle (degrees)'); legend('Actual', 'Predicted','Location','best'); grid on;
        end
        sgtitle('Time-Frequency Feature Model: Predicted vs Actual Angles');
    end

    % --- Performance Comparison Plot ---
    figure('Name', 'Performance Comparison per Angle', 'Position', [200, 200, 1000, 700]);
    plot_tf_comparison = ~isempty(YPred_tf); % Flag to decide if TF bars should be plotted

    % RMSE Comparison
    subplot(2, 1, 1);
    if plot_tf_comparison
        bar_data_rmse = [rmse_td', rmse_tf'];
        b_rmse = bar(bar_data_rmse);
        legend('TD Features', 'TF Features','Location','best');
    else
        bar_data_rmse = rmse_td';
        b_rmse = bar(bar_data_rmse);
        legend('TD Features','Location','best');
    end
    set(gca, 'XTickLabel', angle_names, 'XTick', 1:num_angles); xtickangle(45);
    ylabel('RMSE (degrees)'); title('RMSE Comparison'); grid on;
    % Add text labels for RMSE
    for i = 1:length(b_rmse)
        xtips = b_rmse(i).XEndPoints; ytips = b_rmse(i).YEndPoints;
        labels = string(arrayfun(@(x) sprintf('%.2f', x), ytips, 'UniformOutput', false));
        text(xtips, ytips, labels, 'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize', 8);
    end

    % R-squared Comparison
    subplot(2, 1, 2);
     if plot_tf_comparison
        bar_data_r2 = [r2_td', r2_tf'];
        b_r2 = bar(bar_data_r2);
        legend('TD Features', 'TF Features', 'Location', 'best');
     else
        bar_data_r2 = r2_td';
        b_r2 = bar(bar_data_r2);
        legend('TD Features', 'Location', 'best');
     end
    set(gca, 'XTickLabel', angle_names, 'XTick', 1:num_angles); xtickangle(45);
    ylabel('R-squared'); title('R-squared Comparison'); grid on;
    ylim([min(-0.1, min(bar_data_r2(:))-0.1) 1]); % Ensure y-axis includes 0 and 1, allows negatives
     % Add text labels for R2
    for i = 1:length(b_r2)
        xtips = b_r2(i).XEndPoints; ytips = b_r2(i).YEndPoints;
        labels = string(arrayfun(@(x) sprintf('%.2f', x), ytips, 'UniformOutput', false));
        text(xtips, ytips, labels, 'HorizontalAlignment','center', 'VerticalAlignment','bottom', 'FontSize', 8);
    end
    sgtitle('Model Performance Comparison per Joint Angle');
end % End check if samples available to plot

fprintf('Visualizations generated. End of script.\n');