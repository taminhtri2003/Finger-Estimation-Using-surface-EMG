% Load the .mat file containing the data
try
    load('s1_full.mat'); % Replace with your actual file name
catch e
    error('Failed to load data file: %s', e.message);
end

% --- Feature Extraction Function ---
function features = extractFeatures(emgData, windowSize, windowOverlap)
    % Pre-allocate features matrix.
    numFeatures = 4; % MAV, RMS, WL, ZC
    numWindows = floor((size(emgData, 1) - windowOverlap) / (windowSize - windowOverlap));
    features = zeros(numWindows, numFeatures * size(emgData, 2));

    for chan = 1:size(emgData, 2) % Iterate over each EMG channel
        feature_idx_offset = (chan - 1) * numFeatures;
        for i = 1:numWindows
            startIdx = (i - 1) * (windowSize - windowOverlap) + 1;
            endIdx = startIdx + windowSize - 1;
            window = emgData(startIdx:endIdx, chan);

            % 1. Mean Absolute Value (MAV)
            features(i, feature_idx_offset + 1) = mean(abs(window));

            % 2. Root Mean Square (RMS)
            features(i, feature_idx_offset + 2) = sqrt(mean(window.^2));

            % 3. Waveform Length (WL)
            features(i, feature_idx_offset + 3) = sum(abs(diff(window)));

            % 4. Zero Crossings (ZC)
            zeroCrossings = sum((window(1:end-1) .* window(2:end)) < 0);
            features(i, feature_idx_offset + 4) = zeroCrossings;
        end
    end
end

% --- Data Preprocessing and Feature Extraction ---
windowSize = 200;  % Example: 200ms window
windowOverlap = 100; % Example: 100ms overlap (50%)

%Pre-allocate
total_windows_emg = 0;

for trial = 1:5
    for task = 1:6
        total_windows_emg = total_windows_emg + floor((size(dsfilt_emg{trial, task}, 1) - windowOverlap) / (windowSize - windowOverlap));
    end
end

num_emg_channels = size(dsfilt_emg{1,1}, 2);
num_features_per_channel = 4; % MAV, RMS, WL, ZC
total_features = num_emg_channels * num_features_per_channel;

combined_emg_features = zeros(total_windows_emg, total_features);
combined_joint_angles_resampled = zeros(total_windows_emg, size(joint_angles{1,1},2));

current_idx = 1;

for trial = 1:5
    for task = 1:6
        % Feature extraction for EMG
        emg_features = extractFeatures(dsfilt_emg{trial, task}, windowSize, windowOverlap);
        num_windows = size(emg_features, 1);
        combined_emg_features(current_idx:(current_idx + num_windows - 1), :) = emg_features;


        % Resample joint angles to match the number of feature windows.
        %Use linear interpolation
        original_time = 1:size(joint_angles{trial, task}, 1);
        resampled_time = linspace(1, size(joint_angles{trial, task}, 1), num_windows);
        resampled_angles = interp1(original_time, joint_angles{trial, task}, resampled_time, 'linear');

        combined_joint_angles_resampled(current_idx:(current_idx + num_windows - 1), :) = resampled_angles;


        current_idx = current_idx + num_windows;
    end
end


% --- GUI Setup ---

% Create GUI figure
fig = figure('Name', 'EMG-Angle Regression Trainer', ...
    'NumberTitle', 'off', ...
    'Position', [100, 100, 1200, 800], ...
    'MenuBar', 'none', ...
    'Resize', 'on');

% Store *FEATURE DATA* in UserData
fig.UserData.combined_emg_features = combined_emg_features;
fig.UserData.combined_joint_angles = combined_joint_angles_resampled; % Use resampled angles
fig.UserData.dsfilt_emg = dsfilt_emg;  % Still needed for task highlighting
fig.UserData.windowSize = windowSize; % Store for task highlighting
fig.UserData.windowOverlap = windowOverlap; %Store for task highlighting.

% --- GUI Components ---
% Use a panel to organize input controls
inputPanel = uipanel('Parent', fig, 'Title', 'Input Selection', ...
    'Position', [0.02 0.75 0.46 0.2]); % Positioned at the top-left

% EMG Channel Selection
uicontrol(inputPanel, 'Style', 'text', 'String', 'Select EMG Channel:', ...
    'Position', [10, 110, 150, 20], 'HorizontalAlignment', 'left');
emg_popup = uicontrol(inputPanel, 'Style', 'popupmenu', ...
    'String', {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'}, ...
    'Position', [160, 110, 150, 25], ...
    'Tag', 'emg_popup', 'BackgroundColor', 'w');

% Joint Angle Selection
uicontrol(inputPanel, 'Style', 'text', 'String', 'Select Target Angle:', ...
    'Position', [10, 60, 150, 20], 'HorizontalAlignment', 'left');
joint_popup = uicontrol(inputPanel, 'Style', 'popupmenu', ...
    'String', {'Thumb 1', 'Thumb 2', 'Index 1', 'Index 2', 'Index 3', ...
    'Middle 1', 'Middle 2', 'Middle 3', 'Ring 1', 'Ring 2', ...
    'Ring 3', 'Little 1', 'Little 2', 'Little 3'}, ...
    'Position', [160, 60, 150, 25], ...
    'Tag', 'joint_popup', 'BackgroundColor', 'w');

% Train Model Button
uicontrol(inputPanel, 'Style', 'pushbutton', 'String', 'Train Model', ...
    'Position', [10, 10, 150, 30], ...
    'Callback', @train_model, ...
    'FontWeight', 'bold'); % Make button text bold

% --- Result Display Panel ---
resultPanel = uipanel('Parent', fig, 'Title', 'Model Performance', ...
                     'Position', [0.5 0.75 0.48 0.2]);

% Performance Metrics (using text boxes within the panel)
uicontrol(resultPanel, 'Style', 'text', 'String', 'Adjusted R²:', ...
          'Position', [10, 60, 100, 20],'HorizontalAlignment', 'left');
r2_text = uicontrol(resultPanel, 'Style', 'text', ...
          'Position', [110, 60, 150, 25], ...
          'Tag', 'r2_text', 'HorizontalAlignment', 'left');

uicontrol(resultPanel, 'Style', 'text', 'String', 'RMSE (deg):', ...
          'Position', [10, 20, 100, 20],'HorizontalAlignment', 'left');
rmse_text = uicontrol(resultPanel, 'Style', 'text', ...
          'Position', [110, 20, 150, 25], ...
          'Tag', 'rmse_text', 'HorizontalAlignment', 'left');

% --- Axes for Visualization ---
% *CRITICAL FIX*: Store axes handles in UserData *during creation*
subplot1 = axes('Parent', fig, 'Position', [0.05, 0.1, 0.4, 0.6]);
subplot2 = axes('Parent', fig, 'Position', [0.55, 0.1, 0.4, 0.6]);
fig.UserData.subplot1 = subplot1; % Store the axes handles
fig.UserData.subplot2 = subplot2;


% --- Modified Callback Function ---
function train_model(~, ~)
    fig = ancestor(gcbo, 'figure');

    % Retrieve FEATURE DATA from UserData
    combined_emg_features = fig.UserData.combined_emg_features;
    combined_angles = fig.UserData.combined_joint_angles;
    dsfilt_emg = fig.UserData.dsfilt_emg;
    windowSize = fig.UserData.windowSize;
    windowOverlap = fig.UserData.windowOverlap;

    % *CRITICAL FIX*: Retrieve axes handles from UserData
    subplot1 = fig.UserData.subplot1;
    subplot2 = fig.UserData.subplot2;


    emg_popup = findobj(fig, 'Tag', 'emg_popup');
    emg_idx = get(emg_popup, 'Value');
    emg_strings = get(emg_popup, 'String');
    emg_label = emg_strings{emg_idx};

    joint_popup = findobj(fig, 'Tag', 'joint_popup');
    angle_idx = get(joint_popup, 'Value');
    angle_strings = get(joint_popup, 'String');
    angle_label = angle_strings{angle_idx};

    r2_text = findobj(fig, 'Tag', 'r2_text');
    rmse_text = findobj(fig, 'Tag', 'rmse_text');

    % --- Feature Selection (Based on EMG Channel) ---
    start_feature = (emg_idx - 1) * 4 + 1;
    end_feature = emg_idx * 4;
    X = combined_emg_features(:, start_feature:end_feature);
    y = combined_angles(:, angle_idx);

    % Split data
    cv = cvpartition(size(X, 1), 'HoldOut', 0.3);
    X_train = X(training(cv), :);
    y_train = y(training(cv), :);
    X_test = X(test(cv), :);
    y_test = y(test(cv), :);

    % Train model
    model = fitlm(X_train, y_train);

    % Predict and evaluate
    y_pred = predict(model, X_test);
    r2 = model.Rsquared.Adjusted;
    rmse = sqrt(mean((y_test - y_pred).^2));

    set(r2_text, 'String', sprintf('%.3f', r2));
    set(rmse_text, 'String', sprintf('%.3f', rmse));

    % --- Plotting ---
    % Use the axes handles retrieved from UserData
    axes(subplot1); % Switch to subplot1
    cla;
    feature_names = {'MAV', 'RMS', 'WL', 'ZC'};
    colors = lines(4);
    hold on;
    for i = 1:4
        plot(X(:, i), 'Color', colors(i,:), 'DisplayName', feature_names{i});
    end
    hold off;
    title(sprintf('EMG Features - %s', emg_label), 'Interpreter', 'none');
    xlabel('Window Index');
    ylabel('Feature Value');
    legend('Location', 'best');
    grid on;

    axes(subplot2); % Switch to subplot2
    cla;
    hold on;
    plot(y, 'k', 'DisplayName', 'Actual');
    test_indices = 1:length(y_test);  % x-axis is now window index
    plot(test_indices, y_pred, 'r--', 'DisplayName', 'Predicted');
    title(sprintf('%s Prediction', angle_label), 'Interpreter', 'none');
    xlabel('Window Index');
    ylabel('Angle (deg)');
    legend('Location', 'best');
    grid on;

    % --- Task Highlighting ---
    axes(subplot2); % Ensure we're drawing on subplot2
    cla; % Clear previous patches
    hold on;
    plot(y, 'k', 'DisplayName', 'Actual');
    test_indices = find(test(cv));
    plot(test_indices, y_pred, 'r--', 'DisplayName', 'Predicted');
    title(sprintf('%s Prediction', angle_label), 'Interpreter', 'none');
    xlabel('Window Index');
    ylabel('Angle (deg)');
    legend('Location', 'best');
    grid on;

    samples_per_task = size(dsfilt_emg{1,1}, 1);
    windows_per_task = floor((samples_per_task - windowOverlap) / (windowSize - windowOverlap));
    task_colors = lines(6);

    current_window = 1;
    for trial = 1:5
        for task = 1:6
            start_idx = current_window;
            end_idx = current_window + windows_per_task - 1;
            end_idx = min(end_idx, length(y)); % Ensure within bounds

            if end_idx > start_idx
                patch([start_idx, end_idx, end_idx, start_idx], ...
                      [min(y), min(y), max(y), max(y)], ...
                      task_colors(task,:), 'FaceAlpha', 0.2, ...
                      'EdgeColor', 'none', 'HandleVisibility', 'off');
            end
            current_window = current_window + windows_per_task;
        end
    end
    hold off;
end