% Multi-Head Model for Finger Angle Estimation from EMG
% This script implements a multi-head neural network approach to predict
% finger joint angles from EMG signals

%% 1. Load and prepare the data
clear; clc; close all;

% Load the data file
disp('Loading data...');
load('s2_full.mat');  % Replace with your actual .mat file name

% Extract dimensions
[num_trials, num_tasks] = size(dsfilt_emg);

% Configure preprocessing parameters
window_size = 100;  % 100 samples window for feature extraction
overlap = 50;       % 50% overlap between windows
emg_fs = 1000;      % Assumed EMG sampling frequency (Hz)
kinematics_fs = 100; % Assumed kinematics sampling frequency (Hz)

% Extract muscle names for reference
muscle_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};
joint_angle_names = {'Thumb 1', 'Thumb 2', 'Index 1', 'Index 2', 'Index 3', ...
                     'Middle 1', 'Middle 2', 'Middle 3', 'Ring 1', 'Ring 2', ...
                     'Ring 3', 'Little 1', 'Little 2', 'Little 3'};

%% 2. Preprocess and extract features
disp('Preprocessing and extracting features...');

% Initialize containers for features and targets
X_train = [];  % EMG features
Y_train = [];  % Joint angles (targets)

for trial = 1:num_trials
    for task = 1:num_tasks
        % Get EMG and joint angle data for this trial and task
        emg_data = dsfilt_emg{trial, task};
        angle_data = joint_angles{trial, task};
        
        % Resample joint angle data to match EMG if sampling rates differ
        if size(emg_data, 1) ~= size(angle_data, 1)
            % Calculate resampling factor
            resample_factor = size(emg_data, 1) / size(angle_data, 1);
            
            % Create time vectors for original and target sampling
            t_orig = linspace(0, 1, size(angle_data, 1));
            t_target = linspace(0, 1, size(emg_data, 1));
            
            % Resample each joint angle
            resampled_angles = zeros(size(emg_data, 1), size(angle_data, 2));
            for j = 1:size(angle_data, 2)
                resampled_angles(:, j) = interp1(t_orig, angle_data(:, j), t_target, 'pchip');
            end
            angle_data = resampled_angles;
        end
        
        % Extract features from each window
        for i = 1:window_size-overlap:size(emg_data, 1)-window_size
            % Extract EMG window
            window_data = emg_data(i:i+window_size-1, :);
            
            % Extract corresponding angle (use the angle at the end of the window)
            target_angle = angle_data(i+window_size-1, :);
            
            % Extract time-domain features from EMG window
            features = extractEMGFeatures(window_data);
            
            % Store features and targets
            X_train = [X_train; features];
            Y_train = [Y_train; target_angle];
        end
    end
end

%% 3. Split data into training and testing sets
disp('Splitting data into training and testing sets...');

% Use 80% for training, 20% for testing
train_ratio = 0.8;
n_samples = size(X_train, 1);
idx = randperm(n_samples);
train_idx = idx(1:round(train_ratio * n_samples));
test_idx = idx(round(train_ratio * n_samples)+1:end);

X_train_final = X_train(train_idx, :);
Y_train_final = Y_train(train_idx, :);
X_test = X_train(test_idx, :);
Y_test = Y_train(test_idx, :);

%% 4. Design and train the multi-head neural network
disp('Training multi-head neural network model...');

% Determine the number of input features and output angles
num_features = size(X_train_final, 2);
num_outputs = size(Y_train_final, 2);

% Create separate model for each joint angle (multi-head approach)
models = cell(1, num_outputs);
predictions = zeros(size(X_test, 1), num_outputs);
train_performance = zeros(1, num_outputs);
test_performance = zeros(1, num_outputs);

for i = 1:num_outputs
    % Create a network
    layers = [
        featureInputLayer(num_features, 'Name', 'input')
        fullyConnectedLayer(128, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        dropoutLayer(0.2, 'Name', 'drop1')
        fullyConnectedLayer(64, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(32, 'Name', 'fc3')
        reluLayer('Name', 'relu3')
        fullyConnectedLayer(1, 'Name', 'output')
        regressionLayer('Name', 'regoutput')
    ];
    
    % Options for training - FIX: Use specific validation data for each model
    options = trainingOptions('adam', ...
        'MaxEpochs', 100, ...
        'MiniBatchSize', 64, ...
        'InitialLearnRate', 0.001, ...
        'GradientThreshold', 1, ...
        'Plots', 'training-progress', ...
        'Verbose', true, ...
        'ValidationData', {X_test, Y_test(:, i)}, ... % Only use the target column for this model
        'ValidationFrequency', 30, ...
        'ValidationPatience', 5);
    
    % Train the network for this joint angle
    fprintf('Training model for %s...\n', joint_angle_names{i});
    models{i} = trainNetwork(X_train_final, Y_train_final(:, i), layers, options);
    
    % Evaluate on training data
    train_pred = predict(models{i}, X_train_final);
    train_performance(i) = sqrt(mean((train_pred - Y_train_final(:, i)).^2));
    
    % Evaluate on test data
    predictions(:, i) = predict(models{i}, X_test);
    test_performance(i) = sqrt(mean((predictions(:, i) - Y_test(:, i)).^2));
    
    fprintf('RMSE for %s: Training = %.4f, Testing = %.4f\n', ...
        joint_angle_names{i}, train_performance(i), test_performance(i));
end

%% 5. Visualize the results
disp('Visualizing results...');

% Plot the RMSE for each joint
figure;
bar([train_performance; test_performance]');
legend('Training', 'Testing');
xlabel('Joint Angle');
xticks(1:num_outputs);
xticklabels(joint_angle_names);
xtickangle(45);
ylabel('RMSE (degrees)');
title('Model Performance by Joint Angle');
grid on;

% Plot predicted vs actual for each joint (first 200 samples)
samples_to_plot = min(200, size(predictions, 1));
time_vector = 1:samples_to_plot;

figure;
for i = 1:num_outputs
    subplot(ceil(num_outputs/2), 2, i);
    plot(time_vector, Y_test(1:samples_to_plot, i), 'b-', 'LineWidth', 1.5);
    hold on;
    plot(time_vector, predictions(1:samples_to_plot, i), 'r--', 'LineWidth', 1.5);
    title(joint_angle_names{i});
    if i == 1 || i == num_outputs-1
        ylabel('Angle (degrees)');
    end
    if i > num_outputs-2
        xlabel('Sample');
    end
    legend('Actual', 'Predicted', 'Location', 'best');
    grid on;
end
sgtitle('Actual vs. Predicted Joint Angles');

% Plot the correlation between predicted and actual values
figure;
for i = 1:num_outputs
    subplot(ceil(num_outputs/2), 2, i);
    scatter(Y_test(:, i), predictions(:, i), 10, 'filled', 'MarkerFaceAlpha', 0.3);
    hold on;
    
    % Add reference line
    min_val = min(min(Y_test(:, i)), min(predictions(:, i)));
    max_val = max(max(Y_test(:, i)), max(predictions(:, i)));
    plot([min_val, max_val], [min_val, max_val], 'k--');
    
    % Calculate correlation
    corr_val = corr(Y_test(:, i), predictions(:, i));
    title(sprintf('%s (r = %.2f)', joint_angle_names{i}, corr_val));
    xlabel('Actual Angle (degrees)');
    ylabel('Predicted Angle (degrees)');
    grid on;
end
sgtitle('Correlation between Actual and Predicted Joint Angles');

% Create a 3D visualization of hand movement
visualizeHandMovement(Y_test(1:200,:), predictions(1:200,:), joint_angle_names);

%% 6. Save the models
disp('Saving models...');
save('emg_joint_angle_models.mat', 'models', 'joint_angle_names', 'muscle_names', 'train_performance', 'test_performance');

disp('Analysis complete.');

%% Helper functions
function features = extractEMGFeatures(window_data)
    % Extract time-domain features from EMG window
    
    % Initialize feature vector
    num_channels = size(window_data, 2);
    features = zeros(1, num_channels * 6);
    
    feature_idx = 1;
    for ch = 1:num_channels
        ch_data = window_data(:, ch);
        
        % 1. Mean absolute value (MAV)
        mav = mean(abs(ch_data));
        
        % 2. Root mean square (RMS)
        rms_val = sqrt(mean(ch_data.^2));
        
        % 3. Waveform length (WL)
        wl = sum(abs(diff(ch_data)));
        
        % 4. Zero crossing rate (ZC)
        zc = sum(diff(sign(ch_data)) ~= 0) / (2 * length(ch_data));
        
        % 5. Slope sign changes (SSC)
        diff_ch = diff(ch_data);
        ssc = sum(diff(sign(diff_ch)) ~= 0) / (2 * length(diff_ch));
        
        % 6. Variance (VAR)
        var_val = var(ch_data);
        
        % Store features
        features(feature_idx:feature_idx+5) = [mav, rms_val, wl, zc, ssc, var_val];
        feature_idx = feature_idx + 6;
    end
end

function visualizeHandMovement(actual_angles, predicted_angles, joint_names)
    % Creates a 3D visualization of hand movement based on joint angles
    
    % Define finger segments lengths (in arbitrary units)
    thumb_segments = [4, 3.5];
    index_segments = [4, 3, 2.5];
    middle_segments = [4.2, 3.2, 2.8];
    ring_segments = [4, 3, 2.5];
    little_segments = [3.5, 2.8, 2.2];
    
    % Create figure
    figure('Position', [100, 100, 1200, 500]);
    
    % Choose a sample to visualize (frame)
    frames_to_visualize = [1, 50, 100, 150];
    
    for frame_idx = 1:length(frames_to_visualize)
        frame = frames_to_visualize(frame_idx);
        
        % Set up subplots for actual and predicted
        subplot(2, length(frames_to_visualize), frame_idx);
        visualizeHand(actual_angles(frame, :), joint_names, thumb_segments, index_segments, middle_segments, ring_segments, little_segments);
        title(sprintf('Actual Hand - Frame %d', frame));
        view(120, 20);
        
        subplot(2, length(frames_to_visualize), length(frames_to_visualize) + frame_idx);
        visualizeHand(predicted_angles(frame, :), joint_names, thumb_segments, index_segments, middle_segments, ring_segments, little_segments);
        title(sprintf('Predicted Hand - Frame %d', frame));
        view(120, 20);
    end
    
    % Create animation
    figure('Position', [100, 100, 1000, 500]);
    
    subplot(1, 2, 1);
    actual_hand = visualizeHand(actual_angles(1, :), joint_names, thumb_segments, index_segments, middle_segments, ring_segments, little_segments);
    title('Actual Hand Movement');
    view(120, 20);
    axis equal;
    set(gca, 'XLim', [-10, 10], 'YLim', [-10, 10], 'ZLim', [-2, 12]);
    
    subplot(1, 2, 2);
    predicted_hand = visualizeHand(predicted_angles(1, :), joint_names, thumb_segments, index_segments, middle_segments, ring_segments, little_segments);
    title('Predicted Hand Movement');
    view(120, 20);
    axis equal;
    set(gca, 'XLim', [-10, 10], 'YLim', [-10, 10], 'ZLim', [-2, 12]);
    
    % Create animation
    frames_to_animate = min(size(actual_angles, 1), 100);
    
    for frame = 1:frames_to_animate
        % Update both visualizations
        updateHandVisualization(subplot(1, 2, 1), actual_angles(frame, :), joint_names, thumb_segments, index_segments, middle_segments, ring_segments, little_segments);
        updateHandVisualization(subplot(1, 2, 2), predicted_angles(frame, :), joint_names, thumb_segments, index_segments, middle_segments, ring_segments, little_segments);
        
        sgtitle(sprintf('Frame %d / %d', frame, frames_to_animate));
        drawnow;
        pause(0.05);
    end
end

function hand_data = visualizeHand(angles, joint_names, thumb_segments, index_segments, middle_segments, ring_segments, little_segments)
    % Visualize a 3D hand model based on joint angles
    
    % Initialize wrist position (origin)
    wrist = [0, 0, 0];
    
    % Extract angles for each joint
    thumb_angles = [angles(1), angles(2)];
    index_angles = [angles(3), angles(4), angles(5)];
    middle_angles = [angles(6), angles(7), angles(8)];
    ring_angles = [angles(9), angles(10), angles(11)];
    little_angles = [angles(12), angles(13), angles(14)];
    
    % Calculate finger positions based on angles
    % For simplicity, we'll place fingers in a fan-like arrangement
    
    % Base finger positions (metacarpals)
    thumb_base = wrist + [-2, -2, 0];
    index_base = wrist + [-1, 0, 0];
    middle_base = wrist + [0, 0, 0];
    ring_base = wrist + [1, 0, 0];
    little_base = wrist + [2, 0, 0];
    
    % Calculate finger joint positions
    % Thumb
    [thumb_positions, thumb_vectors] = calculateFingerPositions(thumb_base, thumb_angles, thumb_segments, [-0.5, 0.5, 0.7]);
    
    % Index
    [index_positions, index_vectors] = calculateFingerPositions(index_base, index_angles, index_segments, [0, 0, 1]);
    
    % Middle
    [middle_positions, middle_vectors] = calculateFingerPositions(middle_base, middle_angles, middle_segments, [0, 0, 1]);
    
    % Ring
    [ring_positions, ring_vectors] = calculateFingerPositions(ring_base, ring_angles, ring_segments, [0, 0, 1]);
    
    % Little
    [little_positions, little_vectors] = calculateFingerPositions(little_base, little_angles, little_segments, [0, 0, 1]);
    
    % Combine all positions for plotting
    all_positions = {
        [wrist; thumb_positions], 
        [wrist; index_positions], 
        [wrist; middle_positions], 
        [wrist; ring_positions], 
        [wrist; little_positions]
    };
    
    % Plot the hand
    hold on;
    
    % Plot palm (simple rectangle)
    palm_corners = [
        -2, -2, 0;   % Thumb corner
        -1, 0, 0;    % Index base
        2, 0, 0;     % Little base
        2, -3, 0;    % Bottom right
        -2, -3, 0    % Bottom left
    ];
    
    fill3(palm_corners(:,1), palm_corners(:,2), palm_corners(:,3), [0.9, 0.8, 0.7], 'FaceAlpha', 0.5);
    
    % Colors for each finger
    colors = {[1, 0.5, 0], [0, 0.7, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]};
    finger_names = {'Thumb', 'Index', 'Middle', 'Ring', 'Little'};
    
    % Plot fingers
    for i = 1:length(all_positions)
        pos = all_positions{i};
        plot3(pos(:,1), pos(:,2), pos(:,3), 'o-', 'Color', colors{i}, 'LineWidth', 2, 'MarkerFaceColor', colors{i});
    end
    
    % Set up the plot
    grid on;
    axis equal;
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    
    % Store data for animation updates
    hand_data.thumb = thumb_positions;
    hand_data.index = index_positions;
    hand_data.middle = middle_positions;
    hand_data.ring = ring_positions;
    hand_data.little = little_positions;
    hand_data.wrist = wrist;
end

function updateHandVisualization(ax_handle, angles, joint_names, thumb_segments, index_segments, middle_segments, ring_segments, little_segments)
    % Updates an existing hand visualization with new angles
    
    % Switch to the specified axes
    axes(ax_handle);
    
    % Clear current axes content
    cla;
    
    % Visualize the hand with new angles
    visualizeHand(angles, joint_names, thumb_segments, index_segments, middle_segments, ring_segments, little_segments);
end

function [positions, vectors] = calculateFingerPositions(base_pos, angles, segment_lengths, base_direction)
    % Calculate finger joint positions based on joint angles
    
    % Convert angles from degrees to radians
    angles_rad = angles * (pi/180);
    
    % Initialize arrays
    num_segments = length(segment_lengths);
    positions = zeros(num_segments + 1, 3);
    vectors = zeros(num_segments, 3);
    
    % Base position
    positions(1, :) = base_pos;
    
    % Initial direction vector (normalized)
    current_dir = base_direction / norm(base_direction);
    
    % Calculate positions of each joint
    for i = 1:num_segments
        % Adjust direction based on joint angle
        % For simplicity, we'll only apply rotation in the z-x plane
        if i == 1
            % First joint angle is relative to base direction
            angle = angles_rad(i);
        else
            % Subsequent angles are relative to previous segment
            angle = angles_rad(i);
        end
        
        % Create rotation matrix (simplified for this example)
        % Rotation around Y axis for finger flexion
        R = [
            cos(angle), 0, sin(angle);
            0, 1, 0;
            -sin(angle), 0, cos(angle)
        ];
        
        % Apply rotation
        current_dir = (R * current_dir')';
        
        % Scale by segment length
        vec = current_dir * segment_lengths(i);
        vectors(i, :) = vec;
        
        % New position
        positions(i+1, :) = positions(i, :) + vec;
    end
end