% Load the data
load('s1.mat');

% --- 1. Feature Extraction from EMG ---
windowSize = 200;
overlap = 100;

% Pre-allocate
trainX = [];
trainY = [];
testX = [];
testY = [];

for trial = 1:5  % Iterate through *all* trials
    for task = 1:7
        emg_data = dsfilt_emg{trial, task};
        kinematics = finger_kinematics{trial, task};

        if isempty(emg_data) || isempty(kinematics)
            fprintf('Data missing for Trial %d, Task %d. Skipping.\n', trial, task);
            continue;
        end

        kinematics_downsampled = downsample(kinematics, 10);
        numWindows = floor((size(emg_data, 1) - overlap) / (windowSize - overlap));

        for win = 1:numWindows
            startIndex = (win - 1) * (windowSize - overlap) + 1;
            endIndex = startIndex + windowSize - 1;
            emg_window = emg_data(startIndex:endIndex, :);

            % --- Feature Calculation (Expanded Feature Set) ---
            mav = mean(abs(emg_window));
            rms = sqrt(mean(emg_window.^2));
            var_emg = var(emg_window);
            wl = sum(abs(diff(emg_window)));
            
            % Add more features:
            zero_crossings = sum(diff(sign(emg_window)) ~= 0, 1);  % Zero crossings
            slope_sign_changes = sum(diff(sign(diff(emg_window))) ~= 0, 1); % Slope sign changes
            % Add frequency-domain features (PSD - Power Spectral Density)
             [pxx, ~] = pwelch(emg_window, [], [], [], 1000); %fs=1000 Hz (downsampled freq)
              mean_freq = meanfreq(pxx,1000);   % Mean frequency
              median_freq = medfreq(pxx, 1000);  % Median Frequency
              peak_freq_power = max(pxx);
              total_power = sum(pxx);

            features = [mav, rms, var_emg, wl, zero_crossings, slope_sign_changes, ...
                        mean_freq, median_freq, peak_freq_power, total_power];

            kinematics_index = round(endIndex / 10);
            kinematics_index = max(1, min(kinematics_index, size(kinematics_downsampled, 1)));
            kinematics_values = kinematics_downsampled(kinematics_index, :);

            % --- Data Splitting (Trials 1-3: Train, Trials 4-5: Test) ---
            if trial <= 3
                trainX = [trainX; features];
                trainY = [trainY; kinematics_values];
            else
                testX = [testX; features];
                testY = [testY; kinematics_values];
            end
        end
    end
end

% --- Data Normalization (Important for many models) ---
[trainX_norm, mu, sigma] = zscore(trainX); % Normalize training data
testX_norm = (testX - mu) ./ sigma;       % Apply same normalization to test data

% --- 2. Model Training (Gradient Boosted Trees - Better Performance) ---
% Use an ensemble of Gradient Boosted Trees (fitrensemble)
trained_models = cell(1, size(trainY, 2));
for i = 1:size(trainY, 2)
      trained_models{i} = fitrensemble(trainX_norm, trainY(:, i), 'Method', 'LSBoost', ...
                                      'NumLearningCycles', 200, 'Learners', 'tree', ...
                                      'LearnRate', 0.1);  % Adjust hyperparameters
end
% --- 3. Prediction ---
predictedY = zeros(size(testY));
for i = 1:size(testY, 2)
    predictedY(:, i) = predict(trained_models{i}, testX_norm);
end

% --- 4. Evaluation (RMSE) ---
rmse = sqrt(mean((predictedY - testY).^2));
overall_rmse = mean(rmse);
fprintf('Overall RMSE: %.4f\n', overall_rmse);


% --- 5. Visualization (Predictions vs. Ground Truth) ---

% --- Time Series Plots (for a few kinematic dimensions) ---
num_dims_to_plot = 6; % Increase number of dimensions to plot
figure;
sgtitle('Predictions vs. Ground Truth');
for dim = 1:num_dims_to_plot
    subplot(num_dims_to_plot, 1, dim);
    plot(testY(:, dim), 'b', 'LineWidth', 1.5, 'DisplayName', 'Ground Truth'); % Thinner line
    hold on;
    plot(predictedY(:, dim), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Prediction'); % Thinner line
    legend('Location', 'best'); % Add legend
    title(sprintf('Dimension %d', dim));
    xlabel('Sample Index');
    ylabel('Position');
    grid on; % Add grid lines
end


% --- Scatter Plots (Predicted vs. Actual) ---
figure;
sgtitle('Scatter Plots (Predicted vs. Actual)');
for dim = 1:num_dims_to_plot
    subplot(num_dims_to_plot, 1, dim);
    scatter(testY(:, dim), predictedY(:, dim), 'b.');
    hold on;
    refline(1, 0); % Add a line of perfect agreement (y = x)
    xlabel('Actual Position');
    ylabel('Predicted Position');
    title(sprintf('Dimension %d', dim));
    axis equal;
     grid on;
end

% --- Error Distribution Histograms ---
figure;
sgtitle('Error Distribution');
for dim = 1:num_dims_to_plot
    subplot(num_dims_to_plot, 1, dim);
    residuals = testY(:, dim) - predictedY(:, dim);
    histogram(residuals, 20); % 20 bins
    xlabel('Prediction Error');
    ylabel('Frequency');
    title(sprintf('Dimension %d Error Distribution', dim));
    grid on;
end

% --- 6. 3D Hand Visualization (Same as before, but uses predictedY) ---
marker_connections = [
    % Thumb
    20, 17; 17, 18; 18, 19;
    % Index Finger
    20, 1; 1, 5; 5, 6; 6, 7;
    % Middle Finger
    20, 2; 2, 8; 8, 9; 9, 10;
    % Ring Finger
    20, 3; 3, 11; 11, 12; 12, 13;
    % Little Finger
    20, 4; 4, 14; 14, 15; 15, 16;
    % Wrist
    21, 22; 22, 23;
];
% --- Animation Loop ---
figure;
% Initialize plot (create dummy plot handles)
numMarkers = 23; %  There are 23 markers
h_markers = plot3(zeros(1,numMarkers), zeros(1,numMarkers), zeros(1,numMarkers), '.', 'MarkerSize', 20, 'Color', 'b'); % Dummy markers
hold on;
h_lines = gobjects(size(marker_connections, 1), 1); % Pre-allocate line handles
for i = 1:size(marker_connections, 1)
      h_lines(i) = line([0, 0], [0, 0], [0, 0], 'Color', 'r', 'LineWidth', 1.5); % Dummy lines
end

hold off;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Hand Movement');
grid on;
axis equal;
view(3);

% Determine axis limits based on predicted data + some padding.  Important!
all_x = predictedY(:, 1:3:end);
all_y = predictedY(:, 2:3:end);
all_z = predictedY(:, 3:3:end);
padding = 50; % Add some space around the hand
xlim([min(all_x(:)) - padding, max(all_x(:)) + padding]);
ylim([min(all_y(:)) - padding, max(all_y(:)) + padding]);
zlim([min(all_z(:)) - padding, max(all_z(:)) + padding]);

for t = 1:size(predictedY, 1)  % Iterate through time steps
    % Extract x, y, z coordinates for the current time step
    x_coords = predictedY(t, 1:3:end);
    y_coords = predictedY(t, 2:3:end);
    z_coords = predictedY(t, 3:3:end);
    
    % Check for consistent lengths
    if ~(length(x_coords) == length(y_coords) && length(y_coords) == length(z_coords))
        warning('Coordinate lengths are inconsistent at time step %d. Skipping frame.', t);
        continue;
    end
    
     % Check for correct number of coordinates
    if length(x_coords) ~= numMarkers
        warning('Incorrect number of coordinates at time step %d.  Expected %d, got %d. Skipping frame.', t, numMarkers, length(x_coords));
        continue;  % Skip this frame
    end

    % Combine into a single matrix for plotting
     jointMarkerPos = [x_coords', y_coords', z_coords'];


    % Update marker positions
     set(h_markers, 'XData', jointMarkerPos(:, 1), 'YData', jointMarkerPos(:, 2), 'ZData', jointMarkerPos(:, 3));

     % Update line positions using the provided handMocapModifyKINE logic
       for i = 1:size(marker_connections, 1)
            set(h_lines(i), 'XData', jointMarkerPos(marker_connections(i,:), 1), ...
                                   'YData', jointMarkerPos(marker_connections(i,:), 2), ...
                                   'ZData', jointMarkerPos(marker_connections(i,:), 3));
      end

    title(sprintf('3D Hand Movement (Time: %d)', t));
    drawnow;
    pause(0.03);
end


% --- Helper Functions (from the provided code) ---

function joint = handJointPosExtract(pos)
joint(:,1) = pos(1:3:end);
joint(:,2) = pos(2:3:end);
joint(:,3) = pos(3:3:end);
end