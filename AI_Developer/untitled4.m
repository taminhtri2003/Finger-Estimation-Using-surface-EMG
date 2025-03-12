% Clear workspace and load necessary libraries
clear; clc;

% Load the .mat file containing the data
load('s1_full.mat'); % Replace 'data.mat' with your actual filename

%% Step 1: Data Validation and Preparation
fprintf('Step 1: Validating and preparing the data...\n');

% Initialize matrices for EMG (X) and joint angles (Y)
X = []; % EMG data
Y = []; % Joint angles

% Check for data consistency between EMG and joint angles
for trial = 1:5
    for task = 1:7
        emgData = dsfilt_emg{trial, task};
        angleData = joint_angles{trial, task};
        
        % Check if number of rows match
        if size(emgData, 1) ~= size(angleData, 1)
            error('Mismatch in trial %d, task %d: EMG has %d rows, angles have %d rows', ...
                trial, task, size(emgData, 1), size(angleData, 1));
        end
        
        % Append to global matrices
        X = [X; emgData]; 
        Y = [Y; angleData];
    end
end

% Verify final sizes of concatenated data
fprintf('Final data sizes:\n');
fprintf('EMG data (X): %d x %d\n', size(X, 1), size(X, 2));
fprintf('Joint angles (Y): %d x %d\n', size(Y, 1), size(Y, 2));

% Normalize the EMG data (z-score normalization)
fprintf('Normalizing EMG data...\n');
X = zscore(X);

% Split the data into training (80%) and testing (20%) sets
fprintf('Splitting data into training and testing sets...\n');
rng(42); % Set random seed for reproducibility
cv = cvpartition(size(X, 1), 'HoldOut', 0.2); % 80-20 split
X_train = X(cv.training, :);
Y_train = Y(cv.training, :);
X_test = X(cv.test, :);
Y_test = Y(cv.test, :);

%% Step 2: Model Training
fprintf('Step 2: Training linear regression models...\n');

% Number of joint angles to predict
num_angles = size(Y, 2);

% Train a separate linear regression model for each joint angle
models = cell(num_angles, 1); % Store models for each joint angle
for i = 1:num_angles
    fprintf('Training model for joint angle %d/%d...\n', i, num_angles);
    models{i} = fitrlinear(X_train, Y_train(:, i), ...
        'ObservationsIn', 'columns', ... % Specify input orientation
        'Lambda', 'auto'); % Automatic regularization parameter selection
end

%% Step 3: Prediction and Evaluation
fprintf('Step 3: Evaluating the models...\n');

% Predict joint angles on the test set
Y_pred = zeros(size(Y_test)); % Initialize predictions
for i = 1:num_angles
    Y_pred(:, i) = predict(models{i}, X_test'); % Predict for each joint angle
end

% Calculate Root Mean Squared Error (RMSE) for each joint angle
rmse = sqrt(mean((Y_test - Y_pred).^2, 1));
fprintf('RMSE for each joint angle:\n');
disp(rmse);

% Calculate overall RMSE
overall_rmse = sqrt(mean((Y_test(:) - Y_pred(:)).^2));
fprintf('Overall RMSE: %.4f\n', overall_rmse);

%% Step 4: Explainability - Extract Coefficients
fprintf('Step 4: Extracting coefficients for explainability...\n');

% Define muscle names for interpretability
muscles = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};

% Extract coefficients for each joint angle
coeffs = zeros(length(muscles), num_angles); % Coefficient matrix
for i = 1:num_angles
    coeffs(:, i) = models{i}.Beta; % Regression coefficients
end

%% Step 5: Visualization of Results
fprintf('Step 5: Visualizing results...\n');

% Plot coefficients for each joint angle
figure;
for i = 1:num_angles
    subplot(3, 5, i); % Arrange subplots in a grid
    bar(coeffs(:, i)); % Bar plot of coefficients
    set(gca, 'XTickLabel', muscles, 'XTick', 1:length(muscles)); % Label muscles
    title(sprintf('Joint Angle %d', i)); % Title for each subplot
    ylabel('Coefficient Value'); % Y-axis label
    xlabel('Muscle'); % X-axis label
    if i == 14
        legend('Contribution'); % Add legend for clarity
    end
end
sgtitle('Muscle Contributions to Joint Angles'); % Overall title

% Plot predicted vs actual joint angles for a few examples
figure;
for i = 1:min(3, num_angles) % Plot for first 3 joint angles
    subplot(1, 3, i);
    plot(Y_test(:, i), 'b', 'DisplayName', 'Actual'); hold on;
    plot(Y_pred(:, i), 'r--', 'DisplayName', 'Predicted');
    title(sprintf('Joint Angle %d', i));
    xlabel('Time');
    ylabel('Angle (degrees)');
    legend;
end
sgtitle('Predicted vs Actual Joint Angles');

%% Step 6: Save Results (Optional)
fprintf('Step 6: Saving results...\n');

% Save predictions, coefficients, and RMSE to a file
save('results.mat', 'Y_pred', 'coeffs', 'rmse', 'overall_rmse');

fprintf('Process completed successfully!\n');