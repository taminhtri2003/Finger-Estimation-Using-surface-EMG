% Load the .mat file
load('s1_full.mat'); % Replace with your file

% Define muscle names and tasks
muscle_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};
task_names = {'Thumb', 'Index', 'Middle', 'Ring', 'Little', 'All Fingers', 'Random'};
%Relevant joint angle indices for tasks 1-5 (Individual finger movements)
relevant_joint_angles = {[1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]};

% Parameters
num_imfs = 5;
time_delay = 5;
embedding_dimension = 10;

% Output folder
output_folder = 'EMG_HHT_JointAngle_Results';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Initialize structures to store all features and corresponding joint angles
all_features_data = [];  % Store features from all tasks (1-5)
all_joint_angles_data = []; % Store corresponding joint angles

% --- Loop through tasks (only tasks 1-5) ---
for task = 1:5  % Only individual finger flexion/extension tasks
    fprintf('Processing Task %d (%s)\n', task, task_names{task});

    % --- Loop through muscles ---
    for muscle = 1:size(dsfilt_emg{1,1}, 2)
        fprintf('  Processing Muscle %s\n', muscle_names{muscle});
        all_imfs = [];

        % --- Loop through trials and perform EMD ---
        for trial = 1:size(dsfilt_emg, 1)
            emg_data = dsfilt_emg{trial, task};
            if isempty(emg_data)
                continue;
            end
            emg_signal = emg_data(:, muscle);
            try {
                [imfs, ~, ~] = emd(emg_signal, 'MaxNumIMF'; num_imfs);
                 all_imfs = [all_imfs; imfs];
             } catch ME {
                fprintf('    EMD failed for Trial %d: %s\n', trial, ME.message);
            }

        end

        % --- HHT and GLCM ---
        if ~isempty(all_imfs)
             try {
                % HHT
                [hs, f, t, imfinsf, imfinse] = hht(all_imfs, 1/0.0005);

                % HHS Image and Saving
                figure;
                hht(all_imfs, 1/0.0005);
                title(['HHS - Task: ' task_names{task} ', Muscle: ' muscle_names{muscle}]);
                hhs_filename = fullfile(output_folder, ['HHS_Task' num2str(task) '_Muscle' muscle_names{muscle} '.png']);
                saveas(gcf, hhs_filename);
                close(gcf);

                % GLCM
                hs_normalized = hs - min(hs(:));
                hs_normalized = hs_normalized ./ max(hs_normalized(:));
                hhs_image = im2uint8(hs_normalized);
                glcms = graycomatrix(hhs_image, 'Offset', [0 1; -1 1; -1 0; -1 -1], 'NumLevels', 256, 'GrayLimits', [0 255]);
                stats = graycoprops(glcms, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});

                % Store GLCM Features
                feature_vector = [];
                feature_names = fieldnames(stats);
                for feature_idx = 1:length(feature_names)
                    feature_name = feature_names{feature_idx};
                    feature_values = stats.(feature_name);
                    feature_vector = [feature_vector, mean(feature_values)]; % Concatenate features
                end
                all_features_data = [all_features_data; feature_vector]; % Append to the overall feature matrix


                % --- Collect Corresponding Joint Angles ---
                 joint_angles_for_trials = [];
                 for trial = 1:size(dsfilt_emg,1)
                      if ~isempty(joint_angles{trial,task})
                         joint_angles_for_trials = [joint_angles_for_trials; joint_angles{trial, task}(:, relevant_joint_angles{task})]; % Append
                      end
                 end
                 % Calculate mean joint angles across trials
                 mean_joint_angles = mean(joint_angles_for_trials, 1);  % Mean across time (rows) for this task
                 all_joint_angles_data = [all_joint_angles_data; mean_joint_angles]; %Append to the overall joint angle matrix

            } catch ME {
                fprintf('    HHT or GLCM failed: %s\n', ME.message);
            }
        else
            fprintf('    No IMFs extracted.\n');
        end
    end % Muscles
end % Tasks


% --- Joint Angle Estimation ---

% 1. Data Preparation (Already done in the loops above)
%    - all_features_data:  Matrix of GLCM features (rows: task/muscle combinations, cols: features)
%    - all_joint_angles_data: Matrix of corresponding *mean* joint angles (rows: task/muscle, cols: joint angles)

% 2. Regression Model Training
%    We'll use a simple linear regression model for demonstration.  You can easily replace this with
%    more sophisticated models (e.g., Support Vector Regression, Neural Networks, etc.).

%    Split data into training and testing sets
[train_indices, test_indices] = dividerand(size(all_features_data, 1), 0.8, 0.2, 0.0); %80% training, 20% testing

train_features = all_features_data(train_indices, :);
train_joint_angles = all_joint_angles_data(train_indices, :);
test_features = all_features_data(test_indices, :);
test_joint_angles = all_joint_angles_data(test_indices, :);

% Train a separate model for *each* joint angle
num_joint_angles = size(all_joint_angles_data, 2);
regression_models = cell(1, num_joint_angles);  % Store models
predictions = zeros(size(test_joint_angles));      % Store predictions

for i = 1:num_joint_angles
    % Train a linear regression model
     regression_models{i} = fitlm(train_features, train_joint_angles(:, i));

    % Make predictions on the test set
     predictions(:, i) = predict(regression_models{i}, test_features);
end

% 3. Evaluation
%    Calculate Root Mean Squared Error (RMSE) for each joint angle

rmse_values = zeros(1, num_joint_angles);
for i = 1:num_joint_angles
    rmse_values(i) = sqrt(mean((predictions(:, i) - test_joint_angles(:, i)).^2));
    fprintf('RMSE for Joint Angle %d: %f\n', i, rmse_values(i));
end

% --- Visualization (Example: Scatter plot for one joint angle) ---
joint_angle_to_plot = 1; % Change this to visualize different joint angles
figure;
scatter(test_joint_angles(:, joint_angle_to_plot), predictions(:, joint_angle_to_plot));
xlabel('Actual Joint Angle');
ylabel('Predicted Joint Angle');
title(['Joint Angle Prediction (Joint ' num2str(joint_angle_to_plot) ')']);
hold on;
plot(test_joint_angles(:, joint_angle_to_plot), test_joint_angles(:, joint_angle_to_plot), 'r--'); % Add a diagonal line for perfect prediction
hold off;
legend('Prediction','Ideal','Location','best');
fprintf('Analysis Complete.\n');