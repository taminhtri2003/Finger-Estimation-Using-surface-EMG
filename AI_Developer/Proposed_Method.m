% MATLAB Code for Generalized Additive Models (GAMs) for Finger Movement Estimation
% (Corrected to check for the 'fitgam' function)

% Assuming you have a .mat file named 'your_data.mat' containing the cell variables:
% dsfilt_emg (5x7 cell), finger_kinematics (5x7 cell), joint_angles (5x7 cell)

% Load the data
load('s4_full.mat');

num_trials = size(joint_angles, 1); % Number of trials (rows)
num_tasks = size(joint_angles, 2); % Number of tasks (columns)
num_joint_angles = 14; % Number of joint angles
muscle_names = ["APL", "FCR", "FDS", "FDP", "ED", "EI", "ECU", "ECR"];
joint_angle_names = ["Thumb1", "Thumb2", "Index1", "Index2", "Index3", ...
                     "Middle1", "Middle2", "Middle3", "Ring1", "Ring2", "Ring3", ...
                     "Little1", "Little2", "Little3"];

% Check if the fitgam function exists (indicates the toolbox is likely available)
if exist('fitgam', 'file') == 2
    % Iterate through each trial and task
    for trial = 1:num_trials
        for task = 1:num_tasks
            fprintf('Processing Trial %d, Task %d...\n', trial, task);

            % Extract EMG and joint angle data for the current cell
            emg_data = dsfilt_emg{trial, task}; % Size: 4000x8
            joint_angle_data = joint_angles{trial, task}; % Size: 4000x14

            % Ensure data has the same number of rows
            if size(emg_data, 1) ~= size(joint_angle_data, 1)
                warning('EMG and Joint Angle data have different lengths in Trial %d, Task %d. Skipping.', trial, task);
                continue;
            end

            % Train a separate GAM for each joint angle
            for j = 1:num_joint_angles
                fprintf('  Estimating Joint Angle: %s...\n', joint_angle_names(j));

                % Response variable: Current joint angle
                response = joint_angle_data(:, j);

                % Predictors: EMG data for the 8 muscles
                predictors = emg_data;

                try
                    % Fit the Generalized Additive Model
                    gam_model = fitgam(predictors, response, 'interactions', ...
                                       'PredictorNames', muscle_names, ...
                                       'ResponseName', joint_angle_names(j), ...
                                       'Basis', 'spline', 'NumBasisFunctions', 5); % Adjust NumBasisFunctions as needed

                    % Display the learned effects (coefficients for linear terms, shapes for splines)
                    disp(['    Learned effects for ', joint_angle_names(j), ':']);
                    disp(gam_model.Trained);

                    % Optional: Visualize the partial effects of each muscle on the current joint angle
                    figure;
                    plotPartialEffects(gam_model);
                    sgtitle(['Trial ', num2str(trial), ', Task ', num2str(task), ...
                             ' - Partial Effects on ', joint_angle_names(j)]);

                catch ME
                    warning('Error fitting GAM for Trial %d, Task %d, Joint Angle %s: %s', ...
                            trial, task, joint_angle_names(j), ME.message);
                end
            end
        end
    end

    disp('Finished processing all trials and tasks.');

else
    error('The function "fitgam" was not found. Please ensure that the Statistics and Machine Learning Toolbox is installed and licensed in your MATLAB environment.');
end