% Load the .mat file
load('s1_full.mat'); % Replace 'your_data.mat' with the actual file name

% Number of trials and tasks
num_trials = size(dsfilt_emg, 1);
num_tasks = 5; % Use only the first 5 tasks
num_muscles = 8; % Number of muscles (EMG sensors)
num_joints = 14; % Number of joint angles
num_trees = 20; % Number of trees in the Random Forest - can be tuned

% Initialize variables
muscle_importance = zeros(num_muscles, num_joints);
total_variance = zeros(num_trials, num_tasks);

% Loop through trials and tasks
for trial = 1:num_trials
    for task = 1:num_tasks
        % Extract EMG and joint angle data
        emg_data = dsfilt_emg{trial, task};
        joint_angles_data = joint_angles{trial, task};

        % Check for empty data
        if ~isempty(emg_data) && ~isempty(joint_angles_data)
            % Calculate the total variance of the joint angles for this trial and task
            total_variance(trial, task) = sum(var(joint_angles_data));

            % Perform regression for each joint angle
            for joint = 1:num_joints
                y = joint_angles_data(:, joint);
                X = emg_data;

                % Create and train a Random Forest regression model
                model = TreeBagger(num_trees, X, y, 'Method', 'regression');

                % Predict joint angles
                y_pred = predict(model, X);
                y_pred = str2double(y_pred); % Convert from cell to double

                % Calculate R-squared (for assessing importance)
                SS_total = sum((y - mean(y)).^2);
                if SS_total > 1e-10  % Check if SS_total is greater than a small threshold
                    SS_residual = sum((y - y_pred).^2);
                    R_squared = 1 - (SS_residual / SS_total);
                else
                    R_squared = 0; % If SS_total is too small, set R_squared to 0
                end

                % Calculate muscle importance using a proxy (variance explained by each predictor)
                % 1.  Fit a separate linear regression for each muscle to predict the current joint angle
                muscle_r_squared = zeros(num_muscles, 1);
                for muscle_idx = 1:num_muscles
                    x_muscle = X(:, muscle_idx); % Extract single muscle activity
                    b = regress(y, [ones(size(x_muscle, 1), 1) x_muscle]); % Linear regression
                    y_pred_muscle = [ones(size(x_muscle, 1), 1) x_muscle] * b;
                    SS_residual_muscle = sum((y - y_pred_muscle).^2);
                    SS_total_muscle = sum((y-mean(y)).^2);
                    if SS_total_muscle > 1e-10
                         muscle_r_squared(muscle_idx) = 1 - (SS_residual_muscle / SS_total_muscle);
                    else
                         muscle_r_squared(muscle_idx) = 0;
                    end
                end
                
                % 2. Accumulate the R-squared value, weighted by the muscle R-squared
                muscle_importance(:, joint) = muscle_importance(:, joint) + (R_squared * muscle_r_squared / (num_trials * num_tasks));
            end
        else
            disp(['Warning: Empty data for trial ' num2str(trial) ', task ' num2str(task)]);
        end
    end
end

% Display results
disp('Muscle Importance:');
disp(muscle_importance);

disp('Total Variance of Joint Angles:');
disp(total_variance);

% Optional: Plot the muscle importance
figure;
bar(muscle_importance);
title('Muscle Importance for Each Joint (Random Forest)');
xlabel('Joint Angle');
ylabel('Importance');
legend({'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'});
