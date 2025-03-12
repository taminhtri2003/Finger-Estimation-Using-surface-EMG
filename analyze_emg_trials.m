% Load the .mat file
load('s1.mat');

% Sensor names (assuming the order is consistent)
sensor_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};

num_trials = size(dsfilt_emg, 1);
num_tasks = size(dsfilt_emg, 2);
num_sensors = length(sensor_names);

% Initialize a structure to store the results
results = struct();

% Loop through each task
for task_idx = 1:num_tasks
    task_name = sprintf('Task_%d', task_idx);
    results.(task_name) = struct();

    % Loop through each sensor
    for sensor_idx = 1:num_sensors
        sensor_name = sensor_names{sensor_idx};
        results.(task_name).(sensor_name) = struct();

        rms_values_across_trials = [];

        % Loop through each trial to calculate RMS for this sensor and task
        for trial_idx = 1:num_trials
            emg_data = dsfilt_emg{trial_idx, task_idx}(:, sensor_idx); % Extract EMG data for current trial, task, and sensor
            rms_value = rms(emg_data); % Calculate Root Mean Square
            rms_values_across_trials = [rms_values_across_trials, rms_value]; % Append RMS value
        end

        % Calculate Coefficient of Variation (CV) across trials for RMS values
        cv_rms = (std(rms_values_across_trials) / mean(rms_values_across_trials)) * 100; % CV in percentage

        % Calculate Standard Deviation (SD) across trials for RMS values
        sd_rms = std(rms_values_across_trials);

        % Store the results in the structure
        results.(task_name).(sensor_name).rms_values = rms_values_across_trials;
        results.(task_name).(sensor_name).cv_rms_percent = cv_rms;
        results.(task_name).(sensor_name).sd_rms = sd_rms;

        % You can add more statistical tests here if needed, like ICC
        % For simplified conceptual ICC, you would need to calculate variance
        % between trials and within trials - for this basic script we focus on CV and SD.
    end
end

% Display the results
disp('Statistical Analysis of EMG Trials:');
disp('------------------------------------');
for task_idx = 1:num_tasks
    task_name = sprintf('Task_%d', task_idx);
    disp(['Results for ', task_name, ':']);
    for sensor_idx = 1:num_sensors
        sensor_name = sensor_names{sensor_idx};
        sensor_results = results.(task_name).(sensor_name);
        disp(['  Sensor: ', sensor_name]);
        disp(['    RMS Values across trials: [', num2str(sensor_results.rms_values), ']']);
        disp(['    CV (RMS) across trials: ', num2str(sensor_results.cv_rms_percent), '%']);
        disp(['    SD (RMS) across trials: ', num2str(sensor_results.sd_rms)]);
    end
    disp('------------------------------------');
end

% Further analysis and visualization can be added:
% - Visualize RMS values across trials using box plots or line plots.
% - Compare CV and SD values across different tasks and sensors to understand
%   variability in different muscles and tasks.