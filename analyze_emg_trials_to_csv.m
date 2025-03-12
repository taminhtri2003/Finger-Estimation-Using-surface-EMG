function analyze_emg_trials_to_csv(mat_file_path, csv_file_path)
% ANALYZE_EMG_TRIALS_TO_CSV Analyzes EMG trials from a .mat file and saves results to CSV.
%   Analyzes EMG data from a .mat file containing 'dsfilt_emg' variable,
%   calculates RMS, CV, and SD of RMS across trials for each sensor and task,
%   and saves these results to a CSV file.
%
%   Args:
%       mat_file_path (char): Path to the input .mat file.
%       csv_file_path (char): Path to the output CSV file.
%
%   Example:
%       analyze_emg_trials_to_csv('s1.mat', 'emg_analysis_results.csv');

    % Load the .mat file
    load(mat_file_path);

    % Sensor names (assuming the order is consistent)
    sensor_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};

    num_trials = size(dsfilt_emg, 1);
    num_tasks = size(dsfilt_emg, 2);
    num_sensors = length(sensor_names);

    % Initialize a structure to store the results (optional, for function return)
    results = struct();

    % Open CSV file for writing
    fileID = fopen(csv_file_path, 'w');
    if fileID == -1
        error('Could not open file for writing: %s', csv_file_path);
    end

    % Write CSV header row
    fprintf(fileID, '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', ...
        'Task', 'Sensor', 'Feature', 'Trial_1', 'Trial_2', 'Trial_3', 'Trial_4', 'Trial_5', 'CV_RMS_Percent', 'SD_RMS');

    % Loop through each task
    for task_idx = 1:num_tasks
        task_name = sprintf('Task_%d', task_idx);
        results.(task_name) = struct(); % Initialize task struct (optional)

        % Loop through each sensor
        for sensor_idx = 1:num_sensors
            sensor_name = sensor_names{sensor_idx};
            results.(task_name).(sensor_name) = struct(); % Initialize sensor struct (optional)

            rms_values_across_trials = [];

            % Loop through each trial to calculate RMS for this sensor and task
            for trial_idx = 1:num_trials
                emg_data = dsfilt_emg{trial_idx, task_idx}(:, sensor_idx); % Extract EMG data
                rms_value = rms(emg_data); % Calculate Root Mean Square
                rms_values_across_trials = [rms_values_across_trials, rms_value]; % Append RMS value
            end

            % Calculate Coefficient of Variation (CV) across trials for RMS values
            cv_rms = (std(rms_values_across_trials) / mean(rms_values_across_trials)) * 100; % CV in percentage

            % Calculate Standard Deviation (SD) across trials for RMS values
            sd_rms = std(rms_values_across_trials);

            % Store the results in the structure (optional)
            results.(task_name).(sensor_name).rms_values = rms_values_across_trials;
            results.(task_name).(sensor_name).cv_rms_percent = cv_rms;
            results.(task_name).(sensor_name).sd_rms = sd_rms;

            % Write data to CSV file
            fprintf(fileID, '%s,%s,%s,', task_name, sensor_name, 'RMS_Amplitude'); % Task, Sensor, Feature
            fprintf(fileID, '%.4f,%.4f,%.4f,%.4f,%.4f,', rms_values_across_trials); % Trial RMS values
            fprintf(fileID, '%.4f,%.4f\n', cv_rms, sd_rms); % CV_RMS_Percent, SD_RMS

        end
    end

    % Close CSV file
    fclose(fileID);

    disp(['EMG analysis results saved to: ', csv_file_path]);

end