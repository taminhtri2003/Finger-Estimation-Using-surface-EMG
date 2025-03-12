function analyze_emg_trials_to_csv_multi_feature(mat_file_path, csv_file_path, feature_list)
% ANALYZE_EMG_TRIALS_TO_CSV_MULTI_FEATURE Analyzes EMG trials, calculates multiple features, and saves to CSV.
%   Analyzes EMG data from a .mat file, calculates a list of specified EMG
%   features (e.g., RMS, MRV, iEMG) across trials for each sensor and task,
%   and saves these results to a CSV file.
%
%   Args:
%       mat_file_path (char): Path to the input .mat file.
%       csv_file_path (char): Path to the output CSV file.
%       feature_list (cell array of strings): Cell array of feature names to calculate.
%                                          Supported features: 'RMS_Amplitude', 'MRV', 'iEMG', 'WL'
%
%   Example:
%       feature_list = {'RMS_Amplitude', 'MRV', 'iEMG', 'WL'};
%       analyze_emg_trials_to_csv_multi_feature('s1.mat', 'emg_multi_feature_results.csv', feature_list);

    % Load the .mat file
    load(mat_file_path);

    % Sensor names (assuming the order is consistent)
    sensor_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};

    num_trials = size(dsfilt_emg, 1);
    num_tasks = size(dsfilt_emg, 2);
    num_sensors = length(sensor_names);
    num_features = length(feature_list);

    % Initialize a structure to store the results (optional, for function return)
    results = struct();

    % Open CSV file for writing
    fileID = fopen(csv_file_path, 'w');
    if fileID == -1
        error('Could not open file for writing: %s', csv_file_path);
    end

    % Write CSV header row
    fprintf(fileID, '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', ...
        'Task', 'Sensor', 'Feature', 'Trial_1', 'Trial_2', 'Trial_3', 'Trial_4', 'Trial_5', 'CV_Value_Percent', 'SD_Value');

    % Loop through each task
    for task_idx = 1:num_tasks
        task_name = sprintf('Task_%d', task_idx);
        results.(task_name) = struct(); % Initialize task struct (optional)

        % Loop through each sensor
        for sensor_idx = 1:num_sensors
            sensor_name = sensor_names{sensor_idx};
            results.(task_name).(sensor_name) = struct(); % Initialize sensor struct (optional)

            % Loop through each feature to calculate
            for feature_idx = 1:num_features
                feature_name = feature_list{feature_idx};
                results.(task_name).(sensor_name).(feature_name) = struct(); % Initialize feature struct

                feature_values_across_trials = [];

                % Loop through each trial to calculate the current feature
                for trial_idx = 1:num_trials
                    emg_data = dsfilt_emg{trial_idx, task_idx}(:, sensor_idx); % Extract EMG data
                    feature_value = calculate_emg_feature(emg_data, feature_name); % Calculate EMG feature
                    feature_values_across_trials = [feature_values_across_trials, feature_value]; % Append feature value
                end

                % Calculate Coefficient of Variation (CV) across trials for feature values
                cv_value = (std(feature_values_across_trials) / mean(feature_values_across_trials)) * 100; % CV in percentage

                % Calculate Standard Deviation (SD) across trials for feature values
                sd_value = std(feature_values_across_trials);

                % Store the results in the structure (optional)
                results.(task_name).(sensor_name).(feature_name).values_across_trials = feature_values_across_trials;
                results.(task_name).(sensor_name).(feature_name).cv_percent = cv_value;
                results.(task_name).(sensor_name).(feature_name).sd = sd_value;

                % Write data to CSV file
                fprintf(fileID, '%s,%s,%s,', task_name, sensor_name, feature_name); % Task, Sensor, Feature
                fprintf(fileID, '%.4f,%.4f,%.4f,%.4f,%.4f,', feature_values_across_trials); % Trial feature values
                fprintf(fileID, '%.4f,%.4f\n', cv_value, sd_value); % CV_Value_Percent, SD_Value
            end
        end
    end

    % Close CSV file
    fclose(fileID);

    disp(['EMG multi-feature analysis results saved to: ', csv_file_path]);

end

function feature_value = calculate_emg_feature(emg_signal, feature_name)
% CALCULATE_EMG_FEATURE Calculates a specified EMG feature.
%   Calculates the value of the specified EMG feature from the given EMG signal.
%
%   Args:
%       emg_signal (vector): EMG signal data.
%       feature_name (char): Name of the feature to calculate.
%                            Supported features: 'RMS_Amplitude', 'MRV', 'iEMG', 'WL'
%
%   Returns:
%       feature_value (double): Calculated feature value.

    switch feature_name
        case 'RMS_Amplitude'
            feature_value = rms(emg_signal);
        case 'MRV' % Mean Rectified Value
            feature_value = mean(abs(emg_signal));
        case 'iEMG' % Integrated EMG
            feature_value = sum(abs(emg_signal));
        case 'WL' % Waveform Length
            feature_value = sum(abs(diff(emg_signal)));
        otherwise
            error('Unsupported EMG feature: %s', feature_name);
    end
end
