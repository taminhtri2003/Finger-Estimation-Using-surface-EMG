function analyze_kinematics_trials_to_csv(mat_file_path, csv_file_path, kinematic_feature_list)
% ANALYZE_KINEMATICS_TRIALS_TO_CSV Analyzes kinematics trials, calculates multiple features, and saves to CSV.
%   Analyzes kinematics data from a .mat file, calculates a list of specified
%   kinematic features across trials for each marker coordinate and task,
%   and saves these results to a CSV file.
%
%   Args:
%       mat_file_path (char): Path to the input .mat file.
%       csv_file_path (char): Path to the output CSV file.
%       kinematic_feature_list (cell array of strings): Cell array of kinematic feature names to calculate.
%                                          Supported features: 'Position_Range'
%
%   Example:
%       kinematic_feature_list = {'Position_Range'};
%       analyze_kinematics_trials_to_csv('s1.mat', 'kinematics_results.csv', kinematic_feature_list);

    % Load the .mat file
    load(mat_file_path);

    % Data structure information
    num_trials = size(finger_kinematics, 1);
    num_tasks = size(finger_kinematics, 2);
    num_markers = 69 / 3; % Assuming 69 columns are x,y,z for each marker
    coordinate_names = {'X', 'Y', 'Z'}; % Coordinate names

    % Initialize a structure to store the results (optional)
    results = struct();

    % Open CSV file for writing
    fileID = fopen(csv_file_path, 'w');
    if fileID == -1
        error('Could not open file for writing: %s', csv_file_path);
    end

    % Write CSV header row
    fprintf(fileID, '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', ...
        'Task', 'Marker_Coordinate', 'Feature', 'Trial_1', 'Trial_2', 'Trial_3', 'Trial_4', 'Trial_5', 'CV_Value_Percent', 'SD_Value');

    % Loop through each task
    for task_idx = 1:num_tasks
        task_name = sprintf('Task_%d', task_idx);
        results.(task_name) = struct(); % Initialize task struct (optional)

        % Loop through each marker
        for marker_idx = 1:num_markers
            marker_name = sprintf('Marker_%d', marker_idx);
            results.(task_name).(marker_name) = struct(); % Initialize marker struct (optional)

            % Loop through each coordinate (X, Y, Z)
            for coord_idx = 1:length(coordinate_names)
                coordinate_name = coordinate_names{coord_idx};
                marker_coordinate_name = [marker_name, '_', coordinate_name];
                results.(task_name).(marker_coordinate_name) = struct(); % Initialize coordinate struct (optional)


                % Loop through each kinematic feature to calculate
                for feature_idx = 1:length(kinematic_feature_list)
                    feature_name = kinematic_feature_list{feature_idx};
                    results.(task_name).(marker_coordinate_name).(feature_name) = struct(); % Initialize feature struct

                    feature_values_across_trials = [];

                    % Loop through each trial to calculate the current feature
                    for trial_idx = 1:num_trials
                        kinematic_data = finger_kinematics{trial_idx, task_idx}(:, (marker_idx-1)*3 + coord_idx); % Extract kinematics data for current trial, task, marker, and coordinate
                        feature_value = calculate_kinematic_feature(kinematic_data, feature_name); % Calculate kinematic feature
                        feature_values_across_trials = [feature_values_across_trials, feature_value]; % Append feature value
                    end

                    % Calculate Coefficient of Variation (CV) across trials for feature values
                    cv_value = (std(feature_values_across_trials) / mean(feature_values_across_trials)) * 100; % CV in percentage

                    % Calculate Standard Deviation (SD) across trials for feature values
                    sd_value = std(feature_values_across_trials);

                    % Store the results in the structure (optional)
                    results.(task_name).(marker_coordinate_name).(feature_name).values_across_trials = feature_values_across_trials;
                    results.(task_name).(marker_coordinate_name).(feature_name).cv_percent = cv_value;
                    results.(task_name).(marker_coordinate_name).(feature_name).sd = sd_value;

                    % Write data to CSV file
                    fprintf(fileID, '%s,%s,%s,', task_name, marker_coordinate_name, feature_name); % Task, Marker_Coordinate, Feature
                    fprintf(fileID, '%.4f,%.4f,%.4f,%.4f,%.4f,', feature_values_across_trials); % Trial feature values
                    fprintf(fileID, '%.4f,%.4f\n', cv_value, sd_value); % CV_Value_Percent, SD_Value

                end
            end
        end
    end

    % Close CSV file
    fclose(fileID);

    disp(['Kinematics multi-feature analysis results saved to: ', csv_file_path]);

end

function feature_value = calculate_kinematic_feature(kinematic_data, feature_name)
% CALCULATE_KINEMATIC_FEATURE Calculates a specified kinematic feature.
%   Calculates the value of the specified kinematic feature from the given kinematic data.
%
%   Args:
%       kinematic_data (vector): Kinematic data (joint position, etc.).
%       feature_name (char): Name of the feature to calculate.
%                            Supported features: 'Position_Range'
%
%   Returns:
%       feature_value (double): Calculated feature value.

    switch feature_name
        case 'Position_Range'
            feature_value = max(kinematic_data) - min(kinematic_data);
        otherwise
            error('Unsupported kinematic feature: %s', feature_name);
    end
end

