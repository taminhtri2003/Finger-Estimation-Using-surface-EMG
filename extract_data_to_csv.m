function extract_data_to_csv(mat_file_path, output_dir)
% Extracts data from a .mat file and saves it to CSV files.
%
% Args:
%     mat_file_path: Path to the .mat file.
%     output_dir: Directory to save the generated CSV files.

try
    load(mat_file_path);
catch ME
    fprintf('Error loading .mat file: %s\n', ME.message);
    return;
end

% Check if variables exist
if ~exist('dsfilt_emg', 'var') || ~exist('finger_kinematics', 'var')
    fprintf('Error: ''dsfilt_emg'' or ''finger_kinematics'' not found in the .mat file.\n');
    return;
end

% Check cell array dimensions.  Important for robust code.
if ~isequal(size(dsfilt_emg), [5, 7]) || ~isequal(size(finger_kinematics), [5, 7])
    fprintf('Error: Unexpected cell array dimensions. Expected [5, 7].\n');
    return;
end


% Create the output directory if it doesn't exist
if ~isfolder(output_dir)
    mkdir(output_dir);
end

for trial = 1:size(dsfilt_emg, 1)
    for task = 1:size(dsfilt_emg, 2)
        % Extract EMG and kinematics data
        try
            emg_data = dsfilt_emg{trial, task};
            kinematics_data = finger_kinematics{trial, task};
        catch ME
            fprintf('Error accessing data at trial %d, task %d: %s\n', trial, task, ME.message);
            continue; % Skip to the next iteration if there's an error
        end

        % Check data dimensions within each cell
        if ~ismatrix(emg_data) || size(emg_data, 2) ~= 8
            fprintf('Error: Invalid emg_data dimensions at trial %d, task %d. Expected Nx8 matrix.\n', trial, task);
            continue;
        end
        if ~ismatrix(kinematics_data) || size(kinematics_data, 2) ~= 69
            fprintf('Error: Invalid kinematics_data dimensions at trial %d, task %d. Expected Nx69 matrix.\n', trial, task);
            continue;
        end
        
        % Combine EMG and kinematics data
        combined_data = [emg_data, kinematics_data];

        % Create column names
        emg_column_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};
        kinematics_column_names = cell(1, 69);
        for i = 1:69
            kinematics_column_names{i} = sprintf('Kinematics_%d', i);
        end
        column_names = [emg_column_names, kinematics_column_names];

        % Convert to table
        data_table = array2table(combined_data, 'VariableNames', column_names);

        % Construct the output file name
        output_file_name = fullfile(output_dir, sprintf('trial_%d_task_%d.csv', trial, task));

        % Write to CSV
        try
            writetable(data_table, output_file_name);
            fprintf('Data for trial %d, task %d saved to: %s\n', trial, task, output_file_name);
        catch ME
            fprintf('Error writing CSV for trial %d, task %d: %s\n', trial, task, ME.message);
        end
    end
end

end

