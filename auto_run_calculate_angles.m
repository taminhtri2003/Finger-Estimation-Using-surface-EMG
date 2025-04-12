% Main script to process kinematics and EMG data files

clearvars; % Clear existing variables from the workspace
clc;       % Clear the command window

% --- Configuration ---
input_file_indices = 1:4; % Process files s5 through s10
variables_to_load = {'finger_kinematics', 'dsfilt_emg'}; % Variables to load from input
output_variable_name = 'joint_angles_cell'; % Name for the calculated angles variable

% --- Processing Loop ---
fprintf('Starting kinematics and EMG processing...\n');

for i = input_file_indices
    % Construct input and output filenames
    input_filename = sprintf('s%d.mat', i);
    output_filename = sprintf('s%d_full.mat', i);

    fprintf('----------------------------------------\n');
    fprintf('Processing file: %s\n', input_filename);

    % Check if the input file exists
    if isfile(input_filename)
        try
            % Load the specified variables from the input file
            fprintf('  Loading variables: %s\n', strjoin(variables_to_load, ', '));
            % Use load directly into workspace for the specified variables
            load(input_filename, variables_to_load{:}); % The {:} expands the cell array

            % Check if the expected variables were loaded successfully
            if exist('finger_kinematics', 'var') && exist('dsfilt_emg', 'var')

                % Validate the loaded finger_kinematics data format (basic check)
                if iscell(finger_kinematics) && isequal(size(finger_kinematics), [5, 7])
                    % Calculate joint angles using the provided function
                    fprintf('  Calculating joint angles...\n');
                    % Assign the result to the specific output variable name
                    joint_angles_cell = calculate_joint_angles_ver2(finger_kinematics);

                    % Save the results (original loaded + calculated) to the new .mat file
                    fprintf('  Saving variables finger_kinematics, joint_angles_cell, dsfilt_emg to: %s\n', output_filename);
                    % Save the variables currently in the workspace
                    save(output_filename, 'finger_kinematics', 'joint_angles_cell', 'dsfilt_emg');

                    fprintf('  Successfully processed and saved.\n');
                else
                    fprintf('  Warning: Variable "finger_kinematics" in %s is not a 5x7 cell array. Skipping processing, but saving original data.\n', input_filename);
                    % Optionally save only the original loaded data if calculation fails
                    % save(output_filename, 'finger_kinematics', 'dsfilt_emg');
                    % For now, we just skip saving if calculation isn't done.
                end
            else
                % Create a list of missing variables for the warning message
                missing_vars = {};
                if ~exist('finger_kinematics', 'var')
                    missing_vars{end+1} = 'finger_kinematics';
                end
                if ~exist('dsfilt_emg', 'var')
                    missing_vars{end+1} = 'dsfilt_emg';
                end
                fprintf('  Warning: Required variable(s) "%s" not found in %s. Skipping.\n', strjoin(missing_vars, '", "'), input_filename);
            end

        catch ME % Catch potential errors during loading or processing
            fprintf('  Error processing file %s: %s\n', input_filename, ME.message);
            fprintf('  Skipping this file.\n');
        end
    else
        fprintf('  Warning: Input file %s not found. Skipping.\n', input_filename);
    end

    % Clear variables for the next iteration to avoid carrying them over
    % Make sure to clear all variables loaded or created in the loop
    clear finger_kinematics dsfilt_emg joint_angles_cell;

end

fprintf('----------------------------------------\n');
fprintf('Processing complete.\n');


% --- Required Functions (Copied from your provided code, with minor enhancements) ---

function joint_angles_cell = calculate_joint_angles_ver2(finger_kinematics)
%CALCULATE_JOINT_ANGLES Calculates joint angles from marker positions (in degrees).
%
%   joint_angles_cell = calculate_joint_angles_ver2(finger_kinematics)
%   calculates joint angles from the finger_kinematics data, which is a
%   5x7 cell array of marker position data (x, y, z coordinates).
%   The angles are returned in DEGREES.
%
%   Args:
%       finger_kinematics: A 5x7 cell array. Each cell contains a matrix
%                         (e.g., 4000x69) representing marker positions.
%                         Each row is a time point, and the columns are
%                         (x, y, z) for 23 markers.
%
%   Returns:
%       joint_angles_cell: A 5x7 cell array. Each cell contains a matrix
%                        where each row is a time point and each column
%                        represents a calculated joint angle (in degrees).

    % Input validation (optional but recommended)
    if ~iscell(finger_kinematics) || ~isequal(size(finger_kinematics), [5, 7])
        error('Input finger_kinematics must be a 5x7 cell array.');
    end

    joint_angles_cell = cell(size(finger_kinematics));
    for trial = 1:size(finger_kinematics, 1)
        for task = 1:size(finger_kinematics, 2)
            kinematics_data = finger_kinematics{trial, task};

            % Basic check on inner matrix dimensions (optional)
            if isempty(kinematics_data)
                 fprintf('Warning: Empty cell found at trial %d, task %d. Assigning empty to joint angles for this cell.\n', trial, task);
                 joint_angles_cell{trial, task} = []; % Assign empty
                 continue; % Skip calculation for this cell
            end
            % Check number of columns (should be 23 markers * 3 coords = 69)
            if size(kinematics_data, 2) ~= 69
                 warning('MATLAB:calculate_joint_angles:WrongColumnCount', ...
                         'Expected 69 columns (23 markers * 3 coords) in cell (%d, %d), found %d. Results may be incorrect.', ...
                         trial, task, size(kinematics_data, 2));
                 % Decide how to handle: error, skip, or proceed with caution.
                 % We proceed here, but caution is advised.
            end

            num_timepoints = size(kinematics_data, 1);
            if num_timepoints == 0
                joint_angles = []; % Handle case with zero timepoints
            else
                 joint_angles = zeros(num_timepoints, 14); % 14 angles total

                 % Use try-catch within angle calculations for robustness if needed
                 try
                     % --- Thumb (2 angles) ---
                     joint_angles(:, 1) = calculate_angle_degrees(kinematics_data, 20, 17, 18);
                     joint_angles(:, 2) = calculate_angle_degrees(kinematics_data, 17, 18, 19);
                     % --- Index (3 angles) ---
                     joint_angles(:, 3) = calculate_angle_degrees(kinematics_data, 20, 1, 5);
                     joint_angles(:, 4) = calculate_angle_degrees(kinematics_data, 1, 5, 6);
                     joint_angles(:, 5) = calculate_angle_degrees(kinematics_data, 5, 6, 7);
                     % --- Middle (3 angles) ---
                     joint_angles(:, 6) = calculate_angle_degrees(kinematics_data, 20, 2, 8);
                     joint_angles(:, 7) = calculate_angle_degrees(kinematics_data, 2, 8, 9);
                     joint_angles(:, 8) = calculate_angle_degrees(kinematics_data, 8, 9, 10);
                     % --- Ring (3 angles) ---
                     joint_angles(:, 9) = calculate_angle_degrees(kinematics_data, 20, 3, 11);
                     joint_angles(:, 10) = calculate_angle_degrees(kinematics_data, 3, 11, 12);
                     joint_angles(:, 11) = calculate_angle_degrees(kinematics_data, 11, 12, 13);
                     % --- Little (3 angles) ---
                     joint_angles(:, 12) = calculate_angle_degrees(kinematics_data, 20, 4, 14);
                     joint_angles(:, 13) = calculate_angle_degrees(kinematics_data, 4, 14, 15);
                     joint_angles(:, 14) = calculate_angle_degrees(kinematics_data, 14, 15, 16);
                 catch ME_angle
                     warning('MATLAB:calculate_joint_angles:AngleCalculationError', ...
                             'Error calculating angles for cell (%d, %d): %s. Assigning empty.', trial, task, ME_angle.message);
                     joint_angles = []; % Assign empty if calculation fails
                 end
            end
            joint_angles_cell{trial, task} = joint_angles;
        end
    end
end

function angles_degrees = calculate_angle_degrees(kinematics_data, marker1_idx, marker2_idx, marker3_idx)
%CALCULATE_ANGLE_DEGREES Calculates the angle between three 3D points (in degrees).
%
%   Args:
%       kinematics_data: A matrix of marker positions (Nx69).
%       marker1_idx, marker2_idx, marker3_idx: Indices of the markers (1-23).
%
%   Returns:
%       angles_degrees:  A column vector of angles (in degrees) for each time point.

    % Check if kinematics_data is empty or has zero timepoints
    if isempty(kinematics_data) || size(kinematics_data, 1) == 0
        angles_degrees = [];
        return;
    end

    % Calculate column indices for x, y, z coordinates
    cols1 = (marker1_idx-1)*3 + (1:3);
    cols2 = (marker2_idx-1)*3 + (1:3);
    cols3 = (marker3_idx-1)*3 + (1:3);

    % Check if indices are within the bounds of the data columns
    max_col_needed = max([cols1, cols2, cols3]);
    num_cols_data = size(kinematics_data, 2);
    if max_col_needed > num_cols_data
        error('Marker index results in column index %d, which exceeds the number of columns in kinematics_data (%d). Check marker indices and data integrity.', ...
              max_col_needed, num_cols_data);
    end

    % Extract marker coordinates (x, y, z)
    marker1 = kinematics_data(:, cols1);
    marker2 = kinematics_data(:, cols2);
    marker3 = kinematics_data(:, cols3);

    % Calculate vectors (vector from marker2 to marker1, and marker2 to marker3)
    v1 = marker1 - marker2; % Vector pointing from joint center (m2) to proximal marker (m1)
    v2 = marker3 - marker2; % Vector pointing from joint center (m2) to distal marker (m3)

    % Calculate dot product and magnitudes
    dot_product = sum(v1 .* v2, 2); % Sum along the columns (x,y,z) for each time point (row)
    mag_v1 = sqrt(sum(v1.^2, 2));
    mag_v2 = sqrt(sum(v2.^2, 2));

    % Calculate the cosine of the angle
    denominator = mag_v1 .* mag_v2;
    % Avoid division by zero or near-zero magnitudes (where angle is undefined or unstable)
    epsilon = 1e-10; % A small threshold
    valid_mask = denominator > epsilon;

    cos_theta = zeros(size(dot_product)); % Initialize with zeros

    % Calculate cosine only for valid entries (where magnitudes are non-negligible)
    cos_theta(valid_mask) = dot_product(valid_mask) ./ denominator(valid_mask);

    % Clamp values to [-1, 1] to prevent complex results from acos due to floating-point inaccuracies
    cos_theta = max(-1, min(1, cos_theta));

    % Calculate angle in radians only for valid entries
    angles_radians = zeros(size(dot_product)); % Initialize
    angles_radians(valid_mask) = acos(cos_theta(valid_mask));

    % Convert radians to degrees
    angles_degrees = rad2deg(angles_radians);

    % Handle cases where the angle was undefined (denominator <= epsilon)
    % They currently result in 0 degrees. Consider setting to NaN if that's more appropriate.
    % angles_degrees(~valid_mask) = NaN; % Uncomment this line if NaN is preferred for undefined angles
end