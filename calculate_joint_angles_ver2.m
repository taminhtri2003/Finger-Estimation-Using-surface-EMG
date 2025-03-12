function joint_angles_cell = calculate_joint_angles_ver2(finger_kinematics)
%CALCULATE_JOINT_ANGLES Calculates joint angles from marker positions (in degrees).
%
%   joint_angles_cell = calculate_joint_angles(finger_kinematics)
%   calculates joint angles from the finger_kinematics data, which is a
%   5x7 cell array of marker position data (x, y, z coordinates).
%   The angles are returned in DEGREES.
%
%   Args:
%       finger_kinematics: A 5x7 cell array. Each cell contains a 4000x69
%                         matrix representing marker positions.  Each row is a
%                         time point, and the columns are (x, y, z) for 23
%                         markers.
%
%   Returns:
%       joint_angles_cell: A 5x7 cell array.  Each cell contains a matrix
%                        where each row is a time point and each column
%                        represents a calculated joint angle (in degrees).
%                        The order of angles is as described in the problem
%                        statement.

    joint_angles_cell = cell(size(finger_kinematics));

    for trial = 1:size(finger_kinematics, 1)
        for task = 1:size(finger_kinematics, 2)
            kinematics_data = finger_kinematics{trial, task};
            num_timepoints = size(kinematics_data, 1);
            joint_angles = zeros(num_timepoints, 14); % 14 angles total

            % --- Thumb (2 angles) ---
            % Angle 1: 20-17 to 17-18
            joint_angles(:, 1) = calculate_angle_degrees(kinematics_data, 20, 17, 18);
            % Angle 2: 17-18 to 18-19
            joint_angles(:, 2) = calculate_angle_degrees(kinematics_data, 17, 18, 19);

            % --- Index (3 angles) ---
            % Angle 3: 20-1 to 1-5
            joint_angles(:, 3) = calculate_angle_degrees(kinematics_data, 20, 1, 5);
            % Angle 4: 1-5 to 5-6
            joint_angles(:, 4) = calculate_angle_degrees(kinematics_data, 1, 5, 6);
            % Angle 5: 5-6 to 6-7
            joint_angles(:, 5) = calculate_angle_degrees(kinematics_data, 5, 6, 7);

            % --- Middle (3 angles) ---
            % Angle 6: 20-2 to 2-8
            joint_angles(:, 6) = calculate_angle_degrees(kinematics_data, 20, 2, 8);
            % Angle 7: 2-8 to 8-9
            joint_angles(:, 7) = calculate_angle_degrees(kinematics_data, 2, 8, 9);
            % Angle 8: 8-9 to 9-10
            joint_angles(:, 8) = calculate_angle_degrees(kinematics_data, 8, 9, 10);

            % --- Ring (3 angles) ---
            % Angle 9: 20-3 to 3-11
            joint_angles(:, 9) = calculate_angle_degrees(kinematics_data, 20, 3, 11);
            % Angle 10: 3-11 to 11-12
            joint_angles(:, 10) = calculate_angle_degrees(kinematics_data, 3, 11, 12);
            % Angle 11: 11-12 to 12-13
            joint_angles(:, 11) = calculate_angle_degrees(kinematics_data, 11, 12, 13);

            % --- Little (3 angles) ---
            % Angle 12: 20-4 to 4-14
            joint_angles(:, 12) = calculate_angle_degrees(kinematics_data, 20, 4, 14);
            % Angle 13: 4-14 to 14-15
            joint_angles(:, 13) = calculate_angle_degrees(kinematics_data, 4, 14, 15);
            % Angle 14: 14-15 to 15-16
            joint_angles(:, 14) = calculate_angle_degrees(kinematics_data, 14, 15, 16);


            joint_angles_cell{trial, task} = joint_angles;
        end
    end
end



function angles_degrees = calculate_angle_degrees(kinematics_data, marker1_idx, marker2_idx, marker3_idx)
%CALCULATE_ANGLE_DEGREES Calculates the angle between three 3D points (in degrees).
%
%   angles_degrees = calculate_angle_degrees(kinematics_data, marker1_idx, marker2_idx, marker3_idx)
%   calculates the angle between the vectors formed by marker1-marker2 and
%   marker2-marker3 for each time point, returning the result in degrees.
%
%   Args:
%       kinematics_data: A matrix of marker positions (4000x69).
%       marker1_idx:     Index of the first marker (1-23).
%       marker2_idx:     Index of the second marker (1-23).
%       marker3_idx:     Index of the third marker (1-23).
%
%   Returns:
%       angles_degrees:  A column vector of angles (in degrees) for each time point.

    % Extract marker coordinates (x, y, z)
    marker1 = kinematics_data(:, (marker1_idx-1)*3 + (1:3));
    marker2 = kinematics_data(:, (marker2_idx-1)*3 + (1:3));
    marker3 = kinematics_data(:, (marker3_idx-1)*3 + (1:3));

    % Calculate vectors
    v1 = marker1 - marker2;
    v2 = marker3 - marker2;

    % Calculate dot product and magnitudes
    dot_product = sum(v1 .* v2, 2);
    mag_v1 = sqrt(sum(v1.^2, 2));
    mag_v2 = sqrt(sum(v2.^2, 2));

    % Calculate angle (in radians) and convert to degrees
    angles_radians = acos(max(-1, min(1, dot_product ./ (mag_v1 .* mag_v2))));
    angles_degrees = rad2deg(angles_radians);  % Convert radians to degrees
end