function joint_angles_cell = calculate_joint_angles_acs(finger_kinematics)
%CALCULATE_JOINT_ANGLES_ACS Calculates 3D joint angles using Anatomical Coordinate Systems (ACS).
%
%   joint_angles_cell = calculate_joint_angles_acs(finger_kinematics)
%   calculates 3D joint angles from the finger_kinematics data using ACS.
%   Angles are returned in DEGREES as Euler angles (ZXY sequence).
%
%   Args:
%       finger_kinematics: A 5x7 cell array. Each cell contains a 4000x69
%                         matrix representing marker positions (x, y, z for 23 markers).
%
%   Returns:
%       joint_angles_cell: A 5x7 cell array. Each cell contains a matrix
%                        where each row is a time point. Columns represent
%                        calculated joint angles (in degrees).
%                        Example Columns for Index Finger:
%                           1-3: MCP [Flex/Ext(Z), Ab/Ad(X), AxialRot(Y)]
%                           4-6: PIP [Flex/Ext(Z), Ab/Ad(X), AxialRot(Y)] (Often only Flex/Ext is reliable)
%                           7-9: DIP [Flex/Ext(Z), Ab/Ad(X), AxialRot(Y)] (Often only Flex/Ext is reliable)
%                        (Order needs to be defined for all fingers/thumb).
%
%   Dependencies: Requires helper function 'rotm2euler_zxy.m'.

    joint_angles_cell = cell(size(finger_kinematics));

    for trial = 1:size(finger_kinematics, 1)
        for task = 1:size(finger_kinematics, 2)
            kinematics_data = finger_kinematics{trial, task};
            num_timepoints = size(kinematics_data, 1);

            % --- Preallocate Output Matrix ---
            % Determine the total number of angles to be calculated.
            % Example: 3(MCP)+3(PIP)+3(DIP) for Index = 9 angles
            % Need to define for all fingers + thumb. Let's assume 9 for index for now.
            num_angles_per_finger = 9; % MCP(3) + PIP(3) + DIP(3)
                                       % Adjust if simplifying PIP/DIP to 1 DoF
            num_total_angles = num_angles_per_finger * 4 + 6; % Approx: 4 fingers * 9 + Thumb(6?) = 42
            % For demonstration, let's just calculate Index finger angles (9)
            num_calculated_angles = 9;
            joint_angles = zeros(num_timepoints, num_calculated_angles);

            % --- Define Coordinate Systems for Each Time Point ---
            for t = 1:num_timepoints
                % Get marker positions at time t
                P = reshape(kinematics_data(t, :), 3, 23)'; % Reshape to 23x3 matrix

                % Define Hand ACS (using markers 20, 21, 22)
                % Origin: P20
                % Z_hand: Normal to plane 21-20-22 (approx dorsal/palmar axis)
                % Y_hand: Approx longitudinal axis
                % X_hand: Approx radial/ulnar axis
                try
                    v21_20 = P(21,:) - P(20,:);
                    v22_20 = P(22,:) - P(20,:);
                    Z_hand_temp = cross(v21_20, v22_20);
                    if norm(Z_hand_temp) < 1e-6; error('Hand markers collinear Z'); end
                    Z_hand = Z_hand_temp / norm(Z_hand_temp);

                    % Define Y axis (e.g., pointing roughly towards fingers)
                    % Could use P23 or midpoint of MCPs. Let's use P(1,:)-P(20,:) as proxy
                    Y_hand_temp = P(1,:) - P(20,:); % Crude approximation
                    if norm(Y_hand_temp) < 1e-6; error('Hand markers collinear Y'); end

                    X_hand = cross(Y_hand_temp, Z_hand);
                    if norm(X_hand) < 1e-6; error('Hand markers collinear X'); end
                    X_hand = X_hand / norm(X_hand);

                    Y_hand = cross(Z_hand, X_hand); % Ensure orthogonality
                    Y_hand = Y_hand / norm(Y_hand); % Normalize just in case

                    R_hand = [X_hand', Y_hand', Z_hand']; % Store as rotation matrix (axes as columns)
                catch ME
                    warning('Timepoint %d: Could not define Hand ACS: %s. Skipping.', t, ME.message);
                    R_hand = nan(3); % Use NaN to indicate failure
                end

                % --- Define Index Finger Segment ACS ---
                % Proximal Phalanx (Markers 1 -> 5)
                try
                    [R_index_prox, Z_index_prox_axis] = define_segment_acs(P(1,:), P(5,:), P(6,:), R_hand(:,3)); % Use P6 for plane normal, Z_hand as fallback Z ref
                catch ME
                     warning('Timepoint %d: Could not define Index Prox ACS: %s. Skipping.', t, ME.message);
                     R_index_prox = nan(3); Z_index_prox_axis = nan(1,3);
                end

                % Middle Phalanx (Markers 5 -> 6)
                try
                    [R_index_mid, Z_index_mid_axis] = define_segment_acs(P(5,:), P(6,:), P(7,:), Z_index_prox_axis); % Use P7 for plane normal, Prox Z as fallback Z ref
                catch ME
                     warning('Timepoint %d: Could not define Index Mid ACS: %s. Skipping.', t, ME.message);
                     R_index_mid = nan(3); Z_index_mid_axis = nan(1,3);
                end

                % Distal Phalanx (Markers 6 -> 7)
                try
                    [R_index_dist, ~] = define_segment_acs(P(6,:), P(7,:), P(7,:), Z_index_mid_axis); % Use P7 again (no further marker), Mid Z as fallback Z ref
                catch ME
                     warning('Timepoint %d: Could not define Index Dist ACS: %s. Skipping.', t, ME.message);
                     R_index_dist = nan(3);
                end


                % --- Calculate Relative Orientations (Joint Angles) ---
                % Index MCP (Hand vs Proximal Phalanx)
                if ~any(isnan(R_hand(:))) && ~any(isnan(R_index_prox(:)))
                    R_mcp = R_hand' * R_index_prox; % Rotation of Proximal relative to Hand
                    angles_mcp = rotm2euler_zxy(R_mcp); % [Flex/Ext(Z), Ab/Ad(X), AxialRot(Y)]
                    joint_angles(t, 1:3) = angles_mcp;
                else
                    joint_angles(t, 1:3) = nan;
                end

                % Index PIP (Proximal vs Middle Phalanx)
                if ~any(isnan(R_index_prox(:))) && ~any(isnan(R_index_mid(:)))
                    R_pip = R_index_prox' * R_index_mid; % Rotation of Middle relative to Proximal
                    angles_pip = rotm2euler_zxy(R_pip); % [Flex/Ext(Z), Ab/Ad(X), AxialRot(Y)]
                    joint_angles(t, 4:6) = angles_pip;
                    % Often PIP Ab/Ad and AxialRot are small/noisy, consider only Flex/Ext (angle_pip(1))
                    % joint_angles(t, 4) = angles_pip(1); % If only storing 1 DoF for PIP
                else
                    joint_angles(t, 4:6) = nan;
                end

                % Index DIP (Middle vs Distal Phalanx)
                 if ~any(isnan(R_index_mid(:))) && ~any(isnan(R_index_dist(:)))
                    R_dip = R_index_mid' * R_index_dist; % Rotation of Distal relative to Middle
                    angles_dip = rotm2euler_zxy(R_dip); % [Flex/Ext(Z), Ab/Ad(X), AxialRot(Y)]
                    joint_angles(t, 7:9) = angles_dip;
                    % Often DIP Ab/Ad and AxialRot are small/noisy, consider only Flex/Ext (angle_dip(1))
                     % joint_angles(t, 7) = angles_dip(1); % If only storing 1 DoF for DIP
                 else
                    joint_angles(t, 7:9) = nan;
                end

            end % End loop over timepoints

            % Store results for the current trial/task
            % Make sure the size matches the preallocation based on all fingers/thumb
            joint_angles_cell{trial, task} = joint_angles(:, 1:num_calculated_angles); % Store only calculated index angles for now

        end % End loop over tasks
    end % End loop over trials
end

% --- Helper Function to Define Segment ACS ---
function [R_segment, Z_axis] = define_segment_acs(p_prox, p_dist, p_next, z_ref_fallback)
% Defines ACS for a segment based on proximal and distal markers.
% Uses p_next to help define the flexion plane/axis.
% z_ref_fallback is used if p_next is collinear or identical.

    % Y-axis: Along the segment
    Y_axis = p_dist - p_prox;
    if norm(Y_axis) < 1e-6; error('Segment markers coincident'); end
    Y_axis = Y_axis / norm(Y_axis);

    % Define Z-axis (Flexion/Extension axis) based on plane normal
    % Use the plane formed by p_prox, p_dist, p_next
    vec1 = p_dist - p_prox; % = Y_axis * norm(p_dist - p_prox)
    vec2 = p_next - p_dist;
    plane_normal = cross(vec1, vec2);

    if norm(plane_normal) < 1e-6 % Markers are collinear or p_next=p_dist
        % Fallback: Use the reference Z axis from the proximal segment/hand
        % Project z_ref_fallback onto the plane normal to Y_axis
         if any(isnan(z_ref_fallback)) || norm(z_ref_fallback) < 1e-6
             error('Fallback Z reference is invalid');
         end
         Z_axis = z_ref_fallback - dot(z_ref_fallback, Y_axis) * Y_axis;
         if norm(Z_axis) < 1e-6
             % If z_ref was parallel to Y_axis, pick an arbitrary perpendicular
             % Find an arbitrary vector not parallel to Y_axis
             temp_vec = [1 0 0];
             if abs(dot(temp_vec, Y_axis)) > 0.99
                 temp_vec = [0 1 0];
             end
             Z_axis = cross(Y_axis, temp_vec);
         end

    else
        % Z-axis is perpendicular to the plane normal and the Y-axis
        % This assumes Z is roughly cross(Y, plane_normal) direction
         Z_axis = cross(Y_axis, plane_normal); % Check direction later if needed
%        Z_axis = plane_normal; % Alternative: Z is normal to the plane
    end

     if norm(Z_axis) < 1e-6; error('Could not define Z axis'); end
     Z_axis = Z_axis / norm(Z_axis);

    % X-axis: Orthogonal to Y and Z
    X_axis = cross(Y_axis, Z_axis);
    X_axis = X_axis / norm(X_axis); % Normalize just in case

    % Ensure Z is orthogonal to refined X and Y
    Z_axis = cross(X_axis, Y_axis);
    Z_axis = Z_axis / norm(Z_axis);

    R_segment = [X_axis', Y_axis', Z_axis']; % Axes as columns
end
