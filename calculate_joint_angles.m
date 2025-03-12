function joint_angles = calculate_joint_angles(finger_kinematics)
    % Input: finger_kinematics <5x7 cell> with <4000x69> data
    % Output: joint_angles <5x7 cell> with angles for each finger/joint
    
    % Marker assignments
    palm = 20;  % Base marker
    finger_markers = {
        [20, 17, 18, 19],    % Thumb (MCP, PIP, DIP)
        [20, 1, 5, 6, 7],   % Index
        [20, 2, 8, 9, 10],  % Middle
        [20, 3, 11, 12, 13],% Ring
        [20, 4, 14, 15, 16] % Little
    };
    
    joint_angles = cell(5, 7);
    
    for trial = 1:5
        for task = 1:7
            % Extract position data (4000 x 69)
            pos_data = finger_kinematics{trial, task};
            num_frames = size(pos_data, 1);
            
            % Initialize angle storage (MCP, PIP, DIP for 5 fingers)
            angles = zeros(num_frames, 15); % 5 fingers x 3 joints
            
            for finger = 1:5
                markers = finger_markers{finger};
                for frame = 1:num_frames
                    % Get 3D positions
                    p0 = pos_data(frame, (markers(1)-1)*3 + 1 : (markers(1)-1)*3 + 3); % Base
                    p1 = pos_data(frame, (markers(2)-1)*3 + 1 : (markers(2)-1)*3 + 3);
                    p2 = pos_data(frame, (markers(3)-1)*3 + 1 : (markers(3)-1)*3 + 3);
                    p3 = pos_data(frame, (markers(4)-1)*3 + 1 : (markers(4)-1)*3 + 3);
                    
                    % Calculate vectors
                    v1 = p1 - p0; % MCP vector
                    v2 = p2 - p1; % PIP vector
                    v3 = p3 - p2; % DIP vector
                    
                    % Calculate angles using dot product
                    angles(frame, (finger-1)*3 + 1) = acosd(dot(v1,v2)/(norm(v1)*norm(v2))); % MCP
                    angles(frame, (finger-1)*3 + 2) = acosd(dot(v2,v3)/(norm(v2)*norm(v3))); % PIP
                    if finger == 1  % Thumb has only 2 joints
                        angles(frame, (finger-1)*3 + 3) = NaN;
                    else
                        angles(frame, (finger-1)*3 + 3) = acosd(dot(v3,p3-p1)/(norm(v3)*norm(p3-p1))); % DIP
                    end
                end
            end
            joint_angles{trial, task} = angles;
        end
    end
end