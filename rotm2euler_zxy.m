function euler_angles_deg = rotm2euler_zxy(R)
%ROTM2EULER_ZXY Converts a 3x3 rotation matrix to ZXY Euler angles (in degrees).
%
%   euler_angles_deg = rotm2euler_zxy(R)
%
%   Input:
%       R: A 3x3 rotation matrix or a 3x3xN array of rotation matrices.
%
%   Output:
%       euler_angles_deg: An Nx3 matrix where each row is [angle_Z, angle_X, angle_Y]
%                         in degrees. The angles represent rotations about
%                         Z, then the new X', then the new Y''.
%                         angle_Z: Flexion/Extension
%                         angle_X: Abduction/Adduction
%                         angle_Y: Axial Rotation
%
%   Note: This implementation handles potential gimbal lock when the second
%         rotation (about X') is +/- 90 degrees.

    euler_angles_rad = zeros(size(R, 3), 3);

    for i = 1:size(R, 3)
        Ri = R(:, :, i);

        % Calculate angle_X (rotation about new X')
        % sin(angle_X) = -R(3,2)
        sin_angle_x = -Ri(3, 2);

        % Check for gimbal lock (angle_X is +/- 90 degrees)
        if abs(sin_angle_x) > 0.9999
            % Gimbal lock case
            angle_x = sign(sin_angle_x) * pi / 2; % +/- 90 degrees

            % Set angle_Y (rotation about Y'') to 0
            angle_y = 0;

            % Calculate angle_Z (rotation about Z) using atan2
            % R(1,1) = cos(angle_z)*cos(angle_y) = cos(angle_z) * 1
            % R(2,1) = sin(angle_z)*cos(angle_y) = sin(angle_z) * 1
            angle_z = atan2(Ri(2, 1), Ri(1, 1));

        else
            % Normal case
            angle_x = asin(sin_angle_x);

            % Calculate angle_Z using atan2
            % R(1,2) = cos(angle_z)*sin(angle_x)*sin(angle_y) + sin(angle_z)*cos(angle_y) -- NO (this is ZYX)
            % For ZXY:
            % R(3,1) = cos(angle_x)*sin(angle_z)
            % R(3,3) = cos(angle_x)*cos(angle_z)
            angle_z = atan2(Ri(3, 1), Ri(3, 3));

            % Calculate angle_Y using atan2
            % R(1,2) = sin(angle_x)
            % R(2,2) = cos(angle_x)*cos(angle_y)
            angle_y = atan2(Ri(1, 2), Ri(2, 2));
        end

        euler_angles_rad(i, :) = [angle_z, angle_x, angle_y];
    end

    euler_angles_deg = rad2deg(euler_angles_rad); % Convert to degrees
end
