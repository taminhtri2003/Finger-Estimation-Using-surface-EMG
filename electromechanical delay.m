% Load the .mat file
load('s1_full.mat'); % Replace 'your_data_file.mat' with the actual file name

% Define muscle names for clarity
muscle_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};

% Define joint angle names for clarity
joint_angle_names = {
    'Thumb 1 (20-17 to 17-18)', 'Thumb 2 (17-18 to 18-19)',
    'Index 1 (20-1 to 1-5)', 'Index 2 (1-5 to 5-6)', 'Index 3 (5-6 to 6-7)',
    'Middle 1 (20-2 to 2-8)', 'Middle 2 (2-8 to 8-9)', 'Middle 3 (8-9 to 9-10)',
    'Ring 1 (20-3 to 3-11)', 'Ring 2 (3-11 to 11-12)', 'Ring 3 (11-12 to 12-13)',
    'Little 1 (20-4 to 4-14)', 'Little 2 (4-14 to 14-15)', 'Little 3 (14-15 to 15-16)'
};

% Assuming a sampling frequency (adjust as needed)
sampling_frequency = 2000; % Hz (e.g., if 1000 samples per second)

% Function to calculate electromechanical delay (EMD)
function emd = calculateEMD(emg, joint_angle, sampling_frequency)
    % Find the onset of muscle activation (EMG)
    emg_threshold = mean(emg) + 3 * std(emg); % Example: threshold at 3 standard deviations above mean
    emg_onset_indices = find(emg > emg_threshold, 1, 'first');

    if isempty(emg_onset_indices)
        emd = NaN; % No EMG onset detected
        return;
    end

    emg_onset_time = emg_onset_indices / sampling_frequency;

    % Find the onset of joint movement (kinematics)
    joint_angle_derivative = diff(joint_angle);
    joint_angle_derivative_threshold = mean(abs(joint_angle_derivative)) + 3 * std(abs(joint_angle_derivative)); % Example
    joint_onset_indices = find(abs(joint_angle_derivative) > joint_angle_derivative_threshold, 1, 'first');

    if isempty(joint_onset_indices)
        emd = NaN; % No joint movement onset detected
        return;
    end

    joint_onset_time = joint_onset_indices / sampling_frequency;

    % Calculate the EMD
    emd = joint_onset_time - emg_onset_time;
end

% Loop through trials and tasks
for trial = 1:size(dsfilt_emg, 1)
    for task = 1:size(dsfilt_emg, 2)
        % Extract EMG and joint angle data for the current trial and task
        current_emg = dsfilt_emg{trial, task};
        current_joint_angles = joint_angles{trial, task};

        % Loop through muscles and joint angles and calculate EMD
        for muscle_idx = 1:size(current_emg, 2)
            for joint_idx = 1:size(current_joint_angles, 2)
                % Calculate EMD for the current muscle and joint angle
                emd = calculateEMD(current_emg(:, muscle_idx), current_joint_angles(:, joint_idx), sampling_frequency);

                % Display the results
                fprintf('Trial: %d, Task: %d, Muscle: %s, Joint Angle: %s, EMD: %.4f seconds\n', ...
                    trial, task, muscle_names{muscle_idx}, joint_angle_names{joint_idx}, emd);
            end
        end
    end
end