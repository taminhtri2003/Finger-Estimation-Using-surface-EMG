% Load the .mat file
data = load('s1_full.mat'); % Replace 'your_data_file.mat' with the actual file name

% Define muscle names and task names for better readability
muscle_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};
task_names = {'Thumb Flex/Ext', 'Index Flex/Ext', 'Middle Flex/Ext', 'Ring Flex/Ext', 'Little Flex/Ext', 'All Fingers Flex/Ext', 'Random Flex/Ext'};
joint_angle_names = {'Thumb 1', 'Thumb 2', 'Index 1', 'Index 2', 'Index 3', 'Middle 1', 'Middle 2', 'Middle 3', 'Ring 1', 'Ring 2', 'Ring 3', 'Little 1', 'Little 2', 'Little 3'};

% Loop through each trial and task
for trial = 1:size(data.dsfilt_emg, 1)
    for task = 1:size(data.dsfilt_emg, 2)

        % Extract the EMG and joint angle data for the current trial and task
        emg_data = data.dsfilt_emg{trial, task};
        joint_angles = data.joint_angles{trial, task};

        % Create a figure for the current trial and task
        figure;
        % --- EMG Visualization ---
        subplot(2, 1, 1); % Create a subplot for EMG data (top half)
        hold on;
        time = (1:size(emg_data, 1)) / 2000;   % Assuming sampling rate was downsampled to 1kHz, create a time vector in seconds

        for i = 1:length(muscle_names)
            plot(time, emg_data(:, i), 'LineWidth', 1.5);
        end
        hold off;
        xlabel('Time (s)');
        ylabel('EMG Amplitude');
        title(['Trial ' num2str(trial) ', Task: ' task_names{task} ' - EMG Activity']); % Moved title here
        legend(muscle_names, 'Location', 'best');
        grid on;

        % --- Joint Angle Visualization ---
        subplot(2, 1, 2); % Create a subplot for joint angle data (bottom half)
        hold on;

        % Determine which joint angles to plot based on the task
        relevant_angles = [];
        switch task
            case 1 % Thumb
                relevant_angles = 1:2;
            case 2 % Index
                relevant_angles = 3:5;
            case 3 % Middle
                relevant_angles = 6:8;
            case 4 % Ring
                relevant_angles = 9:11;
            case 5 % Little
                relevant_angles = 12:14;
            case 6 % All fingers
                relevant_angles = 1:14;
            case 7 % Random
                relevant_angles = 1:14; % Plot all for random movements
        end
        
         for i = relevant_angles
                plot(time, joint_angles(:, i), 'LineWidth', 1.5);
         end
        
        hold off;
        xlabel('Time (s)');
        ylabel('Joint Angle (degrees)');  % Assuming angles are in degrees
        title(['Trial ' num2str(trial) ', Task: ' task_names{task} ' - Joint Angles']); % Moved title here
        legend(joint_angle_names(relevant_angles), 'Location', 'best'); % Show legend only for relevant angles
        grid on;
       
        %Adjust plot to use all available space. No longer needed with improved positioning.
        % Removed the loop that adjusted subplot positions

    end
end