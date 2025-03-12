function visualize_trial_data(mat_file_path)
% Visualizes EMG and Kinematics data from a .mat file.
%
% Args:
%     mat_file_path: Path to the .mat file.

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

% Check cell array dimensions
if ~isequal(size(dsfilt_emg), [5, 7]) || ~isequal(size(finger_kinematics), [5, 7])
    fprintf('Error: Unexpected cell array dimensions. Expected [5, 7].\n');
    return;
end

emg_labels = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};
num_kinematics = size(finger_kinematics{1,1}, 2); % Get number of kinematics columns

for trial = 1:size(dsfilt_emg, 1)
    for task = 1:size(dsfilt_emg, 2)
        try
            emg_data = dsfilt_emg{trial, task};
            kinematics_data = finger_kinematics{trial, task};
        catch ME
             fprintf('Error accessing data at trial %d, task %d: %s\n', trial, task, ME.message);
            continue
        end
        
        % --- EMG Visualization ---
        figure; % Create a new figure for each trial/task

        subplot(2,1,1); % Top subplot for EMG
        hold on;
        for i = 1:size(emg_data, 2)
            plot(emg_data(:, i), 'DisplayName', emg_labels{i});
        end
        hold off;
        title(sprintf('EMG Data - Trial %d, Task %d', trial, task));
        xlabel('Time (samples)');
        ylabel('Amplitude');
        legend('Location', 'best'); % Add a legend
        grid on;

        % --- Kinematics Visualization ---
        subplot(2,1,2); % Bottom subplot for kinematics
        hold on;
       
        % Plot x, y, and z components for each marker (assuming groups of 3)
        num_markers = num_kinematics / 3;
        if mod(num_kinematics, 3) ~= 0
              fprintf('Warning: Number of kinematics columns (%d) is not a multiple of 3.  Plotting may be inaccurate.\n', num_kinematics);
              num_markers = floor(num_kinematics/3);  % Plot what we can
        end

        for i = 1:num_markers
             start_col = (i - 1) * 3 + 1;
             end_col = start_col + 2;
             
              % Check if end_col exceeds the bounds of kinematics_data
             if end_col > size(kinematics_data, 2)
                fprintf('Warning: Skipping marker %d due to insufficient columns in kinematics data.\n', i);
                break; % Exit the marker loop
             end

            plot3(kinematics_data(:, start_col), kinematics_data(:, start_col + 1), kinematics_data(:, start_col + 2), ...
                  'DisplayName', sprintf('Marker %d', i));
        end
        hold off;

        title(sprintf('Kinematics Data - Trial %d, Task %d', trial, task));
        xlabel('X Position');
        ylabel('Y Position');
        zlabel('Z Position');
        legend('Location', 'best');
        grid on;
        view(3); % Set a 3D view

        % --- Optional: Add overall figure title ---
        sgtitle(sprintf('Trial %d, Task %d Visualization', trial, task)); % Super title
    end
end

end
