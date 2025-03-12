function compare_tasks_visualization(mat_file_path)
% Visualizes and compares EMG and Kinematics data across different tasks
% from a .mat file.  Focuses on inter-task comparisons.

try
    load(mat_file_path);
catch ME
    fprintf('Error loading .mat file: %s\n', ME.message);
    return;
end

if ~exist('dsfilt_emg', 'var') || ~exist('finger_kinematics', 'var')
    fprintf('Error: ''dsfilt_emg'' or ''finger_kinematics'' not found.\n');
    return;
end
if ~isequal(size(dsfilt_emg), [5, 7]) || ~isequal(size(finger_kinematics), [5, 7])
    fprintf('Error: Unexpected cell array dimensions. Expected [5, 7].\n');
    return;
end

emg_labels = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};
num_tasks = size(dsfilt_emg, 2);
num_trials = size(dsfilt_emg, 1);
num_kinematics = size(finger_kinematics{1,1}, 2);

% --- EMG Comparison Across Tasks ---
for trial = 1:num_trials
    figure; % New figure for each trial
    sgtitle(sprintf('EMG Comparison - Trial %d', trial));

    for emg_channel = 1:length(emg_labels)
        subplot(length(emg_labels), 1, emg_channel);
        hold on;

        for task = 1:num_tasks
            try
                emg_data = dsfilt_emg{trial, task};
            catch ME
                fprintf('Error at Trial %d, Task%d, EMG channel %d: %s', trial, task, emg_channel, ME.message);
                continue;
            end
            
            % Use a consistent time vector (handling potential length differences)
             num_samples = size(emg_data, 1);
             time_vector = (0:num_samples-1)';

            plot(time_vector, emg_data(:, emg_channel), 'DisplayName', sprintf('Task %d', task));
        end

        hold off;
        title(emg_labels{emg_channel});
        ylabel('Amplitude');
        xlabel('Time (samples)');
        legend('Location', 'best');
        grid on;
    end
     % Link x-axes for synchronized zooming/panning
    all_axes = findobj(gcf, 'Type', 'axes');
    linkaxes(all_axes, 'x');
end


% --- Kinematics Comparison Across Tasks (3D Trajectory) ---
for trial = 1:num_trials
    figure;  % New figure for each trial
    sgtitle(sprintf('Kinematics Comparison (3D Trajectory) - Trial %d', trial));
    hold on;

   num_markers = num_kinematics / 3;
    if mod(num_kinematics, 3) ~= 0
      fprintf('Warning: Number of kinematics columns (%d) is not a multiple of 3. Plotting may be inaccurate.\n', num_kinematics);
      num_markers = floor(num_kinematics/3);
   end
    for task = 1:num_tasks
      try
          kinematics_data = finger_kinematics{trial, task};
      catch ME
          fprintf('Error at Trial %d, Task %d: %s', trial, task, ME.message);
          continue;
      end

      for i = 1:num_markers
            start_col = (i-1)*3 + 1;
            end_col = start_col + 2;
            
            if end_col > size(kinematics_data,2)
                fprintf('Warning: Insufficient columns in kinematics. Skipping.\n');
                break;
            end
            
            plot3(kinematics_data(:, start_col), kinematics_data(:, start_col+1), kinematics_data(:, start_col+2), ...
                  'DisplayName', sprintf('Task %d, Marker %d', task, i), 'LineWidth', 1.5); 
      end
    end
    hold off;
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('3D Kinematics Trajectories');
    legend('Location', 'best');
    grid on;
    view(3);
end

% --- Kinematics Comparison Across Tasks (Separate X, Y, Z) ---
for trial = 1:num_trials
     for marker = 1:num_markers % Iterate through each marker
        figure;
        sgtitle(sprintf('Kinematics Comparison (X, Y, Z) - Trial %d, Marker %d', trial, marker));

       for coord = 1:3  % 1 for X, 2 for Y, 3 for Z
            subplot(3, 1, coord); % One subplot for each coordinate (X, Y, Z)
            hold on;

            for task = 1:num_tasks
                try
                    kinematics_data = finger_kinematics{trial, task};
                catch ME
                   fprintf('Error at Trial %d, Task %d: %s\n',trial, task, ME.message);
                   continue;
                end
                 start_col = (marker - 1) * 3 + coord; % Calculate correct column index
                
                 if start_col > size(kinematics_data, 2)
                    fprintf('Warning: start_col exceeds dimensions. Skipping.\n');
                    continue;  % Skip this task
                end
                  % Use a consistent time vector
                 num_samples = size(kinematics_data, 1);
                 time_vector = (0:num_samples-1)';
                
                plot(time_vector, kinematics_data(:, start_col), 'DisplayName', sprintf('Task %d', task));

            end
            hold off;
            if coord == 1
               title('X Position');
            elseif coord == 2
                title('Y Position');
            else
                title('Z Position');
            end

            ylabel('Position');
            xlabel('Time (samples)');
            legend('Location','best');
            grid on;

       end
           % Link x-axes within each marker's figure
        all_axes = findobj(gcf, 'Type', 'axes');
        linkaxes(all_axes, 'x');

     end
end

end

