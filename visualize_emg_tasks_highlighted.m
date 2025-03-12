function visualize_emg_tasks_highlighted(mat_file_path)
% Visualizes EMG data, highlighting each task with a colored box.
% Overlays EMG signals from all tasks for each channel.

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
colors = jet(num_tasks);  % Generate a colormap for the tasks

for trial = 1:num_trials
    figure;
    sgtitle(sprintf('Trial %d - EMG with Task Highlighting', trial));

    for emg_channel = 1:length(emg_labels)
        subplot(length(emg_labels), 1, emg_channel);
        hold on;
        
        % Find maximum y-value across all tasks for this channel & trial (for box height)
        max_y = -Inf;
        for task = 1:num_tasks
            try
                emg_data = dsfilt_emg{trial, task};
            catch
                continue; %Skip task on error
            end
            max_y = max(max_y, max(emg_data(:, emg_channel)));
        end
        
        %Precalculate min_y for all tasks
        min_y = Inf;
        for task = 1:num_tasks;
            try
                emg_data = dsfilt_emg{trial,task};
            catch
                continue;
            end
             min_y = min(min_y, min(emg_data(:, emg_channel)));
        end
        
        % Draw colored boxes for each task
        for task = 1:num_tasks
            try
               emg_data = dsfilt_emg{trial,task};
            catch ME
                fprintf('Error at Trial %d, Channel %d, Task %d: %s\n', trial, emg_channel, task, ME.message);
                continue;
            end
             num_samples = size(emg_data, 1);
             time_vector = (0:num_samples-1)';
             
             % Define the rectangle for the task
            x_start = (task - 1) * num_samples; %Each task on a "shifted timeline"
            x_end   = task * num_samples;
            
            rectangle('Position', [x_start, min_y, num_samples, max_y-min_y], ...
                      'FaceColor', [colors(task, :) 0.2], 'EdgeColor', 'none'); % 0.2 alpha for transparency

            % Overlay the EMG signal (shifted in time)
            plot(time_vector + x_start, emg_data(:, emg_channel), 'Color', colors(task,:), 'DisplayName', sprintf('Task %d', task));

            %Add task number as text
            text(x_start + num_samples / 2, max_y * 0.9, sprintf('Task %d', task), ...
                 'HorizontalAlignment', 'center', 'Color', 'black', 'FontWeight', 'bold');

        end


        hold off;
        title(emg_labels{emg_channel});
        ylabel('Amplitude');
        xlabel('Time (shifted by task)');
        % No legend needed, as tasks are indicated by boxes.
        grid on;

    end
      % Link x-axes for synchronized zooming/panning
        all_axes = findobj(gcf, 'Type', 'axes');
        linkaxes(all_axes, 'x');
end

end
