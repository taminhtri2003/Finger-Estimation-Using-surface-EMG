function visualize_kinematics_per_marker_task_highlighted(mat_file_path)
% Visualizes kinematics data per marker, highlighting each task with boxes.
% Compares X, Y, and Z coordinates of each marker across tasks.

try
    load(mat_file_path);
catch ME
    fprintf('Error loading .mat file: %s\n', ME.message);
    return;
end

if ~exist('finger_kinematics', 'var')
    fprintf('Error: ''finger_kinematics'' not found in the .mat file.\n');
    return;
end
if ~isequal(size(finger_kinematics), [5, 7])
    fprintf('Error: Unexpected cell array dimensions. Expected [5, 7].\n');
    return;
end

num_tasks = size(finger_kinematics, 2);
num_trials = size(finger_kinematics, 1);
num_kinematics = size(finger_kinematics{1,1}, 2);
num_markers = num_kinematics / 3;
if mod(num_kinematics, 3) ~= 0
  fprintf('Warning: Number of kinematics is not a multiple of 3.\n');
  num_markers = floor(num_kinematics/3);
end
colors = jet(num_tasks);  % Colors for each task

for trial = 1:num_trials
    for marker = 1:num_markers
        figure;
        sgtitle(sprintf('Trial %d, Marker %d - Kinematics Comparison', trial, marker));

        for coord = 1:3  % X, Y, Z
            subplot(3, 1, coord);
            hold on;

             % Find max/min across all tasks for this marker and coordinate
            max_y = -Inf;
            min_y = Inf;
            for task = 1:num_tasks
                try
                    kinematics_data = finger_kinematics{trial, task};
                 catch
                    continue
                end
                start_col = (marker - 1) * 3 + coord;
                if start_col > size(kinematics_data, 2)
                    continue;
                end
                max_y = max(max_y, max(kinematics_data(:, start_col)));
                min_y = min(min_y, min(kinematics_data(:, start_col)));
            end

            for task = 1:num_tasks
                try
                    kinematics_data = finger_kinematics{trial, task};
                catch ME
                    fprintf('Error at Trial %d, Task %d, Marker %d: %s\n', trial, task, marker, ME.message);
                   continue; %skip this task if there's an error
                end
                start_col = (marker - 1) * 3 + coord;  % Correct column
                 if start_col > size(kinematics_data, 2)
                     fprintf('Warning: start_col exceeds kinematic data dimenstion\n');
                     continue;
                 end

                 num_samples = size(kinematics_data, 1);
                 time_vector = (0:num_samples - 1)';

                % Define rectangle position
                x_start = (task - 1) * num_samples;
                x_end = task * num_samples;
                
                % Draw rectangle
                rectangle('Position', [x_start, min_y, num_samples, max_y-min_y], ...
                          'FaceColor', [colors(task, :) 0.2], 'EdgeColor', 'none');

                % Plot the kinematic data (shifted)
                plot(time_vector + x_start, kinematics_data(:, start_col), 'Color', colors(task, :), 'DisplayName', sprintf('Task %d', task));
                
                 % Add task number as text
                text(x_start + num_samples / 2, max_y * 0.9, sprintf('Task %d', task), ...
                     'HorizontalAlignment', 'center', 'Color', 'black','FontWeight', 'bold');
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
            xlabel('Time (shifted by task)');
            % No legend needed - tasks are indicated by boxes and text.
            grid on;
        end
         %Link the x-axes
         all_axes = findobj(gcf,'Type','axes');
         linkaxes(all_axes,'x');
    end
end

end

