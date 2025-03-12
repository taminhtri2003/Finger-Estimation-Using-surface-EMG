function compare_trials_within_tasks(mat_file_path, output_dir)
% Compares EMG and kinematics data between trials within each task.
% Generates plots (line and box plots) for statistical comparison.

try
    load(mat_file_path);
catch ME
    fprintf('Error loading .mat file: %s\n', ME.message);
    return;
end

% --- Input Validation and Setup ---
if ~exist('dsfilt_emg', 'var') || ~exist('finger_kinematics', 'var')
    fprintf('Error: ''dsfilt_emg'' or ''finger_kinematics'' not found.\n');
    return;
end
if ~isequal(size(dsfilt_emg), [5, 7]) || ~isequal(size(finger_kinematics), [5, 7])
    fprintf('Error: Unexpected cell array dimensions. Expected [5, 7].\n');
    return;
end

if ~isfolder(output_dir)
    mkdir(output_dir);
end

emg_labels = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};
num_tasks = size(dsfilt_emg, 2);
num_trials = size(dsfilt_emg, 1);
num_kinematics = size(finger_kinematics{1,1}, 2);
num_markers = num_kinematics / 3;
if mod(num_kinematics, 3) ~= 0
    fprintf('Warning: Kinematics data length not a multiple of 3.\n');
    num_markers = floor(num_kinematics/3);
end


% --- EMG Comparison ---
fprintf('Performing EMG Trial Comparison...\n');
for task = 1:num_tasks
    for channel = 1:length(emg_labels)
        figure;
        hold on;
        emg_data_all_trials = cell(1, num_trials); % Store data for boxplot (cell array)

        for trial = 1:num_trials
            try
                emg_data = dsfilt_emg{trial, task}(:, channel);
            catch ME
                fprintf('Error accessing EMG data: %s\n', ME.message);
                emg_data = []; % Set to empty if error
            end

            if ~isempty(emg_data)
                % Consistent time vector
                num_samples = size(emg_data, 1);
                time_vector = (0:num_samples - 1)';

                plot(time_vector, emg_data, 'DisplayName', sprintf('Trial %d', trial));
                emg_data_all_trials{trial} = emg_data; % Store for boxplot
            else
                fprintf('Warning: EMG data empty for Trial %d, Task %d, Channel %s\n', trial, task, emg_labels{channel});
            end
        end

        hold off;
        title(sprintf('EMG Comparison - Task %d, Channel %s', task, emg_labels{channel}));
        xlabel('Time (samples)');
        ylabel('Amplitude');
        legend('Location', 'best');
        grid on;

        % Save the line plot figure
        fig_filename_line = fullfile(output_dir, sprintf('emg_comparison_line_task%d_channel%s.png', task, emg_labels{channel}));
        saveas(gcf, fig_filename_line);
        close(gcf);


        % --- Boxplot for EMG comparison across trials ---
        % Prepare data for boxplot (convert cell array to matrix, padding with NaNs)
        max_length = 0;
        for i = 1:num_trials
            if ~isempty(emg_data_all_trials{i})
                max_length = max(max_length, length(emg_data_all_trials{i}));
            end
        end
        emg_data_matrix = NaN(max_length, num_trials); % Initialize with NaNs
        for i = 1:num_trials
            if ~isempty(emg_data_all_trials{i})
               emg_data_matrix(1:length(emg_data_all_trials{i}), i) = emg_data_all_trials{i};
            end
        end
        
        figure;
        boxplot(emg_data_matrix, 'Labels', arrayfun(@(x) sprintf('Trial %d', x), 1:num_trials, 'UniformOutput', false));
        title(sprintf('EMG Boxplot Across Trials - Task %d, Channel %s', task, emg_labels{channel}));
        xlabel('Trial');
        ylabel('Amplitude');
        grid on;
        % Save the box plot
        fig_filename_box = fullfile(output_dir, sprintf('emg_comparison_box_task%d_channel%s.png', task, emg_labels{channel}));
        saveas(gcf, fig_filename_box);
        close(gcf);
    end
end

% --- Kinematics Comparison ---
fprintf('Performing Kinematics Trial Comparison...\n');
for task = 1:num_tasks
    for marker = 1:num_markers
        for coord = 1:3  % X, Y, Z
            figure;
            hold on;
            kin_data_all_trials = cell(1, num_trials); % For boxplot

            for trial = 1:num_trials
                try
                    kinematics_data = finger_kinematics{trial, task};
                    start_col = (marker - 1) * 3 + coord;
                    
                    if start_col > size(kinematics_data, 2)
                         fprintf('Warning: start_col exceeds kinematic data dimension.\n');
                         kin_data = [];  % Set to empty to avoid errors in plotting
                    else
                        kin_data = kinematics_data(:, start_col);
                    end
                catch ME
                    fprintf('Error accessing kinematics data: %s\n', ME.message);
                    kin_data = []; % Set to empty if there's an error
                end

                if ~isempty(kin_data)
                    % Use a consistent time vector
                    num_samples = size(kin_data, 1);
                    time_vector = (0:num_samples - 1)';

                    plot(time_vector, kin_data, 'DisplayName', sprintf('Trial %d', trial));
                    kin_data_all_trials{trial} = kin_data; % Store for boxplot

                else
                    fprintf('Warning: Kinematics data empty for Trial %d, Task %d, Marker %d, Coord %d\n', trial, task, marker, coord);
                end

            end
            hold off;

            if coord == 1
                coord_label = 'X';
            elseif coord == 2
                coord_label = 'Y';
            else
                coord_label = 'Z';
            end
            title(sprintf('Kinematics Comparison - Task %d, Marker %d, %s', task, marker, coord_label));
            xlabel('Time (samples)');
            ylabel('Position');
            legend('Location', 'best');
            grid on;

            % Save the line plot
            fig_filename_line = fullfile(output_dir, sprintf('kinematics_comparison_line_task%d_marker%d_%s.png', task, marker, coord_label));
            saveas(gcf, fig_filename_line);
            close(gcf);

           % --- Boxplot for kinematics comparison across trials ---
           %Prepare data for boxplot
            max_length = 0;
            for i = 1:num_trials
               if ~isempty(kin_data_all_trials{i})
                   max_length = max(max_length, length(kin_data_all_trials{i}));
               end
            end
            kin_data_matrix = NaN(max_length, num_trials);
            for i = 1:num_trials
                if ~isempty(kin_data_all_trials{i})
                   kin_data_matrix(1:length(kin_data_all_trials{i}),i) = kin_data_all_trials{i};
                end
            end


            figure;
            boxplot(kin_data_matrix, 'Labels', arrayfun(@(x) sprintf('Trial %d', x), 1:num_trials, 'UniformOutput', false));
            title(sprintf('Kinematics Boxplot Across Trials - Task %d, Marker %d, %s', task, marker, coord_label));
            xlabel('Trial');
            ylabel('Position');
            grid on;

            % Save the box plot
            fig_filename_box = fullfile(output_dir, sprintf('kinematics_comparison_box_task%d_marker%d_%s.png', task, marker, coord_label));
            saveas(gcf, fig_filename_box);
            close(gcf);

        end
    end
end

fprintf('Trial comparison analysis complete. Plots saved to: %s\n', output_dir);
end