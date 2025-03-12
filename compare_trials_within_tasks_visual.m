function compare_trials_within_tasks_visual(mat_file_path, output_dir)
% Compares EMG and kinematics data between trials within each task.
% Generates plots (line and box plots, and *heatmaps*) for visual comparison.

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
fprintf('Performing EMG Trial Comparison (Visualizations)...\n');
for task = 1:num_tasks
    for channel = 1:length(emg_labels)
        emg_data_all_trials = cell(1, num_trials);
        max_length = 0;

        % Collect and preprocess data for all trials
        for trial = 1:num_trials
            try
                emg_data = dsfilt_emg{trial, task}(:, channel);
                emg_data_all_trials{trial} = emg_data;
                max_length = max(max_length, length(emg_data));
            catch ME
                fprintf('Error accessing EMG data: %s\n', ME.message);
                emg_data_all_trials{trial} = []; % Set to empty if error
            end
        end
        
        % Pad with NaNs to create a matrix
        emg_data_matrix = NaN(max_length, num_trials);
        for i=1:num_trials
            if ~isempty(emg_data_all_trials{i})
                emg_data_matrix(1:length(emg_data_all_trials{i}), i) = emg_data_all_trials{i};
            end
        end

        % --- 1. Line Plots (as before) ---
        figure;
        hold on;
        for trial = 1:num_trials
            if ~isempty(emg_data_all_trials{trial})
                num_samples = size(emg_data_all_trials{trial}, 1);
                time_vector = (0:num_samples - 1)';
                plot(time_vector, emg_data_all_trials{trial}, 'DisplayName', sprintf('Trial %d', trial));
            end
        end
        hold off;
        title(sprintf('EMG Comparison - Task %d, Channel %s', task, emg_labels{channel}));
        xlabel('Time (samples)');
        ylabel('Amplitude');
        legend('Location', 'best');
        grid on;
        fig_filename_line = fullfile(output_dir, sprintf('emg_comparison_line_task%d_channel%s.png', task, emg_labels{channel}));
        saveas(gcf, fig_filename_line);
        close(gcf);


        % --- 2. Box Plots (as before) ---
        figure;
        boxplot(emg_data_matrix, 'Labels', arrayfun(@(x) sprintf('Trial %d', x), 1:num_trials, 'UniformOutput', false));
        title(sprintf('EMG Boxplot Across Trials - Task %d, Channel %s', task, emg_labels{channel}));
        xlabel('Trial');
        ylabel('Amplitude');
        grid on;
        fig_filename_box = fullfile(output_dir, sprintf('emg_comparison_box_task%d_channel%s.png', task, emg_labels{channel}));
        saveas(gcf, fig_filename_box);
        close(gcf);


        % --- 3. Heatmap (NEW) ---
        figure;
        imagesc(emg_data_matrix);  % Use the NaN-padded matrix
        colormap('jet');  % Choose a colormap
        colorbar;
        title(sprintf('EMG Heatmap Across Trials - Task %d, Channel %s', task, emg_labels{channel}));
        xlabel('Trial');
        ylabel('Time (samples)');
        set(gca, 'XTick', 1:num_trials);  % Set x-ticks to trial numbers
        set(gca, 'XTickLabel', arrayfun(@(x) sprintf('Trial %d', x), 1:num_trials, 'UniformOutput', false));
        fig_filename_heatmap = fullfile(output_dir, sprintf('emg_comparison_heatmap_task%d_channel%s.png', task, emg_labels{channel}));
        saveas(gcf, fig_filename_heatmap);
        close(gcf);
    end
end



% --- Kinematics Comparison ---
fprintf('Performing Kinematics Trial Comparison (Visualizations)...\n');
for task = 1:num_tasks
    for marker = 1:num_markers
        for coord = 1:3  % X, Y, Z
            kin_data_all_trials = cell(1, num_trials);
            max_length = 0;

            % Collect data and determine max length
            for trial = 1:num_trials
                try
                    kinematics_data = finger_kinematics{trial, task};
                    start_col = (marker - 1) * 3 + coord;

                    if start_col > size(kinematics_data, 2)
                         fprintf('Warning: start_col exceeds kinematic data dimension.\n');
                         kin_data_all_trials{trial} = [];
                         continue;
                    end

                    kin_data = kinematics_data(:, start_col);
                    kin_data_all_trials{trial} = kin_data;
                    max_length = max(max_length, length(kin_data));
                catch ME
                    fprintf('Error accessing kinematics data: %s\n', ME.message);
                    kin_data_all_trials{trial} = []; % Set to empty if there's an error
                end
            end

            % Pad with NaNs
            kin_data_matrix = NaN(max_length, num_trials);
            for i = 1:num_trials
                if ~isempty(kin_data_all_trials{i})
                    kin_data_matrix(1:length(kin_data_all_trials{i}), i) = kin_data_all_trials{i};
                end
            end

            % Coordinate label
            if coord == 1
                coord_label = 'X';
            elseif coord == 2
                coord_label = 'Y';
            else
                coord_label = 'Z';
            end

            % --- 1. Line Plots (as before) ---
            figure;
            hold on;
            for trial = 1:num_trials
                if ~isempty(kin_data_all_trials{trial})
                   num_samples = size(kin_data_all_trials{trial}, 1);
                   time_vector = (0:num_samples-1)';
                   plot(time_vector,kin_data_all_trials{trial},'DisplayName', sprintf('Trial%d',trial));
                end
            end
            hold off;
            title(sprintf('Kinematics Comparison - Task %d, Marker %d, %s', task, marker, coord_label));
            xlabel('Time (samples)');
            ylabel('Position');
            legend('Location', 'best');
            grid on;
            fig_filename_line = fullfile(output_dir, sprintf('kinematics_comparison_line_task%d_marker%d_%s.png', task, marker, coord_label));
            saveas(gcf, fig_filename_line);
            close(gcf);

            % --- 2. Box Plots (as before) ---
            figure;
            boxplot(kin_data_matrix, 'Labels', arrayfun(@(x) sprintf('Trial %d', x), 1:num_trials, 'UniformOutput', false));
            title(sprintf('Kinematics Boxplot Across Trials - Task %d, Marker %d, %s', task, marker, coord_label));
            xlabel('Trial');
            ylabel('Position');
            grid on;
            fig_filename_box = fullfile(output_dir, sprintf('kinematics_comparison_box_task%d_marker%d_%s.png', task, marker, coord_label));
            saveas(gcf, fig_filename_box);
            close(gcf);


            % --- 3. Heatmap (NEW) ---
            figure;
            imagesc(kin_data_matrix);
            colormap('jet');
            colorbar;
            title(sprintf('Kinematics Heatmap Across Trials - Task %d, Marker %d, %s', task, marker, coord_label));
            xlabel('Trial');
            ylabel('Time (samples)');
            set(gca, 'XTick', 1:num_trials);
            set(gca, 'XTickLabel', arrayfun(@(x) sprintf('Trial %d', x), 1:num_trials, 'UniformOutput', false));
            fig_filename_heatmap = fullfile(output_dir, sprintf('kinematics_comparison_heatmap_task%d_marker%d_%s.png', task, marker, coord_label));
            saveas(gcf, fig_filename_heatmap);
            close(gcf);
        end
    end
end

fprintf('Trial comparison (visualizations) complete. Plots saved to: %s\n', output_dir);
end