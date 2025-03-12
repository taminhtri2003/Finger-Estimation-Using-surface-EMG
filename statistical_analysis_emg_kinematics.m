function statistical_analysis_emg_kinematics(mat_file_path, output_dir)
% Performs statistical analysis on EMG and Kinematics data and visualizes results.
% Generates tables and plots showing statistical measures.

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

% Create output directory if it doesn't exist
if ~isfolder(output_dir)
    mkdir(output_dir);
end

emg_labels = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};
num_tasks = size(dsfilt_emg, 2);
num_trials = size(dsfilt_emg, 1);
num_kinematics = size(finger_kinematics{1,1}, 2);
num_markers = num_kinematics / 3;
if mod(num_kinematics, 3) ~= 0
     fprintf('Warning: Number of kinematics is not a multiple of 3.\n');
     num_markers = floor(num_kinematics/3);
end


% --- EMG Statistical Analysis ---
fprintf('Performing EMG Statistical Analysis...\n');
emg_stats_table = cell(num_trials * length(emg_labels), num_tasks + 1); % +1 for Trial/Channel info
emg_stats_header = [{'Trial', 'Channel'}, arrayfun(@(x) sprintf('Task %d', x), 1:num_tasks, 'UniformOutput', false)];
emg_stats_row = 1;


for trial = 1:num_trials
    for channel = 1:length(emg_labels)
        emg_stats_table{emg_stats_row, 1} = trial;  % Trial number
        emg_stats_table{emg_stats_row, 2} = emg_labels{channel}; % Channel name
        
        for task = 1:num_tasks
            try
               emg_data = dsfilt_emg{trial, task}(:, channel);
            catch ME
                 fprintf('Error at Trial %d, Channel %d, Task %d:%s\n', trial, channel,task, ME.message);
                 emg_stats_table{emg_stats_row, task + 2} = NaN; % Put NaN if error
                 continue;
            end
            
            % Calculate EMG statistics (mean, std, RMS, max, min)
            emg_mean = mean(emg_data);
            emg_std = std(emg_data);
            emg_rms = rms(emg_data);
            emg_max = max(emg_data);
            emg_min = min(emg_data);

            % Store statistics in the cell array
            emg_stats_table{emg_stats_row, task + 2} = sprintf('Mean: %.2f, Std: %.2f, RMS: %.2f, Max: %.2f, Min: %.2f', ...
                                                              emg_mean, emg_std, emg_rms, emg_max, emg_min);
        end
         emg_stats_row = emg_stats_row + 1;
    end
end

% Convert EMG stats cell array to a table and save
emg_stats_table = cell2table(emg_stats_table, 'VariableNames', emg_stats_header);
writetable(emg_stats_table, fullfile(output_dir, 'emg_statistics.csv'));
fprintf('EMG statistics saved to: %s\n', fullfile(output_dir, 'emg_statistics.csv'));



% --- Kinematics Statistical Analysis ---
fprintf('Performing Kinematics Statistical Analysis...\n');
kinematics_stats_table = cell(num_trials * num_markers * 3, num_tasks + 2); % +2 for Trial/Marker/Coord
kinematics_stats_header = [{'Trial', 'Marker', 'Coordinate'}, arrayfun(@(x) sprintf('Task %d', x), 1:num_tasks, 'UniformOutput', false)];
kinematics_stats_row = 1;

for trial = 1:num_trials
    for marker = 1:num_markers
        for coord = 1:3  % X, Y, Z
              kinematics_stats_table{kinematics_stats_row, 1} = trial;
              kinematics_stats_table{kinematics_stats_row, 2} = marker;
              if coord == 1
                kinematics_stats_table{kinematics_stats_row, 3} = 'X';
              elseif coord == 2;
                  kinematics_stats_table{kinematics_stats_row, 3} = 'Y';
              else
                  kinematics_stats_table{kinematics_stats_row, 3} = 'Z';
              end


            for task = 1:num_tasks;
                try
                    kinematics_data = finger_kinematics{trial, task};
                catch ME;
                    fprintf('Error at Trial %d, Task %d: %s', trial, task, ME.message);
                    kinematics_stats_table{kinematics_stats_row, task + 3} = NaN;
                    continue
                end
                 start_col = (marker - 1) * 3 + coord;
                  if start_col > size(kinematics_data, 2)
                        kinematics_stats_table{kinematics_stats_row, task + 3} = NaN; % Assign NaN if out of bounds
                        fprintf('Warning: start_col exceeds kinematic data dimension.\n');
                        continue;  % Skip to next task
                  end
                kin_data = kinematics_data(:, start_col);


                % Calculate kinematics statistics
                kin_mean = mean(kin_data);
                kin_std = std(kin_data);
                kin_max = max(kin_data);
                kin_min = min(kin_data);
                kin_range = kin_max - kin_min;

                % Store in cell array
                kinematics_stats_table{kinematics_stats_row, task + 3} = sprintf('Mean: %.2f, Std: %.2f, Max: %.2f, Min: %.2f, Range: %.2f', ...
                                                                          kin_mean, kin_std, kin_max, kin_min, kin_range);
            end
             kinematics_stats_row = kinematics_stats_row + 1;
        end
    end
end

% Convert Kinematics stats to table and save
kinematics_stats_table = cell2table(kinematics_stats_table, 'VariableNames', kinematics_stats_header);
writetable(kinematics_stats_table, fullfile(output_dir, 'kinematics_statistics.csv'));
fprintf('Kinematics statistics saved to: %s\n', fullfile(output_dir, 'kinematics_statistics.csv'));

% --- Statistical Visualization (Box Plots) ---
fprintf('Creating Statistical Visualizations (Box Plots)...\n');

% EMG Box Plots (per channel, across tasks)
for trial = 1:num_trials
    for channel = 1:length(emg_labels)
        figure;
        hold on;
        emg_data_for_boxplot = zeros(0,num_tasks); %Initialize

        for task = 1:num_tasks;
            try
                emg_data = dsfilt_emg{trial,task}(:,channel);
                emg_data_for_boxplot(1:length(emg_data),task) = emg_data;
            catch
                emg_data_for_boxplot(:,task) = NaN; %Fill with NaN
                continue
            end
        end

        boxplot(emg_data_for_boxplot, 'Labels', arrayfun(@(x) sprintf('Task %d', x), 1:num_tasks, 'UniformOutput', false));
        title(sprintf('EMG Box Plot - Trial %d, Channel %s', trial, emg_labels{channel}));
        ylabel('Amplitude');
        xlabel('Task');
        grid on;
        hold off;
         % Save the figure
        fig_filename = fullfile(output_dir, sprintf('emg_boxplot_trial%d_channel%s.png', trial, emg_labels{channel}));
        saveas(gcf, fig_filename);
        close(gcf);
    end
end

% Kinematics Box Plots (per marker and coordinate, across tasks)
for trial = 1:num_trials
    for marker = 1:num_markers
        for coord = 1:3
              figure;
              hold on;
              kin_data_for_boxplot = zeros(0,num_tasks);

              for task = 1:num_tasks
                try
                    kinematics_data = finger_kinematics{trial, task};
                catch
                    continue;
                end
                  start_col = (marker - 1) * 3 + coord;
                  if start_col > size(kinematics_data, 2)
                        continue;
                  end

                 kin_data_for_boxplot(1:length(kinematics_data(:,start_col)), task) = kinematics_data(:,start_col);
              end
               boxplot(kin_data_for_boxplot, 'Labels', arrayfun(@(x) sprintf('Task %d', x), 1:num_tasks, 'UniformOutput', false));
                if coord == 1
                    coord_label = 'X';
                elseif coord == 2
                    coord_label = 'Y';
                else
                    coord_label = 'Z';
                end
                title(sprintf('Kinematics Box Plot - Trial %d, Marker %d, %s', trial, marker, coord_label));
                ylabel('Position');
                xlabel('Task');
                grid on;
                hold off;

                 % Save the figure
                fig_filename = fullfile(output_dir, sprintf('kinematics_boxplot_trial%d_marker%d_%s.png', trial, marker, coord_label));
                saveas(gcf, fig_filename);
                close(gcf);
        end
    end
end

fprintf('Statistical analysis complete.  Tables and plots saved to: %s\n', output_dir);

end

