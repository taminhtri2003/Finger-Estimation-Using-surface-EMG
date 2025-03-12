function trial_comparison_stats_to_csv(mat_file_path, output_dir)
% Compares EMG and kinematics data between trials within each task,
% performs statistical tests (ANOVA and t-tests), and exports results to CSV.
% Creates separate CSV files for each task.

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


% --- Main Loop (Iterate through Tasks) ---
for task = 1:num_tasks
    fprintf('Performing analysis for Task %d...\n', task);

    % --- Prepare data structures for this task ---
    emg_results = cell(length(emg_labels) * num_trials, 5);  % Trial, Channel, Mean, Std, RMS
    emg_row = 1;
    kin_results = cell(num_markers * 3 * num_trials, 6); % Trial, Marker, Coord, Mean, Std, Range
    kin_row = 1;

    anova_emg_results = cell(length(emg_labels), 3); % Channel, ANOVA p-value, ANOVA table
    anova_kin_results = cell(num_markers * 3, 4);   % Marker, Coord, ANOVA p-value, ANOVA table
    ttest_emg_results = cell(length(emg_labels), 3);% Channel, t-test p-adj, t-test h
    ttest_kin_results = cell(num_markers*3, 4); % Marker, Coord, t-test p-adj, ttest h



    % --- EMG Analysis (for the current task) ---
    for channel = 1:length(emg_labels)
        emg_data_all_trials = cell(1, num_trials);
        for trial = 1:num_trials
            try
                emg_data = dsfilt_emg{trial, task}(:, channel);
                emg_data_all_trials{trial} = emg_data;

                % Calculate and store EMG statistics
                emg_results{emg_row, 1} = trial;
                emg_results{emg_row, 2} = emg_labels{channel};
                emg_results{emg_row, 3} = mean(emg_data);
                emg_results{emg_row, 4} = std(emg_data);
                emg_results{emg_row, 5} = rms(emg_data);
                emg_row = emg_row + 1;

            catch ME
                fprintf('Error in EMG: Trial %d, Task %d, Channel %s: %s\n', trial, task, emg_labels{channel}, ME.message);
                 emg_results{emg_row, 1} = trial;
                 emg_results{emg_row, 2} = emg_labels{channel};
                 emg_results{emg_row, 3} = NaN;
                 emg_results{emg_row, 4} = NaN;
                 emg_results{emg_row, 5} = NaN;
                 emg_row = emg_row + 1;
            end
        end

        % --- Statistical Tests (EMG) ---
         max_length = 0;
         for i = 1:num_trials
            if ~isempty(emg_data_all_trials{i})
                max_length = max(max_length, length(emg_data_all_trials{i}));
            end
         end
         emg_data_matrix = NaN(max_length,num_trials);
         for i = 1:num_trials
            if ~isempty(emg_data_all_trials{i})
                emg_data_matrix(1:length(emg_data_all_trials{i}), i) = emg_data_all_trials{i};
            end
         end


        if num_trials > 2
            [p_anova, tbl_anova, ~] = anova1(emg_data_matrix, [], 'off');
            anova_emg_results{channel, 1} = emg_labels{channel};
            anova_emg_results{channel, 2} = p_anova;
            anova_emg_results{channel, 3} = tbl_anova; % Store the entire table
        else
            anova_emg_results{channel, 1} = emg_labels{channel};
            anova_emg_results{channel, 2} = NaN;
            anova_emg_results{channel, 3} = {};
        end

        p_values_ttest = NaN(num_trials, num_trials);
        for trial1 = 1:num_trials
            for trial2 = (trial1 + 1):num_trials
                if ~isempty(emg_data_all_trials{trial1}) && ~isempty(emg_data_all_trials{trial2})
                    [~, p_values_ttest(trial1, trial2)] = ttest2(emg_data_all_trials{trial1}, emg_data_all_trials{trial2});
                end
            end
        end
        [h_bonf, p_adj_bonf] = bonf_holm(p_values_ttest(:));
        p_adj_bonf = reshape(p_adj_bonf, size(p_values_ttest));
        h_bonf = reshape(h_bonf, size(p_values_ttest));

        ttest_emg_results{channel, 1} = emg_labels{channel};
        ttest_emg_results{channel, 2} = p_adj_bonf;
        ttest_emg_results{channel, 3} = h_bonf;
    end


    % --- Kinematics Analysis (for the current task) ---
      for marker = 1:num_markers
        for coord = 1:3  % X, Y, Z
            kin_data_all_trials = cell(1, num_trials);

            for trial = 1:num_trials
                try
                    kinematics_data = finger_kinematics{trial, task};
                    start_col = (marker - 1) * 3 + coord;
                    if start_col > size(kinematics_data, 2)
                       fprintf('Warning: start_col out of bounds. Trial %d, Task %d, Marker %d, Coord %d\n', trial, task, marker, coord);
                       kin_data_all_trials{trial} = []; % Set to empty
                       continue;
                    end

                    kin_data = kinematics_data(:, start_col);
                    kin_data_all_trials{trial} = kin_data;

                    % Calculate and store kinematics statistics
                    kin_results{kin_row, 1} = trial;
                    kin_results{kin_row, 2} = marker;
                    kin_results{kin_row, 3} = coord; % 1=X, 2=Y, 3=Z
                    kin_results{kin_row, 4} = mean(kin_data);
                    kin_results{kin_row, 5} = std(kin_data);
                    kin_results{kin_row, 6} = max(kin_data) - min(kin_data);
                    kin_row = kin_row + 1;

                catch ME
                    fprintf('Error in Kinematics: Trial %d, Task %d, Marker %d, Coord %d: %s\n', trial, task, marker, coord, ME.message);
                    kin_results{kin_row, 1} = trial;
                    kin_results{kin_row, 2} = marker;
                    kin_results{kin_row, 3} = coord;
                    kin_results{kin_row, 4} = NaN;
                    kin_results{kin_row, 5} = NaN;
                    kin_results{kin_row, 6} = NaN;
                    kin_row = kin_row + 1;

                end
            end

            % --- Statistical Tests (Kinematics) ---

            max_length = 0;
            for i = 1:num_trials
                if ~isempty(kin_data_all_trials{i})
                    max_length = max(max_length, length(kin_data_all_trials{i}));
                end
            end
            kin_data_matrix = NaN(max_length,num_trials);
            for i = 1:num_trials
                if ~isempty(kin_data_all_trials{i})
                    kin_data_matrix(1:length(kin_data_all_trials{i}),i) = kin_data_all_trials{i};
                end
            end


            if num_trials > 2
                [p_anova, tbl_anova, ~] = anova1(kin_data_matrix, [], 'off');
                 if coord == 1
                    coord_label = 'X';
                elseif coord == 2
                    coord_label = 'Y';
                else
                    coord_label = 'Z';
                end
                anova_kin_results{(marker-1)*3 + coord, 1} = marker;
                anova_kin_results{(marker-1)*3 + coord, 2} = coord_label;
                anova_kin_results{(marker-1)*3 + coord, 3} = p_anova;
                anova_kin_results{(marker-1)*3 + coord, 4} = tbl_anova;
            else
                if coord == 1
                    coord_label = 'X';
                elseif coord == 2
                    coord_label = 'Y';
                else
                    coord_label = 'Z';
                end
                anova_kin_results{(marker-1)*3 + coord, 1} = marker;
                 anova_kin_results{(marker-1)*3 + coord, 2} = coord_label;
                anova_kin_results{(marker-1)*3 + coord, 3} = NaN;
                anova_kin_results{(marker-1)*3 + coord, 4} = {};
            end

            p_values_ttest = NaN(num_trials, num_trials);
            for trial1 = 1:num_trials
                for trial2 = (trial1 + 1):num_trials
                    if ~isempty(kin_data_all_trials{trial1}) && ~isempty(kin_data_all_trials{trial2})
                        [~, p_values_ttest(trial1, trial2)] = ttest2(kin_data_all_trials{trial1}, kin_data_all_trials{trial2});
                    end
                end
            end
            [h_bonf, p_adj_bonf] = bonf_holm(p_values_ttest(:));
            p_adj_bonf = reshape(p_adj_bonf, size(p_values_ttest));
            h_bonf = reshape(h_bonf,size(p_values_ttest));
            
            if coord == 1
                    coord_label = 'X';
                elseif coord == 2
                    coord_label = 'Y';
                else
                    coord_label = 'Z';
            end
            ttest_kin_results{(marker-1)*3 + coord, 1} = marker;
            ttest_kin_results{(marker-1)*3 + coord, 2} = coord_label;
            ttest_kin_results{(marker-1)*3 + coord, 3} = p_adj_bonf;
            ttest_kin_results{(marker-1)*3 + coord, 4} = h_bonf;

        end
    end


    % --- Create and Save CSV Files (for the current task) ---

    % EMG Statistics
    emg_stats_table = cell2table(emg_results, 'VariableNames', {'Trial', 'Channel', 'Mean', 'Std', 'RMS'});
    emg_stats_filename = fullfile(output_dir, sprintf('emg_statistics_task%d.csv', task));
    writetable(emg_stats_table, emg_stats_filename);
    fprintf('EMG statistics for Task %d saved to: %s\n', task, emg_stats_filename);

    % Kinematics Statistics
    kin_stats_table = cell2table(kin_results, 'VariableNames', {'Trial', 'Marker', 'Coordinate', 'Mean', 'Std', 'Range'});
    kin_stats_filename = fullfile(output_dir, sprintf('kinematics_statistics_task%d.csv', task));
    writetable(kin_stats_table, kin_stats_filename);
    fprintf('Kinematics statistics for Task %d saved to: %s\n', task, kin_stats_filename);


    % EMG ANOVA Results
    for i = 1:size(anova_emg_results, 1)
      if ~isempty(anova_emg_results{i,3})
        anova_table_emg = cell2table(anova_emg_results{i, 3}(2:end,:), 'VariableNames', anova_emg_results{i, 3}(1,:));
        anova_emg_filename = fullfile(output_dir, sprintf('emg_anova_task%d_%s.csv', task, anova_emg_results{i, 1}));
        writetable(anova_table_emg, anova_emg_filename);
        fprintf('EMG ANOVA results for Task %d, Channel %s saved to: %s\n', task, anova_emg_results{i, 1}, anova_emg_filename);
      end
    end
    

    % Kinematics ANOVA Results
      for i = 1:size(anova_kin_results, 1)
         if ~isempty(anova_kin_results{i,4})
            if anova_kin_results{i,2} == 'X'
                coord_num = 1;
            elseif anova_kin_results{i,2} == 'Y'
                coord_num = 2;
            else
                coord_num = 3;
            end
            anova_table_kin = cell2table(anova_kin_results{i, 4}(2:end,:), 'VariableNames', anova_kin_results{i, 4}(1,:));
            anova_kin_filename = fullfile(output_dir, sprintf('kinematics_anova_task%d_marker%d_coord%d.csv', task, anova_kin_results{i,1}, coord_num));
            writetable(anova_table_kin, anova_kin_filename);
            fprintf('Kinematics ANOVA results for Task %d, Marker %s, Coord %s saved to: %s\n', task, anova_kin_results{i,1}, anova_kin_results{i,2} , anova_kin_filename);
         end
      end

    % EMG T-Test Results
    for i = 1:size(ttest_emg_results, 1)
       if ~isempty(ttest_emg_results{i,2})
        p_values_table = array2table(ttest_emg_results{i, 2}, ...
                                    'RowNames', arrayfun(@(x) sprintf('Trial %d', x), 1:num_trials, 'UniformOutput', false), ...
                                    'VariableNames', arrayfun(@(x) sprintf('Trial %d', x), 1:num_trials, 'UniformOutput', false));
        h_values_table = array2table(ttest_emg_results{i,3},...
                                     'RowNames', arrayfun(@(x) sprintf('Trial%d',x), 1:num_trials, 'UniformOutput', false),...
                                     'VariableNames', arrayfun(@(x) sprintf('Trial %d', x), 1:num_trials,'UniformOutput', false));
        ttest_emg_p_filename = fullfile(output_dir, sprintf('emg_ttest_p_values_task%d_%s.csv', task, ttest_emg_results{i, 1}));
        ttest_emg_h_filename = fullfile(output_dir, sprintf('emg_ttest_h_values_task%d_%s.csv', task, ttest_emg_results{i, 1}));

        writetable(p_values_table, ttest_emg_p_filename, 'WriteRowNames', true);
        writetable(h_values_table, ttest_emg_h_filename, 'WriteRowNames', true);
        fprintf('EMG t-test results for Task %d, Channel %s saved to CSV files.\n', task, ttest_emg_results{i, 1});
       end
    end

    % Kinematics T-Test Results
     for i = 1:size(ttest_kin_results,1)
        if ~isempty(ttest_kin_results{i,3})
             if ttest_kin_results{i,2} == 'X'
                coord_num = 1;
            elseif ttest_kin_results{i,2} == 'Y'
                coord_num = 2;
            else
                coord_num = 3;
            end
            p_values_table = array2table(ttest_kin_results{i, 3}, ...
                                        'RowNames', arrayfun(@(x) sprintf('Trial %d', x), 1:num_trials, 'UniformOutput', false), ...
                                        'VariableNames', arrayfun(@(x) sprintf('Trial %d', x), 1:num_trials, 'UniformOutput', false));
            h_values_table = array2table(ttest_kin_results{i,4},...
                                         'RowNames', arrayfun(@(x) sprintf('Trial%d',x), 1:num_trials, 'UniformOutput', false),...
                                         'VariableNames', arrayfun(@(x) sprintf('Trial %d', x), 1:num_trials, 'UniformOutput', false));

            ttest_kin_p_filename = fullfile(output_dir, sprintf('kinematics_ttest_p_values_task%d_marker%d_coord%d.csv', task, ttest_kin_results{i,1}, coord_num));
            ttest_kin_h_filename = fullfile(output_dir, sprintf('kinematics_ttest_h_values_task%d_marker%d_coord%d.csv', task, ttest_kin_results{i,1}, coord_num));
            writetable(p_values_table, ttest_kin_p_filename, 'WriteRowNames', true);
            writetable(h_values_table, ttest_kin_h_filename, 'WriteRowNames', true);
            fprintf('Kinematics t-test results for Task %d, Marker %d, Coord %d saved to CSV files.\n', task, ttest_kin_results{i, 1}, coord_num);

        end
     end
end

fprintf('Trial comparison (statistical tests) complete. Results saved to: %s\n', output_dir);
end