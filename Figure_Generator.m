% MATLAB Script to Generate Paper Figures - Multi-Subject Analysis
%
% Processes data files (e.g., s5_full.mat, s6_full.mat, ...) for multiple subjects.
% Generates per-subject figures, calculates statistics, performs cross-subject
% comparisons, and saves a summary statistics table.

clearvars;
clc;
close all; % Close any existing figures

fprintf('Starting multi-subject figure generation script...\n');

% --- Parameters ---
subject_ids = 1:10; % Specify the subject IDs to process
fs = 200; % !!! IMPORTANT: Set your actual sampling rate in Hz !!!
output_dir = 'paper_figures_multi_subject'; % Directory to save figures and table

% --- UPDATED: Define angle names based on user description ---
angle_names = {'Thumb (20-17-18)', 'Thumb (17-18-19)', ... % Angles 1-2
               'Index (20-1-5)', 'Index (1-5-6)', 'Index (5-6-7)', ... % Angles 3-5
               'Middle (20-2-8)', 'Middle (2-8-9)', 'Middle (8-9-10)', ... % Angles 6-8
               'Ring (20-3-11)', 'Ring (3-11-12)', 'Ring (11-12-13)', ... % Angles 9-11
               'Little (20-4-14)', 'Little (4-14-15)', 'Little (14-15-16)'}; % Angles 12-14
num_angles = length(angle_names);

% --- UPDATED: Define EMG channel names based on user description ---
emg_channel_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};
num_emg_channels_ref = length(emg_channel_names); % Should be 8
fprintf('  Using %d reference EMG channels: %s\n', num_emg_channels_ref, strjoin(emg_channel_names, ', '));

% --- Analysis Parameters ---
task_for_avg_plots = 1; % Task index (1-7) used for average angle/EMG plots (Fig 1, 2) AND for cross-subject aggregation
angle_for_task_comp = 5; % Angle index (1-14) used for task comparison plots (Fig 3) -> Corresponds to 'Index (5-6-7)'
trial_for_detail = 1; % Trial index (1-5) used for detail plots (Fig 4, 5)
task_for_detail = 1; % Task index (1-7) used for detail plots (Fig 4, 5)
angle_indices_detail = [4, 5]; % Example: 'Index (1-5-6)' (4), 'Index (5-6-7)' (5) for Fig 5
emg_channel_detail = 1; % Example: EMG Channel 1 ('APL') for Fig 5
emg_lowpass_cutoff_hz = 5; % Cutoff frequency for low-pass filtering EMG envelope for correlation

% --- Setup ---
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('Created output directory: %s\n', output_dir);
end

% --- Data Storage for Cross-Subject Analysis ---
all_subject_stats = struct(); % Store detailed stats per subject/task
all_subject_avg_angles_task1 = []; % Store Task 1 avg angle traces [time x angle x subject] (Only for task_for_avg_plots)
all_subject_avg_emg_task1 = []; % Store Task 1 avg EMG traces [time x emg x subject] (Only for task_for_avg_plots)
all_subject_task_comp_metric = []; % Store task comparison metric [subject x task]
valid_subject_indices = []; % Keep track of subjects processed successfully

% --- Subject Processing Loop ---
% num_emg_channels_ref is now defined above
% emg_channel_names is now defined above

for s_idx = 1:length(subject_ids)
    subject_id = subject_ids(s_idx);
    filename = sprintf('s%d_full.mat', subject_id);
    fprintf('\n--- Processing Subject %d (%s) ---\n', subject_id, filename);

    % --- Load Data for Current Subject ---
    if ~isfile(filename)
        warning('File not found: %s. Skipping subject %d.', filename, subject_id);
        continue; % Skip to the next subject
    end

    temp_num_emg_current_subj = 0; % Initialize EMG channel count for this subject specifically

    try
        % --- UPDATED: Load joint_angles variable name ---
        % Check if 'joint_angles_cell' exists, otherwise try 'joint_angles' based on user description
        vars_in_file = who('-file', filename);
        if ismember('joint_angles_cell', vars_in_file)
            joint_angles_var_name = 'joint_angles_cell';
        elseif ismember('joint_angles', vars_in_file)
            joint_angles_var_name = 'joint_angles';
             warning('Subject %d: Variable "joint_angles_cell" not found, using "joint_angles" instead.', subject_id);
        else
             warning('Subject %d: Neither "joint_angles_cell" nor "joint_angles" found. Skipping.', subject_id);
             continue;
        end

        loaded_data = load(filename, 'finger_kinematics', joint_angles_var_name, 'dsfilt_emg');
        fprintf('  Data loaded for subject %d.\n', subject_id);

        % Basic validation
        required_vars = {'finger_kinematics', joint_angles_var_name, 'dsfilt_emg'};
        if ~all(isfield(loaded_data, required_vars))
             warning('Subject %d: File %s is missing one or more required variables. Skipping.', subject_id, filename);
             continue;
        end

        finger_kinematics = loaded_data.finger_kinematics;
        joint_angles_cell = loaded_data.(joint_angles_var_name); % Use dynamic name
        dsfilt_emg = loaded_data.dsfilt_emg;

        % Validate loaded cell array sizes
        if ~iscell(joint_angles_cell) || ~isequal(size(joint_angles_cell),[5,7])
             warning('Subject %d: "%s" is not a 5x7 cell array. Skipping.', subject_id, joint_angles_var_name);
             continue;
        end
         if ~iscell(dsfilt_emg) || ~isequal(size(dsfilt_emg),[5,7])
             warning('Subject %d: "dsfilt_emg" is not a 5x7 cell array. Skipping.', subject_id);
             continue;
        end
         if ~iscell(finger_kinematics) || ~isequal(size(finger_kinematics),[5,7])
             warning('Subject %d: "finger_kinematics" is not a 5x7 cell array. Skipping.', subject_id);
             continue;
        end


        % Get dimensions and time vector (assuming consistent within subject)
        [num_trials, num_tasks] = size(joint_angles_cell);
        num_timepoints = 0;
        found_data = false;
        for tr = 1:num_trials
            for ta = 1:num_tasks
                if ~isempty(joint_angles_cell{tr, ta}) && size(joint_angles_cell{tr, ta}, 2) == num_angles % Check angle columns
                    num_timepoints = size(joint_angles_cell{tr, ta}, 1);
                    % Check EMG data for this specific cell
                    if ~isempty(dsfilt_emg{tr, ta})
                         temp_num_emg_current_subj = size(dsfilt_emg{tr, ta}, 2);
                         if size(dsfilt_emg{tr, ta}, 1) ~= num_timepoints
                              warning('Subject %d, T%d, Tr%d: EMG timepoints (%d) differ from angles (%d). Skipping EMG for this cell.', subject_id, ta, tr, size(dsfilt_emg{tr, ta}, 1), num_timepoints);
                              temp_num_emg_current_subj = 0; % Treat as 0 if time mismatch
                         end
                    else
                        temp_num_emg_current_subj = 0; % No EMG data in this cell
                    end
                    found_data = true;
                    break; % Found non-empty angle data, proceed
                elseif ~isempty(joint_angles_cell{tr, ta}) && size(joint_angles_cell{tr, ta}, 2) ~= num_angles
                     warning('Subject %d, T%d, Tr%d: Angle data has %d columns, expected %d. Skipping cell.', subject_id, ta, tr, size(joint_angles_cell{tr, ta}, 2), num_angles);
                end
            end
            if found_data, break; end
        end

        if ~found_data || num_timepoints == 0
            warning('Subject %d: Could not find any valid non-empty data cells. Skipping.', subject_id);
            continue;
        end
        time_vec = (0:num_timepoints-1)' / fs;

        % --- UPDATED: Check consistency against fixed reference ---
        if temp_num_emg_current_subj ~= num_emg_channels_ref && temp_num_emg_current_subj > 0
             warning('Subject %d: Number of EMG channels found (%d) differs from reference (%d). Check data consistency. Per-subject EMG analysis will use %d channels.', subject_id, temp_num_emg_current_subj, num_emg_channels_ref, temp_num_emg_current_subj);
        elseif temp_num_emg_current_subj == 0 && num_emg_channels_ref > 0
             warning('Subject %d: No EMG data found in first checked cell, while reference count is %d.', subject_id, num_emg_channels_ref);
        elseif temp_num_emg_current_subj == num_emg_channels_ref && num_emg_channels_ref > 0
             fprintf('  Found %d EMG channels, consistent with reference.\n', temp_num_emg_current_subj);
        end


    catch ME_load
        warning('Error loading or validating data for subject %d from %s: %s. Skipping.', subject_id, filename, ME_load.message);
        continue; % Skip to next subject
    end

    % --- Generate Per-Subject Figures ---
    subj_output_dir = fullfile(output_dir, sprintf('Subject_%d', subject_id));
    if ~exist(subj_output_dir, 'dir'), mkdir(subj_output_dir); end

    % --- Generate Fig 1 for all tasks ---
    fprintf('  Generating Fig 1 series: Average Joint Angles (Tasks 1-%d)...\n', num_tasks);
    for task_idx_fig1 = 1:num_tasks % Loop through all tasks
        try
            % Call the plotting function for the current task
            temp_avg_angles = plot_average_angles(joint_angles_cell, task_idx_fig1, time_vec, angle_names, subj_output_dir, subject_id);

            % Store data for cross-subject analysis ONLY for the designated task (task_for_avg_plots)
            if task_idx_fig1 == task_for_avg_plots
                if ~isempty(temp_avg_angles) && size(temp_avg_angles,1) == num_timepoints && size(temp_avg_angles,2) == num_angles
                     all_subject_avg_angles_task1 = cat(3, all_subject_avg_angles_task1, temp_avg_angles);
                else
                     warning('Subject %d: Avg angle data for Task %d not suitable for cross-subject analysis aggregation.', subject_id, task_for_avg_plots);
                end
            end
        catch ME_fig1
            warning('Subject %d: Could not generate Figure 1 for Task %d: %s', subject_id, task_idx_fig1, ME_fig1.message);
        end
    end % End loop through tasks for Fig 1


    % Fig 2: Avg EMG (Only for task_for_avg_plots)
    % Use temp_num_emg_current_subj to decide if EMG plots can be made FOR THIS SUBJECT
    if temp_num_emg_current_subj > 0
        try
            fprintf('  Generating Fig 2: Average Rectified EMG (Task %d)...\n', task_for_avg_plots);
            % Use the correct channel names based on the number found for this subject
            % If inconsistent, use generic names for the plot, but reference names for storage check
            if temp_num_emg_current_subj == num_emg_channels_ref
                current_emg_names_for_plot = emg_channel_names;
            else
                current_emg_names_for_plot = arrayfun(@(x) sprintf('EMG %d', x), 1:temp_num_emg_current_subj, 'UniformOutput', false);
            end
            avg_emg_task1 = plot_average_emg(dsfilt_emg, task_for_avg_plots, time_vec, current_emg_names_for_plot, subj_output_dir, subject_id);

             % Store for cross-subject analysis ONLY if consistent with reference
             if ~isempty(avg_emg_task1) && size(avg_emg_task1,1) == num_timepoints && size(avg_emg_task1,2) == num_emg_channels_ref
                 all_subject_avg_emg_task1 = cat(3, all_subject_avg_emg_task1, avg_emg_task1);
             elseif ~isempty(avg_emg_task1) && size(avg_emg_task1,2) ~= num_emg_channels_ref % Check added for isempty case
                 warning('Subject %d: Avg EMG data (Task %d) channel count (%d) mismatch with reference (%d). Not stored for cross-subject analysis.', subject_id, task_for_avg_plots, size(avg_emg_task1,2), num_emg_channels_ref);
             end
        catch ME % Catch errors specifically from plot_average_emg
            warning('Subject %d: Could not generate Figure 2: %s', subject_id, ME.message); % Report specific error
        end
    else
         fprintf('  Skipping Fig 2 (Avg EMG) for Subject %d as no EMG data was found for this subject.\n', subject_id);
    end

    % Fig 3: Task Comparison
    try
        fprintf('  Generating Fig 3: Task Comparison (Peak Angle %d: %s)...\n', angle_for_task_comp, angle_names{angle_for_task_comp});
        task_comp_metric = plot_task_comparison(joint_angles_cell, angle_for_task_comp, angle_names, subj_output_dir, subject_id);
        if length(task_comp_metric) == num_tasks
            if isempty(all_subject_task_comp_metric)
                 all_subject_task_comp_metric = nan(length(subject_ids), num_tasks); % Initialize if first subject
            end
            all_subject_task_comp_metric(s_idx, :) = task_comp_metric(:)'; % Store row vector
        else
            warning('Subject %d: Task comparison metric not suitable for cross-subject analysis.', subject_id);
             if isempty(all_subject_task_comp_metric)
                 all_subject_task_comp_metric = nan(length(subject_ids), num_tasks);
             end
             all_subject_task_comp_metric(s_idx, :) = nan; % Fill with NaN
        end
    catch ME
        warning('Subject %d: Could not generate Figure 3: %s', subject_id, ME.message);
         if isempty(all_subject_task_comp_metric)
             all_subject_task_comp_metric = nan(length(subject_ids), num_tasks);
         end
         all_subject_task_comp_metric(s_idx, :) = nan; % Fill with NaN on error
    end

    % Fig 4: Correlation Heatmap
    % Use temp_num_emg_current_subj to decide if possible FOR THIS SUBJECT
    if temp_num_emg_current_subj > 0
        try
            fprintf('  Generating Fig 4: EMG-Angle Correlation (Tr %d, Ta %d)...\n', trial_for_detail, task_for_detail);
            % Use correct names for plot labels
             if temp_num_emg_current_subj == num_emg_channels_ref
                current_emg_names_for_plot = emg_channel_names;
            else
                current_emg_names_for_plot = arrayfun(@(x) sprintf('EMG %d', x), 1:temp_num_emg_current_subj, 'UniformOutput', false);
            end
            plot_correlation_heatmap(joint_angles_cell, dsfilt_emg, trial_for_detail, task_for_detail, fs, emg_lowpass_cutoff_hz, angle_names, current_emg_names_for_plot, subj_output_dir, subject_id);
        catch ME
            warning('Subject %d: Could not generate Figure 4: %s', subject_id, ME.message);
        end
    else
        fprintf('  Skipping Fig 4 (Correlation) for Subject %d as no EMG data was found for this subject.\n', subject_id);
    end

    % Fig 5: Single Trial Detail
    % Use temp_num_emg_current_subj for check
    if temp_num_emg_current_subj > 0 && emg_channel_detail <= temp_num_emg_current_subj
        try
            fprintf('  Generating Fig 5: Single Trial Detail (Tr %d, Ta %d)...\n', trial_for_detail, task_for_detail);
             if temp_num_emg_current_subj == num_emg_channels_ref
                current_emg_names_for_plot = emg_channel_names;
            else
                current_emg_names_for_plot = arrayfun(@(x) sprintf('EMG %d', x), 1:temp_num_emg_current_subj, 'UniformOutput', false);
            end
            plot_single_trial_detail(joint_angles_cell, dsfilt_emg, trial_for_detail, task_for_detail, angle_indices_detail, emg_channel_detail, time_vec, angle_names, current_emg_names_for_plot, subj_output_dir, subject_id);
        catch ME
            warning('Subject %d: Could not generate Figure 5: %s', subject_id, ME.message);
        end
    else
         fprintf('  Skipping Fig 5 (Single Trial EMG Detail) for Subject %d due to no EMG data or invalid channel index (%d > %d).\n', subject_id, emg_channel_detail, temp_num_emg_current_subj);
         % Optionally plot just the angles if EMG isn't available/valid
         try
             fprintf('  Generating Fig 5 (Angles Only): Single Trial Detail (Tr %d, Ta %d)...\n', trial_for_detail, task_for_detail);
             plot_single_trial_detail(joint_angles_cell, [], trial_for_detail, task_for_detail, angle_indices_detail, 0, time_vec, angle_names, {}, subj_output_dir, subject_id); % Pass empty EMG, 0 index
         catch ME_angle_only
              warning('Subject %d: Could not generate Figure 5 (Angles Only): %s', subject_id, ME_angle_only.message);
         end
    end

    % --- Calculate & Store Statistics ---
    try
        fprintf('  Calculating statistics for Subject %d...\n', subject_id);
        % Pass the reference EMG names for consistent table structure
        subject_stats = calculate_subject_stats(joint_angles_cell, dsfilt_emg, num_tasks, angle_names, emg_channel_names); % Use reference names
        all_subject_stats(s_idx).subject_id = subject_id;
        all_subject_stats(s_idx).stats = subject_stats;
        valid_subject_indices = [valid_subject_indices, s_idx]; % Mark subject as successfully processed
        fprintf('  Statistics calculated.\n');
    catch ME_stats
        warning('Subject %d: Could not calculate statistics: %s', subject_id, ME_stats.message);
        all_subject_stats(s_idx).subject_id = subject_id; % Store ID even if stats failed
        all_subject_stats(s_idx).stats = []; % Mark stats as empty/failed
    end


end % --- End Subject Processing Loop ---

% --- Generate Cross-Subject Figures & Table ---
fprintf('\n--- Performing Cross-Subject Analysis ---\n');

if ~isempty(valid_subject_indices) % Check if any subjects were processed successfully
    num_valid_subjects = length(valid_subject_indices);
    fprintf('  Based on %d successfully processed subjects.\n', num_valid_subjects);

    cross_subj_output_dir = fullfile(output_dir, 'Cross_Subject_Analysis');
    if ~exist(cross_subj_output_dir, 'dir'), mkdir(cross_subj_output_dir); end

    % Fig Cross 1: Grand Average Angles (Task 1)
    if size(all_subject_avg_angles_task1, 3) >= num_valid_subjects && size(all_subject_avg_angles_task1,3) > 0 % Check if aggregation happened
        try
            fprintf('  Generating Cross-Subject Figure 1: Grand Average Angles (Task %d)...\n', task_for_avg_plots);
            plot_grand_average(all_subject_avg_angles_task1, time_vec, angle_names, 'Angles (degrees)', sprintf('Grand Average Joint Angles (Task %d)', task_for_avg_plots), cross_subj_output_dir, 'CrossSubj_Fig1_GrandAvgAngles');
        catch ME
            warning(ME.identifier,'Could not generate Cross-Subject Figure 1: %s', ME.message);
        end
    else
         warning('Skipping Grand Average Angles plot due to insufficient or inconsistent data across subjects (Aggregated data from %d subjects).', size(all_subject_avg_angles_task1, 3));
    end

    % Fig Cross 2: Grand Average EMG (Task 1)
    if size(all_subject_avg_emg_task1, 3) >= num_valid_subjects && ~isempty(emg_channel_names) && size(all_subject_avg_emg_task1,3) > 0 % Check if aggregation happened
         try
            fprintf('  Generating Cross-Subject Figure 2: Grand Average Rectified EMG (Task %d)...\n', task_for_avg_plots);
            plot_grand_average(all_subject_avg_emg_task1, time_vec, emg_channel_names, 'Avg Rectified EMG', sprintf('Grand Average Rectified EMG (Task %d)', task_for_avg_plots), cross_subj_output_dir, 'CrossSubj_Fig2_GrandAvgEMG');
        catch ME
            warning(ME.identifier,'Could not generate Cross-Subject Figure 2: %s', ME.message);
        end
    else
         warning('Skipping Grand Average EMG plot due to insufficient/inconsistent data or no EMG channels (Aggregated data from %d subjects).', size(all_subject_avg_emg_task1, 3));
    end

    % Fig Cross 3: Boxplot Task Comparison
    if size(all_subject_task_comp_metric, 1) >= length(subject_ids) % Check if array was initialized correctly
        valid_metric_data = all_subject_task_comp_metric(valid_subject_indices, :); % Select only valid subjects' rows
        if ~all(isnan(valid_metric_data(:))) && ~isempty(valid_metric_data) % Check if there's any non-NaN data
            try
                fprintf('  Generating Cross-Subject Figure 3: Boxplot Task Comparison (Angle %d: %s)...\n', angle_for_task_comp, angle_names{angle_for_task_comp});
                plot_cross_subject_boxplot(valid_metric_data, angle_for_task_comp, angle_names, cross_subj_output_dir, 'CrossSubj_Fig3_BoxplotTaskComp');
            catch ME
                warning(ME.identifier,'Could not generate Cross-Subject Figure 3: %s', ME.message);
            end
        else
             warning('Skipping Boxplot Task Comparison due to all NaN or empty metric data.');
        end
    else
         warning('Skipping Boxplot Task Comparison due to inconsistent metric data structure.');
    end


    % Fig 6: Generate Stats Table & Save CSV
    try
        fprintf('  Generating Fig 6: Statistical Summary Table...\n');
        % Pass reference EMG names for table header consistency
        stats_table = generate_stats_table(all_subject_stats(valid_subject_indices), num_tasks, angle_names, emg_channel_names);
        if ~isempty(stats_table)
            table_filename = fullfile(output_dir, 'Stats_Summary_Table.csv');
            writetable(stats_table, table_filename);
            fprintf('  Saved statistics table to: %s\n', table_filename);
        else
             warning('Statistics table is empty. Not saved.');
        end
    catch ME
        warning(ME.identifier,'Could not generate or save statistics table: %s', ME.message);
    end

else
    fprintf('\nNo subjects processed successfully. Skipping cross-subject analysis and table generation.\n');
end

fprintf('\n--- Multi-subject processing complete ---\n');


% --- Helper Functions ---

% (Include updated versions of plot_average_angles, plot_average_emg,
%  plot_task_comparison, plot_single_trial_detail from previous response,
%  modified to accept subject_id and save to subj_output_dir, and return
%  data needed for cross-subject plots where applicable)

% --- Modified Plotting Functions (Returning Data) ---

function avg_angles = plot_average_angles(angles_cell, task_idx, time_vec, angle_names, out_dir, subject_id)
    % Plots the average (+/- std dev) of each joint angle across trials for a specific task.
    % Returns the average angle matrix [time x angle] for cross-subject analysis.

    avg_angles = []; % Initialize return value
    [num_trials, ~] = size(angles_cell);
    num_angles = length(angle_names);
    num_timepoints = length(time_vec);

    % Aggregate data for the specified task across all trials
    task_angles_all_trials = []; % Will be num_timepoints x num_angles x num_trials
    valid_trials = 0;
    for trial = 1:num_trials
        % Check data validity more carefully
        current_cell_data = angles_cell{trial, task_idx};
        if ~isempty(current_cell_data) && ismatrix(current_cell_data) && ...
           size(current_cell_data, 1) == num_timepoints && size(current_cell_data, 2) == num_angles && ...
           ~any(isnan(current_cell_data(:))) % Check for NaNs too
            task_angles_all_trials = cat(3, task_angles_all_trials, current_cell_data);
            valid_trials = valid_trials + 1;
        else
             % Provide more specific warning if possible
             msg = 'empty'; if ~isempty(current_cell_data), msg = sprintf('size [%d %d] vs expected [%d %d] or contains NaN', size(current_cell_data,1), size(current_cell_data,2), num_timepoints, num_angles); end
             % warning('S%d T%d Tr%d: Angle data invalid (%s). Skipping for avg.', subject_id, task_idx, trial, msg); % Reduce verbosity
        end
    end

    if valid_trials < 1 % Allow plotting even for 1 trial, but SD needs >= 2
        % warning('S%d T%d: Cannot compute average angles: No valid trials found.', subject_id, task_idx); % Reduce verbosity
        return;
    end

    % Calculate mean and standard deviation across trials
    mean_angles = mean(task_angles_all_trials, 3);
    if valid_trials >= 2
        std_angles = std(task_angles_all_trials, 0, 3);
    else
        std_angles = zeros(size(mean_angles)); % No SD for 1 trial
    end
    avg_angles = mean_angles; % Return the mean data

    % Create figure
    num_plots = num_angles;
    num_cols = 3; num_rows = ceil(num_plots / num_cols);
    fig_h = figure('Name', sprintf('S%d Avg Angles T%d', subject_id, task_idx), 'NumberTitle', 'off', 'Position', [100, 100, 1200, 800], 'Visible', 'off');
    sgtitle(sprintf('Subject %d - Avg Joint Angles (Task %d, N=%d trials)', subject_id, task_idx, valid_trials), 'Interpreter', 'none');

    for i = 1:num_angles
        subplot(num_rows, num_cols, i);
        hold on;
        plot(time_vec, mean_angles(:, i), 'b', 'LineWidth', 1.5);
        if valid_trials >= 2
            fill([time_vec; flipud(time_vec)], [mean_angles(:, i) - std_angles(:, i); flipud(mean_angles(:, i) + std_angles(:, i))], ...
                 'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        end
        hold off;
        grid on; xlabel('Time (s)'); ylabel('Angle (deg)'); title(angle_names{i}, 'Interpreter', 'none'); xlim([time_vec(1), time_vec(end)]);
    end

    fig_filename = fullfile(out_dir, sprintf('Fig1_AvgAngles_S%d_T%d.png', subject_id, task_idx));
    try
        print(fig_h, fig_filename, '-dpng', '-r300');
        % fprintf('   Saved: %s\n', fig_filename);
    catch ME_save
        warning('Could not save figure %s: %s', fig_filename, ME_save.message);
    end
    close(fig_h);
end

function avg_emg = plot_average_emg(emg_cell, task_idx, time_vec, emg_names_for_plot, out_dir, subject_id)
    % Plots the average (+/- std dev) of rectified EMG across trials for a specific task.
    % Returns the average rectified EMG matrix [time x emg] for cross-subject analysis.
    % Uses emg_names_for_plot for labeling the figure for this subject.

    avg_emg = []; % Initialize return value
    [num_trials, ~] = size(emg_cell);
    num_emg_channels_subj = length(emg_names_for_plot); % Channels for this subject
    num_timepoints = length(time_vec);

    if num_emg_channels_subj == 0, return; end % Skip if no channels for this subject

    % Aggregate data
    task_emg_all_trials = []; valid_trials = 0;
    for trial = 1:num_trials
         current_cell_data = emg_cell{trial, task_idx};
         if ~isempty(current_cell_data) && ismatrix(current_cell_data) && ...
            size(current_cell_data, 1) == num_timepoints && size(current_cell_data, 2) == num_emg_channels_subj && ...
            ~any(isnan(current_cell_data(:)))
            rectified_emg = abs(current_cell_data); % Rectify EMG
            task_emg_all_trials = cat(3, task_emg_all_trials, rectified_emg);
            valid_trials = valid_trials + 1;
         else
             msg = 'empty'; if ~isempty(current_cell_data), msg = sprintf('size [%d %d] vs expected [%d %d] or contains NaN', size(current_cell_data,1), size(current_cell_data,2), num_timepoints, num_emg_channels_subj); end
             % warning('S%d T%d Tr%d: EMG data invalid (%s). Skipping for avg.', subject_id, task_idx, trial, msg); % Reduce verbosity
         end
    end

    if valid_trials < 1
        % warning('S%d T%d: Cannot compute average EMG: No valid trials found.', subject_id, task_idx); % Reduce verbosity
        return;
    end

    % Calculate mean and standard deviation
    mean_emg = mean(task_emg_all_trials, 3);
     if valid_trials >= 2
        std_emg = std(task_emg_all_trials, 0, 3);
    else
        std_emg = zeros(size(mean_emg));
    end
    avg_emg = mean_emg; % Return mean data

    % Create figure
    num_plots = num_emg_channels_subj;
    num_cols = 2; num_rows = ceil(num_plots / num_cols);
    fig_h = figure('Name', sprintf('S%d Avg EMG T%d', subject_id, task_idx), 'NumberTitle', 'off', 'Position', [150, 150, 1000, 700], 'Visible', 'off');
    sgtitle(sprintf('Subject %d - Avg Rectified EMG (Task %d, N=%d trials)', subject_id, task_idx, valid_trials), 'Interpreter', 'none');

    for i = 1:num_emg_channels_subj
        subplot(num_rows, num_cols, i);
        hold on;
        plot(time_vec, mean_emg(:, i), 'r', 'LineWidth', 1.5);
        if valid_trials >= 2
            fill([time_vec; flipud(time_vec)], [mean_emg(:, i) - std_emg(:, i); flipud(mean_emg(:, i) + std_emg(:, i))], ...
                 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        end
        hold off;
        grid on; xlabel('Time (s)'); ylabel('Rect. EMG Amp.'); title(emg_names_for_plot{i}, 'Interpreter', 'none'); xlim([time_vec(1), time_vec(end)]);
        % --- FIX for Figure 2 Error ---
        % Replace ylim(bottom=0) with axes handle manipulation
        ax=gca; current_ylim=ax.YLim; ax.YLim = [0, current_ylim(2)];
        % --- End FIX ---
    end

    fig_filename = fullfile(out_dir, sprintf('Fig2_AvgEMG_S%d_T%d.png', subject_id, task_idx));
     try
        print(fig_h, fig_filename, '-dpng', '-r300');
        % fprintf('   Saved: %s\n', fig_filename);
    catch ME_save
        warning('Could not save figure %s: %s', fig_filename, ME_save.message);
    end
    close(fig_h);
end

function task_means = plot_task_comparison(angles_cell, angle_idx, angle_names, out_dir, subject_id)
    % Compares a metric (peak value) of a specific angle across tasks for one subject.
    % Returns the mean metric value per task [1 x num_tasks] for cross-subject analysis.

    [num_trials, num_tasks] = size(angles_cell);
    task_means = nan(1, num_tasks); % Initialize return value with NaN

    if angle_idx > length(angle_names) || angle_idx < 1, error('Invalid angle_idx.'); end
    angle_name = angle_names{angle_idx};
    num_angles_expected = length(angle_names);

    task_stds = nan(1, num_tasks);
    valid_task_counts = zeros(1, num_tasks);
    num_timepoints_ref = []; % Reference timepoints from first valid trial/task

    for task = 1:num_tasks
        task_metric_values = [];
        for trial = 1:num_trials
            current_cell_data = angles_cell{trial, task};
            if ~isempty(current_cell_data) && ismatrix(current_cell_data) && size(current_cell_data, 2) == num_angles_expected && ~any(isnan(current_cell_data(:)))
                % Check timepoint consistency
                current_timepoints = size(current_cell_data, 1);
                if isempty(num_timepoints_ref)
                    num_timepoints_ref = current_timepoints;
                elseif current_timepoints ~= num_timepoints_ref
                     % warning('S%d T%d Tr%d: Inconsistent timepoints (%d vs %d). Skipping trial for task comp.', subject_id, task, trial, current_timepoints, num_timepoints_ref); % Reduce verbosity
                     continue;
                end

                angle_data = current_cell_data(:, angle_idx);
                metric_value = max(angle_data); % Peak Value
                task_metric_values = [task_metric_values; metric_value]; %#ok<AGROW>
            else
                 msg = 'empty'; if ~isempty(current_cell_data), msg = sprintf('size [%d %d] vs expected [~ %d] or contains NaN', size(current_cell_data,1), size(current_cell_data,2), num_angles_expected); end
                 % warning('S%d T%d Tr%d: Angle data invalid (%s). Skipping trial for task comp.', subject_id, task, trial, msg); % Reduce verbosity
            end
        end

        if ~isempty(task_metric_values)
            task_means(task) = mean(task_metric_values);
            task_stds(task) = std(task_metric_values);
            valid_task_counts(task) = length(task_metric_values);
        end % Keep NaN if no valid trials
    end

    % Create bar chart
    fig_h = figure('Name', sprintf('S%d Task Comp Ang%d', subject_id, angle_idx), 'NumberTitle', 'off', 'Position', [200, 200, 800, 500], 'Visible', 'off');
    hold on;
    valid_tasks_mask = ~isnan(task_means);
    if any(valid_tasks_mask)
        bar(find(valid_tasks_mask), task_means(valid_tasks_mask));
        errorbar(find(valid_tasks_mask), task_means(valid_tasks_mask), task_stds(valid_tasks_mask), '.', 'Color', 'k', 'LineWidth', 1);
    else
        bar(1:num_tasks, nan(1,num_tasks)); % Plot empty bar if no valid data
        text(num_tasks/2 + 0.5, 0.5, 'No Valid Data', 'HorizontalAlignment','center'); % Adjusted position slightly
    end
    hold off;
    grid on;

    xlabel('Task Index'); ylabel(sprintf('Peak Angle (deg) - %s', angle_name), 'Interpreter', 'none');
    title(sprintf('Subject %d - Peak "%s" Across Tasks (Avg ± SD over trials)', subject_id, angle_name), 'Interpreter', 'none');
    xticks(1:num_tasks); xticklabels(arrayfun(@(x) sprintf('T%d (N=%d)', x, valid_task_counts(x)), 1:num_tasks, 'UniformOutput', false));
    xlim([0.5, num_tasks + 0.5]);
    current_ylim = ylim; % Get current auto-limits
    if current_ylim(1) > 0, ylim([0, current_ylim(2)]); end % Ensure y starts at 0 if all positive


    fig_filename = fullfile(out_dir, sprintf('Fig3_TaskComp_S%d_A%d.png', subject_id, angle_idx));
     try
        print(fig_h, fig_filename, '-dpng', '-r300');
        % fprintf('   Saved: %s\n', fig_filename);
    catch ME_save
        warning('Could not save figure %s: %s', fig_filename, ME_save.message);
    end
    close(fig_h);
end


% --- New Plotting Functions ---

function plot_correlation_heatmap(angles_cell, emg_cell, trial_idx, task_idx, fs, lp_cutoff, angle_names, emg_names_subj, out_dir, subject_id)
    % Calculates and plots the correlation matrix between angles and filtered EMG.
    % Uses emg_names_subj for labeling specific to this subject's channels.

    if isempty(angles_cell{trial_idx, task_idx}) || isempty(emg_cell{trial_idx, task_idx})
        warning('S%d T%d Tr%d: Angle or EMG data empty. Skipping correlation.', subject_id, task_idx, trial_idx);
        return;
    end

    angles_data = angles_cell{trial_idx, task_idx};
    emg_data = emg_cell{trial_idx, task_idx};
    num_angles = length(angle_names);
    num_emg_subj = length(emg_names_subj); % Use subject-specific count

    if size(angles_data, 2) ~= num_angles || size(emg_data, 2) ~= num_emg_subj
         warning('S%d T%d Tr%d: Angle(%d)/EMG(%d) channel mismatch with names Angle(%d)/EMG(%d). Skipping correlation.', ...
                 subject_id, task_idx, trial_idx, size(angles_data, 2), size(emg_data, 2), num_angles, num_emg_subj);
         return;
    end
    if size(angles_data, 1) ~= size(emg_data, 1) || size(angles_data,1) == 0
         warning('S%d T%d Tr%d: Angle/EMG timepoint mismatch or zero length. Skipping correlation.', subject_id, task_idx, trial_idx);
         return;
    end

    % Preprocess EMG: Rectify and low-pass filter to get envelope
    emg_rectified = abs(emg_data);
    try
        % Requires Signal Processing Toolbox
        emg_filtered = lowpass(emg_rectified, lp_cutoff, fs);
    catch % Handle missing toolbox
         warning('Signal Processing Toolbox "lowpass" function not found or failed. Using rectified EMG for correlation.');
         emg_filtered = emg_rectified;
    end


    % Combine data and calculate correlation matrix
    combined_data = [angles_data, emg_filtered];
    % Check for columns with zero variance (causes NaN in corrcoef)
    zero_var_cols = std(combined_data, 0, 1) < 1e-10; % Check std dev along time dimension
    if any(zero_var_cols)
        warning('S%d T%d Tr%d: Constant data found in some channels (Indices: %s). Correlation may contain NaNs.', subject_id, task_idx, trial_idx, sprintf('%d ', find(zero_var_cols)));
        combined_data(:, zero_var_cols) = combined_data(:, zero_var_cols) + randn(size(combined_data,1), sum(zero_var_cols))*1e-9; % Add tiny noise
    end

    try
        corr_matrix = corrcoef(combined_data);
    catch ME_corr
         warning('S%d T%d Tr%d: Error calculating corrcoef: %s. Skipping correlation.', subject_id, task_idx, trial_idx, ME_corr.message);
         return;
    end


    % Extract the EMG vs Angle correlation submatrix
    emg_angle_corr = corr_matrix(1:num_angles, num_angles+1:end); % Rows: Angles, Cols: EMG

    % Create heatmap
    fig_h = figure('Name', sprintf('S%d Corr T%d Tr%d', subject_id, task_idx, trial_idx), 'NumberTitle', 'off', 'Position', [250, 250, 800, 700], 'Visible', 'off');
    try
        % --- FIX for Heatmap Error ---
        % Pass emg_angle_corr directly (size num_angles x num_emg_subj)
        % xvalues = emg_names_subj (length num_emg_subj) -> matches columns
        % yvalues = angle_names (length num_angles) -> matches rows
        heatmap(emg_names_subj, angle_names, emg_angle_corr, ... % REMOVED TRANSPOSE
                'Colormap', coolwarm, ...
                'ColorLimits', [-1, 1], ...
                'Title', sprintf('Subject %d - EMG vs Angle Correlation (Task %d, Trial %d)', subject_id, task_idx, trial_idx), ...
                'XLabel', 'EMG Channels', ...
                'YLabel', 'Joint Angles', ...
                'MissingDataColor', [0.8 0.8 0.8], 'MissingDataLabel', 'NaN');
        % --- End FIX ---

        % Adjust font size if needed
        h = gca; h.FontSize = 8;
    catch ME_heatmap
         warning('S%d T%d Tr%d: Error creating heatmap: %s', subject_id, task_idx, trial_idx, ME_heatmap.message);
         close(fig_h); return;
    end


    fig_filename = fullfile(out_dir, sprintf('Fig4_Correlation_S%d_T%d_Tr%d.png', subject_id, task_idx, trial_idx));
     try
        print(fig_h, fig_filename, '-dpng', '-r300');
        % fprintf('   Saved: %s\n', fig_filename);
    catch ME_save
        warning('Could not save figure %s: %s', fig_filename, ME_save.message);
    end
    close(fig_h);

    % Define coolwarm colormap function if not built-in
    function cmap = coolwarm()
        x = linspace(-1, 1, 256)';
        r = 0.23 + 0.77 * (1 + x) / 2;
        b = 0.23 + 0.77 * (1 - x) / 2;
        g = 1 - 0.8 * abs(x); % Adjust green component for better white point
        cmap = max(0, min(1, [r, g, b])); % Clamp values between 0 and 1
    end
end

function plot_single_trial_detail(angles_cell, emg_cell, trial_idx, task_idx, angle_indices, emg_idx, time_vec, angle_names, emg_names_subj, out_dir, subject_id)
    % Plots selected joint angles and an EMG channel for a single trial.
    % Handles case where emg_cell is empty or emg_idx is 0.
    % Uses emg_names_subj for labeling specific to this subject's channels.

    if isempty(angles_cell{trial_idx, task_idx})
        warning('S%d T%d Tr%d: Angle data empty. Skipping detail plot.', subject_id, task_idx, trial_idx);
        return;
    end
    angles_data = angles_cell{trial_idx, task_idx};
    num_timepoints = length(time_vec);
    num_angles_expected = length(angle_names);

    emg_data = [];
    emg_title = 'EMG Data Not Available';
    plot_emg = false;
    num_emg_subj = length(emg_names_subj); % Channels for this subject

    if ~isempty(emg_cell) && iscell(emg_cell) && ~isempty(emg_cell{trial_idx, task_idx}) && emg_idx > 0
        current_emg_cell_data = emg_cell{trial_idx, task_idx};
        if size(current_emg_cell_data, 2) >= emg_idx % Check if index is valid for actual data
            if emg_idx <= num_emg_subj % Check consistency with names provided
                 emg_data = current_emg_cell_data(:, emg_idx);
                 if size(emg_data, 1) == num_timepoints
                     plot_emg = true;
                     emg_title = sprintf('EMG Channel: %s', emg_names_subj{emg_idx});
                 else
                     warning('S%d T%d Tr%d: EMG timepoints mismatch angles. Skipping EMG plot.', subject_id, task_idx, trial_idx);
                 end
            else
                 warning('S%d T%d Tr%d: EMG index %d out of bounds for names provided (%d). Skipping EMG plot.', subject_id, task_idx, trial_idx, emg_idx, num_emg_subj);
            end
        else
             warning('S%d T%d Tr%d: EMG index %d out of bounds for data columns (%d). Skipping EMG plot.', subject_id, task_idx, trial_idx, emg_idx, size(current_emg_cell_data, 2));
        end
    end

    if size(angles_data,1) ~= num_timepoints || size(angles_data,2) ~= num_angles_expected
        warning('S%d T%d Tr%d: Angle timepoints/channel mismatch. Skipping detail plot.', subject_id, task_idx, trial_idx);
        return;
    end

    fig_h = figure('Name', sprintf('S%d Detail T%d Tr%d', subject_id, task_idx, trial_idx), 'NumberTitle', 'off', 'Position', [300, 300, 900, 600], 'Visible', 'off');
    sgtitle(sprintf('Subject %d - Single Trial Detail (Task %d, Trial %d)', subject_id, task_idx, trial_idx), 'Interpreter', 'none');

    % Plot Angles
    subplot(2, 1, 1);
    hold on;
    colors = lines(length(angle_indices)); legends_angle = {};
    for i = 1:length(angle_indices)
        idx = angle_indices(i);
        if idx > 0 && idx <= num_angles_expected
            plot(time_vec, angles_data(:, idx), 'LineWidth', 1.5, 'Color', colors(i,:));
            legends_angle{end+1} = angle_names{idx}; %#ok<AGROW>
        else
             warning('S%d T%d Tr%d: Invalid angle index %d requested.', subject_id, task_idx, trial_idx, idx);
        end
    end
    hold off; grid on; ylabel('Angle (degrees)'); title('Joint Angles');
    if ~isempty(legends_angle), legend(legends_angle, 'Interpreter', 'none', 'Location', 'best'); end
    xlim([time_vec(1), time_vec(end)]);

    % Plot EMG
    subplot(2, 1, 2);
    if plot_emg
        plot(time_vec, emg_data, 'r', 'LineWidth', 1);
        ylabel('EMG Amplitude');
    else
        text(0.5, 0.5, 'EMG Data Not Plotted', 'HorizontalAlignment', 'center');
        set(gca, 'ytick', [], 'ycolor', 'none'); % Hide y-axis if no data
    end
    grid on; title(emg_title, 'Interpreter', 'none'); xlim([time_vec(1), time_vec(end)]); xlabel('Time (s)');

    fig_filename = fullfile(out_dir, sprintf('Fig5_Detail_S%d_T%d_Tr%d_A%s_E%d.png', ...
        subject_id, task_idx, trial_idx, sprintf('%d', angle_indices), emg_idx));
     try
        print(fig_h, fig_filename, '-dpng', '-r300');
        % fprintf('   Saved: %s\n', fig_filename);
    catch ME_save
        warning('Could not save figure %s: %s', fig_filename, ME_save.message);
    end
    close(fig_h);
end


% --- Statistics Calculation ---

function subject_stats = calculate_subject_stats(angles_cell, emg_cell, num_tasks, angle_names, ref_emg_names)
    % Calculates summary statistics (mean peak angle, mean avg rectified EMG)
    % averaged across trials for each task for one subject.
    % Uses ref_emg_names for structure consistency, but checks actual data size.

    [num_trials, ~] = size(angles_cell);
    num_angles = length(angle_names);
    num_ref_emg = length(ref_emg_names); % Expected EMG channels based on reference
    subject_stats = struct();
    num_timepoints_ref = []; % Reference timepoints from first valid trial/task

    for task = 1:num_tasks
        task_peak_angles_all_trials = nan(num_trials, num_angles);
        task_avg_emg_all_trials = nan(num_trials, num_ref_emg); % Use reference size for storage consistency
        valid_angle_trials = 0;
        valid_emg_trials = 0;

        for trial = 1:num_trials
            angles_data = [];
            emg_data = [];
            current_timepoints = NaN;

            % Process Angles
            current_angle_cell_data = angles_cell{trial, task};
            if ~isempty(current_angle_cell_data) && ismatrix(current_angle_cell_data) && size(current_angle_cell_data, 2) == num_angles && ~any(isnan(current_angle_cell_data(:)))
                 current_timepoints = size(current_angle_cell_data, 1);
                 if isempty(num_timepoints_ref)
                     num_timepoints_ref = current_timepoints;
                 elseif current_timepoints ~= num_timepoints_ref
                      % warning('Stats T%d Tr%d: Inconsistent angle timepoints (%d vs %d). Skipping trial for stats.', task, trial, current_timepoints, num_timepoints_ref); % Reduce verbosity
                      continue; % Skip whole trial if inconsistent
                 end
                 angles_data = current_angle_cell_data;
                 task_peak_angles_all_trials(trial, :) = max(angles_data, [], 1); % Peak angle per channel
                 valid_angle_trials = valid_angle_trials + 1;
            end

            % Process EMG (only if angles were valid for timepoint check)
            if ~isnan(current_timepoints) && num_ref_emg > 0 && ~isempty(emg_cell) && iscell(emg_cell) && ~isempty(emg_cell{trial, task})
                 current_emg_cell_data = emg_cell{trial, task};
                 if size(current_emg_cell_data, 1) == current_timepoints % Check consistency with angles
                     num_actual_emg_subj = size(current_emg_cell_data, 2);
                     if num_actual_emg_subj == num_ref_emg % Check channel consistency
                         if ~any(isnan(current_emg_cell_data(:)))
                             emg_data = current_emg_cell_data;
                             task_avg_emg_all_trials(trial, :) = mean(abs(emg_data), 1); % Mean rectified EMG per channel
                             valid_emg_trials = valid_emg_trials + 1;
                         end
                     % else % Channel count mismatch already warned in main loop
                     end
                 end
            end
        end % End trial loop for task

        % Store mean stats for the task (average across trials)
        if valid_angle_trials > 0
            subject_stats(task).mean_peak_angles = mean(task_peak_angles_all_trials, 1, 'omitnan');
        else
            subject_stats(task).mean_peak_angles = nan(1, num_angles);
        end
        if valid_emg_trials > 0 && num_ref_emg > 0
            subject_stats(task).mean_avg_rect_emg = mean(task_avg_emg_all_trials, 1, 'omitnan');
        else
             subject_stats(task).mean_avg_rect_emg = nan(1, num_ref_emg);
        end
         subject_stats(task).num_valid_angle_trials = valid_angle_trials;
         subject_stats(task).num_valid_emg_trials = valid_emg_trials; % Based on reference channel count

    end % End task loop
end


% --- Stats Table Generation ---

function stats_table = generate_stats_table(all_subject_stats, num_tasks, angle_names, ref_emg_names)
    % Creates a MATLAB table summarizing the calculated statistics.
    % Uses ref_emg_names for consistent table structure.

    num_subjects = length(all_subject_stats);
    num_angles = length(angle_names);
    num_ref_emg = length(ref_emg_names); % Use reference count

    if num_subjects == 0, stats_table = table(); return; end

    % Create variable names for the table
    var_names = {'SubjectID', 'Task'};
    peak_angle_vars = arrayfun(@(x) matlab.lang.makeValidName(sprintf('PeakAngle_%s', angle_names{x})), 1:num_angles, 'UniformOutput', false); % Ensure valid names
    avg_emg_vars = {};
    if num_ref_emg > 0
        avg_emg_vars = arrayfun(@(x) matlab.lang.makeValidName(sprintf('AvgRectEMG_%s', ref_emg_names{x})), 1:num_ref_emg, 'UniformOutput', false); % Ensure valid names
    end
    var_names = [var_names, peak_angle_vars, avg_emg_vars];

    % Preallocate cell array to hold table data
    table_data = cell(num_subjects * num_tasks, length(var_names));
    row_idx = 1;

    for s = 1:num_subjects
        subject_id = all_subject_stats(s).subject_id;
        subject_stats = all_subject_stats(s).stats;

        if isempty(subject_stats), continue; end % Skip if stats failed for subject

        for task = 1:num_tasks
            table_data{row_idx, 1} = subject_id;
            table_data{row_idx, 2} = task;

            % Add peak angles
            if isfield(subject_stats(task), 'mean_peak_angles') && length(subject_stats(task).mean_peak_angles) == num_angles
                 table_data(row_idx, 3:(3+num_angles-1)) = num2cell(subject_stats(task).mean_peak_angles);
            else % Fill with NaN if missing/wrong size
                 table_data(row_idx, 3:(3+num_angles-1)) = num2cell(nan(1, num_angles));
            end

            % Add avg EMG (based on reference number of channels)
            if num_ref_emg > 0
                if isfield(subject_stats(task), 'mean_avg_rect_emg') && length(subject_stats(task).mean_avg_rect_emg) == num_ref_emg
                     table_data(row_idx, (3+num_angles):end) = num2cell(subject_stats(task).mean_avg_rect_emg);
                else % Fill with NaN if missing/wrong size
                     table_data(row_idx, (3+num_angles):end) = num2cell(nan(1, num_ref_emg));
                end
            end
            row_idx = row_idx + 1;
        end
    end

    % Remove empty rows if any subjects were skipped entirely
    table_data = table_data(1:row_idx-1, :);

    % Convert cell array to table
    if ~isempty(table_data)
        stats_table = cell2table(table_data, 'VariableNames', var_names);
    else
        stats_table = table(); % Return empty table if no data
    end
end


% --- Cross-Subject Plotting Functions ---

function plot_grand_average(all_subject_data, time_vec, channel_names, y_label_text, title_text, out_dir, base_filename)
    % Plots the grand average (+/- std dev across subjects) for multi-channel time series data.
    % all_subject_data: [time x channel x subject]

    if isempty(all_subject_data) || ndims(all_subject_data) ~= 3 || size(all_subject_data,3) < 2 % Need at least 2 subjects for SD
        warning('Insufficient or invalid data (%d subjects) provided for grand average plot: %s. Skipping.', size(all_subject_data,3), base_filename);
        return;
    end

    num_channels = length(channel_names);
    num_subjects = size(all_subject_data, 3);
    num_timepoints = length(time_vec);

    if size(all_subject_data, 1) ~= num_timepoints || size(all_subject_data, 2) ~= num_channels
         warning('Dimension mismatch in grand average data: %s. Skipping.', base_filename);
         return;
    end

    % Calculate grand mean and std dev across subjects
    grand_mean = mean(all_subject_data, 3, 'omitnan'); % Use omitnan
    grand_std = std(all_subject_data, 0, 3, 'omitnan'); % Use omitnan

    % Create figure
    num_plots = num_channels;
    num_cols = ceil(sqrt(num_plots)); num_rows = ceil(num_plots / num_cols);

    fig_h = figure('Name', title_text, 'NumberTitle', 'off', 'Position', [100, 100, 1200, 800], 'Visible', 'off');
    sgtitle(sprintf('%s (N=%d Subjects, Avg ± SD across subjects)', title_text, num_subjects), 'Interpreter', 'none');

    for i = 1:num_channels
        subplot(num_rows, num_cols, i);
        hold on;
        plot(time_vec, grand_mean(:, i), 'k', 'LineWidth', 1.5); % Grand mean in black
        fill([time_vec; flipud(time_vec)], [grand_mean(:, i) - grand_std(:, i); flipud(grand_mean(:, i) + grand_std(:, i))], ...
             'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none'); % SD in grey
        hold off;
        grid on; xlabel('Time (s)'); ylabel(y_label_text); title(channel_names{i}, 'Interpreter', 'none');
        xlim([time_vec(1), time_vec(end)]);
        if contains(lower(y_label_text), 'emg'), ax=gca; current_ylim=ax.YLim; ax.YLim = [0, current_ylim(2)]; end % Ensure EMG >= 0
    end

    fig_filename = fullfile(out_dir, sprintf('%s.png', base_filename));
     try
        print(fig_h, fig_filename, '-dpng', '-r300');
        fprintf('   Saved: %s\n', fig_filename);
    catch ME_save
        warning('Could not save figure %s: %s', fig_filename, ME_save.message);
    end
    close(fig_h);
end


function plot_cross_subject_boxplot(subject_task_metric, angle_idx, angle_names, out_dir, base_filename)
    % Creates boxplots comparing a metric across tasks, showing distribution across subjects.
    % subject_task_metric: [subject x task] matrix of the metric values

    if isempty(subject_task_metric) || ndims(subject_task_metric) ~= 2
        warning('Invalid data provided for cross-subject boxplot: %s. Skipping.', base_filename);
        return;
    end

    [num_subjects, num_tasks] = size(subject_task_metric);
    angle_name = angle_names{angle_idx};

    fig_h = figure('Name', 'Cross-Subject Task Comparison', 'NumberTitle', 'off', 'Position', [200, 200, 900, 500], 'Visible', 'off');

    % Create boxplot - requires data in a single vector and a grouping variable
    metric_vector = subject_task_metric(:); % Reshape matrix into a vector
    task_group = repmat((1:num_tasks)', num_subjects, 1); % Create grouping variable for tasks

    valid_data_mask = ~isnan(metric_vector); % Exclude NaN values if any subjects failed

    if ~any(valid_data_mask)
        warning('No valid (non-NaN) data for cross-subject boxplot: %s. Skipping.', base_filename);
        close(fig_h);
        return;
    end

    boxplot(metric_vector(valid_data_mask), task_group(valid_data_mask), 'Labels', arrayfun(@(x) sprintf('Task %d', x), 1:num_tasks, 'UniformOutput', false));

    grid on;
    ylabel(sprintf('Peak Angle (deg) - %s', angle_name), 'Interpreter', 'none');
    xlabel('Task Index');
    title(sprintf('Cross-Subject Distribution of Peak "%s" (N=%d Subjects)', angle_name, num_subjects), 'Interpreter', 'none');

    fig_filename = fullfile(out_dir, sprintf('%s_A%d.png', base_filename, angle_idx));
     try
        print(fig_h, fig_filename, '-dpng', '-r300');
        fprintf('   Saved: %s\n', fig_filename);
    catch ME_save
        warning('Could not save figure %s: %s', fig_filename, ME_save.message);
    end
    close(fig_h);
end
