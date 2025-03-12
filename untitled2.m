% Biophysical Model-Informed Signal Processing for Finger Movement Estimation
% Merging Trials (Tasks 1-6), Removing Task 7, and Synergy Analysis
% MODIFIED to MERGE Kinematics for Correlation Analysis

% Load the data
load('s1.mat'); % Ensure s1.mat is in your MATLAB path

% --- 1. Set Parameters ---
num_synergies = 3;      % Number of muscle synergies to extract (adjust as needed)
smoothing_window_size = 100; % Window size for moving average smoothing of EMG (adjust as needed)
emg_sampling_rate = 1000; % Assumed EMG sampling rate in Hz (adjust if different)
kin_sampling_rate = 100;  % Assumed Kinematics sampling rate in Hz (adjust if different)

num_trials = size(dsfilt_emg, 1); % Get number of trials
num_tasks_original = size(dsfilt_emg, 2); % Original number of tasks
tasks_to_process = 1:6; % Process Tasks 1 to 6 (remove Task 7)
num_tasks_processed = length(tasks_to_process);

% --- 2. Merge Trials for Tasks 1-6 and Extract/Merge Kinematics ---
merged_emg_data = cell(1, num_tasks_processed); % Cell array for merged EMG (1 row, tasks as columns)
merged_kinematic_data = cell(1, num_tasks_processed); % Cell array for MERGED kinematics

for task_index = 1:num_tasks_processed
    task_num = tasks_to_process(task_index);
    merged_emg_task = []; % Initialize for each task
    merged_kin_task = []; % Initialize merged kinematics for each task
    for trial_num = 1:num_trials
        try
            emg_data_cell = dsfilt_emg{trial_num, task_num};
            merged_emg_task = [merged_emg_task; emg_data_cell]; % Vertically concatenate trials

            kin_data_cell = finger_kinematics{trial_num, task_num};
            merged_kin_task = [merged_kin_task; kin_data_cell]; % Vertically concatenate kinematics too!

        catch
            warning('Data missing for Trial %d, Task %d. Skipping trial for merging.', trial_num, task_num);
        end
    end
    merged_emg_data{task_index} = merged_emg_task; % Store merged EMG for the task
    merged_kinematic_data{task_index} = merged_kin_task; % Store merged kinematics for the task
    fprintf('Trials merged for Task %d.\n', task_num);
end

disp('Trial merging complete for Tasks 1-6.');


% --- 3. Initialize Storage for Results ---
W_tasks = cell(1, num_tasks_processed);
H_tasks = cell(1, num_tasks_processed);
VAF_tasks = zeros(1, num_tasks_processed);
Correlation_tasks = cell(1, num_tasks_processed);


% --- 4. Loop through Processed Tasks (1-6) ---
for task_index = 1:num_tasks_processed
    task_num = tasks_to_process(task_index);

    fprintf('Processing Task %d (Merged Trials)...\n', task_num);

    % --- 5. Data Preprocessing ---
    emg_matrix_nmf = merged_emg_data{task_index}; % Get merged EMG data for the task
    kinematic_matrix = merged_kinematic_data{task_index}; % Get MERGED kinematic data

    % Rectification of EMG signal (absolute value)
    rect_emg = abs(emg_matrix_nmf);

    % Smoothing EMG using moving average filter
    smooth_emg = filter(ones(1,smoothing_window_size)/smoothing_window_size, 1, rect_emg);
    emg_matrix_nmf = smooth_emg;

    % --- 6. Non-negative Matrix Factorization (NMF) ---
    [W, H] = nnmf(emg_matrix_nmf', num_synergies);

    % --- 7. Calculate Variance Accounted For (VAF) ---
    EMG_reconstructed = W * H;
    VAF_channel = zeros(1,size(emg_matrix_nmf,2));
    for channel = 1:size(emg_matrix_nmf,2)
        VAF_channel(channel) = 1 - var(emg_matrix_nmf(:,channel) - EMG_reconstructed(channel,:)') / var(emg_matrix_nmf(:,channel));
    end
    VAF_total = mean(VAF_channel) * 100;

    % --- 8. Correlation Analysis (Example - First 3 Kinematic Columns) ---
    if ~isempty(kinematic_matrix)
        kinematic_subset = kinematic_matrix(:, 1:3); % Example: Use first 3 kinematic columns
        % --- MODIFIED LINE: Use H' (transposed H) and kinematic_subset ---
        correlation_matrix = corr(H', kinematic_subset);
        Correlation_tasks{task_index} = correlation_matrix;
    else
        Correlation_tasks{task_index} = [];
    end


    % --- 9. Store Results for Task ---
    W_tasks{task_index} = W;
    H_tasks{task_index} = H;
    VAF_tasks(task_index) = VAF_total;

    fprintf('  VAF for Task %d (Merged Trials): %.2f%%\n', task_num, VAF_total);

end % Task loop

disp('Finished processing merged trials for Tasks 1-6.');

% --- 10. Save Combined Results to a .mat File ---
output_filename = 'synergy_analysis_results_merged_trials_tasks1to6.mat';
save(output_filename, 'W_tasks', 'H_tasks', 'VAF_tasks', 'Correlation_tasks', 'num_synergies', 'smoothing_window_size', 'emg_sampling_rate', 'kin_sampling_rate', 'tasks_to_process');
disp(['Results saved to: ', output_filename]);


% --- 11. Visualizations (Rest of Visualization Code remains the same) ---

% --- 11.1 VAF Bar Plot for Tasks ---
figure('Name', 'VAF for Tasks 1-6 (Merged Trials, 3 Synergies)', 'Position', [100, 100, 800, 400]);
bar(tasks_to_process, VAF_tasks);
title(['VAF (%) for Tasks 1-6 (Merged Trials, ', num2str(num_synergies), ' Synergies)']);
xlabel('Task Number (1-6)');
ylabel('VAF (%)');
xticks(tasks_to_process);
ylim([min(VAF_tasks)-2, max(VAF_tasks)+2]);
for i = 1:num_tasks_processed
    text(tasks_to_process(i), VAF_tasks(i), sprintf('%.1f', VAF_tasks(i)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end


% --- 11.2. Visualize Synergy Vectors (W) for each Task (Example for the first task, you can loop or choose tasks) ---
task_to_visualize_W = 1; % Change this to visualize W for different tasks (1 to 6)
W_current_task = W_tasks{task_to_visualize_W};
figure('Name', ['Synergy Vectors (W) - Task ', num2str(tasks_to_process(task_to_visualize_W))], 'Position', [100, 100, 800, 400]);
bar(W_current_task);
title(['Synergy Vectors (W) - Task ', num2str(tasks_to_process(task_to_visualize_W)), ' (Merged Trials, ', num2str(num_synergies), ' Synergies)']);
xlabel('Muscle Index (1-8: APL, FCR, FDS, FDP, ED, EI, ECU, ECR)');
ylabel('Synergy Weight');
legendStrings_W = cell(1, num_synergies);
for i = 1:num_synergies
    legendStrings_W{i} = ['Synergy ', num2str(i)];
end
legend(legendStrings_W, 'Location', 'eastoutside');
set(gca, 'XTickLabel', {"APL", "FCR", "FDS", "FDP", "ED", "EI", "ECU", "ECR"}); % Corrected line


% --- 11.3. Visualize Synergy Activation Coefficients (H) for a Task (Example for the first task) ---
task_to_visualize_H = 1;
H_current_task = H_tasks{task_to_visualize_H};
time_vector_emg_task = (1:size(H_current_task, 2))/emg_sampling_rate;
figure('Name', ['Synergy Activation (H) - Task ', num2str(tasks_to_process(task_to_visualize_H))], 'Position', [100, 100, 1200, 400]);
plot(time_vector_emg_task, H_current_task');
title(['Synergy Activation Coefficients (H) - Task ', num2str(tasks_to_process(task_to_visualize_H)), ' (Merged Trials, ', num2str(num_synergies), ' Synergies)']);
xlabel('Time (s)');
ylabel('Activation Level');
legendStrings_H = cell(1, num_synergies);
for i = 1:num_synergies
    legendStrings_H{i} = ['Synergy ', num2str(i)];
end
legend(legendStrings_H, 'Location', 'eastoutside');


disp('Code execution completed.');
disp(['Results saved to: ', output_filename]);
disp('Inspect figures for results and VAF bar plot.');
disp('Analyze saved .mat file for W_tasks, H_tasks, VAF_tasks, Correlation_tasks.');