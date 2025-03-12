% Load the .mat file
load('s1.mat');

% --- EMG Preprocessing Parameters ---
bandpass_freq_range = [10, 500]; % Hz
bandpass_order = 4;
lowpass_cutoff_freq = 4; % Hz
lowpass_order = 4;
original_emg_fs = 2000; % Hz
motion_data_fs = 200;   % Hz
downsample_factor = original_emg_fs / motion_data_fs; % 10

% --- Butterworth filter design ---
% Bandpass filter
[bandpass_b, bandpass_a] = butter(bandpass_order, bandpass_freq_range/(original_emg_fs/2), 'bandpass');
% Lowpass filter
[lowpass_b, lowpass_a] = butter(lowpass_order, lowpass_cutoff_freq/(original_emg_fs/2), 'low');

% Initialize cell array to store preprocessed EMG data
preprocessed_emg = cell(size(dsfilt_emg));

% --- Channel Names for EMG data ---
emg_channel_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};

% --- Task descriptions for reference ---
task_descriptions = {
    'Thumb flexion/extension',
    'Index flexion/extension',
    'Middle flexion/extension',
    'Ring flexion/extension',
    'Little flexion/extension',
    'All fingers flexion/extension',
    'Random free finger movement'
};

% --- Create folder to save CSV files ---
folderName = 'CSV_Export';
if ~exist(folderName, 'dir')
    mkdir(folderName);
end

% --- Process EMG data for all trials and tasks ---
disp('Processing EMG data...');
for trial = 1:size(dsfilt_emg, 1)
    for task = 1:size(dsfilt_emg, 2)
        emg_data = dsfilt_emg{trial, task}; % Get EMG data for current trial and task

        if ~isempty(emg_data)
            % 1. Band-pass filter
            filtered_emg = filtfilt(bandpass_b, bandpass_a, emg_data);

            % 2. Rectify
            rectified_emg = abs(filtered_emg);

            % 3. Normalize by overall peak rectified sEMG
            peak_rectified_emg = max(rectified_emg(:));
            if peak_rectified_emg > 0
                normalized_emg = rectified_emg / peak_rectified_emg;
            else
                normalized_emg = rectified_emg;
            end

            % 4. Low-pass filter (after normalization)
            lowpass_filtered_emg = filtfilt(lowpass_b, lowpass_a, normalized_emg);

            % 5. Downsample (after low-pass filtering)
            downsampled_emg = lowpass_filtered_emg(1:downsample_factor:end, :);

            preprocessed_emg{trial, task} = downsampled_emg; % Store preprocessed EMG
            
            disp(['Processed EMG data: Trial ', num2str(trial), ', Task ', num2str(task), ' (', task_descriptions{task}, ')']);
        else
            preprocessed_emg{trial, task} = []; % Keep empty cells if original data is empty
        end
    end
end

% --- Create training and testing datasets ---
training_data = {};
testing_data = {};

% --- Combine EMG and Kinematics data ---
disp('Combining EMG and Kinematics data...');

% Define kinematics channel naming scheme (x,y,z for 23 markers)
kin_marker_names = cell(1, 23);
for i = 1:23
    kin_marker_names{i} = ['Marker_', num2str(i)];
end

kin_channel_names = cell(1, 69);
coord_names = {'x', 'y', 'z'};
k = 1;
for i = 1:23
    for j = 1:3
        kin_channel_names{k} = [kin_marker_names{i}, '_', coord_names{j}];
        k = k + 1;
    end
end

for trial = 1:size(preprocessed_emg, 1)
    for task = 1:size(preprocessed_emg, 2)
        if ~isempty(preprocessed_emg{trial, task}) && ~isempty(finger_kinematics{trial, task})
            % Get preprocessed EMG data
            emg_data = preprocessed_emg{trial, task};
            
            % Get kinematics data
            kin_data = finger_kinematics{trial, task};
            
            % Check if the number of time points match
            emg_size = size(emg_data, 1);
            kin_size = size(kin_data, 1);
            
            % Handle potential size mismatch
            if emg_size ~= kin_size
                warning(['Size mismatch between EMG and Kinematics data for Trial ', num2str(trial), ', Task ', num2str(task)]);
                min_size = min(emg_size, kin_size);
                emg_data = emg_data(1:min_size, :);
                kin_data = kin_data(1:min_size, :);
            end
            
            % Combine EMG and kinematics data
            combined_data = [emg_data, kin_data];
            
            % Add trial and task metadata
            trial_task_info = [trial * ones(size(emg_data, 1), 1), task * ones(size(emg_data, 1), 1)];
            combined_data = [trial_task_info, combined_data];
            
            % Store to appropriate dataset based on trial number
            if trial <= 3
                training_data{end+1} = combined_data;
                disp(['Added to training set: Trial ', num2str(trial), ', Task ', num2str(task), ' (', task_descriptions{task}, ')']);
            else
                testing_data{end+1} = combined_data;
                disp(['Added to testing set: Trial ', num2str(trial), ', Task ', num2str(task), ' (', task_descriptions{task}, ')']);
            end
        end
    end
end

% --- Concatenate data for training and testing sets ---
if ~isempty(training_data)
    training_data_concat = vertcat(training_data{:});
    disp(['Training data size: ', num2str(size(training_data_concat, 1)), ' rows x ', num2str(size(training_data_concat, 2)), ' columns']);
else
    error('No training data available');
end

if ~isempty(testing_data)
    testing_data_concat = vertcat(testing_data{:});
    disp(['Testing data size: ', num2str(size(testing_data_concat, 1)), ' rows x ', num2str(size(testing_data_concat, 2)), ' columns']);
else
    warning('No testing data available');
    testing_data_concat = [];
end

% --- Create column headers ---
headers = ['Trial', 'Task', emg_channel_names, kin_channel_names];

% --- Export to CSV files ---
disp('Exporting to CSV files...');

% Training data
training_csv_path = fullfile(folderName, 'training_data_trial1to3.csv');
fid = fopen(training_csv_path, 'w');
% Write header
fprintf(fid, '%s', headers{1});
for i = 2:length(headers)
    fprintf(fid, ',%s', headers{i});
end
fprintf(fid, '\n');
% Write data
for i = 1:size(training_data_concat, 1)
    fprintf(fid, '%g', training_data_concat(i, 1));
    for j = 2:size(training_data_concat, 2)
        fprintf(fid, ',%g', training_data_concat(i, j));
    end
    fprintf(fid, '\n');
end
fclose(fid);
disp(['Training data saved to: ', training_csv_path]);

% Testing data
if ~isempty(testing_data_concat)
    testing_csv_path = fullfile(folderName, 'testing_data_trial4to5.csv');
    fid = fopen(testing_csv_path, 'w');
    % Write header
    fprintf(fid, '%s', headers{1});
    for i = 2:length(headers)
        fprintf(fid, ',%s', headers{i});
    end
    fprintf(fid, '\n');
    % Write data
    for i = 1:size(testing_data_concat, 1)
        fprintf(fid, '%g', testing_data_concat(i, 1));
        for j = 2:size(testing_data_concat, 2)
            fprintf(fid, ',%g', testing_data_concat(i, j));
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
    disp(['Testing data saved to: ', testing_csv_path]);
end

% --- Create an additional CSV with task descriptions for reference ---
task_desc_path = fullfile(folderName, 'task_descriptions.csv');
fid = fopen(task_desc_path, 'w');
fprintf(fid, 'Task_ID,Description\n');
for i = 1:length(task_descriptions)
    fprintf(fid, '%d,%s\n', i, task_descriptions{i});
end
fclose(fid);
disp(['Task descriptions saved to: ', task_desc_path]);

% --- Create a summary log file ---
summary_path = fullfile(folderName, 'data_summary.txt');
fid = fopen(summary_path, 'w');
fprintf(fid, 'EMG and Kinematics Data Export Summary\n');
fprintf(fid, '=====================================\n\n');
fprintf(fid, 'Data source: s1.mat\n\n');

fprintf(fid, 'EMG Channels (8):\n');
for i = 1:length(emg_channel_names)
    fprintf(fid, '  %d. %s\n', i, emg_channel_names{i});
end
fprintf(fid, '\n');

fprintf(fid, 'Kinematics Channels (69):\n');
fprintf(fid, '  23 markers with x, y, z coordinates\n\n');

fprintf(fid, 'Training Data (Trials 1-3):\n');
fprintf(fid, '  Rows: %d\n', size(training_data_concat, 1));
fprintf(fid, '  Columns: %d (2 metadata + 8 EMG + 69 kinematics)\n\n', size(training_data_concat, 2));

if ~isempty(testing_data_concat)
    fprintf(fid, 'Testing Data (Trials 4-5):\n');
    fprintf(fid, '  Rows: %d\n', size(testing_data_concat, 1));
    fprintf(fid, '  Columns: %d (2 metadata + 8 EMG + 69 kinematics)\n\n', size(testing_data_concat, 2));
else
    fprintf(fid, 'No testing data available.\n\n');
end

fprintf(fid, 'Task Descriptions:\n');
for i = 1:length(task_descriptions)
    fprintf(fid, '  %d. %s\n', i, task_descriptions{i});
end

fclose(fid);
disp(['Summary saved to: ', summary_path]);
disp('Export completed successfully!');