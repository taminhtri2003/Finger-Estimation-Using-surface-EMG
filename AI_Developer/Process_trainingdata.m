% Load the .mat file containing the data
load('s1_full.mat'); % Replace with actual filename if different

% Define column names for the CSV files
emg_names = {'EMG_APL', 'EMG_FCR', 'EMG_FDS', 'EMG_FDP', 'EMG_ED', 'EMG_EI', 'EMG_ECU', 'EMG_ECR'};
angle_names = {'Thumb1', 'Thumb2', 'Index1', 'Index2', 'Index3', 'Middle1', 'Middle2', 'Middle3', ...
               'Ring1', 'Ring2', 'Ring3', 'Little1', 'Little2', 'Little3'};
column_names = [{'Trial', 'Task'}, emg_names, angle_names];

% --- Process Training Data (Trials 1, 2, 3) ---
train_data = [];
for i = 1:3 % Trials 1, 2, 3
    for j = 1:5 % All 7 tasks
        % Extract EMG and joint angle data
        emg = dsfilt_emg{i,j}; % Should be 40000x8
        angles = joint_angles{i,j}; % Should be 4000x14
        
        % Diagnostic: Print sizes
        fprintf('Trial %d, Task %d: EMG size = %dx%d, Angles size = %dx%d\n', ...
                i, j, size(emg,1), size(emg,2), size(angles,1), size(angles,2));
        
        % Downsample EMG to match the number of rows in angles
        target_rows = size(angles, 1); % Get the number of rows in angles (should be 4000)
        if size(emg, 1) == 40000
            % Simple downsampling by taking every 10th sample
            emg_downsampled = emg(1:10:end, :); % Should be 4000x8
            if size(emg_downsampled, 1) ~= target_rows
                % If still not matching, resample to exact size
                emg_downsampled = resample(emg, target_rows, size(emg,1));
            end
        else
            % If EMG size is unexpected, resample directly to match angles
            emg_downsampled = resample(emg, target_rows, size(emg,1));
        end
        
        % Verify sizes match
        if size(emg_downsampled, 1) ~= size(angles, 1)
            error('After adjustment, EMG rows (%d) do not match Angles rows (%d) for Trial %d, Task %d', ...
                  size(emg_downsampled,1), size(angles,1), i, j);
        end
        
        % Combine EMG and joint angles horizontally
        data = [emg_downsampled, angles]; % Should be target_rows x 22 (8 EMG + 14 angles)
        
        % Add trial and task identifier columns
        trial_col = repmat(i, target_rows, 1);
        task_col = repmat(j, target_rows, 1);
        data_with_id = [trial_col, task_col, data];
        
        % Vertically concatenate
        train_data = [train_data; data_with_id];
    end
end

% --- Process Testing Data (Trials 4, 5) ---
test_data = [];
for i = 4:5 % Trials 4, 5
    for j = 1:5 % All 7 tasks
        % Extract EMG and joint angle data
        emg = dsfilt_emg{i,j};
        angles = joint_angles{i,j};
        
        % Diagnostic: Print sizes
        fprintf('Trial %d, Task %d: EMG size = %dx%d, Angles size = %dx%d\n', ...
                i, j, size(emg,1), size(emg,2), size(angles,1), size(angles,2));
        
        % Downsample EMG to match angles
        target_rows = size(angles, 1);
        if size(emg, 1) == 40000
            emg_downsampled = emg(1:10:end, :);
            if size(emg_downsampled, 1) ~= target_rows
                emg_downsampled = resample(emg, target_rows, size(emg,1));
            end
        else
            emg_downsampled = resample(emg, target_rows, size(emg,1));
        end
        
        % Verify sizes match
        if size(emg_downsampled, 1) ~= size(angles, 1)
            error('After adjustment, EMG rows (%d) do not match Angles rows (%d) for Trial %d, Task %d', ...
                  size(emg_downsampled,1), size(angles,1), i, j);
        end
        
        % Combine EMG and joint angles horizontally
        data = [emg_downsampled, angles];
        
        % Add trial and task identifier columns
        trial_col = repmat(i, target_rows, 1);
        task_col = repmat(j, target_rows, 1);
        data_with_id = [trial_col, task_col, data];
        
        % Vertically concatenate
        test_data = [test_data; data_with_id];
    end
end

% --- Export to CSV Files ---
train_table = array2table(train_data, 'VariableNames', column_names);
test_table = array2table(test_data, 'VariableNames', column_names);

writetable(train_table, 'train.csv');
writetable(test_table, 'test.csv');

disp('CSV files "train.csv" and "test.csv" have been created successfully.');