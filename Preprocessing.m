% Load the .mat file
load('s1.mat');

% --- EMG Preprocessing Parameters (from image description) ---
bandpass_freq_range = [10, 500]; % Hz
bandpass_order = 4;
lowpass_cutoff_freq = 4; % Hz
lowpass_order = 4;
original_emg_fs = 2000; % Hz (from text description)
motion_data_fs = 200;    % Hz (from text description)
downsample_factor = original_emg_fs / motion_data_fs; % 10

% --- Butterworth filter design ---
% Bandpass filter
[bandpass_b, bandpass_a] = butter(bandpass_order, bandpass_freq_range/(original_emg_fs/2), 'bandpass');
% Lowpass filter
[lowpass_b, lowpass_a] = butter(lowpass_order, lowpass_cutoff_freq/(original_emg_fs/2), 'low'); % Using original EMG fs before downsampling

% Initialize cell array to store preprocessed EMG data
preprocessed_emg = cell(size(dsfilt_emg));

% --- Channel Names for plot titles and filenames ---
channel_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};

% --- Create folder to save figures ---
folderName = 'Preprocessing_Plots';
if ~exist(folderName, 'dir')
    mkdir(folderName);
end

% --- Loop through Trials (1), Tasks (1-5), and Channels (1-8) for visualization and processing ---
for trial = 1:size(dsfilt_emg, 1)
    for task = 1:5 % Loop through tasks 1 to 5 as requested
        emg_data = dsfilt_emg{trial, task}; % Get EMG data for current trial and task

        if ~isempty(emg_data)
            % 1. Band-pass filter
            filtered_emg = filtfilt(bandpass_b, bandpass_a, emg_data);

            % 2. Rectify
            rectified_emg = abs(filtered_emg);

            % 3. Normalize by overall peak rectified sEMG (peak within this cell for now)
            peak_rectified_emg = max(rectified_emg(:)); % Find peak value in all channels and time points of this cell
            if peak_rectified_emg > 0 % Avoid division by zero if peak is zero
                normalized_emg = rectified_emg / peak_rectified_emg;
            else
                normalized_emg = rectified_emg; % If peak is zero, normalization has no effect
            end

            % 4. Low-pass filter (after normalization)
            lowpass_filtered_emg = filtfilt(lowpass_b, lowpass_a, normalized_emg);

            % 5. Downsample (after low-pass filtering)
            downsampled_emg = lowpass_filtered_emg(1:downsample_factor:end, :);

            preprocessed_emg{trial, task} = downsampled_emg; % Store preprocessed EMG


            for channel = 1:size(emg_data, 2) % Loop through all channels
                time_original = (0:length(emg_data)-1) / original_emg_fs;
                time_downsampled = (0:length(downsampled_emg)-1) / motion_data_fs;

                figure('Position', [100, 100, 1200, 800], 'Visible', 'off'); % Set visible off to speed up, can remove 'Visible','off' to show plots
                % if you want to see them during processing

                subplot(6, 1, 1);
                plot(time_original, emg_data(:, channel));
                title(['1. Raw sEMG Signal (Channel ', channel_names{channel}, ')']);
                ylabel('Amplitude');
                xlabel('Time (s)');
                xlim([time_original(1), time_original(end)]);

                subplot(6, 1, 2);
                plot(time_original, filtered_emg(:, channel));
                title(['2. After Band-Pass Filter (10-500 Hz)']);
                ylabel('Amplitude');
                xlabel('Time (s)');
                xlim([time_original(1), time_original(end)]);

                subplot(6, 1, 3);
                plot(time_original, rectified_emg(:, channel));
                title('3. After Rectification');
                ylabel('Amplitude');
                xlabel('Time (s)');
                xlim([time_original(1), time_original(end)]);

                subplot(6, 1, 4);
                plot(time_original, normalized_emg(:, channel));
                title('4. After Normalization');
                ylabel('Normalized Amplitude');
                xlabel('Time (s)');
                xlim([time_original(1), time_original(end)]);

                subplot(6, 1, 5);
                plot(time_original, lowpass_filtered_emg(:, channel));
                title('5. After Low-Pass Filter (4 Hz)');
                ylabel('Amplitude');
                xlabel('Time (s)');
                xlim([time_original(1), time_original(end)]);

                subplot(6, 1, 6);
                plot(time_downsampled, downsampled_emg(:, channel));
                title('6. After Downsampling (to 200 Hz)');
                ylabel('Amplitude');
                xlabel('Time (s)');
                xlim([time_downsampled(1), time_downsampled(end)]);


                plot_title = ['EMG Preprocessing Stages - Trial ', num2str(trial), ', Task ', num2str(task), ', Channel: ', channel_names{channel}];
                sgtitle(plot_title);

                % --- Save the figure ---
                filename = fullfile(folderName, sprintf('Trial_%d_Task_%d_Channel_%s_Preprocessing.png', trial, task, channel_names{channel}));
                saveas(gcf, filename);
                close(gcf); % Close figure to free memory when running in loop

                disp(['Saved plot: ', filename]); % Optional progress display
            end

        else
            preprocessed_emg{trial, task} = []; % Keep empty cells if original data is empty
        end
    end
end

% --- Optional: Verify the size and sampling rate of preprocessed data ---
% For example, check the size of the first non-empty cell after processing
for trial = 1:size(preprocessed_emg, 1)
    for task = 1:size(preprocessed_emg, 2)
        if ~isempty(preprocessed_emg{trial, task})
            processed_data_size = size(preprocessed_emg{trial, task});
            disp(['Size of preprocessed EMG data (Trial ', num2str(trial), ', Task ', num2str(task), '): ', num2str(processed_data_size)]);
            expected_rows = size(dsfilt_emg{trial, task}, 1) / downsample_factor;
            actual_rows = processed_data_size(1);
            if abs(actual_rows - expected_rows) < 1.5 % Allow for slight rounding errors due to integer division
                 disp(['Expected rows (approx): ', num2str(expected_rows), ', Actual rows: ', num2str(actual_rows), ' - Downsampling seems correct.']);
            else
                 disp(['WARNING: Downsampling might not be correct. Expected rows (approx): ', num2str(expected_rows), ', Actual rows: ', num2str(actual_rows)]);
            end
            % Break after finding the first non-empty cell to avoid too much output after first check
            return;
        end
    end
end


disp(['Preprocessing of dsfilt_emg completed. Plots are saved in folder: ', folderName]);
disp('Results are in the variable "preprocessed_emg".');