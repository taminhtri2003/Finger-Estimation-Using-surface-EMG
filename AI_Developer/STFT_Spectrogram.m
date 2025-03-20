% Load the .mat file
load('s1_full.mat');

% Muscle names
muscle_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};

% --- Visualization Options ---
trial_num = 1; % Example: Visualize the first trial
tasks_to_use = [1, 2, 3, 4, 5, 6, 7]; % Example: All tasks

% --- STFT Parameters ---
window_length = 256;  % Length of the analysis window (samples).  Adjust as needed.
overlap = 128;       % Overlap between adjacent windows (samples). Adjust as needed.
nfft = 512;          % Number of FFT points (should be >= window_length).  Adjust.
fs = 2000;           % **IMPORTANT:** Replace with your actual sampling rate (samples/second)

% --- Processing and Visualization ---

for task_idx = 1:length(tasks_to_use)
    task_num = tasks_to_use(task_idx);
    emg_data = dsfilt_emg{trial_num, task_num};

    if isempty(emg_data)
        warning(['Skipping Trial ', num2str(trial_num), ', Task ', num2str(task_num), ' due to missing data.']);
        continue;
    end

    % Create a figure for the current task
    figure;
    set(gcf,'Position',[100, 100, 1200, 800]);
    sgtitle(['STFT/Spectrogram - Trial ', num2str(trial_num), ', Task ', num2str(task_num)]);


    for muscle_idx = 1:size(emg_data, 2)
        % Extract the EMG signal for the current muscle
        signal = emg_data(:, muscle_idx);

        % --- Calculate the Spectrogram ---
        [S, F, T] = spectrogram(signal, window_length, overlap, nfft, fs);

        % --- Plotting ---
        subplot(2, 4, muscle_idx); % Arrange subplots in a 2x4 grid (for 8 muscles)

        % Plot the spectrogram.  'yaxis' makes frequency the y-axis.
        imagesc(T, F, 20*log10(abs(S))); % Use imagesc for better visualization
        axis xy; % Put origin in lower left corner
        colormap(jet); % Use the 'jet' colormap (you can try others)
        colorbar;      % Show color scale

        title([muscle_names{muscle_idx}]);
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
        ylim([0, fs/2]);  % Limit frequency axis to Nyquist frequency

    end
end