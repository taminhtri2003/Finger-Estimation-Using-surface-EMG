% Load the .mat file
load('s1_full.mat');

% Muscle names
muscle_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};

% --- Visualization Options ---
trial_num = 1;
tasks_to_compare = [1, 2, 6, 7];
time_start = 1;
time_end = 4000;
decomposition_level = 4;

% --- Data Extraction and Wavelet Decomposition ---
wavelet_coeffs = cell(length(tasks_to_compare), size(dsfilt_emg{1,1}, 2));
wavelet_lengths = cell(length(tasks_to_compare), size(dsfilt_emg{1,1}, 2));

for task_idx = 1:length(tasks_to_compare)
    task_num = tasks_to_compare(task_idx);
    emg_data = dsfilt_emg{trial_num, task_num};

    if isempty(emg_data)
        error(['No EMG data for Trial ', num2str(trial_num), ', Task ', num2str(task_num)]);
    end

    for i = 1:size(emg_data, 2)
        [C, L] = wavedec(emg_data(time_start:time_end, i), decomposition_level, 'db4');
        wavelet_coeffs{task_idx, i} = C;
        wavelet_lengths{task_idx, i} = L;
    end
end

% --- Visualization (Separate Figures for Each Level) ---

% Figure for Original EMG Signals (all muscles and tasks)
figure;
set(gcf,'Position',[100 100 1800 900]);

for muscle_idx = 1:size(dsfilt_emg{1,1}, 2)
    subplot(2, 4, muscle_idx); % 2x4 grid for 8 muscles
    hold on;
    for task_idx = 1:length(tasks_to_compare)
        task_num = tasks_to_compare(task_idx);
        emg_data = dsfilt_emg{trial_num, task_num};
        plot(time_start:time_end, emg_data(time_start:time_end, muscle_idx), 'DisplayName', ['Task ' num2str(task_num)]);
    end
    hold off;
    title(['Original EMG - ', muscle_names{muscle_idx}]);
    ylabel('Amplitude');
    xlabel('Sample');
    legend('Location', 'northeast');
    xlim([time_start, time_end]);
end
sgtitle(['Original EMG Signals (Trial ', num2str(trial_num), ')']);



% Figure for Approximation Coefficients
figure;
set(gcf,'Position',[100 100 1800 900]);
sgtitle(['Approximation Coefficients (Level ', num2str(decomposition_level), ', Trial ', num2str(trial_num),')']);

for muscle_idx = 1:size(dsfilt_emg{1,1}, 2)
    subplot(2, 4, muscle_idx);
    hold on;
    for task_idx = 1:length(tasks_to_compare)
        cA = appcoef(wavelet_coeffs{task_idx, muscle_idx}, wavelet_lengths{task_idx, muscle_idx}, 'db4', decomposition_level);
        A = wrcoef('a', wavelet_coeffs{task_idx, muscle_idx}, wavelet_lengths{task_idx, muscle_idx},'db4',decomposition_level);
        plot(linspace(time_start,time_end,length(A)), A, 'DisplayName', ['Task ' num2str(tasks_to_compare(task_idx))]);
    end
    hold off;
    title([muscle_names{muscle_idx}]);
    ylabel('Amplitude');
    xlabel('Sample (Downsampled)');
    legend('Location', 'northeast');
    xlim([time_start, time_end]);
end



% Figures for Detail Coefficients (one figure per level)
for level = 1:decomposition_level
    figure;
     set(gcf,'Position',[100 100 1800 900]);
    sgtitle(['Detail Coefficients (Level ', num2str(decomposition_level - level + 1), ', Trial ', num2str(trial_num), ')']);

    for muscle_idx = 1:size(dsfilt_emg{1,1}, 2)
        subplot(2, 4, muscle_idx); % Arrange subplots in a 2x4 grid
        hold on;
        for task_idx = 1:length(tasks_to_compare)
            cD = detcoef(wavelet_coeffs{task_idx, muscle_idx}, wavelet_lengths{task_idx, muscle_idx}, level);
             D = wrcoef('d', wavelet_coeffs{task_idx, muscle_idx}, wavelet_lengths{task_idx, muscle_idx}, 'db4', level);
            plot(linspace(time_start,time_end, length(D)), D, 'DisplayName', ['Task ' num2str(tasks_to_compare(task_idx))]);
        end
        hold off;
        title([muscle_names{muscle_idx}]);
        ylabel('Amplitude');
        xlabel('Sample (Downsampled)');
        legend('Location', 'northeast');
         xlim([time_start, time_end]);
    end
end