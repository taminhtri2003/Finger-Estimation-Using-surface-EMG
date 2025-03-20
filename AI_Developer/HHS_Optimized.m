% Load the .mat file
load('s1_full.mat'); % Contains dsfilt_emg and joint_angles

% Define constants
num_trials = 5;
num_tasks = 5;
num_muscles = 8;
signal_length = 4000;
window_size = 100;
step_size = 10;
freq_bins = 50;
glcm_features = 4;
total_features = num_muscles * glcm_features;

% Output folder for HHS images
output_folder = 'Hilbert_Huang_Spectrum_Images';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Initialize storage
all_features = [];
all_targets = [];

% Simulated time and frequency axes (for plotting)
% Assuming signal_length = 4000 samples corresponds to 40 seconds
time_seconds = linspace(0, 40, signal_length); % 0 to 40 seconds
freq_hz = linspace(0, 10, freq_bins); % Frequency from 0 to 10 Hz

% Process each trial and task
for trial = 1:num_trials
    for task = 1:num_tasks
        fprintf('Processing Trial %d, Task %d\n', trial, task);
        
        emg_data = dsfilt_emg{trial, task};
        joint_data = joint_angles{trial, task};
        hhs_images = cell(1, num_muscles); % For feature extraction
        
        % EMD and HHT
        for muscle = 1:num_muscles
            emg = emg_data(:, muscle);
            imfs = emd(emg); % 4000-by-num_imfs
            num_imfs = size(imfs, 2);
            
            % Ensure we have at least 3 IMFs
            if num_imfs < 3
                warning('Not enough IMFs for Trial %d, Task %d, Muscle %d. Skipping.', trial, task, muscle);
                continue;
            end
            
            % Compute HHS for IMF2, IMF3, and their combination
            hhs_imf2 = zeros(signal_length, freq_bins); % IMF2
            hhs_imf3 = zeros(signal_length, freq_bins); % IMF3
            amplitude = zeros(num_imfs, signal_length);
            frequency = zeros(num_imfs, signal_length);
            
            % Hilbert Transform for all IMFs
            for imf = 1:num_imfs
                analytic_signal = hilbert(imfs(:, imf));
                amplitude(imf, :) = abs(analytic_signal)';
                phase = unwrap(angle(analytic_signal));
                frequency(imf, 1) = 0;
                frequency(imf, 2:end) = diff(phase)' / (2 * pi);
            end
            
            % Compute maximum frequency for binning
            valid_freqs = frequency(:);
            valid_freqs = valid_freqs(valid_freqs > 0);
            f_max = ifelse(~isempty(valid_freqs), prctile(valid_freqs, 99), 1);
            
            % HHS for IMF2
            for t = 1:signal_length
                freq = frequency(2, t); % IMF2
                if freq > 0 && freq <= f_max
                    bin = min(floor((freq / f_max) * freq_bins) + 1, freq_bins);
                    hhs_imf2(t, bin) = hhs_imf2(t, bin) + amplitude(2, t);
                end
            end
            
            % HHS for IMF3
            for t = 1:signal_length
                freq = frequency(3, t); % IMF3
                if freq > 0 && freq <= f_max
                    bin = min(floor((freq / f_max) * freq_bins) + 1, freq_bins);
                    hhs_imf3(t, bin) = hhs_imf3(t, bin) + amplitude(3, t);
                end
            end
            
            % HHS for IMF2+3 (sum of amplitudes)
            hhs_imf2_3 = hhs_imf2 + hhs_imf3;
            
            % Combined HHS for all IMFs (for feature extraction)
            hhs_combined = zeros(signal_length, freq_bins);
            for t = 1:signal_length
                for imf = 1:num_imfs
                    freq = frequency(imf, t);
                    if freq > 0 && freq <= f_max
                        bin = min(floor((freq / f_max) * freq_bins) + 1, freq_bins);
                        hhs_combined(t, bin) = hhs_combined(t, bin) + amplitude(imf, t);
                    end
                end
            end
            
            % Store normalized combined HHS for feature extraction (grayscale)
            max_hhs = max(hhs_combined(:));
            if max_hhs > 0
                hhs_normalized = uint8(255 * hhs_combined / max_hhs);
            else
                hhs_normalized = zeros(size(hhs_combined), 'uint8');
            end
            hhs_images{muscle} = hhs_normalized;
            
            % Convert HHS to dB for plotting (like the example)
            hhs_imf2_db = 20 * log10(hhs_imf2 + eps); % Add eps to avoid log(0)
            hhs_imf3_db = 20 * log10(hhs_imf3 + eps);
            hhs_imf2_3_db = 20 * log10(hhs_imf2_3 + eps);
            
            % Plot HHS images in subplots
            figure('Position', [100, 100, 800, 600], 'Visible', 'off');
            
            % Subplot 1: IMF2
            subplot(3, 1, 1);
            imagesc(time_seconds, freq_hz, hhs_imf2_db');
            set(gca, 'YDir', 'normal');
            colormap('jet');
            title('IMF2');
            ylabel('Frequency (Hz)');
            clim([-20, 0]); % Adjust based on data
            
            % Subplot 2: IMF3
            subplot(3, 1, 2);
            imagesc(time_seconds, freq_hz, hhs_imf3_db');
            set(gca, 'YDir', 'normal');
            colormap('jet');
            title('IMF3');
            ylabel('Frequency (Hz)');
            clim([-20, 0]);
            
            % Subplot 3: IMF2+3
            subplot(3, 1, 3);
            imagesc(time_seconds, freq_hz, hhs_imf2_3_db');
            set(gca, 'YDir', 'normal');
            colormap('jet');
            title('IMF2+3');
            xlabel('Time (s)');
            ylabel('Frequency (Hz)');
            clim([-20, 0]);
            
            % Add a single colorbar for all subplots
            h = colorbar('Position', [0.92, 0.1, 0.02, 0.8]); % Right side, spanning all subplots
            h.Label.String = 'Amplitude (dB)';
            
            % Export the figure
            filename = sprintf('HHS_Trial%d_Task%d_Muscle%d.png', trial, task, muscle);
            filepath = fullfile(output_folder, filename);
            saveas(gcf, filepath);
            close(gcf);
            fprintf('Exported Hilbert-Huang Spectrum: %s\n', filename);
        end
        
        % Sliding window feature extraction
        start_t = floor(window_size / 2) + 1; % 51
        end_t = signal_length - floor(window_size / 2); % 3950
        num_steps = floor((end_t - start_t) / step_size) + 1; % 390
        fprintf('Extracting features: %d steps\n', num_steps);
        
        for t_idx = 0:num_steps-1
            t = start_t + t_idx * step_size;
            if mod(t_idx, 50) == 0
                fprintf('Step %d/%d (t=%d)\n', t_idx+1, num_steps, t);
            end
            
            win_start = t - floor(window_size / 2);
            win_end = t + floor(window_size / 2) - 1;
            
            feature_vector = [];
            for muscle = 1:num_muscles
                sub_image = hhs_images{muscle}(win_start:win_end, :);
                glcm = graycomatrix(sub_image, 'Offset', [0 1], 'NumLevels', 256, 'Symmetric', true);
                stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
                features = [stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity];
                feature_vector = [feature_vector, features];
            end
            
            all_features = [all_features; feature_vector];
            all_targets = [all_targets; joint_data(t, :)];
        end
    end
end

% Display summary
fprintf('Features: %d samples x %d dimensions\n', size(all_features, 1), size(all_features, 2));
fprintf('Targets: %d samples x %d dimensions\n', size(all_targets, 1), size(all_targets, 2));

% Train regression model with optimized parameters
net = feedforwardnet([32 16]);
net.trainParam.epochs = 50;
net.trainParam.lr = 0.01;
net.trainParam.min_grad = 1e-6;
net.trainParam.max_fail = 10;
net.trainFcn = 'trainlm';
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Train the network
tic;
net = train(net, all_features', all_targets');
training_time = toc;
fprintf('Regression model trained in %.2f seconds.\n', training_time);

% Helper function
function y = ifelse(condition, true_val, false_val)
    if condition
        y = true_val;
    else
        y = false_val;
    end
end