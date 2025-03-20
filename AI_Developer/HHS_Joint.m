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

% Initialize storage
all_features = [];
all_targets = [];

% Process each trial and task
for trial = 1:num_trials
    for task = 1:num_tasks
        fprintf('Processing Trial %d, Task %d\n', trial, task);
        
        emg_data = dsfilt_emg{trial, task};
        joint_data = joint_angles{trial, task};
        hhs_images = cell(1, num_muscles);
        
        % EMD and HHT
        for muscle = 1:num_muscles
            emg = emg_data(:, muscle);
            imfs = emd(emg);
            num_imfs = size(imfs, 2);
            amplitude = zeros(num_imfs, signal_length);
            frequency = zeros(num_imfs, signal_length);
            
            for imf = 1:num_imfs
                analytic_signal = hilbert(imfs(:, imf));
                amplitude(imf, :) = abs(analytic_signal)';
                phase = unwrap(angle(analytic_signal));
                frequency(imf, 1) = 0;
                frequency(imf, 2:end) = diff(phase)' / (2 * pi);
            end
            
            valid_freqs = frequency(:);
            valid_freqs = valid_freqs(valid_freqs > 0);
            f_max = ifelse(~isempty(valid_freqs), prctile(valid_freqs, 99), 1);
            
            hhs = zeros(signal_length, freq_bins);
            for t = 1:signal_length
                for imf = 1:num_imfs
                    freq = frequency(imf, t);
                    if freq > 0 && freq <= f_max
                        bin = min(floor((freq / f_max) * freq_bins) + 1, freq_bins);
                        hhs(t, bin) = hhs(t, bin) + amplitude(imf, t);
                    end
                end
            end
            
            max_hhs = max(hhs(:));
            if max_hhs > 0
                hhs_images{muscle} = uint8(255 * hhs / max_hhs);
            else
                hhs_images{muscle} = zeros(size(hhs), 'uint8');
            end
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

% Train regression model
net = feedforwardnet([64 32]);
net = train(net, all_features', all_targets');
fprintf('Regression model trained.\n');

% Helper function
function y = ifelse(condition, true_val, false_val)
    if condition
        y = true_val;
    else
        y = false_val;
    end
end