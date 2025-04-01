% Load the .mat file
load('s1_full.mat'); % Replace with your file name

% --- Parameters (SOME WILL BE AUTOCALIBRATED) ---
fs = 2000;  % Sampling frequency (Hz) - **CRUCIAL: Replace with your actual value**
window_size_kinematics = round(0.01 * fs); % Smoothing window for kinematics (e.g., 10ms)
min_delay = round(0.01 * fs); % Minimum EMD (e.g., 10ms)
max_delay = round(0.2 * fs);  % Maximum EMD (e.g., 200ms)
onset_threshold_kinematics = 0.02; % Kinematic onset threshold

% EMG-to-Activation Model Parameters (Fuglevand Model) - Initial Guesses
A_initial = -0.56;
P_initial = 3;
tau1_initial = 0.04;
tau2_initial = 0.06;

% Autocalibration Parameters
calibration_task = 6;
calibration_trials = 1:3;
optimization_options = optimoptions('fmincon', ...
    'Display', 'iter', ...
    'MaxFunctionEvaluations', 3000, ...
    'MaxIterations', 1000, ...
    'Algorithm', 'sqp', ...
    'FiniteDifferenceType', 'central', ...
    'StepTolerance', 1e-8, ...
    'ConstraintTolerance', 1e-6);

% Muscle names (for labeling)
muscle_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};

% Pre-allocate
EMD = cell(size(dsfilt_emg, 1), size(dsfilt_emg, 2));
muscle_activation = cell(size(dsfilt_emg, 1), size(dsfilt_emg, 2));
calibrated_params = struct();


% --- Autocalibration ---
for muscle = 1:size(dsfilt_emg{1,1}, 2)
    calibration_emg = [];
    calibration_kinematics = [];
    for trial = calibration_trials
        if ~isempty(dsfilt_emg{trial, calibration_task})
           calibration_emg = [calibration_emg; dsfilt_emg{trial, calibration_task}(:, muscle)];
            % Use joint 1 as a *placeholder*.  You MUST replace this with the correct joint(s) for each muscle.
           calibration_kinematics = [calibration_kinematics; joint_angles{trial, calibration_task}(:, 1)];
        end
    end
    if isempty(calibration_emg)
      warning('No calibration data found for muscle %s', muscle_names{muscle});
      continue;
    end

    calibration_emg = calibration_emg + 1e-6; % Avoid zero values

    objective_function = @(params) emg_to_kinematics_error(params, calibration_emg, calibration_kinematics, fs, window_size_kinematics);

    lb = [-1, 1, 0.01, 0.01];
    ub = [0, 5, 0.1,  0.15];
    x0 = [A_initial, P_initial, tau1_initial, tau2_initial];

    try
       [calibrated_params_muscle, fval] = fmincon(objective_function, x0, [], [], [], [], lb, ub, [], optimization_options);

        calibrated_params.(muscle_names{muscle}).A = calibrated_params_muscle(1);
        calibrated_params.(muscle_names{muscle}).P = calibrated_params_muscle(2);
        calibrated_params.(muscle_names{muscle}).tau1 = calibrated_params_muscle(3);
        calibrated_params.(muscle_names{muscle}).tau2 = calibrated_params_muscle(4);
        calibrated_params.(muscle_names{muscle}).fval = fval;
        fprintf('Calibrated parameters for %s: A = %.4f, P = %.4f, tau1 = %.4f, tau2 = %.4f, fval = %.4f\n', ...
            muscle_names{muscle}, calibrated_params_muscle(1), calibrated_params_muscle(2), ...
            calibrated_params_muscle(3), calibrated_params_muscle(4), fval);
    catch ME
       warning('Optimization failed for muscle %s: %s', muscle_names{muscle}, ME.message);
        calibrated_params.(muscle_names{muscle}).A = NaN;
        calibrated_params.(muscle_names{muscle}).P = NaN;
        calibrated_params.(muscle_names{muscle}).tau1 = NaN;
        calibrated_params.(muscle_names{muscle}).tau2 = NaN;
        calibrated_params.(muscle_names{muscle}).fval = NaN;
        continue;
    end
end


% --- Main Loop ---
for trial = 1:size(dsfilt_emg, 1)
    for task = 1:size(dsfilt_emg, 2)

        emg_data = dsfilt_emg{trial, task};
        kinematic_data = joint_angles{trial, task};

        if isempty(emg_data) || isempty(kinematic_data)
            EMD{trial, task} = NaN(size(emg_data,2), size(kinematic_data,2));
            muscle_activation{trial, task} = [];
            warning('EMG or Kinematic data missing for trial %d, task %d', trial, task);
            continue;
        end

        activation_signals = zeros(size(emg_data));
        emg_onset = zeros(1, size(emg_data, 2));

        for muscle = 1:size(emg_data, 2)
            if ~isnan(calibrated_params.(muscle_names{muscle}).A)
                A = calibrated_params.(muscle_names{muscle}).A;
                P = calibrated_params.(muscle_names{muscle}).P;
                tau1 = calibrated_params.(muscle_names{muscle}).tau1;
                tau2 = calibrated_params.(muscle_names{muscle}).tau2;
            else
                A = A_initial;
                P = P_initial;
                tau1 = tau1_initial;
                tau2 = tau2_initial;
                warning('Using initial parameters for muscle %s.', muscle_names{muscle});
            end

            rectified_emg = abs(emg_data(:, muscle)) + 1e-6;  % Add small constant
            normalized_emg = rectified_emg / max(rectified_emg);

            u = zeros(size(normalized_emg));
            activation = zeros(size(normalized_emg));
            for t = 2:length(normalized_emg)
                u(t) = (normalized_emg(t)^P) / (normalized_emg(t)^P + A^P);
                % Corrected ternary operator usage:
                tau = (u(t) >= activation(t-1)) * tau1 + (u(t) < activation(t-1)) * tau2;
                alpha = 1 - exp(-1 / (tau * fs));
                activation(t) = alpha * u(t) + (1 - alpha) * activation(t-1);
            end
            activation_signals(:, muscle) = activation;

            threshold = 0.05; % Fixed threshold, or relative to max.
            onset_index = find(activation >= threshold, 1, 'first');
            emg_onset(muscle) = onset_index;
        end
        muscle_activation{trial, task} = activation_signals;

         kinematic_onset = zeros(1, size(kinematic_data, 2));
        for joint = 1:size(kinematic_data, 2)
            smoothed_kinematics = movmean(kinematic_data(:, joint), window_size_kinematics);
            kinematic_velocity = diff(smoothed_kinematics) * fs;
            kinematic_velocity = [0; kinematic_velocity];

            threshold_vel = onset_threshold_kinematics * max(abs(kinematic_velocity));
            onset_index = find(abs(kinematic_velocity) > threshold_vel, 1, 'first');

            if ~isempty(onset_index)
                kinematic_onset(joint) = onset_index;
            else
                kinematic_onset(joint) = NaN;
            end
        end

        % --- EMD Calculation ---
         emd_values = NaN(size(emg_data,2), size(kinematic_data,2));
        for muscle = 1:size(emg_data,2)
           for joint = 1:size(kinematic_data,2)
                if ~isnan(emg_onset(muscle)) && ~isnan(kinematic_onset(joint))
                    delay = kinematic_onset(joint) - emg_onset(muscle);
                    if delay >= min_delay && delay <= max_delay
                        emd_values(muscle,joint) = delay / fs;
                    end
                end
           end
        end
        EMD{trial, task} = emd_values;
    end
end


% --- Visualization (Across Tasks and Muscles) ---

num_tasks = size(dsfilt_emg, 2);
num_muscles = size(dsfilt_emg{1,1}, 2);  % Assuming all cells have same # muscles
num_joints = size(joint_angles{1,1},2); %same assumption.

for trial = 1:size(dsfilt_emg,1)
  for joint = 1:num_joints

    figure; % Create a new figure for each trial and joint.
    sgtitle(['Trial ' num2str(trial) ', Joint ' num2str(joint)]);

    for task = 1:num_tasks
        for muscle = 1:num_muscles
             if ~isempty(dsfilt_emg{trial, task}) && ~isempty(muscle_activation{trial,task})
                subplot(num_tasks, num_muscles, (task - 1) * num_muscles + muscle);

                emg = dsfilt_emg{trial, task}(:, muscle);
                activation = muscle_activation{trial, task}(:, muscle);
                joint_angle = joint_angles{trial, task}(:, joint);  % Use the correct joint
                time = (0:length(emg)-1) / fs;
                
                % Find valid EMD
                valid_emd =  EMD{trial, task}(muscle,joint);

                plot(time, emg, 'b', time, activation, 'g', time, joint_angle, 'r');
                hold on;
                
                if ~isnan(emg_onset(muscle))
                    xline(emg_onset(muscle) / fs, 'b--', 'LineWidth', 1); % EMG Onset
                end
                if ~isnan(kinematic_onset(joint))
                    xline(kinematic_onset(joint) / fs, 'r--', 'LineWidth', 1); % Kinematic Onset
                end
                if ~isnan(valid_emd)
                   text(kinematic_onset(joint)/fs + 0.02 , mean(joint_angle), sprintf('EMD=%.3f', valid_emd), 'FontSize',8);
                end

                hold off;
                title([muscle_names{muscle}, ' (Task ', num2str(task), ')']);

                if task == num_tasks && muscle == 1
                     legend('EMG', 'Activation', 'Joint Angle', 'EMG Onset', 'Kinematic Onset');
                end
                if task == num_tasks
                    xlabel('Time (s)');
                end
                if muscle == 1
                  ylabel('Amplitude');
                end
                
                ylim([-0.1, 1.1]); % Consistent y-axis limits
             else
                subplot(num_tasks, num_muscles, (task - 1) * num_muscles + muscle);
                title([muscle_names{muscle}, ' (Task ', num2str(task), ') - No Data']);
             end

        end
    end
  end
end

% --- Statistical Analysis ---
fprintf('\nAverage EMD (seconds) for each Muscle and Task:\n');
for task = 1:size(EMD, 2)
    fprintf('Task %d:\n', task);
    for muscle = 1:size(emg_data,2)
        all_emds = [];
        for trial = 1:size(EMD, 1)
           for joint = 1:size(kinematic_data,2)
              if ~isempty(EMD{trial, task}) && ~isnan(EMD{trial, task}(muscle,joint)) %check for valid EMD value
                    all_emds = [all_emds, EMD{trial,task}(muscle,joint)];
              end
           end
        end
        if ~isempty(all_emds)
            mean_emd = mean(all_emds, 'omitnan');
            std_emd = std(all_emds, 'omitnan');
            fprintf('  %s: Mean = %.4f s, Std = %.4f s\n', muscle_names{muscle}, mean_emd, std_emd);
        else
            fprintf('  %s: No valid EMD data\n', muscle_names{muscle});
        end
    end
    fprintf('\n');
end


% --- Helper Function: emg_to_kinematics_error ---
function error = emg_to_kinematics_error(params, emg_data, kinematic_data, fs, window_size_kinematics)
    A = params(1);
    P = params(2);
    tau1 = params(3);
    tau2 = params(4);

    rectified_emg = abs(emg_data);
    normalized_emg = rectified_emg / max(rectified_emg);

    u = zeros(size(normalized_emg));
    activation = zeros(size(normalized_emg));
    for t = 2:length(normalized_emg)
        u(t) = (normalized_emg(t)^P) / (normalized_emg(t)^P + A^P);
        %correct use of ternary operator
        tau = (u(t) >= activation(t-1)) * tau1 + (u(t) < activation(t-1)) * tau2;
        alpha = 1 - exp(-1 / (tau * fs));
        activation(t) = alpha * u(t) + (1 - alpha) * activation(t-1);
    end

    predicted_kinematics = activation;
    smoothed_kinematics = movmean(kinematic_data, window_size_kinematics);
    min_length = min(length(predicted_kinematics), length(smoothed_kinematics));
    predicted_kinematics = predicted_kinematics(1:min_length);
    smoothed_kinematics = smoothed_kinematics(1:min_length);
    error = sqrt(mean((predicted_kinematics - smoothed_kinematics).^2));
end