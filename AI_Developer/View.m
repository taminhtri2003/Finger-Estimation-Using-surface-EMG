% Load the .mat file containing the data
try
    load('s2_full.mat'); % Replace with your actual file name
catch e
    error('Failed to load data file: %s', e.message);
end

% Combine tasks 1 to 6 across all trials
combined_emg = [];
combined_joint_angles = [];

for trial = 1:5
    for task = 1:6 % Updated to include 6 tasks
        combined_emg = [combined_emg; dsfilt_emg{trial, task}];
        combined_joint_angles = [combined_joint_angles; joint_angles{trial, task}];
    end
end

% Create GUI figure and store data in its UserData
fig = figure('Name', 'EMG and Joint Angle Selector', ...
             'NumberTitle', 'off', ...
             'Position', [100, 100, 900, 600]); % Wider figure for colorbar
fig.UserData.combined_emg = combined_emg;
fig.UserData.combined_joint_angles = combined_joint_angles;
fig.UserData.axes_handle = axes('Position', [0.1, 0.1, 0.7, 0.3]); % Adjusted width

% GUI components
uicontrol('Style', 'text', 'String', 'Select EMG Channel:', ...
          'Position', [50, 550, 150, 20]);
emg_popup = uicontrol('Style', 'popupmenu', ...
                     'String', {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'}, ...
                     'Position', [200, 550, 150, 20], ...
                     'Tag', 'emg_popup');

uicontrol('Style', 'text', 'String', 'Select Joint Angle:', ...
          'Position', [50, 500, 150, 20]);
joint_popup = uicontrol('Style', 'popupmenu', ...
                       'String', {'Thumb 1', 'Thumb 2', 'Index 1', 'Index 2', 'Index 3', ...
                                  'Middle 1', 'Middle 2', 'Middle 3', 'Ring 1', 'Ring 2', ...
                                  'Ring 3', 'Little 1', 'Little 2', 'Little 3'}, ...
                       'Position', [200, 500, 150, 20], ...
                       'Tag', 'joint_popup');

uicontrol('Style', 'pushbutton', 'String', 'Plot Selected Signals', ...
          'Position', [50, 450, 150, 30], ...
          'Callback', @plot_selected_signals);

% Callback function
function plot_selected_signals(~, ~)
    % Get figure handle
    fig = gcf;
    
    % Retrieve data from figure's UserData
    combined_emg = fig.UserData.combined_emg;
    combined_joint_angles = fig.UserData.combined_joint_angles;
    axes_handle = fig.UserData.axes_handle;
    
    % Get selected indices and labels
    emg_popup = findobj(fig, 'Tag', 'emg_popup');
    emg_idx = get(emg_popup, 'Value');
    emg_labels = get(emg_popup, 'String');
    emg_label = emg_labels{emg_idx};
    
    joint_popup = findobj(fig, 'Tag', 'joint_popup');
    joint_idx = get(joint_popup, 'Value');
    joint_labels = get(joint_popup, 'String');
    joint_label = joint_labels{joint_idx};
    
    % Get signals
    emg_signal = combined_emg(:, emg_idx);
    joint_signal = combined_joint_angles(:, joint_idx);
    
    % Clear previous plot
    cla(axes_handle);
    hold(axes_handle, 'on');
    
    % Normalize signals for better visualization
    emg_signal_norm = (emg_signal - min(emg_signal)) / (max(emg_signal) - min(emg_signal));
    joint_signal_norm = (joint_signal - min(joint_signal)) / (max(joint_signal) - min(joint_signal));
    
    % Plot signals with proper labels
    plot(axes_handle, emg_signal_norm, 'b', 'DisplayName', ['EMG: ', emg_label]);
    plot(axes_handle, joint_signal_norm, 'r', 'DisplayName', ['Joint Angle: ', joint_label]);
    
    % Task highlighting parameters
    samples_per_task = size(dsfilt_emg{1,1}, 1); % 4000 samples/task
    num_trials = 5;
    task_colors = lines(6); % 6 colors for 6 tasks
    task_labels = {'Thumb', 'Index', 'Middle', 'Ring', 'Little', 'All Fingers'};
    
    % Calculate signal range
    min_val = 0;    % Since we're using normalized signals
    max_val = 1;
    
    % Create shaded regions for all tasks across all trials
    for trial = 1:num_trials
        for task = 1:6 % Updated for 6 tasks
            start_idx = (trial-1)*6*samples_per_task + (task-1)*samples_per_task + 1; % 6 tasks
            end_idx = start_idx + samples_per_task - 1;
            
            % Use task number (1-6) to determine color
            color_idx = mod(task-1, 6) + 1;
            
            % Create semi-transparent patch
            patch(axes_handle, [start_idx, end_idx, end_idx, start_idx], ...
                  [min_val, min_val, max_val, max_val], ...
                  task_colors(color_idx, :), ...
                  'FaceAlpha', 0.2, 'EdgeColor', 'none');
        end
    end
    
    % Add colorbar for task differentiation
    delete(findobj(fig, 'Tag', 'colorbar_ax')); % Remove existing colorbar
    colorbar_ax = axes('Parent', fig, 'Position', [0.85, 0.1, 0.05, 0.3], ...
                      'Visible', 'off', 'Tag', 'colorbar_ax');
    imagesc(colorbar_ax, [1:6], [1:6]');
    colormap(colorbar_ax, task_colors);
    colorbar(colorbar_ax, 'Ticks', 1:6, 'TickLabels', task_labels);
    ylabel(colorbar_ax, 'Task');
    
    % Add labels and legend
    xlabel(axes_handle, 'Time Samples');
    ylabel(axes_handle, 'Normalized Amplitude/Angle');
    title(axes_handle, 'EMG and Joint Angle Signals with Task Highlighting');
    legend(axes_handle, 'Location', 'best');
    grid(axes_handle, 'on');
    hold(axes_handle, 'off');
end