function KinematicEMGApp()
% A MATLAB app to visualize EMG and Kinematic data from .mat files.

% --- UI Components ---

% Main figure
appFig = uifigure('Name', 'EMG and Kinematics Data Viewer', 'Position', [100, 100, 800, 600]);

% File selection panel
filePanel = uipanel(appFig, 'Title', 'File Selection', 'Position', [20, 480, 760, 100]);
filePathLabel = uilabel(filePanel, 'Text', 'Select .mat file:', 'Position', [10, 50, 100, 22]);
filePathField = uieditfield(filePanel, 'text', 'Position', [120, 50, 520, 22]);
fileBrowseButton = uibutton(filePanel, 'push', 'Text', 'Browse', 'Position', [650, 50, 80, 22], ...
                            'ButtonPushedFcn', @browseButtonCallback);

% Visualization selection panel
visPanel = uipanel(appFig, 'Title', 'Visualization Options', 'Position', [20, 280, 200, 180]);
visTypeLabel = uilabel(visPanel, 'Text', 'Select Visualization:', 'Position', [10, 130, 150, 22]);
visTypeDropdown = uidropdown(visPanel, 'Items', {'Trial Data', 'EMG Task Highlighted', 'Kinematics Marker Highlighted'}, ...
                             'Position', [10, 100, 180, 22], 'Value', 'Trial Data');

% Trial and Task Selection (for Trial Data Visualization)
trialTaskPanel = uipanel(appFig, 'Title', 'Trial/Task Selection (Trial Data Only)', 'Position', [240, 280, 280, 180], 'Visible', 'on');
trialLabel = uilabel(trialTaskPanel, 'Text', 'Trial:', 'Position', [10, 130, 50, 22]);
trialDropdown = uidropdown(trialTaskPanel, 'Items', {}, 'Position', [70, 130, 180, 22],'Enable','off');
taskLabel = uilabel(trialTaskPanel, 'Text', 'Task:', 'Position', [10, 90, 50, 22]);
taskDropdown = uidropdown(trialTaskPanel, 'Items', {}, 'Position', [70, 90, 180, 22], 'Enable','off');

% Plot button
plotButton = uibutton(appFig, 'push', 'Text', 'Plot', 'Position', [540, 350, 220, 40], ...
                      'ButtonPushedFcn', @plotButtonCallback, 'FontWeight', 'bold');

% Axes for plotting (initially hidden)
plotAxes = uiaxes(appFig, 'Position', [20, 20, 760, 240], 'Visible', 'off');


% --- Callbacks ---

    function browseButtonCallback(~, ~)
        [filename, pathname] = uigetfile('*.mat', 'Select a .mat file');
        if isequal(filename, 0) || isequal(pathname, 0)
            % User canceled
            return;
        end
        filePathField.Value = fullfile(pathname, filename);
        
        % Load data and update dropdowns if possible.
        try
            data = load(filePathField.Value);
             if isfield(data, 'dsfilt_emg') && isfield(data,'finger_kinematics')
                 trialDropdown.Items = arrayfun(@(x) ['Trial ' num2str(x)], 1:size(data.dsfilt_emg,1), 'UniformOutput', false);
                 trialDropdown.Enable = 'on';
                 trialDropdown.Value = trialDropdown.Items{1};

                 taskDropdown.Items = arrayfun(@(x) ['Task ' num2str(x)], 1:size(data.dsfilt_emg,2), 'UniformOutput', false);
                 taskDropdown.Enable = 'on';
                 taskDropdown.Value = taskDropdown.Items{1};
             else
                 uialert(appFig, 'Selected file does not contain the required variables (dsfilt_emg and finger_kinematics).', 'File Error');
                 trialDropdown.Enable = 'off';
                 taskDropdown.Enable = 'off';
             end

        catch ME
            uialert(appFig, ['Error loading file: ', ME.message], 'File Error');
             trialDropdown.Enable = 'off';
             taskDropdown.Enable = 'off';
        end

    end

   function plotButtonCallback(~, ~)
       filePath = filePathField.Value;
        if isempty(filePath)
            uialert(appFig, 'Please select a .mat file.', 'File Not Selected');
            return;
        end

        try
           data = load(filePath);
           if ~isfield(data, 'dsfilt_emg') || ~isfield(data,'finger_kinematics')
               uialert(appFig, 'Selected file does not contain the required variables (dsfilt_emg and finger_kinematics).','File Error');
               return;
           end

            visType = visTypeDropdown.Value;
            cla(plotAxes); % Clear previous plots
            plotAxes.Visible = 'on';  % Make axes visible

            switch visType
                case 'Trial Data'
                    %Get selected trial and task
                    trialNum = str2double(trialDropdown.Value(7:end));
                    taskNum = str2double(taskDropdown.Value(6:end));
                    visualize_trial_data_single(data, trialNum, taskNum, plotAxes);
                case 'EMG Task Highlighted'
                     visualize_emg_tasks_highlighted_app(data, plotAxes);
                case 'Kinematics Marker Highlighted'
                     visualize_kinematics_per_marker_task_highlighted_app(data, plotAxes);

            end
        catch ME
            uialert(appFig, ['Error during plotting: ', ME.message], 'Plotting Error');
            plotAxes.Visible = 'off'; % Hide axes on error
        end
   end


   % --- Helper Visualization Functions (Modified for App) ---
     function visualize_trial_data_single(data, trial, task, ax)
        % Modified visualize_trial_data to plot on a specified axes
        emg_data = data.dsfilt_emg{trial, task};
        kinematics_data = data.finger_kinematics{trial, task};
        emg_labels = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};
        num_kinematics = size(kinematics_data, 2);

        % EMG Plot
        yyaxis(ax, 'left');
        cla(ax); %Clear left y-axis
        hold(ax, 'on');
        for i = 1:size(emg_data, 2)
            plot(ax, emg_data(:, i), 'DisplayName', emg_labels{i});
        end
        hold(ax, 'off');
        title(ax, sprintf('Trial %d, Task %d', trial, task));
        xlabel(ax, 'Time (samples)');
        ylabel(ax, 'EMG Amplitude');
        legend(ax, 'Location', 'best');
        grid(ax, 'on');


        % Kinematics Plot (3D)
        yyaxis(ax, 'right');  % Switch to the right y-axis
        cla(ax);  %Clear right y-axis
        hold(ax, 'on');
        num_markers = num_kinematics / 3;
        for i = 1:num_markers
            start_col = (i - 1) * 3 + 1;
            end_col = start_col + 2;
             if end_col > size(kinematics_data, 2)
                fprintf('Warning: Skipping marker %d due to insufficient columns in kinematics data.\n', i);
                break; % Exit the marker loop
             end
            plot3(ax, kinematics_data(:, start_col), kinematics_data(:, start_col + 1), kinematics_data(:, start_col + 2), ...
                  'DisplayName', sprintf('Marker %d', i));
        end
        hold(ax, 'off');
        ylabel(ax, 'Kinematics Position');  % Label for the right y-axis
        legend(ax, 'Location', 'best');
        grid(ax, 'on');
        view(ax, 3);

        %Link x-axes (although there is only one x-axis, it's good practice)
        all_axes = findobj(ax.Parent,'Type','axes');
        linkaxes(all_axes,'x');


    end



    function visualize_emg_tasks_highlighted_app(data, ax)
        % Modified for app: Plots on the provided axes 'ax'
        emg_labels = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};
        num_tasks = size(data.dsfilt_emg, 2);
        num_trials = size(data.dsfilt_emg, 1);
        colors = jet(num_tasks);

        for trial = 1:num_trials
            for emg_channel = 1:length(emg_labels)
               
                %Create a new subplot within the provided axes
                subAx = subplot(length(emg_labels),1,emg_channel,'Parent',ax);
                hold(subAx, 'on');

                 % Find maximum y-value across all tasks for this channel & trial (for box height)
                max_y = -Inf;
                min_y = Inf;
                for task = 1:num_tasks
                    try
                         emg_data = data.dsfilt_emg{trial, task};
                    catch
                        continue;
                    end
                     max_y = max(max_y, max(emg_data(:, emg_channel)));
                     min_y = min(min_y, min(emg_data(:,emg_channel)));
                end


                for task = 1:num_tasks
                     try
                        emg_data = data.dsfilt_emg{trial, task};
                     catch ME
                        fprintf('Error at Trial %d, Channel %d, Task %d: %s\n', trial, emg_channel, task, ME.message);
                        continue;
                    end
                    num_samples = size(emg_data, 1);
                    time_vector = (0:num_samples - 1)';

                    x_start = (task - 1) * num_samples;
                    x_end = task * num_samples;

                    rectangle(subAx,'Position', [x_start, min_y, num_samples, max_y-min_y], ...
                              'FaceColor', [colors(task, :) 0.2], 'EdgeColor', 'none');
                    plot(subAx, time_vector + x_start, emg_data(:, emg_channel), 'Color', colors(task, :), 'DisplayName', sprintf('Task %d', task));
                    text(subAx, x_start + num_samples / 2, max_y * 0.9, sprintf('Task %d', task), ...
                         'HorizontalAlignment', 'center', 'Color', 'black', 'FontWeight', 'bold');
                end

                hold(subAx, 'off');
                title(subAx, [emg_labels{emg_channel},sprintf(' - Trial %d',trial)]);
                ylabel(subAx, 'Amplitude');
                xlabel(subAx, 'Time (shifted by task)');
                grid(subAx, 'on');

            end
        end
          % Link x-axes for synchronized zooming/panning
            all_axes = findobj(ax, 'Type', 'axes');
            linkaxes(all_axes, 'x');
    end

    function visualize_kinematics_per_marker_task_highlighted_app(data, ax)
        % Modified for app:  Plots on the provided axes 'ax'

        num_tasks = size(data.finger_kinematics, 2);
        num_trials = size(data.finger_kinematics, 1);
        num_kinematics = size(data.finger_kinematics{1,1}, 2);
        num_markers = num_kinematics / 3;
         if mod(num_kinematics, 3) ~= 0
              fprintf('Warning: Number of kinematics is not a multiple of 3.\n');
              num_markers = floor(num_kinematics/3);
         end
        colors = jet(num_tasks);

        for trial = 1:num_trials
            for marker = 1:num_markers
                for coord = 1:3  % X, Y, Z
                    % Create subplots within the provided axes 'ax'
                    subAx = subplot(num_markers, 3, (marker - 1) * 3 + coord, 'Parent', ax);
                    hold(subAx, 'on');

                    %Find max/min
                    max_y = -Inf;
                    min_y = Inf;
                    for task = 1:num_tasks
                        try
                            kinematics_data = data.finger_kinematics{trial,task};
                        catch
                            continue;
                        end
                        start_col = (marker - 1) * 3 + coord;
                        if start_col > size(kinematics_data,2)
                            continue;
                        end
                        max_y = max(max_y,max(kinematics_data(:,start_col)));
                        min_y = min(min_y, min(kinematics_data(:,start_col)));
                    end

                    for task = 1:num_tasks
                        try
                            kinematics_data = data.finger_kinematics{trial, task};
                        catch ME
                            fprintf('Error at Trial %d, Task %d, Marker %d: %s\n', trial, task, marker, ME.message);
                            continue;
                        end
                        start_col = (marker - 1) * 3 + coord;
                         if start_col > size(kinematics_data, 2)
                             fprintf('Warning: start_col exceeds kinematic data dimenstion\n');
                             continue;
                         end
                        num_samples = size(kinematics_data, 1);
                        time_vector = (0:num_samples - 1)';

                        x_start = (task - 1) * num_samples;
                        x_end = task * num_samples;

                        rectangle(subAx, 'Position', [x_start, min_y, num_samples, max_y-min_y], ...
                                  'FaceColor', [colors(task, :) 0.2], 'EdgeColor', 'none');
                        plot(subAx, time_vector + x_start, kinematics_data(:, start_col), 'Color', colors(task, :), 'DisplayName', sprintf('Task %d', task));
                        text(subAx, x_start + num_samples / 2, max_y * 0.9, sprintf('Task %d', task), ...
                             'HorizontalAlignment', 'center', 'Color', 'black', 'FontWeight', 'bold');
                    end

                    hold(subAx, 'off');
                    if coord == 1
                        title(subAx, sprintf('Marker %d - X', marker));
                    elseif coord == 2
                        title(subAx, sprintf('Marker %d - Y', marker));
                    else
                        title(subAx, sprintf('Marker %d - Z', marker));
                    end
                    ylabel(subAx, 'Position');
                    xlabel(subAx, 'Time (shifted by task)');
                    grid(subAx, 'on');
                end
            end
        end
          %Link the x-axes
         all_axes = findobj(ax,'Type','axes');
         linkaxes(all_axes,'x');
    end

end