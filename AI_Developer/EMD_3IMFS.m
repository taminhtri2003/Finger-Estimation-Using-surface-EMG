% Load the .mat file
load('s1_full.mat'); % Replace 'your_data_file.mat' with your file

% Define muscle names
muscle_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};

% Parameters for EMD and Chaos Analysis
num_imfs = 5;
embedding_dimension = 10;
time_delay = 5;

% Create a folder to save the figures
output_folder = 'EMG_Chaos_Analysis_Results';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% --- Loop through tasks ---
for task = 1:size(dsfilt_emg, 2)
    fprintf('Processing Task %d\n', task);

    % --- Loop through muscles ---
    for muscle = 1:size(dsfilt_emg{1,1}, 2) % Assuming all cells have the same number of muscles
        fprintf('  Processing Muscle %s\n', muscle_names{muscle});

        % Initialize a figure for the combined plots
        figure('Position', [100, 100, 1200, 800]); % Larger figure for better visualization

        % --- Loop through trials ---
        for trial = 1:size(dsfilt_emg, 1)
            emg_data = dsfilt_emg{trial, task};

            if isempty(emg_data)
                fprintf('    Skipping Trial %d (empty data)\n', trial);
                continue;
            end

            emg_signal = emg_data(:, muscle);

            % --- EMD ---
            try
                [imfs, residual, ~] = emd(emg_signal, 'MaxNumIMF', num_imfs);
            catch ME
                fprintf('    EMD failed for Trial %d: %s\n', trial, ME.message);
                continue; % Skip to the next trial
            end

            % --- Plotting (within the trial loop, but on the same figure) ---

            % Subplot 1: Original EMG Signal
            subplot(3, size(dsfilt_emg, 1), trial); % 3 rows, number of trials columns, position based on trial
            plot(emg_signal);
            title(['Trial ' num2str(trial) ': Original EMG']);
            if trial == 1  % Only add y-label to the first trial subplot
                ylabel('Amplitude');
            end
            xlabel('Time'); % Add x-label to all subplots


            % Subplot 2: IMFs
           subplot(3, size(dsfilt_emg, 1), trial + size(dsfilt_emg, 1)); % Shift to the second row
            hold on;
            for i = 1:min(3, size(imfs, 2))
                plot(imfs(:, i), 'DisplayName', ['IMF ' num2str(i)]);
            end
            hold off;

             if trial == 1
                 ylabel('Amplitude'); %Add ylabel to the first subplot in this row.
             end

            if trial == size(dsfilt_emg, 1)
                legend('Location', 'best'); % add a single legend to last subplot.
            end

            title(['Trial ' num2str(trial) ': First 3 IMFs']);
            xlabel('Time');

            % --- Chaos Analysis and Subplot 3: Reconstructed Phase Space ---
            if ~isempty(imfs)
                try
                    reconstructed_phase_space = zeros(length(imfs(:,1)) - (embedding_dimension - 1) * time_delay, embedding_dimension);
                    for i = 1:embedding_dimension
                         reconstructed_phase_space(:, i) = imfs((1 + (i - 1) * time_delay):(length(imfs(:,1)) - (embedding_dimension - i) * time_delay),1);
                    end

                     % Lyapunov Exponent
                     [lyap_exp, ~] = lyapunovExponent(reconstructed_phase_space, 1/0.0005, 'TimeDelay', time_delay, 'MinSeparation', 2);
                     fprintf('    Trial %d: Largest Lyapunov Exponent = %f\n', trial, max(lyap_exp));

                     %Correlation Dimension
                     correlation_dimension = correlationDimension(reconstructed_phase_space);
                     fprintf('  Muscle %s: Correlation Dimension = %f\n', muscle_names{muscle}, correlation_dimension);

                catch ME
                    fprintf('    Chaos analysis failed for Trial %d: %s\n', trial, ME.message);
                    lyap_exp = NaN;
                end

                 subplot(3, size(dsfilt_emg, 1), trial + 2 * size(dsfilt_emg, 1));
                if size(reconstructed_phase_space, 2) >= 3
                    plot3(reconstructed_phase_space(:, 1), reconstructed_phase_space(:, 2), reconstructed_phase_space(:, 3));
                    xlabel('IMF1(t)');
                    ylabel('IMF1(t + \tau)');
                    zlabel('IMF1(t + 2\tau)');
                elseif size(reconstructed_phase_space,2) == 2
                    plot(reconstructed_phase_space(:,1), reconstructed_phase_space(:,2));
                    xlabel('IMF1(t)');
                    ylabel('IMF1(t+\tau)');
                else
                    plot(reconstructed_phase_space(:,1));
                    xlabel('time');
                    ylabel('IMF1(t)');
                end

                if trial == 1
                    ylabel('Phase Space');
                end

                title(['Trial ' num2str(trial) ': Phase Space, LyapExp=' num2str(max(lyap_exp), '%.3f')]);
                 grid on;
            end

        end % end loop over trials

        % --- Use suplabel for super title, x-label, and y-label ---
        suplabel(['Task ' num2str(task) ' - Muscle ' muscle_names{muscle} ' - All Trials'], 't'); % 't' for title
        suplabel('Time', 'x');          % 'x' for x-axis label
        %suplabel('Amplitude/Phase Space', 'y'); % 'y' for y-axis, but this one is tricky...

        % --- End of suplabel section ---
        
        % Save the figure
        filename = fullfile(output_folder, ['Task' num2str(task) '_Muscle' muscle_names{muscle} '_Combined.png']);
        saveas(gcf, filename);
        close(gcf); % Close the figure after saving

    end % end loop over muscles
end % end loop over tasks

fprintf('Analysis Complete. Figures saved in %s\n', output_folder);