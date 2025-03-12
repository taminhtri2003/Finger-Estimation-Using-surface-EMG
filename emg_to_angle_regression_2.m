function [models, predictions, Y_all, X_all] = emg_to_angle_regression_2(dsfilt_emg, joint_angles)
    % Inputs:
    %   dsfilt_emg: <5x7 cell> containing EMG data, each cell <40000x8>
    %   joint_angles: <5x7 cell> containing joint angles, each cell <4000x15>
    % Outputs:
    %   models: Cell array of trained TreeBagger models (one per joint, empty if NaN)
    %   predictions: Matrix of predicted joint angles (same size as Y_all)
    %   Y_all: Matrix of actual joint angles (same size as predictions)
    %   X_all: Matrix of extracted features (samples x features)

    % Parameters
    window_size = 100;    % Samples for feature extraction
    step_size = 50;      % Sliding window step
    num_features = 3;     % RMS, MAV, Variance
    num_muscles = 8;      % Number of EMG channels
    
    % Initialize storage
    X_all = [];  % Features
    Y_all = [];  % Target angles (matrix with 14 columns)

    % Feature extraction and data preparation
    for trial = 1:5
        for task = 1:7  % Corrected loop: tasks go from 1 to 7
            emg_data = dsfilt_emg{trial, task};
            angle_data = joint_angles{trial, task};
            
            % Calculate number of windows
            num_windows = floor((size(emg_data, 1) - window_size) / step_size) + 1;
            features = zeros(num_windows, num_muscles * num_features);
            
            % Extract features from EMG data
            for win = 1:num_windows
                start_idx = (win-1) * step_size + 1;
                end_idx = start_idx + window_size - 1;
                window_data = emg_data(start_idx:end_idx, :);
                
                % Compute features
                features(win, 1:num_muscles) = rms(window_data);                    % RMS
                features(win, num_muscles+1:2*num_muscles) = mean(abs(window_data)); % MAV
                features(win, 2*num_muscles+1:3*num_muscles) = var(window_data);    % Variance
            end
            
            % Match angle data to feature windows (downsample angles)
            angle_idx = round(linspace(1, size(angle_data, 1), num_windows));
            X_all = [X_all; features];
            Y_all = [Y_all; angle_data(angle_idx, :)];
        end
    end
    
    % Determine number of joints from Y_all
    num_joints = size(Y_all, 2);
    
    % Train a separate model for each joint angle
    models = cell(1, num_joints);    % Store models for each joint
    predictions = zeros(size(Y_all)); % Store predictions (same size as Y_all)

    for joint = 1:num_joints
        % Extract response vector for this joint
        Y_joint = Y_all(:, joint);
        
        if all(isnan(Y_joint))
            % Skip training if all targets are NaN (e.g., thumb DIP)
            models{joint} =;  % No model for this joint
            predictions(:, joint) = NaN;  % Set predictions to NaN
        else
            % Train TreeBagger model for this joint
            models{joint} = TreeBagger(50, X_all, Y_joint, 'Method', 'regression');
            % Generate predictions for this joint
            predictions(:, joint) = predict(models{joint}, X_all);
        end
    end
    
    % Optional: Display a message for debugging
    fprintf('Trained %d models. Skipped %d joints with all NaN values.\n', ...
        sum(~cellfun(@isempty, models)), sum(cellfun(@isempty, models)));
end