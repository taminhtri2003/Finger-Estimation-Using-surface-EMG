function analyze_kinematics_similarity_variability(mat_file_path)
% ANALYZE_KINEMATICS_SIMILARITY_VARIABILITY_DTW_M Analyzes kinematics data for motion similarity and variability using dtw.m.
%   Calculates Dynamic Time Warping (DTW) distances (using provided dtw.m function)
%   and performs Principal Component Analysis (PCA) on kinematics data from a .mat file
%   to quantify motion similarity and variability across trials.
%
%   This version uses the provided 'dtw.m' function instead of pdist2('dtw').
%
%   Args:
%       mat_file_path (char): Path to the input .mat file (e.g., 's1.mat').

    % Load the .mat file
    load(mat_file_path);

    % Sensor names (assuming the order is consistent) -  Using Marker names for kinematics now
    marker_names_prefix = 'Marker_'; % Or 'Marker' if you prefer
    coordinate_names = {'X', 'Y', 'Z'}; % Coordinate names
    num_markers = 69 / 3; % Assuming 69 columns are x,y,z for each marker

    num_trials = size(finger_kinematics, 1);
    num_tasks = size(finger_kinematics, 2);


    % Results structure
    results = struct();

    % Loop through each task
    for task_idx = 1:num_tasks
        task_name = sprintf('Task_%d', task_idx);
        results.(task_name) = struct();

        % Loop through each marker
        for marker_idx = 1:num_markers
            marker_name = sprintf('%s%d', marker_names_prefix, marker_idx); % e.g., 'Marker_1'
            results.(task_name).(marker_name) = struct();

            % Loop through each coordinate (X, Y, Z)
            for coord_idx = 1:length(coordinate_names)
                coordinate_name = coordinate_names{coord_idx};
                marker_coordinate_name = [marker_name, '_', coordinate_name];
                results.(task_name).(marker_name).(coordinate_name) = struct();

                % --- Dynamic Time Warping (DTW) using provided dtw.m ---
                dtw_distances = [];
                trial_kinematics = cell(1, num_trials); % To store kinematics for all trials for this marker/coord

                for trial_idx = 1:num_trials
                    trial_kinematics{trial_idx} = finger_kinematics{trial_idx, task_idx}(:, (marker_idx-1)*3 + coord_idx);
                end

                % Calculate DTW distances between all pairs of trials
                for trial1_idx = 1:num_trials
                    for trial2_idx = trial1_idx+1:num_trials % Avoid redundant and self-comparisons
                        traj1 = trial_kinematics{trial1_idx};
                        traj2 = trial_kinematics{trial2_idx};
                        % Using the provided dtw function
                        d = dtw(traj1, traj2); % Calling the provided dtw function
                        dtw_distances = [dtw_distances, d];
                    end
                end

                avg_dtw_distance = mean(dtw_distances);
                results.(task_name).(marker_name).(coordinate_name).avg_dtw_distance = avg_dtw_distance;


                % --- Principal Component Analysis (PCA) ---
                kinematics_matrix = [];
                for trial_idx = 1:num_trials
                    kinematics_matrix = [kinematics_matrix, trial_kinematics{trial_idx}]; % Time x Trials matrix
                end

                % Apply PCA
                [coeff, score, latent, ~, explained] = pca(kinematics_matrix); % Data is already centered by pca

                results.(task_name).(marker_name).(coordinate_name).pca_explained_variance = explained;
                results.(task_name).(marker_name).(coordinate_name).pca_coeff = coeff; % Optional: store coefficients if needed
                results.(task_name).(marker_name).(coordinate_name).pca_score = score; % Optional: store scores if needed


            end
        end
    end

    % --- Display Results ---
    disp('Kinematics Similarity and Variability Analysis Results (using dtw.m):');
    disp('------------------------------------------------------');

    for task_idx = 1:num_tasks
        task_name = sprintf('Task_%d', task_idx);
        disp(['Task: ', task_name]);
        for marker_idx = 1:num_markers
            marker_name = sprintf('%s%d', marker_names_prefix, marker_idx);
            for coord_idx = 1:length(coordinate_names)
                coordinate_name = coordinate_names{coord_idx};
                marker_coordinate_name = [marker_name, '_', coordinate_name];

                dtw_dist = results.(task_name).(marker_name).(coordinate_name).avg_dtw_distance;
                pca_explained = results.(task_name).(marker_name).(coordinate_name).pca_explained_variance;

                disp(['  ', marker_coordinate_name, ':']);
                disp(['    Average DTW Distance: ', num2str(dtw_dist, '%.4f')]);
                disp(['    PCA Explained Variance (First 3 PCs): ', num2str(pca_explained(1:3)', '%.2f%% ')]); % Display first 3 PCs explained variance

            end
        end
        disp('------------------------------------------------------');
    end

end

% --- dtw.m function (place this in the same directory or MATLAB path) ---
% % Copyright (C) 2013 Quan Wang <wangq10@rpi.edu>,
% % Signal Analysis and Machine Perception Laboratory,
% % Department of Electrical, Computer, and Systems Engineering,
% % Rensselaer Polytechnic Institute, Troy, NY 12180, USA
% % dynamic time warping of two signals
% function d=dtw(s,t,w)
% % s: signal 1, size is ns*k, row for time, colume for channel
% % t: signal 2, size is nt*k, row for time, colume for channel
% % w: window parameter
% %      if s(i) is matched with t(j) then |i-j|<=w
% % d: resulting distance
% if nargin<3
%     w=Inf;
% end
% ns=size(s,1);
% nt=size(t,1);
% if size(s,2)~=size(t,2)
%     error('Error in dtw(): the dimensions of the two input signals do not match.');
% end
% w=max(w, abs(ns-nt)); % adapt window size
% %% initialization
% D=zeros(ns+1,nt+1)+Inf; % cache matrix
% D(1,1)=0;
% %% begin dynamic programming
% for i=1:ns
%     for j=max(i-w,1):min(i+w,nt)
%         oost=norm(s(i,:)-t(j,:));
%         D(i+1,j+1)=oost+min( [D(i,j+1), D(i+1,j), D(i,j)] );
%
%     end
% end
% d=D(ns+1,nt+1);
% end
