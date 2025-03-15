% Main script to run everything
load('s4.mat'); % Load your .mat file
joint_angles = calculate_joint_angles_ver2(finger_kinematics);
muscle_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};
% Call the function with three outputs
% Example: Run the regression (replace with your actual function call)
[models, predictions, Y_all] = emg_to_angle_regression(dsfilt_emg, joint_angles);

% Visualize performance for a specific joint (e.g., joint 1)
visualize_joint_predictions(models, Y_all, predictions); % No joint index provided

% Visualize R² values across all joints
visualize_r2_bar(Y_all, predictions);

% Visualize error distributions across all joints
visualize_error_boxplot(Y_all, predictions);

visualize_feature_importance(models, muscle_names);
