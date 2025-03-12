function visualize_shap_importance(models, X_all, muscle_names)
    % Inputs:
    %   models: Cell array of trained TreeBagger models (one per joint)
    %   X_all:  Matrix of features (samples x features)
    %   muscle_names: Cell array of muscle names (e.g., {'APL', 'FCR', ...})

    num_joints = length(models);
    num_muscles = length(muscle_names);
    num_features_per_muscle = 3; % RMS, MAV, Variance

    % Validate inputs
    if num_muscles ~= 8
        error('Expected 8 muscle names.');
    end

    figure('Name', 'SHAP Importance by Joint', 'NumberTitle', 'off');

    for joint_index = 1:num_joints
        subplot(2, ceil(num_joints/2), joint_index); % Arrange subplots

        if ~isempty(models{joint_index})
            % Calculate SHAP values
            % Assuming you have a function called 'shapleyValues'
            % (You might need to adapt this based on your SHAP implementation)
            shap_values = shapleyValues(models{joint_index}, X_all);

            % Aggregate SHAP values by muscle
            muscle_shap_values = zeros(size(X_all, 1), num_muscles);
            for muscle_idx = 1:num_muscles
                feature_start_idx = (muscle_idx - 1) * num_features_per_muscle + 1;
                feature_end_idx = feature_start_idx + num_features_per_muscle - 1;
                muscle_shap_values(:, muscle_idx) = sum(shap_values(:, feature_start_idx:feature_end_idx), 2);
            end

            % Calculate mean absolute SHAP value for each muscle
            mean_abs_shap = mean(abs(muscle_shap_values));

            % Normalize mean absolute SHAP values
            mean_abs_shap = mean_abs_shap / sum(mean_abs_shap);

            % Create bar chart
            bar(mean_abs_shap);
            set(gca, 'XTick', 1:num_muscles, 'XTickLabel', muscle_names);
            title(['Joint ', num2str(joint_index)]);
            xlabel('Muscle');
            ylabel('Mean Absolute SHAP Value (Normalized)');
            xtickangle(45);
            grid on;
        else
            text(0.5, 0.5, 'No model for this joint', ...
                 'HorizontalAlignment', 'center', 'FontSize', 10);
            axis off;
        end
    end

    sgtitle('SHAP Importance by Muscle for Each Joint');
end