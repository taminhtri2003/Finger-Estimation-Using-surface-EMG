function visualize_feature_importance(models, muscle_names)
    % Inputs:
    %   models: Cell array of trained TreeBagger models (one per joint)
    %   muscle_names: Cell array of muscle names (e.g., {'APL', 'FCR', ...})

    num_joints = length(models);
    num_muscles = length(muscle_names);

    % Validate inputs
    if num_muscles ~= 8
        error('Expected 8 muscle names.');
    end

    figure('Name', 'Feature Importance by Joint', 'NumberTitle', 'off');
    
    for joint_index = 1:num_joints
        subplot(2, ceil(num_joints/2), joint_index); % Arrange subplots

        if ~isempty(models{joint_index})
            if exist('oobPermutedPredictorImportance', 'file')
                % Get feature importance from the TreeBagger model
                importances = oobPermutedPredictorImportance(models{joint_index});
                
                % Group importances by muscle (assuming 3 features per muscle)
                muscle_importances = sum(reshape(importances, 3, num_muscles), 1);
                
                % Normalize muscle importances to sum to 1
                muscle_importances = muscle_importances / sum(muscle_importances);
                
                % Create bar chart
                bar(muscle_importances);
                set(gca, 'XTick', 1:num_muscles, 'XTickLabel', muscle_names);
                title(['Joint ', num2str(joint_index)]);
                xlabel('Muscle');
                ylabel('Normalized Importance');
                xtickangle(45);
                grid on;
            else
                text(0.5, 0.5, 'Feature importance not available', ...
                     'HorizontalAlignment', 'center', 'FontSize', 10);
                axis off;
                warning('oobPermutedPredictorImportance not found. Install Statistics and Machine Learning Toolbox.');
            end
        else
            text(0.5, 0.5, 'No model for this joint', ...
                 'HorizontalAlignment', 'center', 'FontSize', 10);
            axis off;
        end
    end
    
    sgtitle('Feature Importance by Muscle for Each Joint');
end