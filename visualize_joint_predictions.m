function visualize_joint_predictions(models, Y_all, predictions)
    % Inputs:
    %   models: Cell array of trained TreeBagger models
    %   Y_all: Matrix of actual joint angles (time x joints)
    %   predictions: Matrix of predicted joint angles (time x joints)
    
    num_joints = size(Y_all, 2);
    
    % Validate inputs
    if ~isequal(size(Y_all), size(predictions))
        error('Y_all and predictions must have the same size.');
    end
    
    for joint_index = 1:num_joints
        % Check for NaN data
        if all(isnan(Y_all(:, joint_index)))
            disp(['Joint ', num2str(joint_index), ' has no data (all NaN). No visualization generated.']);
            continue; % Skip to the next joint
        end
        
        % Extract data for the specified joint
        actual = Y_all(:, joint_index);
        predicted = predictions(:, joint_index);
        
        % Remove NaN entries for scatter and histogram
        valid_idx = ~isnan(actual) & ~isnan(predicted);
        actual_valid = actual(valid_idx);
        predicted_valid = predicted(valid_idx);
        errors = predicted_valid - actual_valid;
        
        % Calculate R²
        r2 = calculate_r2(actual, predicted);
        
        % Calculate MSE
        mse = mean((actual_valid - predicted_valid).^2);
        
        % Calculate RMSE
        rmse = sqrt(mse);
        
        % Create a figure for each joint
        figure('Name', ['Joint ', num2str(joint_index), ' Prediction Analysis'], 'NumberTitle', 'off');
        
        % 1. Time Series Plot
        subplot(2, 2, 1);
        plot(actual, 'b-', 'DisplayName', 'Actual');
        hold on;
        plot(predicted, 'r--', 'DisplayName', 'Predicted');
        hold off;
        legend('Location', 'best');
        title('Time Series');
        xlabel('Time Step');
        ylabel('Angle (degrees)');
        grid on;
        
        % 2. Scatter Plot with R², MSE, and RMSE
        subplot(2, 2, 2);
        scatter(actual_valid, predicted_valid, 10, 'filled', 'DisplayName', 'Actual vs Predicted');
        hold on;
        min_val = min([actual_valid; predicted_valid]);
        max_val = max([actual_valid; predicted_valid]);
        plot([min_val max_val], [min_val max_val], 'k--', 'DisplayName', 'y = x');
        hold off;
        legend('Location', 'best');
        title(['Scatter Plot (R² = ', num2str(r2, '%.4f'), ...
               ', MSE = ', num2str(mse, '%.4f'), ...
               ', RMSE = ', num2str(rmse, '%.4f'), ')']);
        xlabel('Actual Angle (degrees)');
        ylabel('Predicted Angle (degrees)');
        grid on;
        
        % 3. Error Histogram
        subplot(2, 2, 3);
        histogram(errors, 20, 'Normalization', 'probability', 'FaceColor', 'm', 'DisplayName', 'Error');
        legend('Location', 'best');
        title('Error Distribution');
        xlabel('Prediction Error (degrees)');
        ylabel('Probability');
        grid on;
        
        % 4. Feature Importance (if model exists)
        if ~isempty(models{joint_index})
            if exist('oobPermutedPredictorImportance', 'file')
                importances = oobPermutedPredictorImportance(models{joint_index});
                muscle_importances = sum(reshape(importances, 3, 8), 1);
                muscle_names = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};
                subplot(2, 2, 4);
                bar(muscle_importances);
                set(gca, 'XTick', 1:8, 'XTickLabel', muscle_names);
                title('Feature Importance by Muscle');
                xlabel('Muscle');
                ylabel('Importance');
                xtickangle(45);
            else
                subplot(2, 2, 4);
                text(0.5, 0.5, 'Feature importance not available', 'HorizontalAlignment', 'center');
                axis off;
                warning('oobPermutedPredictorImportance not found. Install Statistics and Machine Learning Toolbox.');
            end
        else
            subplot(2, 2, 4);
            text(0.5, 0.5, 'No model for this joint', 'HorizontalAlignment', 'center');
            axis off;
        end
        
        % Adjust layout
        sgtitle(['Prediction Analysis for Joint ', num2str(joint_index)]);
    end
end