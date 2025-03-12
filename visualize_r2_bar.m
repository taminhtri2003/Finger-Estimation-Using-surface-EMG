function visualize_r2_bar(Y_all, predictions)
    % Inputs:
    %   Y_all: Matrix of actual joint angles (time x joints)
    %   predictions: Matrix of predicted joint angles (time x joints)

    % 1. Input Validation
    % Check if inputs are valid and have matching sizes
    if ~isequal(size(Y_all), size(predictions))
        error('Input matrices Y_all and predictions must have the same size.');
    end

    num_joints = size(Y_all, 2);

    % 2. R² Calculation with NaN Handling
    r2_values = zeros(1, num_joints);
    for joint = 1:num_joints
        actual = Y_all(:, joint);
        predicted = predictions(:, joint);

        % Handle cases where actual values are all the same (variance is zero)
        if all(actual == actual(1))
            r2_values(joint) = NaN; % Or you might set it to 0, depending on your interpretation
        else
            % Remove NaN values before calculating R²
            valid_indices = ~isnan(actual) & ~isnan(predicted);
            if sum(valid_indices) > 1  % Need at least 2 points to calculate R²
                r2_values(joint) = calculate_r2(actual(valid_indices), predicted(valid_indices));
            else
                r2_values(joint) = NaN; % Not enough data to calculate R²
            end
        end
    end

    % 3. Visualization Enhancements
    figure;
    bar(1:num_joints, r2_values);
    title('R² Values for Each Joint', 'FontSize', 14, 'FontWeight', 'bold'); % Enhanced title
    xlabel('Joint Index', 'FontSize', 12);
    ylabel('R²', 'FontSize', 12);
    set(gca, 'XTick', 1:num_joints);
    ylim([-1 1]); % Set y-axis limits for R² (typically -1 to 1)
    grid on;      % Add grid lines for better readability

    % Add text labels to the bars
    for i = 1:num_joints
        if ~isnan(r2_values(i))
            text(i, r2_values(i), sprintf('%.2f', r2_values(i)), ...
                 'HorizontalAlignment', 'center', ...
                 'VerticalAlignment', 'bottom', ...
                 'FontSize', 8);
        else
            text(i, 0, 'NaN', ...
                 'HorizontalAlignment', 'center', ...
                 'VerticalAlignment', 'top', ...
                 'FontSize', 8);
        end
    end
end