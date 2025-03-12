function visualize_error_boxplot(Y_all, predictions)
    % Inputs:
    %   Y_all: Matrix of actual joint angles (time x joints)
    %   predictions: Matrix of predicted joint angles (time x joints)

    errors_all = predictions - Y_all;
    figure;
    boxplot(errors_all, 'Labels', string(1:size(Y_all, 2)));
    title('Prediction Errors for Each Joint');
    xlabel('Joint Index');
    ylabel('Error (degrees)');
end