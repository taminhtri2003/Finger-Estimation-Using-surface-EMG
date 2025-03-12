% Load the data
load('model_predictions.mat');

% Example: Visualize predictions for task 1, output dimension 1
task_num = 1;
output_dim = 1;

figure;
plot(X_1(:,1), y_true_1(:, output_dim), 'o', 'DisplayName', 'True Values');  % Assuming first column of X is relevant
hold on;
plot(X_1(:,1), y_pred_1(:, output_dim), 'x', 'DisplayName', 'Predictions');
hold off;
xlabel('Input Feature 1'); % Adjust as needed
ylabel(['Output Dimension ', num2str(output_dim)]);
title(['Task ', num2str(task_num), ' - Output ', num2str(output_dim)]);
legend show;

% Example: Visualize MSE for all tasks:
mse_values = [];
for i = 1:7
    y_true = eval(['y_true_', num2str(i)]);  % Dynamically get variable names
    y_pred = eval(['y_pred_', num2str(i)]);
    mse = mean((y_true(:) - y_pred(:)).^2); % Calculate for all output dimensions
    mse_values(i) = mse;
end

figure;
bar(mse_values);
xlabel('Task Number');
ylabel('Mean Squared Error');
title('MSE for each task');