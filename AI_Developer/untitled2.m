%% Visualization: Comparison of RMSE and R^2
for j = 1:size(Y_all, 2)  % Iterate through each joint
    figure;
    % RMSE Comparison
    subplot(1, 2, 1);
    % Concatenate horizontally *within* the bar function
    bar([rmse_combined(j), rmse_individual(:, j)', rmse_group(:, j)']);
    xticks(1:(1 + num_muscles + num_groups));
    xticklabels({ 'Combined', muscle_names{:}, group_names{:} }); % FIX HERE
    xtickangle(45);
    ylabel('RMSE (degrees)');
    title(['RMSE Comparison for ' joint_names{j}]);
    grid on;
    set(gca, 'FontSize', 10);
    % R^2 Comparison
    subplot(1, 2, 2);
    % Concatenate horizontally *within* the bar function
    bar([r2_combined(j), r2_individual(:, j)', r2_group(:, j)']);
    xticks(1:(1 + num_muscles + num_groups));
    xticklabels({ 'Combined', muscle_names{:}, group_names{:} }); % FIX HERE
    xtickangle(45);
    ylabel('R^2');
    title(['R^2 Comparison for ' joint_names{j}]);
    grid on;
     set(gca, 'FontSize', 10);

    % Save the figure
    filename = sprintf('Joint_%s_RMSE_R2_Comparison.png', joint_names{j});
    saveas(gcf, filename);
    close(gcf); % Close figure to prevent display
end

%% Overall average performance across all joints.
overall_rmse_combined = sqrt(mean(rmse_combined.^2));
overall_r2_combined = mean(r2_combined);
overall_rmse_individual = squeeze(sqrt(mean(rmse_individual.^2, 2)));
overall_r2_individual = squeeze(mean(r2_individual,2));
overall_rmse_group = squeeze(sqrt(mean(rmse_group.^2, 2)));
overall_r2_group = squeeze(mean(r2_group, 2));

figure;
% RMSE Comparison
subplot(1, 2, 1);
% Concatenate horizontally *within* the bar function
bar([overall_rmse_combined, overall_rmse_individual', overall_rmse_group']);
xticks(1:(1 + num_muscles + num_groups));
xticklabels({ 'Combined', muscle_names{:}, group_names{:} }); % FIX HERE
xtickangle(45);
ylabel('Overall RMSE (degrees)');
title('Overall RMSE Comparison');
grid on;
set(gca, 'FontSize', 10);

% R^2 Comparison
subplot(1, 2, 2);
% Concatenate horizontally *within* the bar function
bar([overall_r2_combined, overall_r2_individual', overall_r2_group']);
xticks(1:(1 + num_muscles + num_groups));
xticklabels({ 'Combined', muscle_names{:}, group_names{:} }); % FIX HERE
xtickangle(45);
ylabel('Overall R^2');
title('Overall R^2 Comparison');
grid on;
set(gca, 'FontSize', 10);

% Save the overall figure
saveas(gcf, 'Overall_RMSE_R2_Comparison.png');
close(gcf); % Close figure to prevent display