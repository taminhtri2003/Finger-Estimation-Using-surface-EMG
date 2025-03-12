function r2 = calculate_r2(actual, predicted)
    % Calculate R² value between actual and predicted data
    valid_idx = ~isnan(actual) & ~isnan(predicted);
    actual = actual(valid_idx);
    predicted = predicted(valid_idx);
    if isempty(actual)
        r2 = NaN;
    else
        mean_actual = mean(actual);
        ss_total = sum((actual - mean_actual).^2);
        ss_res = sum((actual - predicted).^2);
        if ss_total == 0
            r2 = 0;
        else
            r2 = 1 - (ss_res / ss_total);
        end
    end
end