function r2 = calculateR2(y_actual, y_pred)
%calculateR2 Calculates the R-squared (Coefficient of Determination).
%
%   R2 = calculateR2(Y_ACTUAL, Y_PRED) calculates the R-squared value
%   comparing the predicted values Y_PRED to the actual values Y_ACTUAL.
%
%   Inputs:
%       y_actual - Vector or matrix of actual target values. If a matrix,
%                  R2 is calculated column-wise.
%       y_pred   - Vector or matrix of predicted values, same size as y_actual.
%
%   Output:
%       r2       - R-squared value(s). If inputs are matrices, R2 is a row
%                  vector containing the R2 value for each column. Returns
%                  NaN for columns where y_actual is constant.
%
%   Formula: R2 = 1 - (SS_res / SS_tot)
%       SS_res = sum((y_actual - y_pred).^2)  (Residual Sum of Squares)
%       SS_tot = sum((y_actual - mean(y_actual)).^2) (Total Sum of Squares)

    % Ensure inputs are numeric and have the same size
    if ~isnumeric(y_actual) || ~isnumeric(y_pred)
        error('Inputs must be numeric.');
    end
    if ~isequal(size(y_actual), size(y_pred))
        error('Inputs y_actual and y_pred must have the same size.');
    end

    % Calculate column-wise mean of actual values
    mean_y_actual = mean(y_actual, 1); 
    
    % Calculate Residual Sum of Squares (SS_res) column-wise
    ss_res = sum((y_actual - y_pred).^2, 1);
    
    % Calculate Total Sum of Squares (SS_tot) column-wise
    ss_tot = sum((y_actual - mean_y_actual).^2, 1);
    
    % Calculate R2, handle cases where SS_tot is close to zero (constant actual data)
    r2 = zeros(1, size(y_actual, 2)); % Preallocate result row vector
    for j = 1:size(y_actual, 2) % Iterate through columns (angles)
        if ss_tot(j) < eps % Check if SS_tot is effectively zero
            % R2 is undefined or meaningless if actual data doesn't vary
            r2(j) = NaN; 
            warning('R2 calculation: Column %d has near-zero total sum of squares (constant actual data). R2 set to NaN.', j);
        else
            r2(j) = 1 - (ss_res(j) / ss_tot(j));
        end
    end
    
end
