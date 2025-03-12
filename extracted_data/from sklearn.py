from sklearn.metrics import r2_score
import numpy as np

def calculate_r2_score(y_true, y_pred):
    """
    Calculates the R-squared (coefficient of determination) score.

    Args:
        y_true: 1D array-like, ground truth (correct) target values.
        y_pred: 1D array-like, estimated target values.

    Returns:
        float: The R-squared score. Best possible score is 1.0, and it
               can be negative (because the model can be arbitrarily worse).
    """
    return r2_score(y_true, y_pred)

def calculate_r2_score_robust(y_true, y_pred):
    """
    Calculates R-squared, handling near-constant y_true values robustly.

    Args:
        y_true: 1D array-like, ground truth (correct) target values.
        y_pred: 1D array-like, estimated target values.

    Returns:
        float: The R-squared score.  Returns 1.0 if y_true is near-constant
               AND predictions are perfect, 0.0 if y_true is near-constant
               and predictions are not perfect, and uses sklearn's r2_score
               otherwise.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if np.allclose(y_true, y_true[0]):  # Check for near-constant values
        if np.allclose(y_true, y_pred):
            return 1.0  # Perfect prediction of a constant
        else:
            return 0.0  # Predictions not perfect
    else:
        return r2_score(y_true, y_pred)


# --- Example Usage ---

# Example 1: Good predictions
y_true1 = [3, -0.5, 2, 7]
y_pred1 = [2.5, 0.0, 2, 8]
r2_1 = calculate_r2_score(y_true1, y_pred1)
print(f"Example 1 (Good Predictions) - R-squared: {r2_1}")
r2_1_robust = calculate_r2_score_robust(y_true1, y_pred1)
print(f"Example 1 (Good Predictions) - Robust R-squared: {r2_1_robust}")

# Example 2: Poor predictions
y_true2 = [1, 2, 3, 4, 5]
y_pred2 = [5, 4, 3, 2, 1]  # Reversed order
r2_2 = calculate_r2_score(y_true2, y_pred2)
print(f"Example 2 (Poor Predictions) - R-squared: {r2_2}")
r2_2_robust = calculate_r2_score_robust(y_true2, y_pred2)
print(f"Example 2 (Poor Predictions) - Robust R-squared: {r2_2_robust}")
# Example 3: Constant y_true, perfect prediction
y_true3 = [2, 2, 2, 2]
y_pred3 = [2, 2, 2, 2]
r2_3 = calculate_r2_score(y_true3, y_pred3)  # sklearn might return NaN or -inf
print(f"Example 3 (Constant y_true, perfect) - R-squared: {r2_3}") # may show nan
r2_3_robust = calculate_r2_score_robust(y_true3, y_pred3)
print(f"Example 3 (Constant y_true, perfect) - Robust R-squared: {r2_3_robust}")


# Example 4: Constant y_true, imperfect prediction
y_true4 = [2, 2, 2, 2]
y_pred4 = [2, 2, 2, 3]  # One value is different
r2_4 = calculate_r2_score(y_true4, y_pred4)  # sklearn might return NaN or -inf
print(f"Example 4 (Constant y_true, imperfect) - R-squared: {r2_4}") # may show nan
r2_4_robust = calculate_r2_score_robust(y_true4, y_pred4)
print(f"Example 4 (Constant y_true, imperfect) - Robust R-squared: {r2_4_robust}")

# Example 5: Near-constant y_true, perfect prediction (within tolerance)
y_true5 = [2.0000, 2.0001, 1.9999, 2.0002]
y_pred5 = [2.0000, 2.0001, 1.9999, 2.0002]
r2_5 = calculate_r2_score(y_true5, y_pred5)
print(f"Example 5 (Near-constant y_true, perfect) - R-squared: {r2_5}")
r2_5_robust = calculate_r2_score_robust(y_true5, y_pred5)
print(f"Example 5 (Near-constant y_true, perfect) - Robust R-squared: {r2_5_robust}")

# Example 6: Near-constant y_true, imperfect prediction
y_true6 = [2.0000, 2.0001, 1.9999, 2.0002]
y_pred6 = [2.0001, 2.0002, 2.0000, 2.0003]  # Slightly different
r2_6 = calculate_r2_score(y_true6, y_pred6)
print(f"Example 6 (Near-constant y_true, imperfect) - R-squared: {r2_6}")
r2_6_robust = calculate_r2_score_robust(y_true6, y_pred6)
print(f"Example 6 (Near-constant y_true, imperfect) - Robust R-squared: {r2_6_robust}")

# Example 7: y_true with zero variance but y_pred is different
y_true7 = np.array([1,1,1,1])
y_pred7 = np.array([2,3,4,5])
r2_7 = calculate_r2_score(y_true7, y_pred7)  # sklearn might return NaN or -inf
print(f"Example 7 (zero variance) - R-squared: {r2_7}") # may show nan
r2_7_robust = calculate_r2_score_robust(y_true7, y_pred7)
print(f"Example 7 (zero variance) - Robust R-squared: {r2_7_robust}")