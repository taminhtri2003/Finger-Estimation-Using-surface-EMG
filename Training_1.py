import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression  # Or any other regressor
from sklearn.metrics import mean_squared_error
import scipy.io  # For saving data in MATLAB-compatible format

def train_and_predict(extracted_data_folder):
    """
    Trains multi-output regression models and saves data for MATLAB visualization.

    Args:
        extracted_data_folder: Path to the folder containing the CSV files.

    Returns:
        A dictionary containing the trained models and their evaluation metrics,
        or None if an error occurs.
    """
    try:
        models = {}
        for task_num in range(1, 8):  # Iterate through tasks 1 to 7
            task_data = []
            for trial_num in range(1, 4):  # Iterate through trials 1 to 3 for training
                file_name = f"trial_{trial_num}_task_{task_num}.csv"
                file_path = os.path.join(extracted_data_folder, file_name)

                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    task_data.append(df)
                else:
                    print(f"Warning: File not found: {file_path}")
                    return None  # Stop if a file is missing

            if task_data:
                combined_data = pd.concat(task_data, ignore_index=True)

                X = combined_data.iloc[:, :8].values  # First 8 columns as input
                y = combined_data.iloc[:, -69:].values  # Last 69 columns as output

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )  # Split data

                base_regressor = LinearRegression()  # Choose your regressor
                model = MultiOutputRegressor(base_regressor)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                models[f"task_{task_num}"] = {"model": model, "mse": mse}

                print(f"Task {task_num} - MSE: {mse}")
            else:
                print(f"No data found for task {task_num}")
                return None  # Stop if no data for a task

        # Prepare data for MATLAB (including all trials for visualization)
        matlab_data = {}
        for task_num in range(1, 8):
            task_data = []
            for trial_num in range(1, 6):  # All trials (1-5) for visualization
                file_name = f"trial_{trial_num}_task_{task_num}.csv"
                file_path = os.path.join(extracted_data_folder, file_name)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    task_data.append(df)
                else:
                    print(f"Warning: File not found: {file_path}")
                    continue  # Skip if file is missing (but continue with other trials)
            
            if task_data: # Make sure there is data before concatinating.
                combined_data = pd.concat(task_data, ignore_index=True)
                X = combined_data.iloc[:, :8].values
                y_true = combined_data.iloc[:, -69:].values

                if f"task_{task_num}" in models: # If model exists, use it.
                    model = models[f"task_{task_num}"]["model"]
                    y_pred = model.predict(X)
                else:
                    print(f"Warning: No model for task {task_num}, using zeros for predictions.")
                    y_pred = np.zeros_like(y_true)  # Or some other default

                matlab_data[f"X_{task_num}"] = X
                matlab_data[f"y_true_{task_num}"] = y_true
                matlab_data[f"y_pred_{task_num}"] = y_pred
            else:
                print(f"No data for task {task_num} to save to matlab file.")

        scipy.io.savemat("model_predictions.mat", matlab_data)
        print("Data saved to model_predictions.mat")
        return models  # Return the trained models

    except Exception as e:
        print(f"An error occurred: {e}")
        return None



# Example usage:
extracted_data_folder = "extracted_data"  # Replace with your folder path
trained_models = train_and_predict(extracted_data_folder)

if trained_models:
    print("Training and data saving complete.")
    # You can now access the trained models (e.g., for making predictions on new data)
    # for task, data in trained_models.items():
    #     model = data["model"]
    #     # ... use the model ...
else:
    print("Training or data saving failed.")