import os
import pandas as pd
import glob

def combine_csv_files(directory, train_trials, test_trials, exclude_tasks):
    """
    Combines CSV files for training and testing, excluding specified tasks.

    Args:
        directory: The directory containing the CSV files.
        train_trials: A list of trial numbers to include in the training set.
        test_trials: A list of trial numbers to include in the testing set.
        exclude_tasks: A list of task numbers to exclude.

    Returns:
        A tuple containing two pandas DataFrames: (train_df, test_df)
    """

    train_files = []
    test_files = []

    # Create file patterns for glob
    for trial in train_trials:
        for task in range(1, 8):  # Tasks are 1 to 7
            if task not in exclude_tasks:
                train_files.append(os.path.join(directory, f"trial_{trial}_task_{task}.csv"))
    for trial in test_trials:
        for task in range(1, 8):  # Tasks are 1 to 7
            if task not in exclude_tasks:
                test_files.append(os.path.join(directory, f"trial_{trial}_task_{task}.csv"))

    # Use glob to find matching files, handling potential file not found errors.
    train_files_found = []
    for pattern in train_files:
        found_files = glob.glob(pattern)
        if found_files:  # Check if any files were found
            train_files_found.extend(found_files)
        else:
            print(f"Warning: No files found for pattern: {pattern}")


    test_files_found = []
    for pattern in test_files:
        found_files = glob.glob(pattern)
        if found_files:
            test_files_found.extend(found_files)
        else:
            print(f"Warning: No files found for pattern: {pattern}")



    # Check if any files were found at all
    if not train_files_found and not test_files_found:
        raise FileNotFoundError("No matching CSV files found in the specified directory.")
    if not train_files_found:
      raise FileNotFoundError("No matching CSV files found for training in the specified directory")
    if not test_files_found:
      raise FileNotFoundError("No matching CSV files found for testing in the specified directory")



    # Concatenate files into DataFrames
    train_dfs = []
    for file in train_files_found:
        try:
            df = pd.read_csv(file)
            train_dfs.append(df)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            # Consider whether to continue or raise the exception depending on your needs


    test_dfs = []
    for file in test_files_found:
        try:
            df = pd.read_csv(file)
            test_dfs.append(df)
        except Exception as e:
            print(f"Error reading file {file}: {e}")



    # Create empty dataframes to handle cases where no files are found for a category
    if not train_dfs:
      train_df = pd.DataFrame()
    else:
      train_df = pd.concat(train_dfs, ignore_index=True)

    if not test_dfs:
      test_df = pd.DataFrame()
    else:
      test_df = pd.concat(test_dfs, ignore_index=True)


    return train_df, test_df


def add_trial_and_task_columns(df, filename):
    """Adds 'Trial' and 'Task' columns to a DataFrame based on the filename.
    Handles potential errors gracefully.
    """
    try:
        # Extract trial and task from filename
        parts = filename.split('_')
        trial = int(parts[1])
        task = int(parts[3].split('.')[0])  # Remove '.csv' extension

        # Add trial and task as new columns
        df['Trial'] = trial
        df['Task'] = task

    except (IndexError, ValueError) as e:
        print(f"Error extracting trial/task from filename {filename}: {e}")
        # Set default values or handle the error as appropriate for your use case
        df['Trial'] = -1  # Use -1 or None as a placeholder
        df['Task'] = -1   # Use -1 or None as a placeholder
    return df


def main():
    """Main function to combine CSV files and save the combined DataFrames."""

    directory = "./"  # Change this to your directory if needed.
    train_trials = [1, 2, 3]
    test_trials = [4, 5]
    exclude_tasks = [7]


    try:
      train_df, test_df = combine_csv_files(directory, train_trials, test_trials, exclude_tasks)

      #add trial and task number to dataframe from the name of csv file
      train_files = []
      test_files = []
      for trial in train_trials:
          for task in range(1, 8):
              if task not in exclude_tasks:
                  train_files.append(f"trial_{trial}_task_{task}.csv")
      for trial in test_trials:
          for task in range(1, 8):
              if task not in exclude_tasks:
                  test_files.append(f"trial_{trial}_task_{task}.csv")


      train_dfs_with_info = []
      for file_name in train_files:
          file_path = os.path.join(directory,file_name)
          if os.path.exists(file_path):
              df = pd.read_csv(file_path)
              df_with_info = add_trial_and_task_columns(df, file_name)
              train_dfs_with_info.append(df_with_info)

      test_dfs_with_info = []
      for file_name in test_files:
          file_path = os.path.join(directory, file_name)
          if os.path.exists(file_path):
              df = pd.read_csv(file_path)
              df_with_info = add_trial_and_task_columns(df, file_name)
              test_dfs_with_info.append(df_with_info)


      if train_dfs_with_info :
        train_df = pd.concat(train_dfs_with_info, ignore_index=True)
      else:
          train_df = pd.DataFrame()

      if test_dfs_with_info:
        test_df = pd.concat(test_dfs_with_info, ignore_index=True)
      else:
          test_df = pd.DataFrame()


      # Save the combined DataFrames to CSV files
      if not train_df.empty:
        train_df.to_csv("train_data.csv", index=False)
        print("Training data saved to train_data.csv")
      else:
          print("No training data to save.")


      if not test_df.empty:
        test_df.to_csv("test_data.csv", index=False)
        print("Testing data saved to test_data.csv")
      else:
        print("No test data to save")
    except FileNotFoundError as e:
        print(e)
    except Exception as ex:
        print("An unexpected error occurred", ex)




if __name__ == "__main__":
    main()