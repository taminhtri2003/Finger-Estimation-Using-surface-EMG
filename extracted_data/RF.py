import function_process 

from function_process import load_data, preprocess_emg, extract_emg_features, process_kinematics, prepare_train_val_sets, train_evaluate_ml_models, train_evaluate_dl_models, visualize_model_comparison, analyze_feature_importance, visualize_predictions, visualize_emg_kinematics_relationship, explain_random_forest_shap

# --- Modified Main Function ---

def main():
    # Load data
    train_data, test_data, emg_cols, kin_cols = load_data('train_data.csv', 'test_data.csv')

    # Preprocess EMG data
    processed_train_emg = preprocess_emg(train_data, emg_cols, plot_example=True)
    processed_test_emg = preprocess_emg(test_data, emg_cols, plot_example=False)

    # Extract features from EMG
    train_emg_features = extract_emg_features(processed_train_emg)
    test_emg_features = extract_emg_features(processed_test_emg)

    # Process kinematics to match EMG feature windows and select first 5 tasks
    train_kin_processed = process_kinematics(train_data, kin_cols)
    test_kin_processed = process_kinematics(test_data, kin_cols)

    # Ensure we are using only the first 5 kinematics columns
    kin_cols_subset = kin_cols[:5]

    # Ensure matching number of samples
    min_samples = min(train_emg_features.shape[0], train_kin_processed.shape[0])
    train_emg_features = train_emg_features.iloc[:min_samples]
    train_kin_processed = train_kin_processed.iloc[:min_samples]

    min_samples_test = min(test_emg_features.shape[0], test_kin_processed.shape[0])
    test_emg_features = test_emg_features.iloc[:min_samples_test]
    test_kin_processed = test_kin_processed.iloc[:min_samples_test]

    # Prepare training and validation sets
    (X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled,
     X_train, X_val, y_train, y_val,
     X_scaler, y_scaler) = prepare_train_val_sets(
         train_emg_features, train_kin_processed
    )

    # Train and evaluate machine learning models
    ml_results = train_evaluate_ml_models(
        X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, y_scaler
    )

    # Train and evaluate deep learning models (placeholder - does nothing)
    dl_results = train_evaluate_dl_models(
        X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, y_scaler
    )

    # Visualize model comparison
    results_df = visualize_model_comparison(ml_results, dl_results)


    # --- XAI with SHAP (if Random Forest was trained) ---
    if 'Random Forest' in ml_results:
        print("\nApplying SHAP for Explainable AI (XAI)...")
        explain_random_forest_shap(
            ml_results['Random Forest']['model'].estimators_[0],  # Access the underlying RF regressor
            X_train_scaled,
            X_val_scaled,
            train_emg_features.columns
        )
    else:
        print("\nRandom Forest model not found. Skipping SHAP analysis.")

    # Analyze feature importance (using Random Forest model)  -- This is the *original* feature importance.
    if 'Random Forest' in ml_results:
        analyze_feature_importance(
             ml_results['Random Forest']['model'].estimators_[0], # Access the underlying RF regressor,
            train_emg_features.columns
        )

    # Visualize predictions for the best model
    best_model_name = results_df.iloc[0]['Model']  # Model with lowest RMSE
    # Use kin_cols_subset for visualization
    visualize_predictions(
        best_model_name,
        ml_results[best_model_name]['predictions'],
        ml_results[best_model_name]['true_values'],
        kin_cols_subset
    )
     # Visualize EMG-kinematics relationships, use kin_cols_subset
    visualize_emg_kinematics_relationship(
        X_train, y_train, train_emg_features.columns, kin_cols_subset
    )


    # Test on the test set using the best model
    print(f"\nEvaluating best model ({best_model_name}) on test set:")

    # Scale test data
    X_test_scaled = X_scaler.transform(test_emg_features)

    # Get the best model
    best_model = ml_results[best_model_name]['model']

    # Predict on test set.  Removed the LSTM special case.
    y_test_pred_scaled = best_model.predict(X_test_scaled)


    # Inverse transform to get original scale
    y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled)

    # Since we don't have ground truth for the test set in this example,
    # we'll just visualize the predictions.  Only plot the number of kinematics columns we have.
    plt.figure(figsize=(14, 8))
    for i in range(min(5, y_test_pred.shape[1])):  # Plot up to 5 kinematics, or fewer if less are predicted.
        plt.plot(y_test_pred[:100, i], label=f'Kinematics {i+1}')

    plt.title(f'Test Set Predictions ({best_model_name})')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("test_predictions.png")
    plt.show()

    print("\nAnalysis complete! The best model for EMG to kinematics mapping was:", best_model_name)
    print(f"With RMSE: {results_df.iloc[0]['RMSE']:.4f}, MAE: {results_df.iloc[0]['MAE']:.4f}, R²: {results_df.iloc[0]['R²']:.4f}")


if __name__ == "__main__":
    main()