classdef TEST_CODE < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                 matlab.ui.Figure
        GridLayout               matlab.ui.container.GridLayout
        LeftPanel                matlab.ui.container.Panel
        LoadDataButton           matlab.ui.control.Button
        DataFileLabel            matlab.ui.control.Label
        FilePathEditFieldLabel   matlab.ui.control.Label
        FilePathEditField        matlab.ui.control.EditField
        TrialDropDownLabel       matlab.ui.control.Label
        TrialDropDown            matlab.ui.control.DropDown
        TaskDropDownLabel        matlab.ui.control.Label
        TaskDropDown             matlab.ui.control.DropDown
        JointAngleDropDownLabel  matlab.ui.control.Label
        JointAngleDropDown       matlab.ui.control.DropDown
        TrainModelButton         matlab.ui.control.Button
        StatusTextAreaLabel      matlab.ui.control.Label
        StatusTextArea           matlab.ui.control.TextArea
        ExplainPredictionButton  matlab.ui.control.Button
        SamplePointSpinnerLabel  matlab.ui.control.Label
        SamplePointSpinner       matlab.ui.control.Spinner
        RightPanel               matlab.ui.container.Panel
        PredictionAxes           matlab.ui.control.UIAxes
        LIMEImportanceAxes       matlab.ui.control.UIAxes
    end

    % Properties that store data and model
    properties (Access = private)
        LoadedData % Structure to hold loaded .mat file data
        SelectedEMG % Currently selected EMG data matrix
        SelectedAngles % Currently selected Joint Angle data matrix
        SelectedTargetAngle % Currently selected target angle vector
        TrainedModel % Stores the trained regression model
        FeatureNames % Names of EMG channels
        JointAngleNames % Names of Joint Angles
        IsDataLoaded = false % Flag to check if data is loaded
        IsModelTrained = false % Flag to check if model is trained
    end

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            % Define feature and joint angle names based on description
            app.FeatureNames = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};
            app.JointAngleNames = { ...
                'Thumb 1 (20-17-18)', 'Thumb 2 (17-18-19)', ...
                'Index 1 (20-1-5)', 'Index 2 (1-5-6)', 'Index 3 (5-6-7)', ...
                'Middle 1 (20-2-8)', 'Middle 2 (2-8-9)', 'Middle 3 (8-9-10)', ...
                'Ring 1 (20-3-11)', 'Ring 2 (3-11-12)', 'Ring 3 (11-12-13)', ...
                'Little 1 (20-4-14)', 'Little 2 (4-14-15)', 'Little 3 (14-15-16)'};

            % Populate Joint Angle Dropdown
            app.JointAngleDropDown.Items = app.JointAngleNames;
            app.JointAngleDropDown.Value = app.JointAngleNames{1}; % Default selection

            % Disable controls until data is loaded
            app.TrialDropDown.Enable = 'off';
            app.TaskDropDown.Enable = 'off';
            app.JointAngleDropDown.Enable = 'off';
            app.TrainModelButton.Enable = 'off';
            app.ExplainPredictionButton.Enable = 'off';
            app.SamplePointSpinner.Enable = 'off';

            app.StatusTextArea.Value = {'App Started. Please load data.'};
        end

        % Button pushed function: LoadDataButton
        function LoadDataButtonPushed(app, event)
            [file, path] = uigetfile('*.mat', 'Select Data File');
            if isequal(file, 0)
                app.StatusTextArea.Value = {'Data loading cancelled.'};
                return;
            end
            filePath = fullfile(path, file);
            app.FilePathEditField.Value = filePath;
            app.StatusTextArea.Value = {'Loading data...'};
            drawnow; % Update UI

            try
                app.LoadedData = load(filePath);
                % Basic validation (check if expected variables exist)
                if ~(isfield(app.LoadedData, 'dsfilt_emg') && ...
                     isfield(app.LoadedData, 'joint_angles') && ...
                     iscell(app.LoadedData.dsfilt_emg) && ...
                     iscell(app.LoadedData.joint_angles) && ...
                     isequal(size(app.LoadedData.dsfilt_emg), [5, 7]) && ...
                     isequal(size(app.LoadedData.joint_angles), [5, 7]))
                    errordlg('Invalid .mat file structure. Required cell variables: dsfilt_emg [5x7], joint_angles [5x7]', 'Load Error');
                    app.StatusTextArea.Value = {'Error: Invalid data file structure.'};
                    app.IsDataLoaded = false;
                    return;
                end

                app.IsDataLoaded = true;
                app.StatusTextArea.Value = {'Data loaded successfully.', 'Select Trial, Task, and Joint Angle.'};

                % Enable controls
                app.TrialDropDown.Enable = 'on';
                app.TaskDropDown.Enable = 'on';
                app.JointAngleDropDown.Enable = 'on';
                app.TrainModelButton.Enable = 'on';

                % Reset model status
                app.IsModelTrained = false;
                app.TrainedModel = [];
                app.ExplainPredictionButton.Enable = 'off';
                app.SamplePointSpinner.Enable = 'off';
                cla(app.PredictionAxes); % Clear axes
                cla(app.LIMEImportanceAxes);

            catch ME
                app.IsDataLoaded = false;
                app.StatusTextArea.Value = {['Error loading data: ', ME.message]};
                errordlg(['Error loading file: ' ME.message], 'Load Error');
            end
        end

        % Value changed function: TrialDropDown, TaskDropDown, JointAngleDropDown
        function DataSelectionChanged(app, event)
            if ~app.IsDataLoaded
                return;
            end

            trialIdx = str2double(app.TrialDropDown.Value);
            taskIdx = str2double(app.TaskDropDown.Value);
            jointAngleName = app.JointAngleDropDown.Value;
            jointAngleIdx = find(strcmp(app.JointAngleNames, jointAngleName));

            if isempty(trialIdx) || isempty(taskIdx) || isempty(jointAngleIdx)
                 app.StatusTextArea.Value = {'Invalid selection.'};
                 return;
            end

            try
                % Extract the selected data slice
                app.SelectedEMG = app.LoadedData.dsfilt_emg{trialIdx, taskIdx};
                app.SelectedAngles = app.LoadedData.joint_angles{trialIdx, taskIdx};

                % --- Input Validation ---
                % Check if data is numeric and not empty
                if ~isnumeric(app.SelectedEMG) || isempty(app.SelectedEMG)
                    error('Selected EMG data is not a numeric matrix or is empty.');
                end
                 if ~isnumeric(app.SelectedAngles) || isempty(app.SelectedAngles)
                    error('Selected angle data is not a numeric matrix or is empty.');
                end

                % Check data dimensions
                if size(app.SelectedEMG, 2) ~= 8
                   error('Selected EMG data does not have 8 columns (features).');
                end
                 if size(app.SelectedAngles, 2) ~= 14
                   error('Selected joint angle data does not have 14 columns.');
                end
                if size(app.SelectedEMG, 1) ~= size(app.SelectedAngles, 1)
                    error('EMG and Angle data have different number of samples (rows).');
                end
                % --- End Input Validation ---

                app.SelectedTargetAngle = app.SelectedAngles(:, jointAngleIdx);

                % Update spinner limits based on selected data length
                numSamples = size(app.SelectedTargetAngle, 1);
                 if numSamples < 1
                    error('Selected data slice has zero samples.');
                end
                app.SamplePointSpinner.Limits = [1, numSamples];
                % Ensure default spinner value is within new limits
                app.SamplePointSpinner.Value = min(app.SamplePointSpinner.Value, numSamples);
                app.SamplePointSpinner.Value = max(1, app.SamplePointSpinner.Value); % Ensure it's at least 1


                app.StatusTextArea.Value = {sprintf('Selected Trial %d, Task %d.', trialIdx, taskIdx), ...
                                            sprintf('Target Angle: %s', jointAngleName), ...
                                            sprintf('%d samples available.', numSamples), ...
                                            'Ready to train model.'};
                app.IsModelTrained = false; % Reset model status on data change
                app.TrainedModel = [];
                app.ExplainPredictionButton.Enable = 'off';
                app.SamplePointSpinner.Enable = 'off';
                cla(app.PredictionAxes); % Clear axes
                cla(app.LIMEImportanceAxes);

            catch ME
                app.StatusTextArea.Value = {['Error selecting data: ', ME.message]};
                errordlg(['Data Selection Error: ' ME.message], 'Error');
                app.SelectedEMG = [];
                app.SelectedTargetAngle = [];
                app.IsModelTrained = false;
                app.TrainedModel = [];
                app.ExplainPredictionButton.Enable = 'off';
                app.SamplePointSpinner.Enable = 'off';
                 % Disable controls if data selection failed critically
                app.TrainModelButton.Enable = 'off';
            end
        end

        % Button pushed function: TrainModelButton
        function TrainModelButtonPushed(app, event)
            if isempty(app.SelectedEMG) || isempty(app.SelectedTargetAngle)
                app.StatusTextArea.Value = {'No data selected for training.'};
                errordlg('Please select valid Trial, Task, and Joint Angle first.', 'Training Error');
                return;
            end

            app.StatusTextArea.Value = {'Training Feedforward Neural Network...'};
            app.TrainModelButton.Enable = 'off'; % Disable button during training
            app.ExplainPredictionButton.Enable = 'off'; % Also disable explain button
            drawnow;

            try
                X = app.SelectedEMG;
                Y = app.SelectedTargetAngle;

                % --- Data Splitting (Simple Holdout) ---
                cv = cvpartition(size(X, 1), 'HoldOut', 0.2); % 80% train, 20% test
                idxTrain = training(cv);
                idxTest = test(cv);
                XTrain = X(idxTrain, :);
                YTrain = Y(idxTrain, :);
                XTest = X(idxTest, :);
                YTest = Y(idxTest, :);

                 % Check if training set is empty
                if isempty(XTrain) || isempty(YTrain)
                    error('Training data partition is empty. Not enough data points for holdout validation.');
                end
                 % Check if test set is empty
                if isempty(XTest) || isempty(YTest)
                    warning('Test data partition is empty. Evaluation metrics will not be calculated.');
                    rmse = NaN;
                    r2 = NaN;
                    YPred = []; % No predictions to plot
                else
                    % Proceed with training only if training data exists
                    % --- Model Training (Feedforward Neural Network) ---
                    layers = [ ...
                        featureInputLayer(size(XTrain, 2), 'Name', 'input', 'Normalization', 'zscore')
                        fullyConnectedLayer(25, 'Name', 'fc1')
                        reluLayer('Name', 'relu1')
                        fullyConnectedLayer(15, 'Name', 'fc2')
                        reluLayer('Name', 'relu2')
                        fullyConnectedLayer(1, 'Name', 'output')
                        regressionLayer('Name', 'regression')];

                    options = trainingOptions('adam', ...
                        'MaxEpochs', 50, ...
                        'MiniBatchSize', 64, ...
                        'InitialLearnRate', 0.01, ...
                        'Shuffle', 'every-epoch', ...
                        'Plots', 'none', ...
                        'Verbose', false);

                    net = trainNetwork(XTrain, YTrain, layers, options);
                    app.TrainedModel = net; % Store the trained network

                    % --- Evaluation ---
                    YPred = predict(app.TrainedModel, XTest);
                    rmse = sqrt(mean((YTest - YPred).^2));
                    % Calculate R-squared, handle potential division by zero if YTest is constant
                    if sum((YTest - mean(YTest)).^2) == 0
                        r2 = NaN; % R-squared is undefined if actual values are constant
                    else
                        r2 = 1 - sum((YTest - YPred).^2) / sum((YTest - mean(YTest)).^2);
                    end
                end

                app.StatusTextArea.Value = {sprintf('Model training complete.'), ...
                                            sprintf('Test Set RMSE: %.4f', rmse), ...
                                            sprintf('Test Set R-squared: %.4f', r2), ...
                                            'Ready for explanation.'};
                app.IsModelTrained = true;
                app.ExplainPredictionButton.Enable = 'on';
                app.SamplePointSpinner.Enable = 'on';

                % --- Visualization ---
                cla(app.PredictionAxes); % Clear previous plots
                if ~isempty(YPred) % Only plot if predictions were made
                    plot(app.PredictionAxes, YTest, 'b-', 'LineWidth', 1.5);
                    hold(app.PredictionAxes, 'on');
                    plot(app.PredictionAxes, YPred, 'r--', 'LineWidth', 1.5);
                    hold(app.PredictionAxes, 'off');
                    legend(app.PredictionAxes, 'Actual', 'Predicted', 'Location', 'best');
                    title(app.PredictionAxes, 'Actual vs. Predicted Joint Angle (Test Set)');
                    xlabel(app.PredictionAxes, 'Sample Index (Test Set)');
                    ylabel(app.PredictionAxes, 'Angle (degrees)');
                    grid(app.PredictionAxes, 'on');
                else
                    title(app.PredictionAxes, 'Actual vs. Predicted Angle (No Test Data)');
                    text(app.PredictionAxes, 0.5, 0.5, 'No test data to plot.', 'HorizontalAlignment', 'center');
                end


            catch ME
                app.StatusTextArea.Value = {['Error during model training: ', ME.message]};
                errordlg(['Training failed: ' ME.message], 'Training Error');
                app.IsModelTrained = false;
                app.TrainedModel = [];
                app.ExplainPredictionButton.Enable = 'off';
                app.SamplePointSpinner.Enable = 'off';
                cla(app.PredictionAxes); % Clear axes on error
                title(app.PredictionAxes, 'Actual vs. Predicted Angle (Training Failed)'); % Update title
            end
            app.TrainModelButton.Enable = 'on'; % Re-enable button
             % Re-enable explain button only if model training succeeded
            if app.IsModelTrained
                app.ExplainPredictionButton.Enable = 'on';
                app.SamplePointSpinner.Enable = 'on';
            end
        end

        % Button pushed function: ExplainPredictionButton
        function ExplainPredictionButtonPushed(app, event)
            if ~app.IsModelTrained || isempty(app.TrainedModel)
                app.StatusTextArea.Value = {'No model trained yet.'};
                errordlg('Please train a model first.', 'Explanation Error');
                return;
            end
             if isempty(app.SelectedEMG) % Add check if EMG data is available
                app.StatusTextArea.Value = {'No EMG data selected for explanation.'};
                errordlg('Please ensure data is selected before explaining.', 'Explanation Error');
                return;
            end

            sampleIdx = app.SamplePointSpinner.Value;
            % Validate sample index against the *currently selected* EMG data size
            if sampleIdx < 1 || sampleIdx > size(app.SelectedEMG, 1)
                app.StatusTextArea.Value = {sprintf('Selected sample index (%d) is out of bounds [1, %d].', sampleIdx, size(app.SelectedEMG, 1))};
                errordlg(sprintf('Sample index must be between 1 and %d.', size(app.SelectedEMG, 1)), 'Explanation Error');
                return;
            end

            queryPoint = app.SelectedEMG(sampleIdx, :);
            actualValue = app.SelectedTargetAngle(sampleIdx); % Get actual value for context

            app.StatusTextArea.Value = {sprintf('Generating LIME explanation for sample %d...', sampleIdx)};
            app.ExplainPredictionButton.Enable = 'off'; % Disable button
            app.TrainModelButton.Enable = 'off'; % Disable train button too
            drawnow;

            try
                % --- LIME Explanation ---
                % Create a LIME explainer object
                predFcn = @(X) predict(app.TrainedModel, X);

                % *** FIX: Convert EMG data to a table for fitting LIME ***
                % Ensure FeatureNames has the correct number of elements
                if numel(app.FeatureNames) ~= size(app.SelectedEMG, 2)
                     error('Number of feature names does not match number of EMG columns.');
                end
                emgTable = array2table(app.SelectedEMG, 'VariableNames', app.FeatureNames);

                % Create LIME explainer using the table's variable names implicitly
                limeExplainer = lime(predFcn);

                % Fit the LIME explainer using the table
                fit(limeExplainer, emgTable);

                % Explain the specific query point.
                % NOTE: The query point should also be a table row or structure
                % that matches the format used in fit. A single-row table is safest.
                queryTable = array2table(queryPoint, 'VariableNames', app.FeatureNames);
                [expData, ~] = explain(limeExplainer, queryTable, 'NumFeatures', 8); % Explain all 8 features

                % Extract feature importances (scores)
                % expData is now a table, access weights and indices differently
                featureScores = expData.FeatureWeights{1}; % Weights are in a cell
                featureNamesExplained = expData.Feature{1}; % Feature names are in a cell

                % --- Create ordered scores based on app.FeatureNames ---
                % Initialize scores vector
                orderedScores = zeros(1, numel(app.FeatureNames));
                % Map the explained scores back to the original order
                for i = 1:numel(featureNamesExplained)
                    % Find the index of the explained feature in the original list
                    originalIndex = find(strcmp(app.FeatureNames, featureNamesExplained{i}));
                    if ~isempty(originalIndex)
                        orderedScores(originalIndex) = featureScores(i);
                    end
                end
                % --- End score ordering ---

                % --- Visualization ---
                cla(app.LIMEImportanceAxes); % Clear previous plot
                bar(app.LIMEImportanceAxes, orderedScores);
                set(app.LIMEImportanceAxes, 'XTick', 1:numel(app.FeatureNames)); % Set tick positions
                set(app.LIMEImportanceAxes, 'XTickLabel', app.FeatureNames);
                xtickangle(app.LIMEImportanceAxes, 45); % Angle labels for readability
                ylabel(app.LIMEImportanceAxes, 'LIME Feature Importance Score');
                title(app.LIMEImportanceAxes, sprintf('LIME Explanation for Sample %d (Actual: %.2f)', sampleIdx, actualValue));
                grid(app.LIMEImportanceAxes, 'on');

                app.StatusTextArea.Value = {sprintf('LIME explanation complete for sample %d.', sampleIdx)};

            catch ME
                app.StatusTextArea.Value = {['Error during LIME explanation: ', ME.message], ME.getReport('basic')}; % Add basic report for more details
                errordlg(['LIME failed: ' ME.message], 'Explanation Error');
                cla(app.LIMEImportanceAxes); % Clear axes on error
                title(app.LIMEImportanceAxes, 'LIME Feature Importance (Error)'); % Update title
            end
            % Re-enable buttons
            app.ExplainPredictionButton.Enable = 'on';
            app.TrainModelButton.Enable = 'on';
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 950 600];
            app.UIFigure.Name = 'EMG Signal XAI Application';

            % Create GridLayout
            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {250, '1x'};
            app.GridLayout.RowHeight = {'1x'};

            % Create LeftPanel
            app.LeftPanel = uipanel(app.GridLayout);
            app.LeftPanel.Layout.Row = 1;
            app.LeftPanel.Layout.Column = 1;
            app.LeftPanel.Title = 'Controls';
            app.LeftPanel.FontWeight = 'bold';
            app.LeftPanel.Scrollable = 'on'; % Allow scrolling if content overflows

             % --- Arrange components vertically using a grid layout inside LeftPanel ---
            LeftPanelGrid = uigridlayout(app.LeftPanel);
            LeftPanelGrid.ColumnWidth = {'1x'};
             % Define row heights - adjust as needed for spacing
            LeftPanelGrid.RowHeight = {'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit', 'fit', '1x'}; % '1x' for Status Area to fill space
            LeftPanelGrid.Padding = [10 10 10 10]; % Add padding

            % Create LoadDataButton
            app.LoadDataButton = uibutton(LeftPanelGrid, 'push');
            app.LoadDataButton.ButtonPushedFcn = createCallbackFcn(app, @LoadDataButtonPushed, true);
            app.LoadDataButton.Icon = fullfile(matlabroot,'toolbox','matlab','icons','file_open.png');
            app.LoadDataButton.Text = 'Load .mat Data File';
            app.LoadDataButton.Layout.Row = 1;
            app.LoadDataButton.Layout.Column = 1;

            % Create DataFileLabel
            app.DataFileLabel = uilabel(LeftPanelGrid);
            app.DataFileLabel.HorizontalAlignment = 'right';
            app.DataFileLabel.Text = 'Data File:';
            app.DataFileLabel.Layout.Row = 2;
            app.DataFileLabel.Layout.Column = 1;
             % Adjust position slightly if needed via Padding or direct Position if not using grid strictly

            % Create FilePathEditField
            app.FilePathEditField = uieditfield(LeftPanelGrid, 'text');
            app.FilePathEditField.Editable = 'off';
            app.FilePathEditField.Layout.Row = 3;
            app.FilePathEditField.Layout.Column = 1;

            % Create TrialDropDownLabel
            app.TrialDropDownLabel = uilabel(LeftPanelGrid);
            app.TrialDropDownLabel.HorizontalAlignment = 'left'; % Align left for better look in grid
            app.TrialDropDownLabel.Text = 'Trial:';
            app.TrialDropDownLabel.Layout.Row = 4;
            app.TrialDropDownLabel.Layout.Column = 1;

            % Create TrialDropDown
            app.TrialDropDown = uidropdown(LeftPanelGrid);
            app.TrialDropDown.Items = {'1', '2', '3', '4', '5'};
            app.TrialDropDown.ValueChangedFcn = createCallbackFcn(app, @DataSelectionChanged, true);
            app.TrialDropDown.Layout.Row = 5;
            app.TrialDropDown.Layout.Column = 1;
            app.TrialDropDown.Value = '1';

            % Create TaskDropDownLabel
            app.TaskDropDownLabel = uilabel(LeftPanelGrid);
            app.TaskDropDownLabel.HorizontalAlignment = 'left';
            app.TaskDropDownLabel.Text = 'Task:';
            app.TaskDropDownLabel.Layout.Row = 6;
            app.TaskDropDownLabel.Layout.Column = 1;

            % Create TaskDropDown
            app.TaskDropDown = uidropdown(LeftPanelGrid);
            app.TaskDropDown.Items = {'1: Thumb', '2: Index', '3: Middle', '4: Ring', '5: Little', '6: All Fingers', '7: Random'};
            app.TaskDropDown.ItemsData = {'1', '2', '3', '4', '5', '6', '7'};
            app.TaskDropDown.ValueChangedFcn = createCallbackFcn(app, @DataSelectionChanged, true);
            app.TaskDropDown.Layout.Row = 7;
            app.TaskDropDown.Layout.Column = 1;
            app.TaskDropDown.Value = '1';

            % Create JointAngleDropDownLabel
            app.JointAngleDropDownLabel = uilabel(LeftPanelGrid);
            app.JointAngleDropDownLabel.HorizontalAlignment = 'left';
            app.JointAngleDropDownLabel.Text = 'Joint Angle:';
            app.JointAngleDropDownLabel.Layout.Row = 8;
            app.JointAngleDropDownLabel.Layout.Column = 1;

            % Create JointAngleDropDown
            app.JointAngleDropDown = uidropdown(LeftPanelGrid);
            app.JointAngleDropDown.Items = {'(Load Data First)'};
            app.JointAngleDropDown.ValueChangedFcn = createCallbackFcn(app, @DataSelectionChanged, true);
            app.JointAngleDropDown.Layout.Row = 9;
            app.JointAngleDropDown.Layout.Column = 1;

            % Create TrainModelButton
            app.TrainModelButton = uibutton(LeftPanelGrid, 'push');
            app.TrainModelButton.ButtonPushedFcn = createCallbackFcn(app, @TrainModelButtonPushed, true);
            app.TrainModelButton.Icon = fullfile(matlabroot,'toolbox','nnet','nnguis','private','tool_training.png');
            app.TrainModelButton.Text = 'Train NN Model';
            app.TrainModelButton.Layout.Row = 10;
            app.TrainModelButton.Layout.Column = 1;

            % Create SamplePointSpinnerLabel
            app.SamplePointSpinnerLabel = uilabel(LeftPanelGrid);
            app.SamplePointSpinnerLabel.HorizontalAlignment = 'left';
            app.SamplePointSpinnerLabel.Text = 'Explain Sample #:';
            app.SamplePointSpinnerLabel.Layout.Row = 11;
            app.SamplePointSpinnerLabel.Layout.Column = 1;

            % Create SamplePointSpinner
            app.SamplePointSpinner = uispinner(LeftPanelGrid);
            app.SamplePointSpinner.Limits = [1 4000];
            app.SamplePointSpinner.Step = 1;
            app.SamplePointSpinner.Layout.Row = 12;
            app.SamplePointSpinner.Layout.Column = 1;
            app.SamplePointSpinner.Value = 100;

            % Create ExplainPredictionButton
            app.ExplainPredictionButton = uibutton(LeftPanelGrid, 'push');
            app.ExplainPredictionButton.ButtonPushedFcn = createCallbackFcn(app, @ExplainPredictionButtonPushed, true);
            app.ExplainPredictionButton.Icon = fullfile(matlabroot,'toolbox','matlab','icons','greencircleicon.gif');
            app.ExplainPredictionButton.Text = 'Explain Prediction (LIME)';
            app.ExplainPredictionButton.Layout.Row = 13;
            app.ExplainPredictionButton.Layout.Column = 1;


            % Create StatusTextAreaLabel
            app.StatusTextAreaLabel = uilabel(LeftPanelGrid);
            app.StatusTextAreaLabel.Text = 'Status:';
            app.StatusTextAreaLabel.Layout.Row = 14; % Adjust row if needed
            app.StatusTextAreaLabel.Layout.Column = 1;
            app.StatusTextAreaLabel.FontWeight = 'bold';

            % Create StatusTextArea
            app.StatusTextArea = uitextarea(LeftPanelGrid);
            app.StatusTextArea.Editable = 'off';
            app.StatusTextArea.Layout.Row = 15; % Adjust row if needed
            app.StatusTextArea.Layout.Column = 1;
            app.StatusTextArea.WordWrap = 'on';


            % Create RightPanel
            app.RightPanel = uipanel(app.GridLayout);
            app.RightPanel.Layout.Row = 1;
            app.RightPanel.Layout.Column = 2;
            app.RightPanel.Title = 'Visualizations';
            app.RightPanel.FontWeight = 'bold';

             % --- Use a grid layout in RightPanel for plots ---
            RightPanelGrid = uigridlayout(app.RightPanel);
            RightPanelGrid.ColumnWidth = {'1x'};
            RightPanelGrid.RowHeight = {'1x', '1x'}; % Two equal rows for plots
            RightPanelGrid.Padding = [10 10 10 10];

            % Create PredictionAxes
            app.PredictionAxes = uiaxes(RightPanelGrid);
            title(app.PredictionAxes, 'Actual vs. Predicted Angle')
            xlabel(app.PredictionAxes, 'Sample Index')
            ylabel(app.PredictionAxes, 'Angle')
            app.PredictionAxes.Layout.Row = 1;
            app.PredictionAxes.Layout.Column = 1;

            % Create LIMEImportanceAxes
            app.LIMEImportanceAxes = uiaxes(RightPanelGrid);
            title(app.LIMEImportanceAxes, 'LIME Feature Importance')
            xlabel(app.LIMEImportanceAxes, 'EMG Channel')
            ylabel(app.LIMEImportanceAxes, 'Importance Score')
            app.LIMEImportanceAxes.Layout.Row = 2;
            app.LIMEImportanceAxes.Layout.Column = 1;

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = TEST_CODE

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end
