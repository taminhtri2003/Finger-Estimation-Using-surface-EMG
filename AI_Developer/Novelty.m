% MATLAB Code for sEMG-based Joint Angle Estimation
% APPROACH: Feature Extraction (Windowed RMS per Channel) + BiLSTM Model
% Aligns with user diagram and addresses sampling rate (200Hz). Includes LIME fix.

clear; clc; close all;

% --- Configuration ---
dataFilePath = 's.mat'; % <<<--- IMPORTANT: Replace path
samplingRate = 200;                  % Hz (NEW)
% --- Feature Extraction Params ---
windowDuration = 0.050; % seconds (e.g., 50ms)
stepDuration = 0.025;   % seconds (e.g., 25ms step -> 50% overlap)
windowSamples = round(windowDuration * samplingRate);
stepSamples = round(stepDuration * samplingRate);
featureSequenceLength = 10; % Number of feature windows per input sequence (TUNABLE)
% --- Data Split Params ---
trainTrialsIdx = [1, 2, 3]; testTrialsIdx = [4, 5]; validationSplitRatio = 0.2;
% --- Model Hyperparameters ---
numHiddenUnitsBiLSTM = 128; % Keep similar capacity for feature sequences
dropoutRate = 0.2;
numDenseUnits = 64; % Intermediate dense layer
l2_reg = 1e-5;      % L2 regularization factor
% --- Training Options ---
maxEpochs = 80; % May need more/fewer epochs with features
miniBatchSize = 64;
initialLearnRate = 0.001;

% --- 1. Load Raw Data ---
fprintf('Loading data from: %s\n', dataFilePath);
try data = load(dataFilePath); fprintf('Data loaded.\n'); if ~isfield(data,'dsfilt_emg')||~isfield(data,'joint_angles'); fprintf('Error: Vars missing.\n'); return; end; dsfilt_emg=data.dsfilt_emg; joint_angles=data.joint_angles; catch ME; fprintf('Error loading: %s\n%s\n', dataFilePath, ME.message); return; end

% --- 2. Process and Split Data by Trial ---
fprintf('Processing and splitting data by trial...\n');
numTrials=size(dsfilt_emg,1); numTasks=size(dsfilt_emg,2); numEMGChannels=-1; numAngles=-1; train_emg_list={}; train_angles_list={}; test_emg_list={}; test_angles_list={}; trainSegCount=0; testSegCount=0;
for i=1:numTrials; for j=1:numTasks; if isempty(dsfilt_emg{i,j})||isempty(joint_angles{i,j}); continue; end; emg_segment=dsfilt_emg{i,j}; angle_segment=joint_angles{i,j}; if numEMGChannels==-1; if ndims(emg_segment)==2&&ndims(angle_segment)==2&&size(emg_segment,1)>0&&size(angle_segment,1)>0; numEMGChannels=size(emg_segment,2); numAngles=size(angle_segment,2); fprintf('Detected %d EMG channels, %d angles.\n', numEMGChannels, numAngles); else; fprintf('Error: Seg (%d,%d) bad dims.\n',i,j); return; end; end
if size(emg_segment,2)~=numEMGChannels||size(angle_segment,2)~=numAngles; continue; end; minLength=min(size(emg_segment,1),size(angle_segment,1)); if minLength<=windowSamples; continue; end; emg_valid=emg_segment(1:minLength,:); angle_valid=angle_segment(1:minLength,:); if any(isnan(emg_valid(:)))||any(isinf(emg_valid(:)))||any(isnan(angle_valid(:)))||any(isinf(angle_valid(:))); continue; end
if ismember(i,trainTrialsIdx); trainSegCount=trainSegCount+1; train_emg_list{trainSegCount}=emg_valid; train_angles_list{trainSegCount}=angle_valid; elseif ismember(i,testTrialsIdx); testSegCount=testSegCount+1; test_emg_list{testSegCount}=emg_valid; test_angles_list{testSegCount}=angle_valid; end; end; end
if isempty(train_emg_list); fprintf('Error: No valid train segments.\n'); return; end; if isempty(test_emg_list); fprintf('Warning: No valid test segments.\n'); end

% --- 3. Feature Extraction (Windowed RMS per Channel) & Align Angles ---
fprintf('Extracting Windowed RMS features (Window: %d samples, Step: %d samples)...\n', windowSamples, stepSamples);
[train_features_all, train_angles_aligned] = extractWindowedRMS(train_emg_list, train_angles_list, windowSamples, stepSamples, numEMGChannels, numAngles);
fprintf('Total Training feature vectors: %d\n', size(train_features_all, 1));
[test_features_all, test_angles_aligned] = extractWindowedRMS(test_emg_list, test_angles_list, windowSamples, stepSamples, numEMGChannels, numAngles);
fprintf('Total Test feature vectors: %d\n', size(test_features_all, 1));

if isempty(train_features_all); fprintf('Error: Feature extraction failed for training data.\n'); return; end

clear train_emg_list train_angles_list test_emg_list test_angles_list data dsfilt_emg joint_angles; % Free memory

% --- 4. Normalize Features and Angles (Fit on Train Only) ---
fprintf('Normalizing features and angles (fit on train)...\n');
% Normalize FEATURES (RMS values)
[feature_mean, feature_std] = calculate_norm_params(train_features_all);
train_features_norm = apply_norm(train_features_all, feature_mean, feature_std);
test_features_norm = apply_norm(test_features_all, feature_mean, feature_std);
% Normalize ANGLES
[angle_mean, angle_std] = calculate_norm_params(train_angles_aligned);
train_angles_norm = apply_norm(train_angles_aligned, angle_mean, angle_std);
test_angles_norm = apply_norm(test_angles_aligned, angle_mean, angle_std);
fprintf('Normalization complete.\n');
clear train_features_all train_angles_aligned test_features_all test_angles_aligned;

% --- 5. Create FEATURE Sequences ---
fprintf('Creating sequences of FEATURES (Length: %d feature windows)...\n', featureSequenceLength);
% --- Train/Val Pool Sequences ---
numTrainFeatureVecs = size(train_features_norm, 1);
if numTrainFeatureVecs < featureSequenceLength; fprintf('Error: Train features (%d) < featSeqLen (%d).\n', numTrainFeatureVecs, featureSequenceLength); return; end
numTrainSequences = numTrainFeatureVecs - featureSequenceLength + 1;
XTrainVal_cells = cell(numTrainSequences, 1); YTrainVal_matrix_angles = zeros(numAngles, numTrainSequences);
fprintf('Creating %d train/val feature sequences...\n', numTrainSequences);
for i = 1:numTrainSequences
    % Input is sequence of RMS feature vectors (8 features per time step)
    XTrainVal_cells{i} = train_features_norm(i : i+featureSequenceLength-1, :)'; % Features x Time (8 x featSeqLen)
    % Target is angle vector at the END of the feature sequence window
    YTrainVal_matrix_angles(:, i) = train_angles_norm(i+featureSequenceLength-1, :)'; % Angles x 1
end
% --- Test Sequences ---
XTest_cells = {}; YTest_matrix_angles = [];
if ~isempty(test_features_norm)
    numTestFeatureVecs = size(test_features_norm, 1);
    if numTestFeatureVecs >= featureSequenceLength
        numTestSequences = numTestFeatureVecs - featureSequenceLength + 1;
        XTest_cells = cell(numTestSequences, 1); YTest_matrix_angles = zeros(numAngles, numTestSequences);
        fprintf('Creating %d test feature sequences...\n', numTestSequences);
        for i = 1:numTestSequences
            XTest_cells{i} = test_features_norm(i : i+featureSequenceLength-1, :)';
            YTest_matrix_angles(:, i) = test_angles_norm(i+featureSequenceLength-1, :)';
        end
    else; fprintf('Warn: Not enough test features (%d) for featSeqLen (%d).\n', numTestFeatureVecs, featureSequenceLength); end
end
clear train_features_norm train_angles_norm test_features_norm test_angles_norm;

% --- 6. Split Train/Validation Sets ---
fprintf('Splitting train/validation feature sequence sets...\n');
numTotalTrainVal = numel(XTrainVal_cells); cv=cvpartition(numTotalTrainVal,'HoldOut',validationSplitRatio); idxTrain=training(cv); idxVal=test(cv);
XTrain=XTrainVal_cells(idxTrain); YTrain_resp_x_obs=YTrainVal_matrix_angles(:,idxTrain); XValidation=XTrainVal_cells(idxVal); YValidation_resp_x_obs=YTrainVal_matrix_angles(:,idxVal);
YTrain=YTrain_resp_x_obs'; YValidation=YValidation_resp_x_obs'; % Transpose Y to Obs x Angles for trainNetwork
fprintf('Final Data split: Train (%d), Val (%d), Test (%d)\n', numel(XTrain), numel(XValidation), numel(XTest_cells));
clear XTrainVal_cells YTrainVal_matrix_angles YTrain_resp_x_obs YValidation_resp_x_obs;

% --- 7. Define BiLSTM Model for FEATURES ---
fprintf('Defining BiLSTM network for RMS features...\n');
% Input layer now expects numEMGChannels (8) features (RMS value per channel)
layers = [
    sequenceInputLayer(numEMGChannels, 'Name', 'input_rms_features', 'Normalization', 'none')

    % BiLSTM Layer to process sequence of features
    bilstmLayer(numHiddenUnitsBiLSTM, 'OutputMode', 'last', 'Name', 'bilstm')
    dropoutLayer(dropoutRate, 'Name', 'drop_lstm')

    % Dense layers for final mapping
    fullyConnectedLayer(numDenseUnits, 'Name', 'fc1', 'WeightRegularizer', 'L2', 'RegularizationFactor', l2_reg)
    layerNormalizationLayer('Name','ln_fc1') % Normalize before ReLU
    reluLayer('Name','relu_fc1')
    dropoutLayer(dropoutRate, 'Name', 'drop_fc1')

    fullyConnectedLayer(numAngles, 'Name', 'fc_output', 'WeightRegularizer', 'L2', 'RegularizationFactor', l2_reg)
    regressionLayer('Name', 'output')
];
% analyzeNetwork(...) remains commented out

% --- 8. Specify Training Options ---
fprintf('Setting training options...\n');
if isempty(XValidation)||isempty(YValidation); ValData={}; ValFreq=30; else; ValData={XValidation, YValidation}; if ~isempty(XTrain); ValFreq=floor(numel(XTrain)/miniBatchSize); if ValFreq<1; ValFreq=1; end; else; ValFreq=30; end; end
options = trainingOptions('adam','MaxEpochs',maxEpochs,'MiniBatchSize',miniBatchSize,'InitialLearnRate',initialLearnRate,'LearnRateSchedule','piecewise','LearnRateDropFactor',0.2,'LearnRateDropPeriod',floor(maxEpochs*0.6),'GradientThreshold',1,'ValidationData',ValData,'ValidationFrequency',ValFreq,'Shuffle','every-epoch','Plots','training-progress','Verbose',true,'ExecutionEnvironment','auto');

% --- 9. Train Network ---
fprintf('Checking data consistency...\n'); % Checks remain the same (Obs vs Rows of Y)
numObsTrainX=numel(XTrain); numObsTrainY=size(YTrain,1); fprintf('Check-Train Pred: %d, Resp: %d\n',numObsTrainX,numObsTrainY); if numObsTrainX==0||numObsTrainY==0; error('No train data.'); end; if numObsTrainX~=numObsTrainY; error('FATAL: Train mismatch: Pred(%d) vs Resp(%d).',numObsTrainX,numObsTrainY); end
if ~isempty(ValData); numObsValX=numel(XValidation); numObsValY=size(YValidation,1); fprintf('Check-Val Pred: %d, Resp: %d\n',numObsValX,numObsValY); if numObsValX~=numObsValY; error('FATAL: Val mismatch: Pred(%d) vs Resp(%d).',numObsValX,numObsValY); end; end; fprintf('Checks passed.\n');

fprintf('Starting network training (BiLSTM on Features)...\n');
net=[]; trainInfo=[];
try if isempty(XTrain)||isempty(YTrain); error('Cannot train: X/YTrain empty.'); end; YTrain=double(YTrain); if ~isempty(ValData); ValData{2}=double(ValData{2}); options.ValidationData=ValData; end
    [net, trainInfo] = trainNetwork(XTrain, YTrain, layers, options); fprintf('Training complete.\n');
catch ME; fprintf('\nError training: %s\nFile: %s, Line: %d\n', ME.message, ME.stack(1).file, ME.stack(1).line); disp(ME.getReport); return; end

% --- 10. Evaluate Network ---
fprintf('\n--- Evaluating Model ---\n'); results=struct(); results.y_pred=[]; results.y_actual=[];
if isempty(XTest_cells)||isempty(YTest_matrix_angles); fprintf('No test data. Skipping.\n');
else
    fprintf('Evaluating on test set...\n');
    try YPred_normalized=predict(net, XTest_cells, 'MiniBatchSize', miniBatchSize); if size(YPred_normalized,2)~=numAngles; if size(YPred_normalized,1)==numAngles; YPred_normalized=YPred_normalized'; else; error('Predict dims unexpected.'); end; end; YTest_transposed=YTest_matrix_angles';
    catch ME; fprintf('\nError predict: %s\n', ME.message); disp(ME.getReport); return; end
    YPred=(YPred_normalized.*angle_std)+angle_mean; YActual=(YTest_transposed.*angle_std)+angle_mean; results.y_pred=YPred; results.y_actual=YActual; % Use ANGLE scaler
    RMSE=sqrt(mean((YPred-YActual).^2,1)); SS_res=sum((YActual-YPred).^2,1); SS_tot=sum((YActual-mean(YActual,1)).^2,1); valid_idx=SS_tot>eps; R2=zeros(1,numAngles); R2(valid_idx)=1-(SS_res(valid_idx)./SS_tot(valid_idx)); OverallRMSE=mean(RMSE); OverallR2=mean(R2(valid_idx));
    results.rmse_per_joint=RMSE; results.r2_per_joint=R2; results.overall_rmse=OverallRMSE; results.overall_r2=OverallR2;
    fprintf('Eval Complete:\n  RMSE: %.4f\n  R2: %.4f\n', OverallRMSE, OverallR2);
end

% --- 11. Visualize Results ---
fprintf('\n--- Visualizing Results ---\n');
if ~isempty(results.y_pred) && ~isempty(results.y_actual)
    % (Visualization code remains the same)
    jointNames=cellstr("J"+(1:numAngles)); if numAngles==14; jointNames={'T1','T2','I1','I2','I3','M1','M2','M3','R1','R2','R3','L1','L2','L3'}; end; numToPlot=min(500,size(results.y_actual,1)); plotIndices=1:numToPlot; timeVector=plotIndices; jointsToPlot=[4,7,1]; jointsToPlot=jointsToPlot(jointsToPlot<=numAngles); if isempty(jointsToPlot); jointsToPlot=1:min(3,numAngles); end
    figure('Name','Test: Pred vs Actual (Feat+BiLSTM)','Pos',[100,100,800,600]); numSubplots=length(jointsToPlot); for i=1:numSubplots; jointIdx=jointsToPlot(i); subplot(numSubplots,1,i); plot(timeVector,results.y_actual(plotIndices,jointIdx),'b-'); hold on; plot(timeVector,results.y_pred(plotIndices,jointIdx),'r--'); hold off; ylabel('Angle'); title(jointNames{jointIdx}); legend('Actual','Pred','Loc','best'); grid on; if i==numSubplots; xlabel('Sample Index'); end; end; sgtitle('Test: Actual vs Predicted (Feat+BiLSTM)');
    figure('Name','Test: Metrics (Feat+BiLSTM)','Pos',[150,150,800,600]); xJoints=1:numAngles; subplot(2,1,1); bar(xJoints,results.rmse_per_joint); ylabel('RMSE'); title('Test RMSE/Joint'); xticks(xJoints); xticklabels(jointNames); xtickangle(45); grid on; xlim([0.5 numAngles+0.5]); subplot(2,1,2); bar(xJoints,results.r2_per_joint); ylabel('R^2'); title('Test R^2/Joint'); xticks(xJoints); xticklabels(jointNames); xtickangle(45); grid on; xlim([0.5 numAngles+0.5]); minR2=min(results.r2_per_joint(results.r2_per_joint>-inf)); if isempty(minR2); minR2=-0.1; end; ylim([min(-0.1, minR2-0.1) 1.1]); sgtitle('Test Evaluation Metrics'); fprintf('Visualization complete.\n');
else; fprintf('Skipping visualization.\n'); end

% --- 12. Explainable AI using LIME (Example for Feature Input) ---
fprintf('\n--- Explainable AI using LIME (Features Example) ---\n');
if ~isempty(results.y_pred) && exist('lime', 'file')
    try
        sampleIdxToExplain = 1;
        if sampleIdxToExplain > numel(XTest_cells); fprintf('Sample idx %d invalid.\n', sampleIdxToExplain);
        else
            xExplain = XTest_cells{sampleIdxToExplain}; % Feature sequence: Features(8) x Time(featSeqLen)
            fprintf('Generating LIME for test feature sequence %d...\n', sampleIdxToExplain);
            predictFcn = @(dataCell) predictWrapper(net, dataCell); % Use the same wrapper
            numImportantFeatures = 20; % Show top N feature importances (Channel*Time step)
            limeExplainer = lime(predictFcn, xExplain, 'NumFeatures', numImportantFeatures);
            figure('Name', sprintf('LIME (Features) - Sample %d', sampleIdxToExplain)); plot(limeExplainer); title(sprintf('LIME Feature Importance - Sample %d', sampleIdxToExplain));
            fprintf('LIME plot generated. Features represent RMS of Channel C at Window Step T.\n');
        end
    catch ME_lime; fprintf('\nError LIME: %s\n', ME_lime.message); disp(ME_lime.getReport); end
else; fprintf('Skipping LIME: No test data or LIME fn not found.\n'); if ~exist('lime', 'file'); fprintf('Note: LIME requires DL Tbx.\n'); end; end

fprintf('\n--- Workflow Complete ---\n');


% === Helper Functions ===

function [features_all, angles_aligned_all] = extractWindowedRMS(emg_list, angles_list, windowSamples, stepSamples, numChannels, numAngles)
    % Extracts windowed RMS features and aligns angles
    features_all_list = {};
    angles_aligned_all_list = {};
    totalFeatVecs = 0;

    numSegments = numel(emg_list);
    for k = 1:numSegments
        emg_seg = emg_list{k};
        angle_seg = angles_list{k};
        nSamples = size(emg_seg, 1);

        % Calculate number of windows for this segment
        numWindows = floor((nSamples - windowSamples) / stepSamples) + 1;
        if numWindows <= 0; continue; end

        segment_features = zeros(numWindows, numChannels);
        segment_angles_aligned = zeros(numWindows, numAngles);

        for i = 1:numWindows
            % Calculate window indices
            startIdx = (i-1) * stepSamples + 1;
            endIdx = startIdx + windowSamples - 1;
            if endIdx > nSamples; break; end % Should not happen with floor logic, but safety

            % Extract window data
            emg_window = emg_seg(startIdx:endIdx, :);

            % Calculate RMS for each channel in the window
            rms_values = sqrt(mean(emg_window.^2, 1)); % Result is 1 x numChannels
            segment_features(i, :) = rms_values;

            % Get the angle at the END of the window
            segment_angles_aligned(i, :) = angle_seg(endIdx, :);
        end
        % Store results for this segment
        if i >= numWindows % Ensure loop completed at least once
             features_all_list{end+1} = segment_features(1:i,:); % Use i in case loop broke early
             angles_aligned_all_list{end+1} = segment_angles_aligned(1:i,:);
             totalFeatVecs = totalFeatVecs + i;
        end
    end

    % Concatenate results from all segments
    if ~isempty(features_all_list)
        features_all = vertcat(features_all_list{:});
        angles_aligned_all = vertcat(angles_aligned_all_list{:});
        assert(size(features_all, 1) == totalFeatVecs, 'Feature vector count mismatch');
        assert(size(angles_aligned_all, 1) == totalFeatVecs, 'Aligned angle count mismatch');
    else
        features_all = [];
        angles_aligned_all = [];
    end
end

function [norm_params_mean, norm_params_std] = calculate_norm_params(data)
    % Calculates mean and std, handling potential all-zero columns
    if isempty(data)
        norm_params_mean = [];
        norm_params_std = [];
        return;
    end
    norm_params_mean = mean(data, 1);
    norm_params_std = std(data, 0, 1);
    norm_params_std(norm_params_std < eps) = 1; % Avoid division by zero
end

function data_norm = apply_norm(data, params_mean, params_std)
    % Applies Z-score normalization using pre-calculated parameters
    if isempty(data) || isempty(params_mean) || isempty(params_std)
        data_norm = data; % Return empty if input is empty or params missing
        return;
    end
    % Ensure consistent dimensions for broadcasting if data is single sample row
    if size(data,1) > 0 && size(params_mean, 2) == size(data, 2)
         data_norm = (data - params_mean) ./ params_std;
    else
         warning('Normalization parameters incompatible with data dimensions. Skipping normalization.');
         data_norm = data; % Return original if dimensions mismatch
    end
end


% --- Local Function: predictWrapper for LIME ---
function Y = predictWrapper(net, X)
% Wrapper for MATLAB's predict function to be used with LIME.
% Ensures input is cell array. Handles potential errors.
    if ~iscell(X); X_cell = {X}; else; X_cell = X; end
    try
        Y_pred = predict(net, X_cell); Y = Y_pred;
    catch ME_pred
        fprintf('Error inside predictWrapper: %s\n', ME_pred.message);
        try numOutputs = net.Layers(end-1).OutputSize; numObs = numel(X_cell); Y = nan(numObs, numOutputs);
        catch; rethrow(ME_pred); end
    end
end