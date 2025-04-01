% Load the .mat file
load('s4_full.mat');

% --- Data Preparation ---

% Number of trials, tasks, muscles, and joint angles
numTrials = size(dsfilt_emg, 1);
numTasks = size(dsfilt_emg, 2);
numMuscles = size(dsfilt_emg{1,1}, 2);
numJointAngles = size(joint_angles{1,1}, 2);

% Muscle-to-Joint Angle Mapping (Crucial for Independent Outputs)
muscleJointMapping = {
    [1], [1, 2];        % APL -> Thumb 1, Thumb 2
    [2], [3, 4, 5];     % FCR -> Index 1, 2, 3
    [3,4], [6, 7, 8];   % FDS, FDP -> Middle 1, 2, 3
    [5], [9, 10, 11];   % ED -> Ring 1, 2, 3
    [6,7,8], [12, 13, 14];% EI, ECU, ECR -> Little 1, 2, 3
    };

% --- Feature Extraction (with Windowing) ---
windowSize = 200;  % Window size for feature extraction (adjust as needed)
overlap = 100;     % Overlap between windows (adjust as needed)

% Mean Absolute Value (MAV) within a window
function mav = calculateWindowedMAV(data, windowSize, overlap)
  numSamples = length(data);
  numWindows = floor((numSamples - windowSize) / overlap) + 1;
  mav = zeros(numWindows, 1);
  startIndex = 1;
  for i = 1:numWindows
    endIndex = min(startIndex + windowSize - 1, numSamples); % Ensure we don't go out of bounds
    windowData = data(startIndex:endIndex);
    mav(i) = mean(abs(windowData));
    startIndex = startIndex + overlap;
  end
end

% Root Mean Square (RMS) within a window
function rmsValue = calculateWindowedRMS(data, windowSize, overlap)
  numSamples = length(data);
  numWindows = floor((numSamples - windowSize) / overlap) + 1;
  rmsValue = zeros(numWindows, 1);
  startIndex = 1;
  for i = 1:numWindows
      endIndex = min(startIndex + windowSize - 1, numSamples);
      windowData = data(startIndex:endIndex);
      rmsValue(i) = sqrt(mean(windowData.^2));
      startIndex = startIndex + overlap;
  end
end

% --- Create and Train Independent LSTM Networks with Attention ---

nets = cell(length(muscleJointMapping), 1);
trainingResults = cell(length(muscleJointMapping), 1);

for groupIdx = 1:length(muscleJointMapping)
    muscleIndices = muscleJointMapping{groupIdx, 1};
    jointIndices = muscleJointMapping{groupIdx, 2};

    % --- Feature and Output Extraction (Windowed) ---
    groupFeatures = [];
    groupOutputs = [];

    for trial = 1:numTrials
        for task = 1:numTasks
            emgData = dsfilt_emg{trial, task};
            jointAnglesData = joint_angles{trial, task};

            % Calculate windowed features
            numWindows = 0;
            windowedFeatures = [];
            for i = 1:length(muscleIndices)
                muscle = muscleIndices(i);
                currentEMG = emgData(:, muscle);
                mav = calculateWindowedMAV(currentEMG, windowSize, overlap);
                rmsValue = calculateWindowedRMS(currentEMG, windowSize, overlap);
                % Ensure all features have the same number of windows
                if i == 1
                  numWindows = length(mav);
                  windowedFeatures = zeros(numWindows, length(muscleIndices) * 2);
                end

                % Handle potential size mismatch due to windowing.
                numMav = length(mav);
                numRms = length(rmsValue);

                minWindows = min([numWindows, numMav, numRms]);

                windowedFeatures(1:minWindows, (i-1)*2 + 1) = mav(1:minWindows);
                windowedFeatures(1:minWindows, (i-1)*2 + 2) = rmsValue(1:minWindows);

            end

            % Extract corresponding windowed outputs
              startIndex = 1;
              windowedOutputs = zeros(numWindows,length(jointIndices));
              for i = 1:numWindows
                  endIndex = min(startIndex + windowSize -1, size(jointAnglesData,1));
                  windowData = jointAnglesData(startIndex:endIndex,jointIndices);
                  %Take the mean value within the window.  Could also take the last value.
                  windowedOutputs(i,:) = mean(windowData,1);
                  startIndex = startIndex + overlap;
              end
            windowedOutputs = windowedOutputs(1:minWindows,:);

            groupFeatures = [groupFeatures; windowedFeatures];
            groupOutputs = [groupOutputs; windowedOutputs];
        end
    end

    % --- Data Splitting ---
    %  Split based on trials, *not* individual windows.
    trainFeatures = [];
    trainOutputs = [];
    testFeatures = [];
    testOutputs = [];

    startIndex = 1; % Keep track of where we are in the concatenated data
    for trial = 1:numTrials
      trialLength = 0;
      for task = 1:numTasks  % Get total samples for *this* trial
        trialLength = trialLength + floor((size(dsfilt_emg{trial, task}, 1) - windowSize) / overlap) + 1;
      end
      
      trialLength = min(trialLength,size(groupOutputs,1) - startIndex + 1); % Limit trial length if we get to the end
      endIndex = startIndex + trialLength -1;
        if trial <= 3
            trainFeatures = [trainFeatures; groupFeatures(startIndex:endIndex,:)];
            trainOutputs = [trainOutputs; groupOutputs(startIndex:endIndex,:)];
        else
            testFeatures = [testFeatures; groupFeatures(startIndex:endIndex,:)];
            testOutputs = [testOutputs; groupOutputs(startIndex:endIndex,:)];
        end
        startIndex = endIndex + 1;  % Update for next trial
        if startIndex > size(groupOutputs,1)
          break; %We have reached the end of the dataset
        end
    end

    % --- Reshape Data for LSTM ---
    % LSTM expects input data in the shape (numFeatures, numSamples, numTimeSteps)
    % We'll treat each window as a time step.
    trainFeatures = permute(trainFeatures, [2, 1]);
    trainOutputs = permute(trainOutputs, [2, 1]);
    testFeatures = permute(testFeatures, [2, 1]);
    testOutputs = permute(testOutputs, [2, 1]);

     % --- Convert to Cell Arrays (required for sequence input)---
    trainFeatures = num2cell(trainFeatures, 1);
    trainOutputs = num2cell(trainOutputs, 1);
    testFeatures = num2cell(testFeatures, 1);
    testOutputs = num2cell(testOutputs, 1);


    % --- Define LSTM Network Architecture with Attention ---
     numFeatures = size(trainFeatures{1}, 1);
     numHiddenUnits = 50; % Adjust as needed
     numResponses = size(trainOutputs{1}, 1);
     layers = [
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
        bilstmLayer(numHiddenUnits/2, 'OutputMode','sequence') % Add a bidirectional LSTM
        attentionLayer() % Add the attention layer
        fullyConnectedLayer(numHiddenUnits)  % Additional fully connected layer
        dropoutLayer(0.5) % Add dropout for regularization
        fullyConnectedLayer(numResponses)
        regressionLayer
        ];

    % --- Training Options ---
      options = trainingOptions('adam', ...  % Adam optimizer
        'MaxEpochs',50, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',0.005, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',20, ...
        'LearnRateDropFactor',0.2, ...
        'Verbose',0, ...
        'Plots','training-progress');

    % --- Train the Network ---
    [net, tr] = trainNetwork(trainFeatures, trainOutputs, layers, options);
    nets{groupIdx} = net;
    trainingResults{groupIdx} = tr;

    % --- Testing ---
    predictedOutputs = predict(net, testFeatures);
    % Convert predictions back to matrix form
    predictedOutputs = cell2mat(predictedOutputs)';

    % --- Performance Evaluation (for this group) ---
    numOutputs = size(testOutputs{1}, 1);
    rmseErrors = zeros(1, numOutputs);
    correlationCoefficients = zeros(1, numOutputs);
    testOutputsMat = cell2mat(testOutputs)'; % Convert for calculations.

    for i = 1:numOutputs
        rmseErrors(i) = sqrt(mean((testOutputsMat(:,i) - predictedOutputs(:,i)).^2));
        r = corrcoef(testOutputsMat(:,i), predictedOutputs(:,i));
        correlationCoefficients(i) = r(1,2);
        disp(['Group ', num2str(groupIdx), ', Joint Angle ', num2str(jointIndices(i)), ': RMSE = ', num2str(rmseErrors(i)), ', R = ', num2str(correlationCoefficients(i))]);
    end
    overallRMSE = mean(rmseErrors);
    overallR = mean(correlationCoefficients);

    disp(['Group ', num2str(groupIdx), ' Overall RMSE: ', num2str(overallRMSE)]);
    disp(['Group ', num2str(groupIdx), ' Overall R: ', num2str(overallR)]);

    % --- Visualizations (for this group) ---
    % 1. Actual vs. Predicted Time Series (Selected Joints)
     figure;
     numJointsToPlot = min(3, numOutputs);
     for jointIdx = 1:numJointsToPlot
        subplot(numJointsToPlot, 1, jointIdx);
        plot(testOutputsMat(:, jointIdx), 'b', 'LineWidth', 1.5);  % Actual
        hold on;
        plot(predictedOutputs(:, jointIdx), 'r--', 'LineWidth', 1.5);  % Predicted
        hold off;
        legend('Actual', 'Predicted');
        xlabel('Window Index');
        ylabel('Joint Angle');
        title(['Group ', num2str(groupIdx), ', Joint Angle ', num2str(jointIndices(jointIdx)), ' (Test)']);
        grid on;
     end
      sgtitle(['Actual vs. Predicted Time Series (Group ', num2str(groupIdx),')']);

    % 2. Scatter Plots with Regression Line
    figure;
    for jointIdx = 1:numOutputs
        subplot(ceil(numOutputs/2), 2, jointIdx);
        scatter(testOutputsMat(:, jointIdx), predictedOutputs(:, jointIdx), 'filled');
        xlabel('Actual');
        ylabel('Predicted');
        title(['Group ', num2str(groupIdx), ', Joint Angle ', num2str(jointIndices(jointIdx))]);
        grid on;
        hold on;
        p = polyfit(testOutputsMat(:, jointIdx), predictedOutputs(:, jointIdx), 1);
        x1 = linspace(min(testOutputsMat(:, jointIdx)), max(testOutputsMat(:, jointIdx)), 100);
        y1 = polyval(p, x1);
        plot(x1, y1, 'r-', 'LineWidth', 1.5);
        refline(1,0);
        hold off
    end
    sgtitle(['Scatter Plots with Regression Line (Group ', num2str(groupIdx),')']);

     % 3. Residual Plots
     figure;
     for jointIdx = 1:numOutputs
        subplot(ceil(numOutputs/2), 2, jointIdx);
        residuals = testOutputsMat(:, jointIdx) - predictedOutputs(:, jointIdx);
        plot(predictedOutputs(:, jointIdx), residuals, '.');
        xlabel('Predicted Value');
        ylabel('Residual');
        title(['Group ', num2str(groupIdx), ', Joint Angle ', num2str(jointIndices(jointIdx))]);
        grid on;
        hold on;
        yline(0, 'r--');
        hold off;
     end
     sgtitle(['Residual Plots (Group ', num2str(groupIdx),')']);
end

% --- Saving the Models ---
save('trained_emg_models_lstm_attention.mat', 'nets');
function attentionWeights = attentionLayer()
% Custom self-attention layer
  attentionWeights = layerGraph;
  attentionWeights = addLayers(attentionWeights, sequenceInputLayer(1, 'Name', 'attention_input')); % Dummy input layer
  attentionWeights = addLayers(attentionWeights,  functionLayer(@(X) sum(X, 2), 'Name', 'sum')); % Sum across sequence length
  attentionWeights = connectLayers(attentionWeights, 'attention_input', 'sum');

  attentionWeights = addLayers(attentionWeights,  functionLayer(@(X, S) X ./ S, 'Name', 'normalize')); % Normalize weights
  attentionWeights = connectLayers(attentionWeights, 'attention_input', 'normalize/in1');
  attentionWeights = connectLayers(attentionWeights, 'sum', 'normalize/in2');

  attentionWeights = addLayers(attentionWeights,  functionLayer(@(X, W) pagemtimes(X, 'transpose', W, 'none'), 'Name', 'context')); % Weighted sum
  attentionWeights = connectLayers(attentionWeights, 'attention_input', 'context/in2');
  attentionWeights = connectLayers(attentionWeights, 'normalize', 'context/in1');
%   attentionWeights = renameLayer(attentionWeights, 'context', 'attentionOutput'); %removed renameLayer

end