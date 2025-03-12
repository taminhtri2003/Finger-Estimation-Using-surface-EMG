% Clear workspace and command window
clear;
clc;

%% 1. Load and Preprocess Data
% Load the .mat file (replace 'your_data_file.mat' with your actual file name)
data = load('s1_full.mat');

% Extract EMG data (5x7 cell, each cell is 4000x8)
dsfilt_emg = data.dsfilt_emg;

% Normalize EMG data
normalized_emg = cell(5, 7);
for i = 1:5
    for j = 1:7
        emg = dsfilt_emg{i, j}; % 4000x8 matrix
        normalized_emg{i, j} = (emg - mean(emg, 1)) ./ std(emg, [], 1); % Normalize per channel
    end
end

% Extract joint angles as labels (5x7 cell, each cell is 4000x14)
joint_angles = data.joint_angles;

%% 2. Define Model Architecture
% Define layers
layers = [
    % Input layer: 8 channels (muscles) x 4000 time points
    imageInputLayer([4000 8 1], 'Name', 'input', 'Normalization', 'none')
    
    % CNN layers for feature extraction
    convolution2dLayer([5 1], 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool1')
    
    convolution2dLayer([5 1], 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'pool2')
    
    % Flatten for attention mechanism
    flattenLayer('Name', 'flatten')
    
    % Attention mechanism: Reduce to 8 outputs (one per channel)
    fullyConnectedLayer(8, 'Name', 'attention_fc')
    softmaxLayer('Name', 'attention_softmax')
    
    % Reshape attention output to match raw features [1, 8, 1]
    functionLayer(@(X) reshape(X, [1, 8, 1]), 'Name', 'reshape_attention')
    
    % Concatenation layer to combine attention output and raw features
    concatenationLayer(1, 2, 'Name', 'concat')
    
    % Flatten concatenated output for fully connected layer
    flattenLayer('Name', 'flatten_concat')
    
    % Fully connected layer for regression (14 joint angles)
    fullyConnectedLayer(14, 'Name', 'fc1')
    regressionLayer('Name', 'output')
];

% Define a custom input for raw features (mean of each channel)
rawFeatureLayer = imageInputLayer([1 8 1], 'Name', 'raw_input');

% Create layer graph
lgraph = layerGraph(layers);

% Add raw feature input layer
lgraph = addLayers(lgraph, rawFeatureLayer);

% Connect raw features to concatenation layer
lgraph = connectLayers(lgraph, 'raw_input', 'concat/in2');

%% 3. Visualize Model Structure
% Visualize the network
figure;
plot(lgraph);
title('Model Architecture');

% Optional: Analyze network to check sizes
% analyzeNetwork(lgraph);

%% 4. Prepare Training Data
% Convert cell data to a format suitable for training
XTrain = cell(35, 1); % 5 trials x 7 tasks = 35 samples
YTrain = cell(35, 1);
rawFeatures = cell(35, 1);

k = 1;
for i = 1:5
    for j = 1:7
        % EMG data as 4000x8x1 (add channel dimension)
        XTrain{k} = reshape(normalized_emg{i, j}, [4000 8 1]);
        % Raw features (mean of each channel)
        rawFeatures{k} = reshape(mean(normalized_emg{i, j}, 1), [1 8 1]);
        % Joint angles as labels
        YTrain{k} = joint_angles{i, j};
        k = k + 1;
    end
end

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 4, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% 5. Train the Model
% Train the network with two inputs: EMG data and raw features
net = trainNetwork({XTrain, rawFeatures}, YTrain, lgraph, options);

%% 6. Explainability: Visualize Attention Weights
% Extract attention weights from a sample
sampleIdx = 1;
sampleEMG = XTrain{sampleIdx};
sampleRaw = rawFeatures{sampleIdx};
predictions = predict(net, {sampleEMG, sampleRaw});

% Get attention layer output
attentionWeights = activations(net, sampleEMG, 'attention_softmax');

% Plot attention weights for 8 muscles
figure;
bar(attentionWeights);
xticks(1:8);
xticklabels({'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'});
xlabel('Muscles');
ylabel('Attention Weight');
title('Muscle Contribution to Prediction');

% Display predictions vs. actual for the sample
figure;
plot(predictions', 'b', 'LineWidth', 1.5);
hold on;
plot(YTrain{sampleIdx}', 'r--', 'LineWidth', 1);
legend('Predicted', 'Actual');
xlabel('Time');
ylabel('Joint Angles');
title('Predicted vs Actual Joint Angles');
hold off;