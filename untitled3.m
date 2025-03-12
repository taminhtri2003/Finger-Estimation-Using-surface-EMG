% Load the data
load('s1.mat');

% Data dimensions
num_trials = 5;
num_tasks = 7;
num_emg_sensors = 8;
num_kinematics = 69;

% Data preparation
X_train = cell(3, num_tasks);
Y_train = cell(3, num_tasks);
X_val = cell(2, num_tasks);
Y_val = cell(2, num_tasks);

% Extract training data (trials 1-3)
for trial = 1:3
    for task = 1:num_tasks
        X_train{trial, task} = dsfilt_emg{trial, task};
        Y_train{trial, task} = finger_kinematics{trial, task};
    end
end

% Extract validation data (trials 4-5)
for trial = 4:5
    for task = 1:num_tasks
        X_val{trial-3, task} = dsfilt_emg{trial, task};
        Y_val{trial-3, task} = finger_kinematics{trial, task};
    end
end

% Preprocess data to match dimensions and normalize
X_train_processed = [];
Y_train_processed = [];
X_val_processed = [];
Y_val_processed = [];

% Process training data
for trial = 1:3
    for task = 1:num_tasks
        emg_data = X_train{trial, task};
        kinematics_data = Y_train{trial, task};
        
        % Downsample EMG to match kinematics length (assuming 10:1 ratio)
        emg_downsampled = emg_data(1:10:end, :);
        
        % Ensure lengths match
        min_length = min(size(emg_downsampled, 1), size(kinematics_data, 1));
        emg_downsampled = emg_downsampled(1:min_length, :);
        kinematics_data = kinematics_data(1:min_length, :);
        
        X_train_processed = [X_train_processed; emg_downsampled];
        Y_train_processed = [Y_train_processed; kinematics_data];
    end
end

% Process validation data
for trial = 1:2
    for task = 1:num_tasks
        emg_data = X_val{trial, task};
        kinematics_data = Y_val{trial, task};
        
        % Downsample EMG
        emg_downsampled = emg_data(1:10:end, :);
        
        % Ensure lengths match
        min_length = min(size(emg_downsampled, 1), size(kinematics_data, 1));
        emg_downsampled = emg_downsampled(1:min_length, :);
        kinematics_data = kinematics_data(1:min_length, :);
        
        X_val_processed = [X_val_processed; emg_downsampled];
        Y_val_processed = [Y_val_processed; kinematics_data];
    end
end

% Normalize data using z-score
[X_train_normalized, mu_X, sigma_X] = zscore(X_train_processed);
[Y_train_normalized, mu_Y, sigma_Y] = zscore(Y_train_processed);

% Apply same normalization to validation data
X_val_normalized = (X_val_processed - mu_X) ./ sigma_X;
Y_val_normalized = (Y_val_processed - mu_Y) ./ sigma_Y;

% Prepare sequences for sequence-to-sequence modeling
window_size = 100;  % Sequence length
stride = 25;        % Stride for overlapping windows

% Create training sequences
X_train_seq = {};
Y_train_seq = {};
idx = 1;
for i = 1:stride:(size(X_train_normalized, 1) - window_size + 1)
    X_train_seq{idx} = X_train_normalized(i:i+window_size-1, :)';  % Channels x Time
    Y_train_seq{idx} = Y_train_normalized(i:i+window_size-1, :)';  % Features x Time
    idx = idx + 1;
end

% Create validation sequences
X_val_seq = {};
Y_val_seq = {};
idx = 1;
for i = 1:stride:(size(X_val_normalized, 1) - window_size + 1)
    X_val_seq{idx} = X_val_normalized(i:i+window_size-1, :)';  % Channels x Time
    Y_val_seq{idx} = Y_val_normalized(i:i+window_size-1, :)';  % Features x Time
    idx = idx + 1;
end

% Convert to dlarray for deep learning
X_train_dlarray = dlarray(cat(3, X_train_seq{:}), 'CBT');  % Channels x Batch x Time
Y_train_dlarray = dlarray(cat(3, Y_train_seq{:}), 'CBT');  % Features x Batch x Time
X_val_dlarray = dlarray(cat(3, X_val_seq{:}), 'CBT');
Y_val_dlarray = dlarray(cat(3, Y_val_seq{:}), 'CBT');

% Define model architecture
numHeads = 4;
headSize = 16;
ffnSize = 128;
numBlocks = 3;

% Initialize model parameters
parameters = initializeMultiHeadAttentionModel(num_emg_sensors, num_kinematics, numHeads, headSize, ffnSize, numBlocks);

% Training parameters
numEpochs = 10;
miniBatchSize = 16;
initialLearnRate = 0.001;
learnRateDropPeriod = 20;
learnRateDropFactor = 0.1;

% Initialize training plot
figure;
lineLossTrain = animatedline('Color', [0.85 0.325 0.098]);
lineLossValidation = animatedline('Color', [0 0.447 0.741]);
xlabel("Iteration");
ylabel("Loss");
legend('Training', 'Validation');
grid on;

% Training loop
numObservations = size(X_train_dlarray, 2);
numIterationsPerEpoch = floor(numObservations / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

avgGradient = [];
avgGradientSq = [];
iteration = 0;
start = tic;

for epoch = 1:numEpochs
    % Shuffle data
    idx = randperm(numObservations);
    X_train_shuffled = X_train_dlarray(:,idx,:);
    Y_train_shuffled = Y_train_dlarray(:,idx,:);
    
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        
        % Extract mini-batch
        idx = (i-1)*miniBatchSize+1:min(i*miniBatchSize, numObservations);
        X_batch = X_train_shuffled(:,idx,:);
        Y_batch = Y_train_shuffled(:,idx,:);
        
        % Adjust learning rate
        learnRate = initialLearnRate * learnRateDropFactor^floor((epoch-1)/learnRateDropPeriod);
        
        % Compute gradients and loss
        [gradients, loss] = dlfeval(@modelGradients, parameters, X_batch, Y_batch);
        
        % Update parameters with ADAM
        [parameters, avgGradient, avgGradientSq] = adamupdate(parameters, gradients, ...
            avgGradient, avgGradientSq, iteration, learnRate);
        
        % Update training plot
        D = duration(0,0,toc(start),Format="hh:mm:ss");
        addpoints(lineLossTrain, iteration, double(gather(extractdata(loss))));
        title("Epoch: " + epoch + ", Elapsed: " + string(D));
        drawnow;
    end
    
    % Compute validation loss
    YPredVal = predictMultiHeadAttention(parameters, X_val_dlarray);
    valLoss = mse(YPredVal, Y_val_dlarray);
    addpoints(lineLossValidation, epoch*numIterationsPerEpoch, double(gather(extractdata(valLoss))));
    fprintf('Epoch %d: Validation Loss = %f\n', epoch, double(gather(extractdata(valLoss))));
end

% Model prediction and visualization
YPred = predictMultiHeadAttention(parameters, X_val_dlarray);
YPred_data = gather(extractdata(YPred));
Y_val_data = gather(extractdata(Y_val_dlarray));

% Reshape sigma_Y and mu_Y for broadcasting
sigma_Y = reshape(sigma_Y, [num_kinematics, 1, 1]);
mu_Y = reshape(mu_Y, [num_kinematics, 1, 1]);

% Denormalize predictions and validation data
YPred_denormalized = YPred_data .* sigma_Y + mu_Y;
Y_val_denormalized = Y_val_data .* sigma_Y + mu_Y;  % If applicable

% Plot sample predictions
figure;
sample_idx = 1;
time_steps = 1:size(YPred_data, 3);
joint_indices = [1, 2, 3];  % Example joint indices
for i = 1:length(joint_indices)
    subplot(length(joint_indices), 1, i);
    pred_values = squeeze(YPred_denormalized(joint_indices(i), sample_idx, :));
    true_values = squeeze(Y_val_denormalized(joint_indices(i), sample_idx, :));
    plot(time_steps, pred_values, 'r-', time_steps, true_values, 'b-');
    legend('Predicted', 'Ground Truth');
    title(['Joint ' num2str(joint_indices(i))]);
    grid on;
end

% Save model
save('emg_to_kinematics_model.mat', 'parameters', 'mu_X', 'sigma_X', 'mu_Y', 'sigma_Y');

% Functions

function parameters = initializeMultiHeadAttentionModel(inputSize, outputSize, numHeads, headSize, ffnSize, numBlocks)
    parameters = struct;
    
    % Input projection
    parameters.inputProjection.Weights = dlarray(initializeGlorot(inputSize, inputSize, 'xavier'));
    parameters.inputProjection.Bias = dlarray(zeros(inputSize, 1, 'single'));
    
    % Attention blocks
    for block = 1:numBlocks
        parameters.attentionBlocks(block).queryWeights = dlarray(initializeGlorot(inputSize, numHeads*headSize, 'xavier'));
        parameters.attentionBlocks(block).keyWeights = dlarray(initializeGlorot(inputSize, numHeads*headSize, 'xavier'));
        parameters.attentionBlocks(block).valueWeights = dlarray(initializeGlorot(inputSize, numHeads*headSize, 'xavier'));
        parameters.attentionBlocks(block).outputWeights = dlarray(initializeGlorot(numHeads*headSize, inputSize, 'xavier'));
        
        parameters.attentionBlocks(block).layerNorm1.Offset = dlarray(zeros(inputSize, 1, 'single'));
        parameters.attentionBlocks(block).layerNorm1.Scale = dlarray(ones(inputSize, 1, 'single'));
        
        parameters.attentionBlocks(block).ffn.fc1.Weights = dlarray(initializeGlorot(inputSize, ffnSize, 'xavier'));
        parameters.attentionBlocks(block).ffn.fc1.Bias = dlarray(zeros(ffnSize, 1, 'single'));
        parameters.attentionBlocks(block).ffn.fc2.Weights = dlarray(initializeGlorot(ffnSize, inputSize, 'xavier'));
        parameters.attentionBlocks(block).ffn.fc2.Bias = dlarray(zeros(inputSize, 1, 'single'));
        
        parameters.attentionBlocks(block).layerNorm2.Offset = dlarray(zeros(inputSize, 1, 'single'));
        parameters.attentionBlocks(block).layerNorm2.Scale = dlarray(ones(inputSize, 1, 'single'));
    end
    
    % Output projection
    parameters.outputProjection.Weights = dlarray(initializeGlorot(inputSize, outputSize, 'xavier'));
    parameters.outputProjection.Bias = dlarray(zeros(outputSize, 1, 'single'));
end

function weights = initializeGlorot(numIn, numOut, type)
    if strcmp(type, 'xavier')
        variance = 2 / (numIn + numOut);
    else
        variance = 2 / numIn;
    end
    weights = randn(numOut, numIn, 'single') * sqrt(variance);
end

function [gradients, loss] = modelGradients(parameters, X, Y)
    Y_pred = predictMultiHeadAttention(parameters, X);
    loss = mse(Y_pred, Y);
    gradients = dlgradient(loss, parameters);
end

function Y_pred = predictMultiHeadAttention(parameters, X)
    Z = fullyConnectTimeDistributed(X, parameters.inputProjection.Weights, parameters.inputProjection.Bias);
    
    for block = 1:numel(parameters.attentionBlocks)
        input = Z;
        Z_norm = layerNormalization(Z, parameters.attentionBlocks(block).layerNorm1.Offset, ...
            parameters.attentionBlocks(block).layerNorm1.Scale);
        Z_attn = multiHeadAttention(Z_norm, Z_norm, Z_norm, ...
            parameters.attentionBlocks(block).queryWeights, ...
            parameters.attentionBlocks(block).keyWeights, ...
            parameters.attentionBlocks(block).valueWeights, ...
            parameters.attentionBlocks(block).outputWeights);
        Z = Z_attn + input;
        
        input = Z;
        Z_norm = layerNormalization(Z, parameters.attentionBlocks(block).layerNorm2.Offset, ...
            parameters.attentionBlocks(block).layerNorm2.Scale);
        Z_ffn = feedForwardNetwork(Z_norm, parameters.attentionBlocks(block).ffn);
        Z = Z_ffn + input;
    end
    
    Y_pred = fullyConnectTimeDistributed(Z, parameters.outputProjection.Weights, parameters.outputProjection.Bias);
end

function output = multiHeadAttention(Q, K, V, WQ, WK, WV, WO)
    % Extract dimensions
    [featureDim, batchSize, seqLength] = size(Q);
    numHeads = 4;         % Number of attention heads
    headDim = 16;         % Dimension per head
    totalHeadDim = numHeads * headDim;  % Total dimension after projection

    % Step 1: Project Q, K, V into multi-head space
    q = fullyConnectTimeDistributed(Q, WQ);  % (totalHeadDim, batchSize, seqLength)
    k = fullyConnectTimeDistributed(K, WK);  % (totalHeadDim, batchSize, seqLength)
    v = fullyConnectTimeDistributed(V, WV);  % (totalHeadDim, batchSize, seqLength)

    % Step 2: Split into multiple heads
    % Reshape to (headDim, numHeads, batchSize, seqLength)
    q = reshape(q, headDim, numHeads, batchSize, seqLength);
    k = reshape(k, headDim, numHeads, batchSize, seqLength);
    v = reshape(v, headDim, numHeads, batchSize, seqLength);

    % Step 3: Compute attention scores
    % Transpose k to (numHeads, batchSize, seqLength, headDim)
    k_trans = permute(k, [2, 3, 4, 1]);  % (numHeads, batchSize, seqLength, headDim)

    % Initialize scores array
    scores = zeros(numHeads, batchSize, seqLength, seqLength, 'single');
    
    % Compute scores for each head and batch
    for h = 1:numHeads
        for b = 1:batchSize
            Q = squeeze(q(:, h, b, :))';  % (seqLength, headDim)
            K = squeeze(k_trans(h, b, :, :))';  % (headDim, seqLength)
            scores(h, b, :, :) = Q * K;  % (seqLength, headDim) * (headDim, seqLength) = (seqLength, seqLength)
        end
    end

    % Step 4: Scale and apply softmax
    scores = scores / sqrt(double(headDim));  % Scale by sqrt(headDim)
    attn_weights = softmax(scores, 4);        % Softmax over the last dimension (seqLength)

    % Step 5: Apply attention weights to values
    attn_output = zeros(headDim, numHeads, batchSize, seqLength, 'single');
    
    for h = 1:numHeads
        for b = 1:batchSize
            V = squeeze(v(:, h, b, :))';  % (seqLength, headDim)
            attn_weights_hb = squeeze(attn_weights(h, b, :, :));  % (seqLength, seqLength)
            attn_out = attn_weights_hb * V;  % (seqLength, seqLength) * (seqLength, headDim) = (seqLength, headDim)
            attn_output(:, h, b, :) = attn_out';  % (headDim, seqLength)
        end
    end

    % Step 6: Combine heads
    % Reshape back to (totalHeadDim, batchSize, seqLength)
    attn_output = reshape(attn_output, totalHeadDim, batchSize, seqLength);

    % Step 7: Final output projection
    output = fullyConnectTimeDistributed(attn_output, WO);
end

function output = layerNormalization(X, offset, scale)
    mean_X = mean(X, 1);
    var_X = var(X, 0, 1);
    output = scale .* ((X - mean_X) ./ sqrt(var_X + eps)) + offset;
end

function output = feedForwardNetwork(X, ffnParams)
    Z = fullyConnectTimeDistributed(X, ffnParams.fc1.Weights, ffnParams.fc1.Bias);
    Z = relu(Z);
    output = fullyConnectTimeDistributed(Z, ffnParams.fc2.Weights, ffnParams.fc2.Bias);
end

function Y = fullyConnectTimeDistributed(X, weights, bias)
    [outputDim, ~] = size(weights);
    [featureDim, batchSize, seqLength] = size(X);
    
    % Reshape X to (featureDim, batchSize * seqLength)
    X_reshaped = reshape(X, featureDim, batchSize * seqLength);
    
    % Apply weights and bias
    if nargin < 3
        Y_reshaped = weights * X_reshaped;
    else
        Y_reshaped = weights * X_reshaped + bias;
    end
    
    % Reshape back to (outputDim, batchSize, seqLength)
    Y = reshape(Y_reshaped, outputDim, batchSize, seqLength);
end

function output = softmax(X, dim)
    X_max = max(X, [], dim);
    X_exp = exp(X - X_max);
    output = X_exp ./ sum(X_exp, dim);
end

function Y = relu(X)
    Y = max(0, X);
end

function loss = mse(Y_pred, Y_true)
    loss = mean((Y_pred - Y_true).^2, 'all');
end