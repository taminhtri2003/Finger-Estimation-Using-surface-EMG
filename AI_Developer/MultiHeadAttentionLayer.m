classdef MultiHeadAttentionLayer < nnet.layer.Layer
    properties
        NumHeads
        HeadSize
        QueryWeights
        KeyWeights
        ValueWeights
        OutputWeights
        DropoutRate
    end

    methods
        function layer = MultiHeadAttentionLayer(numHeads, headSize, dropoutRate)
            layer.NumHeads = numHeads;
            layer.HeadSize = headSize;
            layer.DropoutRate = dropoutRate;
        end

        function layer = initialize(layer, inputSize)
            % Xavier initialization
            layer.QueryWeights = dlarray(randn(inputSize, layer.HeadSize * layer.NumHeads) * sqrt(2 / (inputSize + layer.HeadSize * layer.NumHeads)));
            layer.KeyWeights = dlarray(randn(inputSize, layer.HeadSize * layer.NumHeads) * sqrt(2 / (inputSize + layer.HeadSize * layer.NumHeads)));
            layer.ValueWeights = dlarray(randn(inputSize, layer.HeadSize * layer.NumHeads) * sqrt(2 / (inputSize + layer.HeadSize * layer.NumHeads)));
            layer.OutputWeights = dlarray(randn(layer.HeadSize * layer.NumHeads, inputSize) * sqrt(2 / (layer.HeadSize * layer.NumHeads + inputSize)));
        end

        function Z = forward(layer, X)
            numSamples = size(X, 1);
            query = X * layer.QueryWeights;
            key = X * layer.KeyWeights;
            value = X * layer.ValueWeights;

            query = reshape(query, numSamples, layer.HeadSize, layer.NumHeads);
            key = reshape(key, numSamples, layer.HeadSize, layer.NumHeads);
            value = reshape(value, numSamples, layer.HeadSize, layer.NumHeads);

            attention = zeros(numSamples, layer.HeadSize, layer.NumHeads);
            for h = 1:layer.NumHeads
                q = query(:, :, h);
                k = key(:, :, h);
                v = value(:, :, h);

                scores = (q * k') / sqrt(layer.HeadSize);
                attention(:, :, h) = v * softmax(scores, 2);
            end

            Z = reshape(attention, numSamples, layer.HeadSize * layer.NumHeads) * layer.OutputWeights;
            Z = dropout(Z, layer.DropoutRate);
        end

        function outSize = outputSize(layer, inputSize)
            outSize = inputSize;
        end
    end
end