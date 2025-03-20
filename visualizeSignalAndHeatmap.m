function visualizeSignalAndHeatmap(signals, time, signalLabels)
  % visualizeSignalAndHeatmap(signals, time, signalLabels)
  %
  % Visualizes original signals and their heatmap representation.
  %
  % Inputs:
  %   signals: A matrix where each row is a signal, and each column is a time sample.
  %   time: A vector representing the time axis.
  %   signalLabels: (Optional) A cell array of strings for signal labels.
  %
  % Example:
  %   time = 0:0.01:10;
  %   signals = [sin(time); cos(2*time); 0.5 * randn(size(time))];
  %   signalLabels = {'Sine', 'Cosine', 'Noise'};
  %   visualizeSignalAndHeatmap(signals, time, signalLabels);
  %
  %   %Example without labels:
  %   visualizeSignalAndHeatmap(signals, time);

  [numSignals, numSamples] = size(signals);

  if nargin < 3 || isempty(signalLabels)
    signalLabels = cell(1, numSignals);
    for i = 1:numSignals
      signalLabels{i} = ['Signal ', num2str(i)];
    end
  end

  if length(signalLabels) ~= numSignals
    error('Number of signal labels must match the number of signals.');
  end

  figure;

  % Part 1: Original Signals Plot
  subplot(2, 1, 1); % Create a subplot for the signals
  hold on;
  for i = 1:numSignals
    plot(time, signals(i, :), 'DisplayName', signalLabels{i});
  end
  xlabel('Time');
  ylabel('Signal Value');
  title('Original Signals');
  legend('show');
  grid on;
  hold off;

  % Part 2: Heatmap Representation
  subplot(2, 1, 2); % Create a subplot for the heatmap
  imagesc(time, 1:numSignals, signals); % Display the signals as a heatmap
  colormap(parula); % Choose a colormap (e.g., parula, jet)
  colorbar; % Show the colorbar
  xlabel('Time');
  ylabel('Signal Index');
  title('Heatmap of Signals');
  yticks(1:numSignals);
  yticklabels(signalLabels);
end