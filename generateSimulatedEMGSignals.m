function [emgSignals, time] = generateSimulatedEMGSignals(numSignals, duration, samplingRate)
  % generateSimulatedEMGSignals(numSignals, duration, samplingRate)
  %
  % Generates a matrix of simulated EMG signals.
  %
  % Inputs:
  %   numSignals: Number of EMG signals to generate.
  %   duration: Duration of the signals in seconds.
  %   samplingRate: Sampling rate in Hz.
  %
  % Outputs:
  %   emgSignals: A matrix where each row is an EMG signal.
  %   time: A vector representing the time axis.
  %
  % Example:
  %   [emgSignals, time] = generateSimulatedEMGSignals(3, 5, 1000);
  %   visualizeSignalAndHeatmap(emgSignals, time, {'EMG 1', 'EMG 2', 'EMG 3'});

  time = 0:1/samplingRate:duration;
  numSamples = length(time);
  emgSignals = zeros(numSignals, numSamples);

  for i = 1:numSignals
    % Generate a base signal with random bursts
    baseSignal = randn(1, numSamples);
    burstLocations = randi([1, numSamples], 1, round(0.1 * numSamples)); % Random burst locations
    burstAmplitude = 5 + 3 * randn(size(burstLocations)); % Random burst amplitudes
    for j = 1:length(burstLocations)
      burstStart = max(1, burstLocations(j) - round(0.01 * samplingRate));
      burstEnd = min(numSamples, burstLocations(j) + round(0.01 * samplingRate));
      baseSignal(burstStart:burstEnd) = baseSignal(burstStart:burstEnd) + burstAmplitude(j);
    end

    % Apply a bandpass filter to simulate EMG characteristics
    lowCutoff = 20; % Hz
    highCutoff = 450; % Hz
    [b, a] = butter(4, [lowCutoff, highCutoff] / (samplingRate / 2), 'bandpass');
    emgSignals(i, :) = filtfilt(b, a, baseSignal);

    % Add some noise
    emgSignals(i, :) = emgSignals(i, :) + 0.5 * randn(1, numSamples);
  end
end