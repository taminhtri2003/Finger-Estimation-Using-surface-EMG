function realtime_emg_visualizer()
% REALTIME_EMG_VISUALIZER_EXTENDED_FILTERS_FFT  Visualizes EMG, with more filters and FFT.

% Load EMG data
try
    load('s1.mat', 'dsfilt_emg');
    EMGData = dsfilt_emg;
catch
    errordlg('Error loading EMG data file s1.mat. Please ensure it is in the MATLAB path.', 'File Load Error');
    return;
end

sensorNames = {'APL', 'FCR', 'FDS', 'FDP', 'ED', 'EI', 'ECU', 'ECR'};
numTasks = size(EMGData, 2);
taskStrings = cell(1, numTasks);
for i = 1:numTasks
    taskStrings{i} = sprintf('Task %d', i);
end

filterSettings = struct(...
    'Type', 'None', ...
    'BandpassLowCutoff', 20, ...
    'BandpassHighCutoff', 500, ...
    'NotchFrequency', 50, ...
    'HighpassCutoff', 10, ...
    'LowpassCutoff', 500, ...
    'WaveletDenoising', false, ...
    'NoiseReductionMA', false, ...
    'MovingAverageWindow', 5);

selectedSensors = true(1, length(sensorNames));
selectedTaskIndex = 1;

% --- Create UI Figure and Controls ---
fig = figure('Name', 'Real-time EMG Visualizer (Extended Filters + FFT)', 'Position', [100, 100, 1000, 850]); % Increased height

% Left Panel for Controls
controlPanel = uipanel('Parent', fig, 'Position', [0, 0, 0.2, 1]);
taskPanel = uipanel('Parent', controlPanel, 'Title', 'Tasks', 'Position', [0, 0.82, 1, 0.18]);
sensorPanel = uipanel('Parent', controlPanel, 'Title', 'Sensors', 'Position', [0, 0.52, 1, 0.3]);
filterPanel = uipanel('Parent', controlPanel, 'Title', 'Filtering', 'Position', [0, 0, 1, 0.52]);

% Task Selection Dropdown
taskDropdownLabel = uicontrol('Parent', taskPanel, 'Style', 'text', 'String', 'Select Task:', 'HorizontalAlignment', 'left', 'Position', [10, 50, 80, 20]);
taskDropdown = uicontrol('Parent', taskPanel, 'Style', 'popupmenu', 'String', taskStrings, 'Value', selectedTaskIndex, 'Position', [10, 20, 150, 30], ...
    'Callback', {@update_plot_callback});

% Sensor Selection Checkboxes
sensorCheckBoxes = [];
for i = 1:length(sensorNames)
    sensorCheckBoxes(i) = uicontrol('Parent', sensorPanel, 'Style', 'checkbox', 'String', sensorNames{i}, 'Value', selectedSensors(i), 'Position', [10, 180 - (i-1)*25, 150, 20], ...
        'Callback', {@update_plot_callback});
end

% Filter Type Dropdown (Extended with Butterworth and Median Filter)
filterTypeLabel = uicontrol('Parent', filterPanel, 'Style', 'text', 'String', 'Filter Type:', 'HorizontalAlignment', 'left', 'Position', [10, 420, 70, 20]);
filterTypeDropdown = uicontrol('Parent', filterPanel, 'Style', 'popupmenu', 'String', {'None', 'Bandpass', 'Notch', 'Highpass', 'Lowpass', 'Wavelet Denoising', 'Moving Avg. Noise Reduction', 'Butterworth Bandpass', 'Butterworth Highpass', 'Butterworth Lowpass', 'Median Filter'}, 'Value', 1, 'Position', [10, 390, 180, 30], ...
    'Callback', {@filter_type_callback});

% Bandpass Filter Settings Panel
bandpassPanel = uipanel('Parent', filterPanel, 'Title', 'Bandpass Settings', 'Position', [0, 270, 1, 0.3], 'Visible', 'off');
bpLowCutoffLabel = uicontrol('Parent', bandpassPanel, 'Style', 'text', 'String', 'Low Cutoff (Hz):', 'HorizontalAlignment', 'left', 'Position', [10, 60, 100, 20]);
bpLowCutoffSpinner = uicontrol('Parent', bandpassPanel, 'Style', 'slider', 'Min', 1, 'Max', 500, 'Value', filterSettings.BandpassLowCutoff, 'Position', [10, 30, 150, 30], ...
    'Callback', {@bandpass_cutoff_callback, 'low'});
bpLowCutoffEdit = uicontrol('Parent', bandpassPanel, 'Style', 'edit', 'String', num2str(filterSettings.BandpassLowCutoff), 'Position', [170, 35, 50, 25],'Callback', {@edit_spinner_callback, bpLowCutoffSpinner, 'low'});
bpHighCutoffLabel = uicontrol('Parent', bandpassPanel, 'Style', 'text', 'String', 'High Cutoff (Hz):', 'HorizontalAlignment', 'left', 'Position', [10, 0, 100, 20]);
bpHighCutoffSpinner = uicontrol('Parent', bandpassPanel, 'Style', 'slider', 'Min', 100, 'Max', 1000, 'Value', filterSettings.BandpassHighCutoff, 'Position', [10, -30, 150, 30], ...
    'Callback', {@bandpass_cutoff_callback, 'high'});
bpHighCutoffEdit = uicontrol('Parent', bandpassPanel, 'Style', 'edit', 'String', num2str(filterSettings.BandpassHighCutoff), 'Position', [170, -25, 50, 25],'Callback', {@edit_spinner_callback, bpHighCutoffSpinner, 'high'});

% Notch Filter Settings Panel
notchPanel = uipanel('Parent', filterPanel, 'Title', 'Notch Settings', 'Position', [0, 270, 1, 0.3], 'Visible', 'off');
notchFreqLabel = uicontrol('Parent', notchPanel, 'Style', 'text', 'String', 'Notch Freq (Hz):', 'HorizontalAlignment', 'left', 'Position', [10, 60, 100, 20]);
notchFreqSpinner = uicontrol('Parent', notchPanel, 'Style', 'slider', 'Min', 40, 'Max', 60, 'Value', filterSettings.NotchFrequency, 'Position', [10, 30, 150, 30], ...
    'Callback', {@notch_freq_callback});
notchFreqEdit = uicontrol('Parent', notchPanel, 'Style', 'edit', 'String', num2str(filterSettings.NotchFrequency), 'Position', [170, 35, 50, 25],'Callback', {@edit_spinner_callback, notchFreqSpinner, 'notch'});

% Highpass Filter Settings Panel
highpassPanel = uipanel('Parent', filterPanel, 'Title', 'Highpass Settings', 'Position', [0, 270, 1, 0.3], 'Visible', 'off');
hpCutoffLabel = uicontrol('Parent', highpassPanel, 'Style', 'text', 'String', 'Cutoff (Hz):', 'HorizontalAlignment', 'left', 'Position', [10, 60, 100, 20]);
hpCutoffSpinner = uicontrol('Parent', highpassPanel, 'Style', 'slider', 'Min', 1, 'Max', 500, 'Value', filterSettings.HighpassCutoff, 'Position', [10, 30, 150, 30], ...
    'Callback', {@highpass_cutoff_callback});
hpCutoffEdit = uicontrol('Parent', highpassPanel, 'Style', 'edit', 'String', num2str(filterSettings.HighpassCutoff), 'Position', [170, 35, 50, 25],'Callback', {@edit_spinner_callback, hpCutoffSpinner, 'highpass'});

% Lowpass Filter Settings Panel
lowpassPanel = uipanel('Parent', filterPanel, 'Title', 'Lowpass Settings', 'Position', [0, 270, 1, 0.3], 'Visible', 'off');
lpCutoffLabel = uicontrol('Parent', lowpassPanel, 'Style', 'text', 'String', 'Cutoff (Hz):', 'HorizontalAlignment', 'left', 'Position', [10, 60, 100, 20]);
lpCutoffSpinner = uicontrol('Parent', lowpassPanel, 'Style', 'slider', 'Min', 100, 'Max', 1000, 'Value', filterSettings.LowpassCutoff, 'Position', [10, 30, 150, 30], ...
    'Callback', {@lowpass_cutoff_callback});
lpCutoffEdit = uicontrol('Parent', lowpassPanel, 'Style', 'edit', 'String', num2str(filterSettings.LowpassCutoff), 'Position', [170, 35, 50, 25],'Callback', {@edit_spinner_callback, lpCutoffSpinner, 'lowpass'});

% Wavelet Denoising Checkbox
waveletDenoisingCheckbox = uicontrol('Parent', filterPanel, 'Style', 'checkbox', 'String', 'Wavelet Denoising', 'Value', filterSettings.WaveletDenoising, 'Position', [10, 230, 180, 20], ...
    'Callback', {@wavelet_denoising_callback});

% Moving Average Noise Reduction Checkbox
noiseReductionMACheckbox = uicontrol('Parent', filterPanel, 'Style', 'checkbox', 'String', 'Moving Avg. Noise Reduction', 'Value', filterSettings.NoiseReductionMA, 'Position', [10, 200, 180, 20], ...
    'Callback', {@noise_reduction_ma_callback});

applyFiltersButton = uicontrol('Parent', filterPanel, 'Style', 'pushbutton', 'String', 'Apply Filters', 'Position', [10, 130, 150, 30], ...
    'Callback', {@apply_filters_callback});
clearPlotButton = uicontrol('Parent', filterPanel, 'Style', 'pushbutton', 'String', 'Clear Plot', 'Position', [10, 80, 150, 30], ...
    'Callback', {@clear_plot_callback});

% EMG Display Axes
emgAxes = axes('Parent', fig, 'Position', [0.22, 0.52, 0.75, 0.45]); % Adjusted position - top axes for EMG
xlabel(emgAxes, 'Time (s)');
ylabel(emgAxes, 'EMG Amplitude (mV)');
title(emgAxes, 'EMG Signals (Time Domain)');
grid(emgAxes, 'on');
hold(emgAxes, 'on');

% FFT Display Axes (New axes below EMG plot)
fftAxes = axes('Parent', fig, 'Position', [0.22, 0.05, 0.75, 0.4]); % Position below EMG axes
xlabel(fftAxes, 'Frequency (Hz)');
ylabel(fftAxes, 'Magnitude');
title(fftAxes, 'EMG Signal Spectrum (Frequency Domain)');
grid(fftAxes, 'on');
hold(fftAxes, 'on');


% --- Update Plot Function ---
    function update_emg_plot()
        cla(emgAxes); % Clear EMG plot
        cla(fftAxes); % Clear FFT plot

        taskDataToPlot = EMGData(:, selectedTaskIndex);
        trialDataToPlot = taskDataToPlot{1};

        sensorIndicesToPlot = find(selectedSensors);
        if isempty(sensorIndicesToPlot)
            title(emgAxes, 'No Sensors Selected');
            title(fftAxes, 'No Sensors Selected'); % Clear FFT title too
            return;
        end

        emgToPlot = trialDataToPlot(:, sensorIndicesToPlot);

        % Apply Filters
        filterType = filterSettings.Type;
        filteredEMG = emgToPlot;
        fs = 1000;

        if strcmp(filterType, 'Bandpass')
            lowCutoff = filterSettings.BandpassLowCutoff;
            highCutoff = filterSettings.BandpassHighCutoff;
            filterDesign = designfilt('bandpassiir','FilterOrder',4, ...
                'CutoffFrequencies',[lowCutoff highCutoff],'SampleRate',fs);
            for sensorIdx = 1:size(emgToPlot, 2)
                filteredEMG(:, sensorIdx) = filtfilt(filterDesign, emgToPlot(:, sensorIdx));
            end
        elseif strcmp(filterType, 'Notch')
            notchFreq = filterSettings.NotchFrequency;
            filterDesign = designfilt('bandstopiir','FilterOrder',2, ...
                'CutoffFrequencies',[notchFreq-2 notchFreq+2],'SampleRate',fs);
            for sensorIdx = 1:size(emgToPlot, 2)
                filteredEMG(:, sensorIdx) = filtfilt(filterDesign, emgToPlot(:, sensorIdx));
            end
        elseif strcmp(filterType, 'Highpass')
            cutoffFreq = filterSettings.HighpassCutoff;
            filterDesign = designfilt('highpassiir','FilterOrder',4, ...
                'CutoffFrequency',cutoffFreq,'SampleRate',fs);
            for sensorIdx = 1:size(emgToPlot, 2)
                filteredEMG(:, sensorIdx) = filtfilt(filterDesign, emgToPlot(:, sensorIdx));
            end
        elseif strcmp(filterType, 'Lowpass')
            cutoffFreq = filterSettings.LowpassCutoff;
            filterDesign = designfilt('lowpassiir','FilterOrder',4, ...
                'CutoffFrequency',cutoffFreq,'SampleRate',fs);
            for sensorIdx = 1:size(emgToPlot, 2)
                filteredEMG(:, sensorIdx) = filtfilt(filterDesign, emgToPlot(:, sensorIdx));
            end
        elseif strcmp(filterType, 'Wavelet Denoising')
            for sensorIdx = 1:size(emgToPlot, 2)
                filteredEMG(:, sensorIdx) = wdenoise(emgToPlot(:, sensorIdx));
            end
        elseif strcmp(filterType, 'Moving Avg. Noise Reduction')
            windowSize = filterSettings.MovingAverageWindow;
             for sensorIdx = 1:size(emgToPlot, 2)
                filteredEMG(:, sensorIdx) = movmean(emgToPlot(:, sensorIdx), windowSize);
            end
        elseif strcmp(filterType, 'Butterworth Bandpass')
            lowCutoff = filterSettings.BandpassLowCutoff;
            highCutoff = filterSettings.BandpassHighCutoff;
            [b, a] = butter(4, [lowCutoff highCutoff]/(fs/2), 'bandpass'); % 4th order Butterworth
            for sensorIdx = 1:size(emgToPlot, 2)
                filteredEMG(:, sensorIdx) = filtfilt(b, a, emgToPlot(:, sensorIdx));
            end
        elseif strcmp(filterType, 'Butterworth Highpass')
            cutoffFreq = filterSettings.HighpassCutoff;
            [b, a] = butter(4, cutoffFreq/(fs/2), 'high');
            for sensorIdx = 1:size(emgToPlot, 2)
                filteredEMG(:, sensorIdx) = filtfilt(b, a, emgToPlot(:, sensorIdx));
            end
        elseif strcmp(filterType, 'Butterworth Lowpass')
            cutoffFreq = filterSettings.LowpassCutoff;
            [b, a] = butter(4, cutoffFreq/(fs/2), 'low');
            for sensorIdx = 1:size(emgToPlot, 2)
                filteredEMG(:, sensorIdx) = filtfilt(b, a, emgToPlot(:, sensorIdx));
            end
         elseif strcmp(filterType, 'Median Filter')
             windowSize = 5; % Fixed window size for median filter for now
             for sensorIdx = 1:size(emgToPlot, 2)
                 filteredEMG(:, sensorIdx) = medfilt1(emgToPlot(:, sensorIdx), windowSize);
             end
        end

        % Debugging output - check filter effect:
        signal_diff = filteredEMG - emgToPlot;
        max_diff = max(abs(signal_diff(:)));
        disp(['Maximum difference after filtering (', filterType, '): ', num2str(max_diff)]);


        timeVector = (0:size(filteredEMG,1)-1) / fs;
        plot(emgAxes, timeVector, filteredEMG);

        xlabel(emgAxes, 'Time (s)');
        ylabel(emgAxes, 'EMG Amplitude (mV)');
        title(emgAxes, ['EMG Signals - ', taskStrings{selectedTaskIndex}, ' (Time Domain)']);
        legend(emgAxes, sensorNames(sensorIndicesToPlot), 'Location', 'best');


        % --- FFT Calculation and Plotting ---
        if ~isempty(filteredEMG)
            sensorDataForFFT = filteredEMG(:, 1); % FFT of the first selected sensor for now
            N = length(sensorDataForFFT);
            Y = fft(sensorDataForFFT);
            P2 = abs(Y/N);
            P1 = P2(1:N/2+1);
            P1(2:end-1) = 2*P1(2:end-1);
            f = fs*(0:(N/2))/N;

            plot(fftAxes, f, P1); % Plot FFT magnitude
            xlabel(fftAxes, 'Frequency (Hz)');
            ylabel(fftAxes, 'Magnitude');
            title(fftAxes, ['EMG Signal Spectrum - ', sensorNames{sensorIndicesToPlot(1)}, ' (Frequency Domain)']); % FFT title
            xlim(fftAxes, [0 fs/2]); % Limit frequency axis to Nyquist frequency
        else
            cla(fftAxes); % Clear FFT axes if no EMG data to process
            title(fftAxes, 'No EMG Data for FFT');
        end


    end


% --- UI Control Callbacks --- (Callbacks remain mostly the same as before)
    function update_plot_callback(~, ~)
        selectedTaskIndex = get(taskDropdown, 'Value');
        for i = 1:length(sensorNames)
            selectedSensors(i) = get(sensorCheckBoxes(i), 'Value');
        end
        update_emg_plot();
    end

    function filter_type_callback(hObject, ~)
        selectedFilterType = hObject.String{hObject.Value};
        filterSettings.Type = selectedFilterType;
        set(bandpassPanel, 'Visible', strcmp(selectedFilterType, 'Bandpass') || strcmp(selectedFilterType, 'Butterworth Bandpass'));
        set(notchPanel, 'Visible', strcmp(selectedFilterType, 'Notch'));
        set(highpassPanel, 'Visible', strcmp(selectedFilterType, 'Highpass')|| strcmp(selectedFilterType, 'Butterworth Highpass'));
        set(lowpassPanel, 'Visible', strcmp(selectedFilterType, 'Lowpass')|| strcmp(selectedFilterType, 'Butterworth Lowpass'));
        set(waveletDenoisingCheckbox, 'Visible', ~strcmp(selectedFilterType, 'None'));
        set(noiseReductionMACheckbox, 'Visible', ~strcmp(selectedFilterType, 'None'));

        panelYPosition = 270;
        % Reposition and show/hide filter settings panels
        set(bandpassPanel, 'Position', [0, panelYPosition, 1, 0.3] , 'Visible', strcmp(selectedFilterType, 'Bandpass')|| strcmp(selectedFilterType, 'Butterworth Bandpass'));
        set(notchPanel, 'Position', [0, panelYPosition, 1, 0.3], 'Visible', strcmp(selectedFilterType, 'Notch'));
        set(highpassPanel, 'Position', [0, panelYPosition, 1, 0.3], 'Visible', strcmp(selectedFilterType, 'Highpass')|| strcmp(selectedFilterType, 'Butterworth Highpass'));
        set(lowpassPanel, 'Position', [0, panelYPosition, 1, 0.3], 'Visible', strcmp(selectedFilterType, 'Lowpass')|| strcmp(selectedFilterType, 'Butterworth Lowpass'));


    end

    function bandpass_cutoff_callback(hObject, ~, cutoffType)
        value = round(get(hObject, 'Value'));
        if strcmp(cutoffType, 'low')
            filterSettings.BandpassLowCutoff = value;
            set(bpLowCutoffEdit, 'String', num2str(value));
        elseif strcmp(cutoffType, 'high')
            filterSettings.BandpassHighCutoff = value;
            set(bpHighCutoffEdit, 'String', num2str(value));
        end
    end

     function notch_freq_callback(hObject, ~)
        value = round(get(hObject, 'Value'));
        filterSettings.NotchFrequency = value;
        set(notchFreqEdit, 'String', num2str(value));
    end

     function highpass_cutoff_callback(hObject, ~)
        value = round(get(hObject, 'Value'));
        filterSettings.HighpassCutoff = value;
        set(hpCutoffEdit, 'String', num2str(value));
    end

    function lowpass_cutoff_callback(hObject, ~)
        value = round(get(hObject, 'Value'));
        filterSettings.LowpassCutoff = value;
        set(lpCutoffEdit, 'String', num2str(value));
    end


    function wavelet_denoising_callback(hObject, ~)
        filterSettings.WaveletDenoising = get(hObject, 'Value');
    end

    function noise_reduction_ma_callback(hObject, ~)
        filterSettings.NoiseReductionMA = get(hObject, 'Value');
    end


    function edit_spinner_callback(hObject, ~, spinnerHandle, filterType)
         valueStr = get(hObject, 'String');
         value = str2double(valueStr);
         if ~isnan(value)
            if strcmp(filterType, 'low')
                filterSettings.BandpassLowCutoff = value;
            elseif strcmp(filterType, 'high')
                filterSettings.BandpassHighCutoff = value;
            elseif strcmp(filterType, 'notch')
                filterSettings.NotchFrequency = value;
            elseif strcmp(filterType, 'highpass')
                filterSettings.HighpassCutoff = value;
            elseif strcmp(filterType, 'lowpass')
                filterSettings.LowpassCutoff = value;
            end
            set(spinnerHandle, 'Value', value);
         else
            % Reset to spinner value if input is not valid
            if strcmp(filterType, 'low')
                set(hObject, 'String', num2str(filterSettings.BandpassLowCutoff));
            elseif strcmp(filterType, 'high')
                set(hObject, 'String', num2str(filterSettings.BandpassHighCutoff));
            elseif strcmp(filterType, 'notch')
                 set(hObject, 'String', num2str(filterSettings.NotchFrequency));
            elseif strcmp(filterType, 'highpass')
                set(hObject, 'String', num2str(filterSettings.HighpassCutoff));
            elseif strcmp(filterType, 'lowpass')
                set(hObject, 'String', num2str(filterSettings.LowpassCutoff));
            end
            errordlg('Invalid input. Please enter a number.', 'Input Error');
         end
    end


    function apply_filters_callback(~, ~)
        update_emg_plot();
    end

    function clear_plot_callback(~, ~)
        cla(emgAxes);
        cla(fftAxes); % Clear FFT axes too
    end


% --- Initial Plot ---
update_emg_plot();

end
