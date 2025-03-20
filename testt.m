% 1. Raw Signal (Example: Sine wave with noise)
time = 0:0.01:10;
rawSignal = sin(time) + 0.2 * randn(size(time));

% 2. Preprocessing (Normalization)
normalizedSignal = (rawSignal - min(rawSignal)) / (max(rawSignal) - min(rawSignal));

% 3. Mapping to RGB (Colormap)
colormap_choice = parula(256); % Choose a colormap
color_indices = round(normalizedSignal * 255) + 1; % Scale to colormap indices
rgb_matrix = ind2rgb(color_indices', colormap_choice); % Convert indices to RGB

% 4. RGB Matrix Creation (Reshape for image display)
rgb_image = reshape(rgb_matrix, [1, length(time), 3]); %Reshape to 1 row, time length columns, and 3 color channels.
rgb_image = repmat(rgb_image,[100,1,1]); %expand the image vertically.

% 5. Visualization
figure;
imshow(rgb_image);
title('Signal Visualized with Colormap');