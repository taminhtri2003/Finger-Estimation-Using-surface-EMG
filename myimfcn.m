function results = myimfcn(im)
%Image Processing Function to Detect Green Dots
%
% IM      - Input image (RGB).
% RESULTS - A scalar structure with the processing results, including
%           locations of green dots.
%
%--------------------------------------------------------------------------

% 1. Convert to HSV color space for easier color thresholding.
imhsv = rgb2hsv(im);

% 2. Define the range for "green" in HSV.  This is the most crucial part
%    and may require tuning based on your specific images and lighting.
%    - Hue (H):  Green typically falls between 0.25 and 0.45 (roughly).
%    - Saturation (S):  We want fairly saturated colors, so set a minimum.
%    - Value (V):  Avoid very dark or very bright regions.

hue_lower = 0.25;  % Lower bound for green hue
hue_upper = 0.45;  % Upper bound for green hue
sat_lower = 0.3;   % Minimum saturation (adjust as needed)
val_lower = 0.2;   % Minimum value (brightness)
val_upper = 0.9;    %max value to prevent from detecting very bright spots

% 3. Create a mask based on the HSV thresholds.
green_mask = (imhsv(:,:,1) >= hue_lower) & (imhsv(:,:,1) <= hue_upper) & ...
             (imhsv(:,:,2) >= sat_lower) & ...
             (imhsv(:,:,3) >= val_lower) & (imhsv(:,:,3) <= val_upper);

% 4. (Optional) Morphological operations to clean up the mask.
%    - Opening (erosion followed by dilation) removes small noise.
%    - Closing (dilation followed by erosion) fills small holes.
se = strel('disk', 3); % Create a disk-shaped structuring element (adjust size)
green_mask = imopen(green_mask, se);
green_mask = imclose(green_mask, se);


% 5. Find connected components (the green dots).
cc = bwconncomp(green_mask);

% 6. Get the centroids of the connected components.
stats = regionprops(cc, 'Centroid');

% 7. Store the results.
results.centroids = cat(1, stats.Centroid); % Concatenate centroids into an N-by-2 matrix
results.green_mask = green_mask; % Store the binary mask
results.num_dots = cc.NumObjects; %store the number of dots
end