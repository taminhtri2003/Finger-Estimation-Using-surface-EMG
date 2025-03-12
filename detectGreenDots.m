function [centroids, numDots] = detectGreenDots(imagePath, varargin)
%DETECTGREENDOTS Detects and counts green dots in an image.
%
%   [CENTROIDS, NUMDOTS] = DETECTGREENDOTS(IMAGEPATH) detects green
%   dots in the image specified by IMAGEPATH and returns their centroids
%   and the total number of dots detected.
%
%   [CENTROIDS, NUMDOTS] = DETECTGREENDOTS(IMAGEPATH, 'PARAM1', VALUE1, 'PARAM2', VALUE2, ...)
%   allows specifying optional parameters to control the detection process:
%
%   'GreenThreshold':  A 3-element vector [R_thresh, G_thresh, B_thresh]
%                       specifying the minimum RGB values for a pixel to be
%                       considered "green".  Default: [50, 100, 50].  It's
%                       generally better to use the HSV or L*a*b* thresholds
%                       for more robust color detection, but this provides a
%                       simple RGB option.
%   'HueThreshold':    A 2-element vector [min_hue, max_hue] specifying the
%                       hue range (in degrees, 0-360) for green.  Default:
%                       [90, 150] (roughly corresponds to green).  Hue is
%                       generally more robust to lighting variations than
%                       direct RGB thresholds.
%   'SaturationThreshold': A 2-element vector [min_sat, max_sat] specifying
%                       the saturation range (0-1) for green. Default:
%                       [0.2, 1].  Lower saturation values can help exclude
%                       washed-out greens.
%   'ValueThreshold':   A 2-element vector [min_val, max_val] specifying the
%                       value (brightness) range (0-1) for green. Default:
%                       [0.2, 1].
%   'LabThreshold':   A 2x3 matrix specifying threshold range of L, a, and b.
%                       Default: [20 80; -80 -10; 10 80];   %L [minL maxL], a [mina, maxa], b[minb maxb]
%                       Lab colorspace is often superior to RGB or HSV
%   'AreaThreshold':   A 2-element vector [min_area, max_area] specifying
%                       the minimum and maximum area (in pixels) of a
%                       connected region to be considered a dot.  Default:
%                       [10, 500].  Helps filter out noise and very large
%                       green objects.
%   'CircularityThreshold': A 2-element vector [min_circ, max_circ]
%                       specifying the minimum and maximum circularity
%                       (4*pi*area/perimeter^2) of a region. A perfect
%                       circle has a circularity of 1.  Default: [0.6, 1].
%                       Helps distinguish dots from other shapes.
%   'DisplayResults':  A boolean value (true/false) indicating whether to
%                       display the original image with the detected dots
%                       marked. Default: false.
%   'ColorSpace' :    A string to select color space for thresholding.
%                       Options: 'RGB', 'HSV', 'Lab'. Default: 'HSV'
%

%   Example:
%     [centroids, numDots] = detectGreenDots('green_dots.jpg', ...
%                                           'HueThreshold', [100, 140], ...
%                                           'AreaThreshold', [20, 200], ...
%                                           'DisplayResults', true);

% --- Input Parsing ---
p = inputParser;
addRequired(p, 'imagePath', @(x) exist(x, 'file') == 2);

% Default parameter values
defaultGreenThreshold = [50, 100, 50];
defaultHueThreshold = [90, 150];
defaultSaturationThreshold = [0.2, 1];
defaultValueThreshold = [0.2, 1];
defaultLabThreshold = [20 80; -80 -10; 10 80];  % L, a, b thresholds
defaultAreaThreshold = [10, 500];
defaultCircularityThreshold = [0.6, 1];
defaultDisplayResults = false;
defaultColorSpace = 'HSV';

% Add optional parameters
addParameter(p, 'GreenThreshold', defaultGreenThreshold, @(x) isnumeric(x) && numel(x) == 3);
addParameter(p, 'HueThreshold', defaultHueThreshold, @(x) isnumeric(x) && numel(x) == 2);
addParameter(p, 'SaturationThreshold', defaultSaturationThreshold, @(x) isnumeric(x) && numel(x) == 2);
addParameter(p, 'ValueThreshold', defaultValueThreshold, @(x) isnumeric(x) && numel(x) == 2);
addParameter(p, 'LabThreshold', defaultLabThreshold, @(x) isnumeric(x) && all(size(x) == [2,3]));
addParameter(p, 'AreaThreshold', defaultAreaThreshold, @(x) isnumeric(x) && numel(x) == 2);
addParameter(p, 'CircularityThreshold', defaultCircularityThreshold, @(x) isnumeric(x) && numel(x) == 2);
addParameter(p, 'DisplayResults', defaultDisplayResults, @islogical);
addParameter(p, 'ColorSpace', defaultColorSpace, @(x) ischar(x) && ismember(x, {'RGB', 'HSV', 'Lab'}));

parse(p, imagePath, varargin{:});

% Get the parsed parameter values
greenThreshold = p.Results.GreenThreshold;
hueThreshold = p.Results.HueThreshold;
saturationThreshold = p.Results.SaturationThreshold;
valueThreshold = p.Results.ValueThreshold;
labThreshold = p.Results.LabThreshold;
areaThreshold = p.Results.AreaThreshold;
circularityThreshold = p.Results.CircularityThreshold;
displayResults = p.Results.DisplayResults;
colorSpace = p.Results.ColorSpace;

% --- Image Loading and Preprocessing ---
img = imread(imagePath);

% --- Color-Based Segmentation ---
switch colorSpace
    case 'RGB'
         % Simple RGB thresholding
        greenMask = (img(:,:,1) >= greenThreshold(1)) & ...
                    (img(:,:,2) >= greenThreshold(2)) & ...
                    (img(:,:,3) >= greenThreshold(3));
    case 'HSV'
        % Convert to HSV
        imgHSV = rgb2hsv(img);
        % Hue is in the range [0, 1] in MATLAB, so scale the input threshold
        hueThresholdRad = hueThreshold / 360;
         % Handle hue wraparound (e.g., red is both near 0 and 1)
        if hueThresholdRad(1) < hueThresholdRad(2)
            greenMask = (imgHSV(:,:,1) >= hueThresholdRad(1)) & (imgHSV(:,:,1) <= hueThresholdRad(2)) & ...
                        (imgHSV(:,:,2) >= saturationThreshold(1)) & (imgHSV(:,:,2) <= saturationThreshold(2)) & ...
                        (imgHSV(:,:,3) >= valueThreshold(1)) & (imgHSV(:,:,3) <= valueThreshold(2));

        else
            greenMask = ((imgHSV(:,:,1) >= hueThresholdRad(1)) | (imgHSV(:,:,1) <= hueThresholdRad(2))) & ...
                        (imgHSV(:,:,2) >= saturationThreshold(1)) & (imgHSV(:,:,2) <= saturationThreshold(2))& ...
                        (imgHSV(:,:,3) >= valueThreshold(1)) & (imgHSV(:,:,3) <= valueThreshold(2));
        end
    case 'Lab'
        % Convert to Lab
        imgLab = rgb2lab(img);
        greenMask = (imgLab(:,:,1) >= labThreshold(1,1)) & (imgLab(:,:,1) <= labThreshold(1,2)) & ...
                    (imgLab(:,:,2) >= labThreshold(2,1)) & (imgLab(:,:,2) <= labThreshold(2,2)) & ...
                    (imgLab(:,:,3) >= labThreshold(3,1)) & (imgLab(:,:,3) <= labThreshold(3,2));


    otherwise  % Should not reach here because of inputParser check
        error('Invalid ColorSpace specified.');
end



% --- Morphological Operations (Noise Removal) ---
% Use imopen to remove small noise and imclose to fill small holes
seOpen = strel('disk', 2); % Adjust size as needed
seClose = strel('disk', 3); % Adjust size as needed
greenMask = imopen(greenMask, seOpen);
greenMask = imclose(greenMask, seClose);

% --- Connected Component Analysis ---
CC = bwconncomp(greenMask);

% --- Feature Extraction ---
stats = regionprops(CC, 'Centroid', 'Area', 'Perimeter');

% --- Filtering Based on Area and Circularity ---
centroids = [];
numDots = 0;

for i = 1:CC.NumObjects
    area = stats(i).Area;
    perimeter = stats(i).Perimeter;
    
    % Handle potential division by zero
    if perimeter > 0
        circularity = (4 * pi * area) / (perimeter^2);
    else
        circularity = 0; % Assign a circularity of 0 if perimeter is 0
    end

    if area >= areaThreshold(1) && area <= areaThreshold(2) && ...
       circularity >= circularityThreshold(1) && circularity <= circularityThreshold(2)
        centroids = [centroids; stats(i).Centroid];
        numDots = numDots + 1;
    end
end


% --- Display Results (Optional) ---

if displayResults
    imshow(img);
    hold on;
    if ~isempty(centroids) % Check if centroids is not empty
        plot(centroids(:,1), centroids(:,2), 'r+', 'MarkerSize', 10, 'LineWidth', 2);
    end
    title(['Detected Green Dots: ', num2str(numDots)]);
    hold off;
end

end