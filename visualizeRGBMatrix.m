function visualizeRGBMatrix(rgbMatrix, titleStr)
  % visualizeRGBMatrix(rgbMatrix, titleStr)
  %
  % Visualizes an RGB matrix as an image.
  %
  % Inputs:
  %   rgbMatrix: A 3D matrix representing an RGB image (height x width x 3).
  %   titleStr: (Optional) A string for the figure title.
  %
  % Example:
  %   % Create a sample RGB matrix (red gradient)
  %   width = 256;
  %   height = 100;
  %   redChannel = repmat(linspace(0, 1, width), height, 1);
  %   greenChannel = zeros(height, width);
  %   blueChannel = zeros(height, width);
  %   rgbMatrix = cat(3, redChannel, greenChannel, blueChannel);
  %   visualizeRGBMatrix(rgbMatrix, 'Red Gradient');
  %
  %   %Example 2: display a random RGB image
  %   rgbMatrix = rand(100,100,3);
  %   visualizeRGBMatrix(rgbMatrix, 'Random RGB Image');

  if ndims(rgbMatrix) ~= 3 || size(rgbMatrix, 3) ~= 3
    error('Input matrix must be a 3D RGB matrix (height x width x 3).');
  end

  figure;
  imshow(rgbMatrix);

  if nargin > 1 && ~isempty(titleStr)
    title(titleStr);
  end
end