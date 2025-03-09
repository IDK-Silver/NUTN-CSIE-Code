%% Main Script

close all;
clearvars;
clc;

% Get the directory of the current script.
% If mfilename returns empty (e.g. when running interactively), use the current folder.
scriptPath = mfilename('fullpath');
if isempty(scriptPath)
    scriptFolder = pwd;
else
    scriptFolder = fileparts(scriptPath);
end

imageSourceFolder = [fullfile(scriptFolder, "res")];


% Define MATLAB default path as a fallback.
fallbackFolder = fullfile(userpath); % Example path in MATLAB installation

% Define image extensions.
imageExtensions = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'};

imagePaths={};

% Try to find images in the script folder first.
if isfolder(imageSourceFolder)
    imagePaths = findImages(imageSourceFolder, imageExtensions);
end

% If no images found, try the fallback MATLAB default path.
if isempty(imagePaths) && isfolder(fallbackFolder)
    imageSourceFolder=fallbackFolder;
    imagePaths = findImages(imageSourceFolder, imageExtensions);
    
    % Using Regex to avoid processing non-target files
    pattern = '^*.handwrite.*\..*$';
    isMatch = cellfun( ...
        @(x) ~isempty(regexp(x, pattern, 'once')), ...
        imagePaths, 'UniformOutput', true ...
    );
    imagePaths = imagePaths(isMatch);
end

if isempty(imagePaths)
    error('No image files found in both script folder and fallback folder.');
end

% Check that all files exist
if ~all(cellfun(@(x) isfile(x), imagePaths))
    error('One or more image files do not exist. Please check the file paths.');
end

if  imageSourceFolder==fallbackFolder
    resultFolder = fullfile(imageSourceFolder, 'result');
else
    resultFolder = fullfile(fileparts(imageSourceFolder), 'result');
end

% Create result folder if it does not exist.
if ~isfolder(resultFolder)
    mkdir(resultFolder);
end
fprintf("The Processed Image will save to :\n  %s\n", resultFolder)

% Process each image.
for i = 1:length(imagePaths)
    % Read the image.
    img = imread(imagePaths{i});
    

    if ndims(img) == 3 && size(img, 3) == 3
        J = stretchlim(img,[0.052 0.26]);
        img = imadjust(img, J, []); 
        img = rgb2gray(img);
    else
        img = gray2;
    end
    
    img = imadjust(img);
    img=imbinarize(img, 0.4);
    
    % Get filename without extension.
    [~, filename, ext] = fileparts(imagePaths{i});
    
    % Save the processed image in the 'result' folder with the same name.
    outputFilePath = fullfile(resultFolder, [filename, ext]); % Keeping original extension.
    imwrite(img, outputFilePath);
    
    fprintf('Processed and saved: %s\n', outputFilePath);
end

%% Local Functions

function imagePaths = findImages(folder, imageExtensions)
    % This function searches for image files in the specified folder
    % matching the provided extensions.
    imageFiles = [];
    for i = 1:length(imageExtensions)
        files = dir(fullfile(folder, imageExtensions{i}));
        imageFiles = [imageFiles; files]; %#ok<AGROW>
    end
    if isempty(imageFiles)
        imagePaths = {};
    else
        imagePaths = fullfile(folder, {imageFiles.name});
    end
end