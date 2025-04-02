function imageProcessingAppLayout
    % This function creates a GUI for restoring old photos, performing tasks
    % such as scratch removal, color balance, contrast enhancement, and color saturation.
    % The user can upload an image, apply the chosen processing steps, preview the results,
    % and save the processed image.

    % Main Figure Configuration
    figureWidth      = 950;
    figureHeight     = 700;
    backgroundColor  = [0.94 0.94 0.94];  % Default MATLAB background color

    mainFigure = figure('Name', 'Old Image Repair', ...
                        'Position', [150, 150, figureWidth, figureHeight], ...
                        'NumberTitle', 'off', ...
                        'Color', backgroundColor, ...
                        'Resize', 'on', ...
                        'CloseRequestFcn', @closeFigureCallback);


    % Top Pane config
    topPanelHeightRatio = 0.30;  % Ratio of total figure height for top panel
    topPanelHandle = uipanel('Parent', mainFigure, ...
                             'Title', '控制面板', ...
                             'FontSize', 11, ...
                             'FontWeight', 'bold', ...
                             'BackgroundColor', backgroundColor, ...
                             'Units', 'normalized', ...
                             'Position', [0.02, 1 - topPanelHeightRatio - 0.02, 0.96, topPanelHeightRatio]);

    % Axes area
    axesBottomMargin = 0.08;
    axesTopMargin    = 0.06;
    axesHeight       = 1 - topPanelHeightRatio - 0.02 - axesBottomMargin - axesTopMargin;
    axesWidthFraction= 0.4;   % Fraction of the figure width for each axes
    axesHorizontalSpacing = 0.08;  % Horizontal spacing between the two axes
    axesTotalWidth   = 2 * axesWidthFraction + axesHorizontalSpacing;
    axesLeftMargin   = (1 - axesTotalWidth) / 2;

    % Store UI handles in a struct
    guiHandles = struct();

    % Original image axes
    guiHandles.originalAxesHandle = axes('Parent', mainFigure, ...
                                         'Position', [axesLeftMargin, axesBottomMargin, axesWidthFraction, axesHeight]);
    title(guiHandles.originalAxesHandle, '原始影像');
    axis(guiHandles.originalAxesHandle, 'image', 'off');

    % Processed image axes
    guiHandles.processedAxesHandle = axes('Parent', mainFigure, ...
                                          'Position', [axesLeftMargin + axesWidthFraction + axesHorizontalSpacing, axesBottomMargin, axesWidthFraction, axesHeight]);
    title(guiHandles.processedAxesHandle, '處理後影像');
    axis(guiHandles.processedAxesHandle, 'image', 'off');

    % --- Configure controls within the Top Panel ---
    set(topPanelHandle, 'Units', 'pixels'); % Temporarily switch to pixels for precise layout
    panelPositionPixels = get(topPanelHandle, 'Position');
    panelPixelWidth  = panelPositionPixels(3);
    panelPixelHeight = panelPositionPixels(4);

    controlMargin       = 15;   % Margin inside the panel
    standardButtonWidth = 120;
    standardButtonHeight= 30;

    % Left Sub-Panel (File Operations)
    filePanelWidth = standardButtonWidth + 2 * controlMargin;
    filePanelHandle = uipanel('Parent', topPanelHandle, ...
                              'Title', '檔案操作', ...
                              'FontSize', 10, ...
                              'BackgroundColor', backgroundColor, ...
                              'Units', 'pixels', ...
                              'Position', [controlMargin, controlMargin, filePanelWidth, panelPixelHeight - 2 * controlMargin]);

    uicontrol('Parent', filePanelHandle, ...
              'Style', 'pushbutton', ...
              'String', '上傳影像', ...
              'Units', 'pixels', ...
              'Position', [controlMargin, panelPixelHeight - 2*controlMargin - standardButtonHeight - 10, standardButtonWidth, standardButtonHeight], ...
              'Callback', @uploadImageCallback);

    guiHandles.saveButtonHandle = uicontrol('Parent', filePanelHandle, ...
                                            'Style', 'pushbutton', ...
                                            'String', '儲存處理後影像', ...
                                            'Units', 'pixels', ...
                                            'Position', [controlMargin, controlMargin + 10, standardButtonWidth, standardButtonHeight], ...
                                            'Callback', @saveProcessedImageCallback, ...
                                            'Enable', 'off');

    % Right Sub-Panel (Processing Options)
    optionsPanelLeftEdge  = controlMargin + filePanelWidth + controlMargin;
    optionsPanelWidth = panelPixelWidth - optionsPanelLeftEdge - controlMargin;
    optionsPanelHandle = uipanel('Parent', topPanelHandle, ...
                                 'Title', '處理選項', ...
                                 'FontSize', 10, ...
                                 'BackgroundColor', backgroundColor, ...
                                 'Units', 'pixels', ...
                                 'Position', [optionsPanelLeftEdge, controlMargin, ...
                                              optionsPanelWidth, panelPixelHeight - 2 * controlMargin]);

    checkboxWidth = 200;
    checkboxHeight = 20;

    % Positions for checkbox grid (2 x 2)
    gridColumn1Left   = controlMargin;
    gridColumn2Left   = gridColumn1Left + checkboxWidth + controlMargin;
    gridRow1Bottom = panelPixelHeight - 2*controlMargin - checkboxHeight - 10;
    gridRow2Bottom = gridRow1Bottom - checkboxHeight - 5;

    guiHandles.checkboxScratchHandle = uicontrol('Parent', optionsPanelHandle, ...
        'Style', 'checkbox', 'String', '1. 去刮痕/噪點 (中值濾波)', ...
        'BackgroundColor', backgroundColor, 'Units', 'pixels', ...
        'Position', [gridColumn1Left, gridRow1Bottom, checkboxWidth, checkboxHeight], ...
        'Value', 1, ...
        'TooltipString', '適用於去除椒鹽噪點與細小刮痕，可能輕微模糊影像');

    guiHandles.checkboxColorBalanceHandle = uicontrol('Parent', optionsPanelHandle, ...
        'Style', 'checkbox', 'String', '2. 色彩平衡 (自動校正)', ...
        'BackgroundColor', backgroundColor, 'Units', 'pixels', ...
        'Position', [gridColumn2Left, gridRow1Bottom, checkboxWidth, checkboxHeight], ...
        'Value', 1, ...
        'TooltipString', '透過灰度世界假設自動調整色彩平衡 (預設)');

    guiHandles.checkboxContrastHandle = uicontrol('Parent', optionsPanelHandle, ...
        'Style', 'checkbox', 'String', '3. 增強對比度 (褪色)', ...
        'BackgroundColor', backgroundColor, 'Units', 'pixels', ...
        'Position', [gridColumn1Left, gridRow2Bottom, checkboxWidth, checkboxHeight], ...
        'Value', 1, ...
        'TooltipString', '使用自適應直方圖均衡 (adapthisteq) 增強影像對比');

    guiHandles.checkboxSaturationHandle = uicontrol('Parent', optionsPanelHandle, ...
        'Style', 'checkbox', 'String', '4. 增強色彩飽和度', ...
        'BackgroundColor', backgroundColor, 'Units', 'pixels', ...
        'Position', [gridColumn2Left, gridRow2Bottom, checkboxWidth, checkboxHeight], ...
        'Value', 0, ...
        'TooltipString', '增加彩色圖像的色彩鮮豔度');

    % Process Button in the Options Panel
    processButtonWidth  = 150;
    processButtonLeftEdge = (optionsPanelWidth - processButtonWidth) / 2;
    processButtonBottomEdge = controlMargin + 10;

    uicontrol('Parent', optionsPanelHandle, ...
              'Style', 'pushbutton', ...
              'String', '處理影像', ...
              'Units', 'pixels', ...
              'FontSize', 10, ...
              'FontWeight', 'bold', ...
              'Position', [processButtonLeftEdge, processButtonBottomEdge, processButtonWidth, standardButtonHeight], ...
              'Callback', @processImageCallback);

    % Set the top panel units back to normalized after layout
    set(topPanelHandle, 'Units', 'normalized');

    % Structure for Storing Images and Handles
    applicationData = struct();
    applicationData.originalImage  = [];
    applicationData.processedImage = [];
    applicationData.imagePathname  = '';
    applicationData.imageFilename  = '';
    applicationData.imageExtension = '';
    applicationData.guiHandles     = guiHandles; % Store the handles struct

    guidata(mainFigure, applicationData); % Store data associated with the figure

    % Display a startup message
    disp('請先上傳影像。');

    % Nested Callback Functions

    function uploadImageCallback(~, ~)
        % Allows user to select and load an image from file.
        % Retrieves current data, opens file dialog, loads image, updates GUI.
        currentData = guidata(mainFigure); % Retrieve data associated with the figure

        [selectedFilename, selectedPathname] = uigetfile( ...
            {'*.jpg;*.jpeg;*.png;*.tif;*.bmp', '影像檔案'}, ...
            '選擇影像');

        if isequal(selectedFilename, 0) || isequal(selectedPathname, 0)
            disp('未選擇檔案。');
            return;
        end

        fullImagePath = fullfile(selectedPathname, selectedFilename);
        try
            loadedImage = imread(fullImagePath);
            disp(['影像維度: ' mat2str(size(loadedImage))]);
        catch readError
            errordlg(['讀取影像失敗: ' readError.message], '讀取錯誤');
            return;
        end

        % Store relevant info in applicationData
        currentData.originalImage  = loadedImage;
        currentData.imagePathname  = selectedPathname;
        currentData.imageFilename  = selectedFilename;
        [~, ~, currentData.imageExtension] = fileparts(selectedFilename);
        currentData.processedImage = []; % Clear previous processed image
        guidata(mainFigure, currentData); % Store updated data

        % Display the original image
        axes(currentData.guiHandles.originalAxesHandle);
        imshow(loadedImage);
        title(currentData.guiHandles.originalAxesHandle, ['原始影像: ' selectedFilename], 'Interpreter', 'none');
        axis(currentData.guiHandles.originalAxesHandle, 'image', 'off');

        % Clear the processed image axes
        axes(currentData.guiHandles.processedAxesHandle);
        cla; % Clear axes content
        title(currentData.guiHandles.processedAxesHandle, '處理後影像');
        axis(currentData.guiHandles.processedAxesHandle, 'image', 'off');

        % Disable save button until processing has been done
        set(currentData.guiHandles.saveButtonHandle, 'Enable', 'off');

        % Reset the checkboxes to default values
        % set(currentData.guiHandles.checkboxScratchHandle,      'Value', 0);
        % set(currentData.guiHandles.checkboxColorBalanceHandle, 'Value', 1);
        % set(currentData.guiHandles.checkboxContrastHandle,     'Value', 1);
        % set(currentData.guiHandles.checkboxSaturationHandle,   'Value', 0);

        disp(['影像已載入: ' fullImagePath]);
    end

    function processImageCallback(~, ~)
        % Applies selected processing steps to the original image.
        % Retrieves data, checks selections, applies steps, updates GUI.
        currentData = guidata(mainFigure);

        if isempty(currentData.originalImage)
            errordlg('請先上傳一張影像！', '錯誤');
            return;
        end

        % Retrieve checkbox states from the handles struct stored in applicationData
        applyScratchRemoval    = get(currentData.guiHandles.checkboxScratchHandle,      'Value');
        applyColorBalance      = get(currentData.guiHandles.checkboxColorBalanceHandle, 'Value');
        applyContrastEnhancement = get(currentData.guiHandles.checkboxContrastHandle,     'Value');
        applySaturationBoost   = get(currentData.guiHandles.checkboxSaturationHandle,   'Value');

        if ~applyScratchRemoval && ~applyColorBalance && ~applyContrastEnhancement && ~applySaturationBoost
            errordlg('請至少選擇一個處理步驟！', '錯誤');
            return;
        end

        currentImage = currentData.originalImage; % Start with the original
        appliedStepNames = {};

        disp('開始處理...');
        startTime = tic; % Start timer

        try
            % 1) Scratch/Noise Removal (Median Filter)
            if applyScratchRemoval
                currentImage = applyScratchReduction(currentImage);
                appliedStepNames{end+1} = '中值濾波';
                disp('  已應用: 中值濾波');
            end

            % 2) Color Balance (Gray World)
            if applyColorBalance && size(currentImage, 3) == 3 % Only for color images
                currentImage = applyColorBalanceGrayworld(currentImage);
                appliedStepNames{end+1} = '色彩平衡';
                disp('  已應用: 色彩平衡');
            elseif applyColorBalance
                disp('  提示: 灰度圖，跳過色彩平衡。');
            end

            % 3) Contrast Enhancement (Adaptive Histogram Equalization)
            if applyContrastEnhancement
                currentImage = applyContrastEnhancementAdapthisteq(currentImage);
                appliedStepNames{end+1} = '增強對比度';
                disp('  已應用: 增強對比度');
            end

            % 4) Saturation Enhancement
            if applySaturationBoost && size(currentImage, 3) == 3 % Only for color images
                saturationFactor = 1.4; % Define saturation factor
                currentImage = applySaturationEnhancement(currentImage, saturationFactor);
                appliedStepNames{end+1} = '增強飽和度';
                disp('  已應用: 增強飽和度');
            elseif applySaturationBoost
                disp('  提示: 灰度圖，跳過增強飽和度。');
            end

            currentData.processedImage = currentImage; % Store the final processed image
            guidata(mainFigure, currentData); % Save the updated data

            % Update the processed image axes
            axes(currentData.guiHandles.processedAxesHandle);
            imshow(currentImage);
            if isempty(appliedStepNames)
                processedTitleString = '處理後影像 (未執行任何步驟)';
            else
                processedTitleString = ['處理後: ' strjoin(appliedStepNames, ' + ')];
            end
            title(currentData.guiHandles.processedAxesHandle, processedTitleString);
            axis(currentData.guiHandles.processedAxesHandle, 'image', 'off');

            % Enable the save button
            set(currentData.guiHandles.saveButtonHandle, 'Enable', 'on');

            processingDuration = toc(startTime); % Stop timer
            disp(['處理完成。耗時: ' num2str(processingDuration, '%.2f') ' 秒。']);

        catch processingError
            errordlg(['影像處理過程中發生錯誤: ' processingError.message], '處理錯誤');
            set(currentData.guiHandles.saveButtonHandle, 'Enable', 'off'); % Disable save on error
            currentData.processedImage = []; % Clear processed image on error
            guidata(mainFigure, currentData); % Save cleared data
        end
    end

    function saveProcessedImageCallback(~, ~)
        % Saves the processed image to the same directory as the original.
        % Retrieves data, constructs output path, saves image.
        currentData = guidata(mainFigure);

        if isempty(currentData.processedImage)
            errordlg('沒有可儲存的處理後影像。', '儲存錯誤');
            return;
        end
        if isempty(currentData.imagePathname) || isempty(currentData.imageExtension)
            errordlg('無法確定原始檔案路徑或類型。', '儲存錯誤');
            return;
        end

        % Construct output filename (e.g., output.jpg) in the original directory
        outputFilename = fullfile(currentData.imagePathname, ['output' currentData.imageExtension]);
        try
            % Ensure image is in uint8 format for standard saving, if it was converted to double
            if isfloat(currentData.processedImage)
                imageToSave = im2uint8(currentData.processedImage);
                disp('儲存前將 double 影像轉換為 uint8。');
            else
                imageToSave = currentData.processedImage;
            end

            imwrite(imageToSave, outputFilename);
            msgbox(['影像已成功儲存為: ' outputFilename], '儲存成功', 'help');
            disp(['影像已儲存至: ' outputFilename]);
        catch saveError
            errordlg(['儲存影像時發生錯誤: ' saveError.message], '儲存失敗');
        end
    end

    function closeFigureCallback(figureHandle, ~)
        % Callback to handle figure closing.
        disp('關閉應用程式...');
        delete(figureHandle); % Delete the main figure window
    end

    % Image Processing Helper Functions

    function outputImage = applyScratchReduction(inputImage)
        % Apply a 3x3 median filter to remove noise or scratches.
        % Handles both grayscale and color images.
        filterDimensions = [3 3];
        if size(inputImage, 3) == 3 % Color image
            outputImage = inputImage; % Initialize output
            for channelIndex = 1:3
                outputImage(:,:,channelIndex) = medfilt2(inputImage(:,:,channelIndex), filterDimensions);
            end
        else % Grayscale image
            outputImage = medfilt2(inputImage, filterDimensions);
        end
    end

    function outputImage = applyColorBalanceGrayworld(inputImage)
        % Perform automatic color balance using the gray-world assumption.
        % Assumes inputImage is a color image (size(inputImage,3)==3).
        originalDataType = class(inputImage);
        imageDouble = im2double(inputImage); % Convert to double for calculations

        meanRed   = mean(imageDouble(:,:,1), 'all');
        meanGreen = mean(imageDouble(:,:,2), 'all');
        meanBlue  = mean(imageDouble(:,:,3), 'all');

        % Prevent division by zero or near-zero
        if meanRed <= 1e-6 || meanGreen <= 1e-6 || meanBlue <= 1e-6
            warning('Gray-world: one or more channels have zero or near-zero mean. Skipping color balance.');
            outputImage = inputImage; % Return original if means are problematic
            return;
        end

        % Calculate scaling factors relative to green channel mean
        scaleRed = meanGreen / meanRed;
        scaleBlue = meanGreen / meanBlue;

        outputImageDouble = imageDouble; % Initialize output
        outputImageDouble(:,:,1) = imageDouble(:,:,1) * scaleRed;  % Scale red channel
        outputImageDouble(:,:,3) = imageDouble(:,:,3) * scaleBlue;  % Scale blue channel
        outputImageDouble = max(0, min(outputImageDouble, 1)); % Clip values to [0, 1]

        % Convert back to original data type
        switch originalDataType
            case 'uint8'
                outputImage = im2uint8(outputImageDouble);
            case 'uint16'
                outputImage = im2uint16(outputImageDouble);
            otherwise % Keep as double if original was double/single
                outputImage = outputImageDouble;
        end
    end

    function outputImage = applyContrastEnhancementAdapthisteq(inputImage)
        % Enhance image contrast using Adaptive Histogram Equalization (adapthisteq).
        % Handles color images by processing the Luminance channel in LAB space.
        if size(inputImage, 3) == 3 % Color image
            originalDataType = class(inputImage);
            % Convert to LAB color space
            try
                labImage = rgb2lab(inputImage);
            catch conversionError
                warning('rgb2lab conversion failed: %s. Skipping contrast enhancement.', conversionError.message);
                outputImage = inputImage; % Return original on error
                return;
            end

            % Extract Luminance channel and normalize to [0, 1] for adapthisteq
            luminanceChannel = labImage(:,:,1) / 100;
            if ~isa(luminanceChannel, 'double') % Ensure double precision
                luminanceChannel = double(luminanceChannel);
            end

            % Apply adaptive histogram equalization
            luminanceEnhanced = adapthisteq(luminanceChannel);

            % Replace original L channel and scale back
            labImage(:,:,1) = luminanceEnhanced * 100;

            % Convert back to RGB and original data type
            try
                switch originalDataType
                    case 'uint8'
                        outputImage = lab2rgb(labImage, 'Out', 'uint8');
                    case 'uint16'
                        outputImage = lab2rgb(labImage, 'Out', 'uint16');
                    otherwise
                        outputImage = lab2rgb(labImage); % Output as double
                end
            catch conversionBackError
                warning('lab2rgb conversion failed: %s. Returning LAB-enhanced image (might look strange).', conversionBackError.message);
                % As a fallback, maybe return the original image, or the LAB image if needed
                outputImage = inputImage; % Safest fallback
            end
        else % Grayscale image
            % Apply adapthisteq directly
            % Handle potential input types issues for adapthisteq
            if isfloat(inputImage) && (max(inputImage(:)) > 1 || min(inputImage(:)) < 0)
                % Normalize float images not in [0,1] range
                imageNormalized = mat2gray(inputImage);
                outputImage = adapthisteq(imageNormalized);
                % Note: Output of adapthisteq on normalized float is double in [0,1].
                % Consider converting back if needed, but often float is fine.
            elseif islogical(inputImage)
                warning('Logical image type is not suitable for adapthisteq. Skipping contrast enhancement.');
                outputImage = inputImage; % Return original logical image
            else % Assume uint8, uint16, or float in [0,1]
                try
                    outputImage = adapthisteq(inputImage);
                catch adapthisteqError
                     warning('adapthisteq failed directly: %s. Skipping contrast enhancement.', adapthisteqError.message);
                    outputImage = inputImage; % Return original on error
                end
            end
        end
    end

    function outputImage = applySaturationEnhancement(inputImage, enhancementFactor)
        % Enhance color saturation by converting to HSV, scaling the S channel.
        % Assumes inputImage is a color image (size(inputImage,3)==3).
        originalDataType = class(inputImage);
        imageDouble = im2double(inputImage); % Convert to double

        try
            % Convert RGB to HSV
            hsvImage = rgb2hsv(imageDouble);
            % Scale the Saturation channel (channel 2), clipping at 1
            hsvImage(:,:,2) = min(hsvImage(:,:,2) * enhancementFactor, 1);
            % Convert back to RGB
            outputImageDouble = hsv2rgb(hsvImage);
        catch hsvError
            warning('HSV conversion or processing failed: %s. Skipping saturation enhancement.', hsvError.message);
            outputImage = inputImage; % Return original on error
            return;
        end

        % Convert back to original data type
        switch originalDataType
            case 'uint8'
                outputImage = im2uint8(outputImageDouble);
            case 'uint16'
                outputImage = im2uint16(outputImageDouble);
            otherwise
                outputImage = outputImageDouble; % Keep as double
        end
    end

end % End of main function imageProcessingAppLayout