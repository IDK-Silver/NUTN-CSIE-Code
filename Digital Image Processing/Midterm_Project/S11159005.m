% MATLAB Script for Automated Image Quality Analysis and Enhancement
%
% Description:
%   Analyzes input images for noise, blur, exposure, and color issues.
%   Iteratively applies corrections based on document techniques (Filtering, 
%   Sharpening, Gamma/Intensity Adjustment, Histogram Equalization, etc.).
%   In each iteration, ALL detected problems are attempted to be fixed.
%   Processing stops when no issues are detected, no further improvement occurs 
%   across a full iteration, or max iterations are reached.
%
% Author:     Yu-Feng
% Repository: https://github.com/IDK-Silver/NUTN-CSIE-Code/tree/main/Digital%20Image%20Processing/Midterm_Project/
%
% Input:
%   - Image files specified in 'image_names', located in 'search_dirs'.
%
% Output:
%   - Processed images: 'A<original_name>.jpg' in 'result_subdir'.
%   - Comparison text files: 'A<original_name>.txt' detailing original/final 
%     analysis and applied steps, saved in 'result_subdir'.
%   - (Optional) Comparison figures: '<original_name>_comparison.png' saved 
%     in 'result_subdir', controlled by 'generate_comparison_png'.
%   - Console output logging the process.
%
% Requirements:
%   - MATLAB R2019b or later (due to im2gray usage).
%   - Image Processing Toolbox.
%
% Script Structure:
%   1. Configuration Section
%   2. Helper Functions (File finding)
%   3. Detection Functions (Noise, Blur, Exposure, Color)
%   4. Processing Functions (Denoise, Deblur/Sharpen, Exposure Fix, Color Fix)
%   5. Main Script Execution (Initialization, Analysis, Iterative Processing, Output)
%
% Version: 2.0 - Final Iterative Logic
%

% =========================================================================
%                       CONFIGURATION SECTION
% =========================================================================
clear; close all; clc; 

image_names = {'01.jpg', '02.jpg', '03.jpg'};               % List of image files to process
search_dirs = {'', 'res', 'images', 'data', 'input'};       % Subdirs to search
result_subdir = '.';                                        % Subdir for output files (relative to script dir)
max_iterations = 5 ;                                        % Max processing iterations per image
generate_comparison_png = true;                             % Set to false to disable saving comparison PNG

% =========================================================================
%                       HELPER & UTILITY FUNCTIONS
% =========================================================================

function found_path = find_file_in_paths(base_dir, filename, search_dirs)
    % Finds the full path of a file by searching in specified directories.
    for i = 1:length(search_dirs)
        try_path = fullfile(base_dir, search_dirs{i}, filename);
        if isfile(try_path)
            found_path = try_path;
            return; 
        end
    end
    error('File not found: %s in any of the specified search directories relative to %s', filename, base_dir);
end

% =========================================================================
%                           DETECTION FUNCTIONS
% =========================================================================

function noise_type = detect_noise_type(img)
    % Detects estimated noise type ('none', 'saltpepper', 'gaussian', 'poisson').
    
    % Parameters
    salt_pepper_frac_thresh = 0.01; 
    gaussian_check_sigma = 2;
    gaussian_noise_std_thresh = 8; % In 0-255 range equivalent
    poisson_rel_diff_thresh = 0.1; 

    noise_type = 'unknown'; % Default if conversion fails
    try 
        if ndims(img) == 3 && size(img,3) == 3, img_gray = rgb2gray(img);
        elseif ndims(img) == 2, img_gray = img;
        else, img_gray = im2gray(img); end
        img_gray_double = im2double(img_gray); 
    catch ME
         warning('Noise detection: Image conversion failed: %s.', sprintf('%s', ME.message)); return;
    end

    totalPixels = numel(img_gray_double);
    low_thresh = 1/255; high_thresh = 254/255; 
    numNearZero = sum(img_gray_double(:) <= low_thresh);
    numNearFull = sum(img_gray_double(:) >= high_thresh);
    fracExtremes = (numNearZero + numNearFull) / totalPixels;

    % Salt & Pepper check
    if fracExtremes > salt_pepper_frac_thresh 
        noise_type = 'saltpepper'; return;
    end

    % Gaussian check
    try
        I_smooth = imgaussfilt(img_gray_double, gaussian_check_sigma); 
        diff_img = img_gray_double - I_smooth;
        noiseStd = std(diff_img(:)) * 255; 
        if noiseStd > gaussian_noise_std_thresh 
            noise_type = 'gaussian'; return;
        end
    catch ME, warning('Noise detection: Gaussian check failed: %s', sprintf('%s', ME.message)); end

    % Poisson check
    img_mean = mean(img_gray_double(:));
    img_var = var(img_gray_double(:));
    if img_mean > 1e-6 && abs(img_var / img_mean - 1) < poisson_rel_diff_thresh 
        noise_type = 'poisson'; return;
    end

    noise_type = 'none'; % If no noise detected
end

function [blur_type, clarity_value, motion_direction] = detect_blur_type(img)
    % Detects estimated blur type ('none', 'motion', 'gaussian').

    % Parameters
    blur_clarity_threshold = 0.0008; 
    motion_gradient_ratio_threshold = 0.35; 
    
    blur_type = 'unknown'; clarity_value = NaN; motion_direction = 'unknown'; % Defaults
    try 
        if ndims(img) == 3 && size(img,3) == 3, img_gray = rgb2gray(img);
        elseif ndims(img) == 2, img_gray = img;
        else, img_gray = im2gray(img); end
         I_gray_d = im2double(img_gray);
    catch ME, warning('Blur detection: Image conversion failed: %s.', sprintf('%s', ME.message)); return; end
    
    % Clarity (Laplacian variance)
    laplacian_kernel = fspecial('laplacian', 0); 
    try
        laplacian_response = imfilter(I_gray_d, laplacian_kernel, 'replicate', 'conv');
        clarity_value = var(laplacian_response(:));
    catch ME, warning('Blur detection: Laplacian filter failed: %s.', sprintf('%s', ME.message)); return; end

    % Blur type determination
    if clarity_value < blur_clarity_threshold
        try
            hx = [-1 0 1]; hy = hx';      
            gx = imfilter(I_gray_d, hx, 'replicate', 'conv');
            gy = imfilter(I_gray_d, hy, 'replicate', 'conv');
            sum_abs_gx = sum(abs(gx(:))); sum_abs_gy = sum(abs(gy(:)));

            if sum_abs_gx + sum_abs_gy < 1e-6, blur_type = 'gaussian'; motion_direction = 'none';
            else
                gradient_ratio = abs(sum_abs_gx - sum_abs_gy) / max(sum_abs_gx, sum_abs_gy);
                if gradient_ratio > motion_gradient_ratio_threshold
                    blur_type = 'motion';
                    if sum_abs_gx < sum_abs_gy, motion_direction = 'horizontal'; else, motion_direction = 'vertical'; end
                else, blur_type = 'gaussian'; motion_direction = 'none'; end
            end
        catch ME, warning('Blur detection: Gradient analysis failed: %s. Assuming "gaussian".', sprintf('%s', ME.message)); blur_type = 'gaussian'; motion_direction = 'none'; end
    else, blur_type = 'none'; motion_direction = 'none'; end
end

function exposure_problem = detect_exposure_problem(img)
    % Detects potential exposure problems ('Underexposed', 'Overexposed', 'Normal').

    % Parameters
    underexposure_mean_thresh = 0.3; overexposure_mean_thresh = 0.7; 
    saturation_percent_thresh = 5.0; 
    saturated_low_val = 1/255; saturated_high_val = 254/255; 
    
    exposure_problem = 'unknown'; % Default
    try 
        if size(img, 3) == 3, img_gray = rgb2gray(img); else, img_gray = img; end
        img_double = im2double(img_gray);
    catch ME, warning('Exposure detection: Image conversion failed: %s.', sprintf('%s', ME.message)); return; end

    meanIntensity = mean(img_double(:));
    totalPixels = numel(img_double);
    percent_saturated_low = sum(img_double(:) <= saturated_low_val) / totalPixels * 100;
    percent_saturated_high = sum(img_double(:) >= saturated_high_val) / totalPixels * 100;

    if meanIntensity < underexposure_mean_thresh || percent_saturated_low > saturation_percent_thresh, exposure_problem = 'Underexposed';
    elseif meanIntensity > overexposure_mean_thresh || percent_saturated_high > saturation_percent_thresh, exposure_problem = 'Overexposed';
    else, exposure_problem = 'Normal'; end
end

function color_problem = detect_color_problem(img)
    % Detects potential color problems ('grayscale', 'color_cast', 'normal').

    % Parameters
    color_cast_rel_dev_threshold = 0.1; 
    
    color_problem = 'unknown'; % Default
    if ndims(img) == 2 || (ndims(img) == 3 && size(img,3) == 1), color_problem = 'grayscale';
    elseif ndims(img) == 3 && size(img,3) == 3
        try 
            img_double = im2double(img);
            means = mean(reshape(img_double, [], 3), 1); 
            mean_overall = mean(means);
            max_deviation = max(abs(means - mean_overall));
            if mean_overall > 1e-6 && (max_deviation / mean_overall) > color_cast_rel_dev_threshold, color_problem = 'color_cast'; else, color_problem = 'normal'; end
        catch ME, warning('Color detection: Calculation failed: %s.', sprintf('%s', ME.message)); end
    else, warning('Color detection: Unexpected image dimensions/channels (%s).', mat2str(size(img))); end
end

% =========================================================================
%                         PROCESSING FUNCTIONS
% =========================================================================

function J = denoise_image(I, noise_type)
    % Denoises image I based on detected noise_type.
    J = I; % Default
    median_filter_size = [3,3]; gaussian_filter_sigma = 1.5;

    switch lower(noise_type)
        case 'saltpepper', J = apply_median(I, median_filter_size); disp('Applied Median Filter for Salt & Pepper Noise.');
        case 'gaussian', J = apply_gaussian(I, gaussian_filter_sigma); disp(['Applied Gaussian Filter (sigma=', num2str(gaussian_filter_sigma), ') for Gaussian Noise.']);
        case 'poisson', J = apply_median(I, median_filter_size); disp('Applied Median Filter (as substitute) for Poisson Noise.'); % Substitute
        case 'none', % No action
        otherwise, if ~strcmpi(noise_type, 'none'), warning('Unknown noise type "%s". No denoising.', noise_type); end
    end
end

function J = apply_median(I, filter_size)
    % Applies median filter, handles errors.
    J = I;
    try
        if ndims(I) == 3, for c = 1:3, J(:,:,c) = medfilt2(I(:,:,c), filter_size); end
        else, J = medfilt2(I, filter_size); end
    catch ME, warning('Median filtering failed: %s', sprintf('%s', ME.message)); J=I; end
end

function J = apply_gaussian(I, sigma)
    % Applies Gaussian filter, handles errors and fallbacks.
    J = I;
    try, J = imgaussfilt(I, sigma);
    catch ME 
        warning('imgaussfilt failed: %s. Trying fallback.', sprintf('%s', ME.message));
        try 
             I_double = im2double(I); J_double = I_double;
             if ndims(I_double) == 3, for c=1:3, J_double(:,:,c) = imgaussfilt(I_double(:,:,c), sigma); end
             else, J_double = imgaussfilt(I_double, sigma); end
             if isa(I,'uint8'), J = im2uint8(J_double); elseif isa(I,'uint16'), J=im2uint16(J_double); else J=J_double; end
         catch ME2, warning('Fallback Gaussian filtering failed: %s.', sprintf('%s', ME2.message)); J=I; end
    end
end

function J = deblur_image(I, blur_type, ~) 
    % Sharpens image I if blur detected, using Laplacian-based sharpening.
    J = I; % Default
    laplacian_alpha = 0.2; sharpening_strength = 0.8; 

    switch lower(blur_type)
        case {'motion', 'gaussian'}
            laplacian_kernel = fspecial('laplacian', laplacian_alpha); 
            try
                I_double = im2double(I); J_double = I_double; 
                if ndims(I_double) == 3
                     for c = 1:3, J_double(:,:,c) = I_double(:,:,c) - sharpening_strength * imfilter(I_double(:,:,c), laplacian_kernel, 'replicate', 'conv'); end
                else, J_double = I_double - sharpening_strength * imfilter(I_double, laplacian_kernel, 'replicate', 'conv'); end
                J_double = max(0, min(1, J_double)); % Clip
                if isa(I, 'uint8'), J = im2uint8(J_double); elseif isa(I, 'uint16'), J = im2uint16(J_double); else J = J_double; end
                disp(['Applied Sharpening (Laplacian based, strength=', num2str(sharpening_strength), ') for Blur Type: ', blur_type]);
            catch ME, warning('Failed during sharpening: %s.', sprintf('%s', ME.message)); J = I; end
        case 'none', % No action
        otherwise, if ~strcmpi(blur_type, 'none'), warning('Unknown blur type "%s". No sharpening.', blur_type); end
    end
end

function J = fix_exposure(I, exposure_problem)
    % Corrects detected exposure problems.
    J = I; % Default
    gamma_underexposed = 0.6; imadjust_overexposed_high_in = 0.9; 

    switch lower(exposure_problem)
        case 'underexposed'
            try 
                J_double = im2double(I) .^ gamma_underexposed;
                if isa(I, 'uint8'), J = im2uint8(J_double); elseif isa(I, 'uint16'), J = im2uint16(J_double); else J = J_double; end
                disp(['Applied Gamma Correction (gamma=', num2str(gamma_underexposed), ') for Underexposure.']);
            catch ME, warning('Gamma correction failed: %s.', sprintf('%s', ME.message)); J=I; end
        case 'overexposed'
             try
                 J_double = imadjust(im2double(I), [0 imadjust_overexposed_high_in], [0 1]);
                 if isa(I, 'uint8'), J = im2uint8(J_double); elseif isa(I, 'uint16'), J = im2uint16(J_double); else J = J_double; end
                 disp('Applied Intensity Adjustment (imadjust) for Overexposure.');
             catch ME, warning('imadjust failed: %s.', sprintf('%s', ME.message)); J=I; end
        case 'normal', % No action
        otherwise, if ~strcmpi(exposure_problem, 'Normal'), warning('Unknown exposure problem "%s". No correction.', exposure_problem); end
    end
end

function J = fix_color_problem(I, color_problem)
    % Fixes color cast or enhances grayscale contrast.
    J = I; % Default
    switch lower(color_problem)
        case 'grayscale'
            try % Enhance contrast using histogram equalization
                if ~isa(I, 'uint8') && ~isa(I, 'uint16'), I_gray = im2uint8(mat2gray(I)); else, I_gray = I; end
                if ndims(I_gray) == 2, J = histeq(I_gray); 
                elseif ndims(I_gray) == 3 && size(I_gray,3) == 1, J_temp = histeq(I_gray(:,:,1)); J = repmat(J_temp, [1, 1, size(I,3)]); if isa(I,'uint8'), J=im2uint8(J); elseif isa(I,'uint16'), J=im2uint16(J); end
                else, warning('Input for grayscale enhancement is not 2D/M-N-1.'); J = I; return; end % Skip if not suitable
                disp('Applied Histogram Equalization for Grayscale Contrast.');
             catch ME, warning('histeq failed: %s.', sprintf('%s', ME.message)); J = I; end
        case 'color_cast'
            % Correct color cast using simple channel stretching
            if ndims(I) == 3 && size(I,3) == 3
                try
                    I_double = im2double(I); J_double = I_double; 
                    for c = 1:3
                        channel = I_double(:,:,c); minC = min(channel(:)); maxC = max(channel(:));
                        if maxC > minC, J_double(:,:,c) = (channel - minC) / (maxC - minC); end
                    end
                    if isa(I, 'uint8'), J = im2uint8(J_double); elseif isa(I, 'uint16'), J = im2uint16(J_double); else J = J_double; end
                    disp('Applied Channel Stretching for Color Cast Correction.');
                catch ME, warning('Failed to correct color cast: %s.', sprintf('%s', ME.message)); J = I; end
            else, warning('Image flagged as color_cast is not RGB.'); end
        case 'normal', % No action
        otherwise, if ~ismember(lower(color_problem), {'normal', 'grayscale', 'unknown'}), warning('Unknown color problem "%s". No correction.', color_problem); end
    end
end


% =========================================================================
%                           MAIN SCRIPT EXECUTION
% =========================================================================

% --- Setup Paths & Environment ---
try 
    script_path_full = mfilename('fullpath');
    if isempty(script_path_full), error('Cannot determine script path.'); end 
    [script_dir,~,~] = fileparts(script_path_full);
catch ME 
     warning('mfilename(''fullpath'') failed (%s). Using pwd.', sprintf('%s', ME.message)); script_dir = pwd;
end
result_path = fullfile(script_dir, result_subdir);
if ~exist(result_path, 'dir'), fprintf('Creating result directory: %s\n', result_path); mkdir(result_path); end
fprintf('Script base directory: %s\n', script_dir);
fprintf('Result directory: %s\n', result_path);

% --- Find Image Files ---
image_paths = cell(size(image_names));
all_found = true;
for i = 1:length(image_names)
    try, image_paths{i} = find_file_in_paths(script_dir, image_names{i}, search_dirs); fprintf('Found: %s -> %s\n', image_names{i}, image_paths{i});
    catch ME, warning('%s - Will skip this image.', sprintf('%s', ME.message)); image_paths{i} = ''; all_found = false; end
end
if ~all_found, fprintf('Warning: Not all images were found.\n'); end
valid_indices = ~cellfun('isempty', image_paths);
image_paths = image_paths(valid_indices); image_names = image_names(valid_indices);
if isempty(image_paths), error('No valid images found to process. Exiting.'); end

% --- Image Analysis Structure Preallocation ---
num_images = length(image_paths);
image_analysis(num_images) = struct('filename', [], 'path', [], 'data', [], 'size', [], 'is_color', [], 'resolution', [], 'color_depth', [], 'noise_type', [], 'blur_type', [], 'clarity_value', [], 'motion_direction', [], 'exposure_problem', [], 'color_problem', []); 

% --- Initial Image Loading and Analysis ---
fprintf('\n--- Starting Initial Image Analysis ---\n');
for i = 1:num_images
    current_filename = image_names{i}; current_path = image_paths{i};
    fprintf('Analyzing image: %s\n', current_filename);
    try
        current_image_data = imread(current_path); img_info = imfinfo(current_path); 
        image_analysis(i).filename = current_filename; image_analysis(i).path = current_path;
        image_analysis(i).data = current_image_data; image_analysis(i).size = size(current_image_data);
        if strcmpi(img_info(1).ColorType, "truecolor"), image_analysis(i).is_color = 'Color'; elseif strcmpi(img_info(1).ColorType, "grayscale"), image_analysis(i).is_color = 'Grayscale'; elseif strcmpi(img_info(1).ColorType, "indexed"), image_analysis(i).is_color = 'Indexed'; else, image_analysis(i).is_color = img_info(1).ColorType; end
        image_analysis(i).resolution = sprintf('%d x %d', img_info(1).Width, img_info(1).Height); image_analysis(i).color_depth = sprintf('%d bits', img_info(1).BitDepth);
        image_analysis(i).noise_type = detect_noise_type(current_image_data);
        [blur_type, clarity_value, motion_direction] = detect_blur_type(current_image_data);
        image_analysis(i).blur_type = blur_type; image_analysis(i).clarity_value = clarity_value; image_analysis(i).motion_direction = motion_direction;
        image_analysis(i).exposure_problem = detect_exposure_problem(current_image_data); image_analysis(i).color_problem = detect_color_problem(current_image_data);
        fprintf('  Initial Analysis - Noise:%s, Blur:%s, Exposure:%s, Color:%s\n', image_analysis(i).noise_type, image_analysis(i).blur_type, image_analysis(i).exposure_problem, image_analysis(i).color_problem);
    catch ME, warning('Failed load/analyze %s: %s', current_filename, sprintf('%s', ME.message)); image_analysis(i).filename = current_filename; image_analysis(i).path = current_path; image_analysis(i).data = []; end
end
valid_load_indices = ~arrayfun(@(x) isempty(x.data), image_analysis); image_analysis = image_analysis(valid_load_indices);
if isempty(image_analysis), error('No images loaded/analyzed. Exiting.'); end
fprintf('\n--- Successfully loaded and initially analyzed %d images. ---\n', length(image_analysis));

% --- Apply Processing Flow (Iterative - Apply All Detected Fixes per Iteration) ---
fprintf('\n--- Starting Iterative Image Processing ---\n');
for i = 1:length(image_analysis)
    fprintf('Processing image: %s (%d/%d)\n', image_analysis(i).filename, i, length(image_analysis));
    original_image_data = image_analysis(i).data; original_info = image_analysis(i); 
    if isempty(original_image_data), fprintf('  Skipping: previous load error.\n'); continue; end

    processed_image = original_image_data; 
    processing_steps_applied_total = {}; 
    
    for iteration = 1:max_iterations
        fprintf('--- Iteration %d ---\n', iteration);
        image_at_iteration_start = processed_image; 
        problems_found_this_iteration = false;
        steps_applied_this_iteration = {}; 

        % Analyze current state
        current_noise = detect_noise_type(processed_image);
        [current_blur, ~, current_motion] = detect_blur_type(processed_image); 
        current_exposure = detect_exposure_problem(processed_image);
        current_color = detect_color_problem(processed_image);

        % Check if problems exist that need fixing
        needs_denoise = ~ismember(lower(current_noise), {'none', 'unknown'});
        needs_deblur = ~ismember(lower(current_blur), {'none', 'unknown'});
        needs_exposure_fix = ~ismember(lower(current_exposure), {'normal', 'unknown'});
        needs_color_fix = strcmpi(current_color, 'color_cast');
        needs_grayscale_enhance = (strcmpi(current_color, 'grayscale') && iteration == 1); % Enhance grayscale only once

        if ~(needs_denoise || needs_deblur || needs_exposure_fix || needs_color_fix || needs_grayscale_enhance)
             fprintf('  No problems detected needing correction in Iteration %d. Stopping.\n', iteration); break; 
        end

        % Apply ALL applicable fixes for this iteration
        if needs_denoise
            problems_found_this_iteration = true; fprintf('  Attempting Step: Denoise (Noise Type: %s)\n', current_noise);
            processed_image = denoise_image(processed_image, current_noise); steps_applied_this_iteration{end+1} = 'Denoise';
        end
        if needs_deblur % Apply deblur after potential denoising
            problems_found_this_iteration = true; fprintf('  Attempting Step: Deblur/Sharpen (Blur Type: %s)\n', current_blur);
            processed_image = deblur_image(processed_image, current_blur, current_motion); steps_applied_this_iteration{end+1} = 'Deblur/Sharpen';
        end
        if needs_exposure_fix % Apply exposure fix after potential denoising/deblurring
            problems_found_this_iteration = true; fprintf('  Attempting Step: Fix Exposure (Exposure Problem: %s)\n', current_exposure);
            processed_image = fix_exposure(processed_image, current_exposure); steps_applied_this_iteration{end+1} = 'Fix Exposure';
        end
        if needs_color_fix % Apply color fix last
            problems_found_this_iteration = true; step_name = 'Fix Color Cast'; fprintf('  Attempting Step: %s (Color Problem: %s)\n', step_name, current_color);
            processed_image = fix_color_problem(processed_image, current_color); steps_applied_this_iteration{end+1} = step_name;
        elseif needs_grayscale_enhance % Apply grayscale enhance last (if applicable)
             problems_found_this_iteration = true; step_name = 'Enhance Grayscale'; fprintf('  Attempting Step: %s (Color Problem: %s)\n', step_name, current_color);
             processed_image = fix_color_problem(processed_image, current_color); steps_applied_this_iteration{end+1} = step_name;
        end

        % Check for convergence (no change after applying all steps)
        if isequal(processed_image, image_at_iteration_start)
             fprintf('  No change detected in image after applying fixes in Iteration %d. Stopping.\n', iteration); break; 
        else
             fprintf('  Image changed in Iteration %d. Applied: %s\n', iteration, strjoin(steps_applied_this_iteration, ', '));
             for step_idx = 1:length(steps_applied_this_iteration), processing_steps_applied_total{end+1} = sprintf('Iter %d: %s', iteration, steps_applied_this_iteration{step_idx}); end
        end

        if iteration == max_iterations, fprintf('  Reached maximum iterations (%d). Stopping.\n', max_iterations); end
    end % End of iterations loop

    % Final Analysis after all iterations
    final_info = struct();
    final_info.noise_type = detect_noise_type(processed_image);
    [final_info.blur_type, final_info.clarity_value, final_info.motion_direction] = detect_blur_type(processed_image);
    final_info.exposure_problem = detect_exposure_problem(processed_image);
    final_info.color_problem = detect_color_problem(processed_image);

    % --- Prepare Output Filenames ---
    [~, name, ~] = fileparts(original_info.filename);
    processed_base_filename = ['A' name]; 
    processed_image_filename = fullfile(result_path, [processed_base_filename '.jpg']); 
    comparison_text_filename = fullfile(result_path, [processed_base_filename '.txt']);
    comparison_png_filename = fullfile(result_path, [name '_comparison.png']); 

    % --- Save Processed Image ---
    try, imwrite(processed_image, processed_image_filename, 'jpg', 'Quality', 95); fprintf('  Saved final processed image to: %s\n', processed_image_filename);
    catch ME, warning('Failed to save final processed image %s: %s', processed_image_filename, sprintf('%s', ME.message)); end
    
    % --- Generate and Save Comparison Text File ---
    fid = -1; % Initialize file ID
    try
        fid = fopen(comparison_text_filename, 'w');
        if fid == -1, error('Cannot open file %s for writing.', comparison_text_filename); end
        fprintf(fid, 'Image Processing Comparison Report for: %s\n', original_info.filename); fprintf(fid, '=================================================\n\n');
        fprintf(fid, '--- Original Image Analysis ---\n'); fprintf(fid, 'Noise Type: %s\n', original_info.noise_type); fprintf(fid, 'Blur Type: %s (Clarity: %.4f, Motion: %s)\n', original_info.blur_type, original_info.clarity_value, original_info.motion_direction); fprintf(fid, 'Exposure Problem: %s\n', original_info.exposure_problem); fprintf(fid, 'Color Problem: %s\n\n', original_info.color_problem);
        fprintf(fid, '--- Applied Processing Steps ---\n');
        if isempty(processing_steps_applied_total), fprintf(fid, 'None\n\n'); else, for step_idx = 1:length(processing_steps_applied_total), fprintf(fid, '%s\n', processing_steps_applied_total{step_idx}); end; fprintf(fid, '\n'); end
        fprintf(fid, '--- Final Image Analysis ---\n'); fprintf(fid, 'Noise Type: %s\n', final_info.noise_type); fprintf(fid, 'Blur Type: %s (Clarity: %.4f, Motion: %s)\n', final_info.blur_type, final_info.clarity_value, final_info.motion_direction); fprintf(fid, 'Exposure Problem: %s\n', final_info.exposure_problem); fprintf(fid, 'Color Problem: %s\n', final_info.color_problem);
        fclose(fid); fprintf('  Saved comparison text file to: %s\n', comparison_text_filename);
    catch ME
        warning('Failed to create/save comparison text file %s: %s', comparison_text_filename, sprintf('%s', ME.message)); if fid ~= -1, fclose(fid); end 
    end

    % --- Create and Save Comparison PNG Figure (Conditional) ---
    if generate_comparison_png
        comp_fig = []; % Initialize figure handle
        try
            comp_fig = figure('Name', ['Compare: ', original_info.filename], 'Position', [50 50 1200 600], 'Visible', 'off'); 
            h_ax1 = subplot(1, 2, 1); imshow(original_info.data); if ndims(original_info.data) == 2, colormap(h_ax1, 'gray'); end; title({'Original Image', sprintf('(%s)', original_info.filename)}, 'FontSize', 10, 'Interpreter', 'none');
            h_ax2 = subplot(1, 2, 2); imshow(processed_image); if ndims(processed_image) == 2, colormap(h_ax2, 'gray'); end; title({'Final Processed Image', sprintf('(%s.jpg)', processed_base_filename)}, 'FontSize', 10, 'Interpreter', 'none');
            if isempty(processing_steps_applied_total), applied_str = 'Applied: None'; else, applied_str = ['Applied: ', strjoin(processing_steps_applied_total, ' -> ')]; end
            sgtitle({['Comparison: ', original_info.filename], applied_str}, 'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none'); 
            saveas(comp_fig, comparison_png_filename); fprintf('  Saved comparison PNG figure to: %s\n', comparison_png_filename);
        catch ME, warning('Failed to create/save comparison PNG for %s: %s', original_info.filename, sprintf('%s', ME.message)); 
        finally, if ~isempty(comp_fig) && ishandle(comp_fig), close(comp_fig); end % Ensure figure closes
        end
    else, fprintf('  Skipped saving comparison PNG figure (generate_comparison_png = false).\n'); end 

    fprintf('Finished processing image: %s\n\n', original_info.filename);
end % --- End of main loop for each image ---

fprintf('\n--- Iterative Image Processing Complete ---\n');