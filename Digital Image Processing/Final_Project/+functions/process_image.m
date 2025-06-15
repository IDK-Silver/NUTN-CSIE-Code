function [final_image, intermediate_stages, report] = process_image(image_path)
    % 讀取影像並轉為灰階
    img_raw = imread(image_path);
    if size(img_raw, 3) == 3
        current_image = rgb2gray(img_raw);
    else
        current_image = img_raw;
    end
    
    % 確保初始影像為 uint8 類型
    if ~isa(current_image, 'uint8')
        current_image = im2uint8(current_image);
    end
    
    report = struct();
    processing_steps = {'Original Grayscale'};
    
    % 初始化 Cell Array 並存入第1張圖 (原始灰階)
    intermediate_stages = {};
    intermediate_stages{end+1} = current_image;

    % --- 自動診斷與處理 ---
    report.lighting = functions.diagnose.diagnose_lighting(current_image);
    report.noise = functions.diagnose.diagnose_noise(current_image);
    report.blur = functions.diagnose.diagnose_blur(current_image);
    
    % 關卡一：光線處理
    current_image = functions.process.correct_lighting(current_image, report.lighting);
    % 確保輸出為 uint8 類型
    if ~isa(current_image, 'uint8')
        current_image = im2uint8(current_image);
    end
    processing_steps{end+1} = 'After Lighting Stage';
    intermediate_stages{end+1} = current_image;

    % 關卡二：去雜訊處理
    if ~strcmp(report.noise, 'none')
        current_image = functions.process.remove_noise(current_image, report.noise);
        % 確保輸出為 uint8 類型
        if ~isa(current_image, 'uint8')
            current_image = im2uint8(current_image);
        end
    end
    processing_steps{end+1} = 'After Noise Stage';
    intermediate_stages{end+1} = current_image;

    % 關卡三：銳化處理
    if report.blur
        current_image = functions.process.apply_sharpening(current_image);
        % 確保輸出為 uint8 類型
        if ~isa(current_image, 'uint8')
            current_image = im2uint8(current_image);
        end
    end
    processing_steps{end+1} = 'After Sharpening Stage';
    intermediate_stages{end+1} = current_image;
    
    % 最終的二值化，其結果由 final_image 獨立回傳
    final_image = functions.process.binarize_adaptive(current_image);
    % 二值化結果轉換為 uint8 (0 和 255)
    if islogical(final_image)
        final_image = uint8(final_image) * 255;
    elseif ~isa(final_image, 'uint8')
        final_image = im2uint8(final_image);
    end
    
    processing_steps{end+1} = 'Final Binary';
    report.applied_steps = processing_steps;
end