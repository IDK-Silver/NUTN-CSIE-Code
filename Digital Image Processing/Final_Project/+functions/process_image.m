function [final_image, intermediate_stages, report] = process_image(image_path)
    % 讀取影像並確保為 uint8 灰階
    img_raw = imread(image_path);
    if size(img_raw, 3) == 3, current_image = rgb2gray(img_raw); else, current_image = img_raw; end
    if ~isa(current_image, 'uint8'), current_image = im2uint8(current_image); end
    
    report = struct();
    intermediate_stages = {};
    actual_applied_steps = {}; 
    
    intermediate_stages{end+1} = current_image; % 存入第1張圖 (原始灰階)

    % --- 自動診斷 ---
    report.lighting = functions.diagnose.diagnose_lighting(current_image);
    report.noise = functions.diagnose.diagnose_noise(current_image);
    report.blur = functions.diagnose.diagnose_blur(current_image);
    
    % --- 【核心修正】: 將處理步驟放回 if 判斷式中 ---
    
    % 關卡一：光線處理
    % 只有在診斷出光線有問題時，才執行校正
    if ~strcmp(report.lighting, 'ok')
        current_image = functions.process.correct_lighting(current_image, report.lighting);
        actual_applied_steps{end+1} = ['光線校正 (' report.lighting ')'];
    end
    intermediate_stages{end+1} = current_image;

    % 關卡二：去雜訊處理
    if ~strcmp(report.noise, 'none')
        current_image = functions.process.remove_noise(current_image, report.noise);
        actual_applied_steps{end+1} = '去雜訊';
    end
    intermediate_stages{end+1} = current_image;

    % 關卡三：銳化處理
    if report.blur
        current_image = functions.process.apply_sharpening(current_image);
        actual_applied_steps{end+1} = '影像銳化';
    end
    intermediate_stages{end+1} = current_image;
    
    % 最終的二值化
    final_image = functions.process.binarize_adaptive(current_image);
    actual_applied_steps{end+1} = '自適應二值化與清理';
    
    report.applied_steps = actual_applied_steps;
    
    % 確保最終輸出為 uint8
    if islogical(final_image), final_image = uint8(final_image) * 255; end
end