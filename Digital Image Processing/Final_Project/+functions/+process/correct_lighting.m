function corrected_image = correct_lighting(image_data, lighting_condition)
    if size(image_data, 3) == 3
        image_data = rgb2gray(image_data);
    end
    
    % 分析影像統計資訊（用於後處理判斷）
    std_intensity = std(double(image_data(:)));
    
    % 根據輸入的診斷結果選擇對應的校正策略
    switch lighting_condition
        case 'underexposed'  % 過暗影像
            % 使用 Gamma 校正提亮，避免雜訊放大
            gamma_value = 0.6;  % Gamma < 1 提亮
            corrected_image = imadjust(image_data, [], [], gamma_value);
            
        case 'overexposed'  % 過亮影像
            % 使用 Gamma 校正壓暗
            gamma_value = 1.4;  % Gamma > 1 壓暗
            corrected_image = imadjust(image_data, [], [], gamma_value);
            
        case 'low_contrast'  % 低對比度影像
            % 使用溫和的對比度增強
            corrected_image = adapthisteq(image_data, ...
                'ClipLimit', 0.015, ...  % 較低的限制避免雜訊
                'Distribution', 'uniform', ...
                'NumTiles', [8 8]);  % 較少的分塊避免過度處理
                
        case 'ok'  % 正常影像
            corrected_image = image_data;
            % 輕微的對比度增強（可選）
            % corrected_image = imadjust(image_data, stretchlim(image_data, [0.01 0.99]), []);
            
        otherwise  % 未知狀況，保持原樣
            corrected_image = image_data;
    end
    
    % 後處理：針對文字影像的雜訊抑制
    if std_intensity > 40  % 如果雜訊較多
        % 使用引導濾波器，保護邊緣同時去雜訊
        if exist('imguidedfilter', 'file')
            corrected_image = imguidedfilter(corrected_image, corrected_image, ...
                'NeighborhoodSize', [5 5], 'DegreeOfSmoothing', 0.1^2);
        else
            % 如果沒有引導濾波器，使用輕微的高斯濾波
            corrected_image = imgaussfilt(corrected_image, 0.5);
        end
    end
end