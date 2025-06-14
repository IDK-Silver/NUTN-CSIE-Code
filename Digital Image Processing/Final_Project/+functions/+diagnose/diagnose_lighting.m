function report = diagnose_lighting(image_path_or_data)
    % 支援路徑或直接傳入影像資料
    if ischar(image_path_or_data) || isstring(image_path_or_data)
        img = imread(image_path_or_data);
    else
        img = image_path_or_data;
    end
    
    % 轉為灰階
    if size(img, 3) == 3
        img_gray = rgb2gray(img);
    else
        img_gray = img;
    end
    
    % 分析影像特性（與 correct_lighting.m 保持一致）
    mean_intensity = mean(img_gray(:));
    std_intensity = std(double(img_gray(:)));
    
    % 判斷光線條件
    if mean_intensity < 80
        report = 'underexposed';
    elseif mean_intensity > 180
        report = 'overexposed';
    else
        % 進一步判斷是否需要對比度增強
        if std_intensity < 30  % 對比度較低
            report = 'low_contrast';
        else
            report = 'ok';
        end
    end
end