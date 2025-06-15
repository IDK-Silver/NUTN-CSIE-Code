function report = diagnose_noise(image_path_or_data)
    if ischar(image_path_or_data) || isstring(image_path_or_data)
        img = imread(image_path_or_data);
    else
        img = image_path_or_data;
    end
    
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    % 確保為 uint8
    if ~isa(img, 'uint8')
        img = im2uint8(img);
    end
    
    % 椒鹽雜訊檢測：關鍵是找出孤立的異常像素
    % 使用中值濾波來識別雜訊
    img_median = medfilt2(img, [3 3]);
    
    % 計算原影像與中值濾波的差異
    diff_img = abs(double(img) - double(img_median));
    
    % 找出差異很大的像素（可能是雜訊）
    noise_threshold = 50;  % 差異閾值
    potential_noise = diff_img > noise_threshold;
    
    % 椒鹽雜訊通常是極值像素且與鄰域差異很大
    salt_pixels = potential_noise & (img >= 240);  % 鹽雜訊（白點）
    pepper_pixels = potential_noise & (img <= 15); % 胡椒雜訊（黑點）
    
    salt_count = sum(salt_pixels(:));
    pepper_count = sum(pepper_pixels(:));
    total_noise = salt_count + pepper_count;
    noise_ratio = total_noise / numel(img);
    
    % 除錯輸出
    fprintf('Min: %.0f, Max: %.0f\n', min(img(:)), max(img(:)));
    fprintf('Salt pixels: %d, Pepper pixels: %d, Total noise: %d\n', ...
        salt_count, pepper_count, total_noise);
    fprintf('Noise ratio: %.6f\n', noise_ratio);
    
    % 更合理的閾值：真正的椒鹽雜訊比例通常很小
    if noise_ratio > 0.0004  % 0.01% 的雜訊像素
        report = 'salt-and-pepper';
    else
        report = 'none';
    end
end