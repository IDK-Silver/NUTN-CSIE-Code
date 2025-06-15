function binary_image = binarize_adaptive(image_data)
    % 確保輸入為灰階且為 uint8 類型
    if size(image_data, 3) == 3
        image_data = rgb2gray(image_data);
    end
    
    % 確保為 uint8 類型
    if ~isa(image_data, 'uint8')
        image_data = im2uint8(image_data);
    end
    
    % [調校點]：sensitivity 和 strel 的大小是關鍵參數
    sensitivity = 0.6; 
    T = adaptthresh(image_data, sensitivity, 'ForegroundPolarity', 'dark', ...
        'NeighborhoodSize', 2*floor(size(image_data,1)/16)+1, 'Statistic', 'gaussian');
    binary_image = imbinarize(image_data, T);
    
    % 形態學運算序列
    % 第一步：閉合運算 - 連接文字筆劃
    se_close = strel('square', 2);
    binary_image = imclose(binary_image, se_close);
    
    % 第二步：開啟運算 - 去除小雜點
    se_open = strel('disk', 1);
    binary_image = imopen(binary_image, se_open);
    
    % 第三步：再次閉合 - 強化文字結構
    se_close_2 = strel('square', 1);
    binary_image = imclose(binary_image, se_close_2);
    
    % 第四步：最終開啟 - 精細化處理
    se_open_2 = strel('disk', 1);
    binary_image = imopen(binary_image, se_open_2);
    
    % 反相處理：使文字為白色，背景為黑色
    binary_image = ~binary_image;
    
    % 擴張處理：增強文字邊緣
    se_dilate = strel('square', 1);
    binary_image = imdilate(binary_image, se_dilate);
    
    % 最終輸出：轉換為 uint8 格式 (0 和 255)
    if islogical(binary_image)
        binary_image = ~uint8(binary_image) * 255;
    elseif ~isa(binary_image, 'uint8')
        binary_image = ~im2uint8(binary_image);
    end
end