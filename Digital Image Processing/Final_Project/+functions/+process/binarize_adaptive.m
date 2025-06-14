function binary_image = binarize_adaptive(image_data)
    if size(image_data, 3) == 3, image_data = rgb2gray(image_data); end
    % [調校點]：sensitivity 和 strel 的大小是關鍵參數
    sensitivity = 0.6; 
    T = adaptthresh(image_data, sensitivity, 'ForegroundPolarity', 'dark', 'NeighborhoodSize', 2*floor(size(image_data,1)/16)+1, 'Statistic', 'gaussian');
    binary_image = imbinarize(image_data, T);
    
    
    % se_close = strel('square', 1.7); % 使用 disk 結構元素
    % binary_image = imopen(binary_image, se_close);

    % se_close_disk = strel('disk', 1); % 使用 disk 結構元素
    % binary_image = imopen(binary_image, se_close_disk);


    se_close = strel('square', 2); % 使用 disk 結構元素
    binary_image = imclose(binary_image, se_close);

    % se_open = strel('disk', 1); % 使用 disk 結構元素
    % binary_image = imopen(binary_image, se_open);

    se_open_2 = strel('disk', 1); % 使用 disk 結構元素
    binary_image = imopen(binary_image, se_open_2);


    

    se_open_3 = strel('square', 1); % 使用 disk 結構元素
    binary_image = imclose(binary_image, se_open_3);

    se_open_4 = strel('disk', 1); % 使用 disk 結構元素
    binary_image = imopen(binary_image, se_open_4);
    


    % se = strel('square', 4); % 使用 disk 結構元素
    % binary_image = imclose(~binary_image, se);
    binary_image = ~binary_image;

    binary_image = imdilate(binary_image, strel('square', 1)); % 擴張處理，增強邊緣

    


end