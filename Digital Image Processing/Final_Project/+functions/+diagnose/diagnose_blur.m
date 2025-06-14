function report = diagnose_blur(image_path_or_data)
    if ischar(image_path_or_data) || isstring(image_path_or_data), img = imread(image_path_or_data); else, img = image_path_or_data; end
    if size(img, 3) == 3, img = rgb2gray(img); end
    laplacian_matrix = imfilter(double(img), fspecial('laplacian'));
    variance = var(laplacian_matrix(:));
    
    % [調校點]：這是一個經驗閾值，請根據您的實驗結果進行調整
    blur_threshold = 100; 
    
    % fprintf('Image blur variance: %f\n', variance); % 可取消此行註解來觀察數值
    
    if variance < blur_threshold, report = true;
    else, report = false;
    end
end