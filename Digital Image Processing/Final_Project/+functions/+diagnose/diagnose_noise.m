function report = diagnose_noise(image_path_or_data)
    if ischar(image_path_or_data) || isstring(image_path_or_data), img = imread(image_path_or_data); else, img = image_path_or_data; end
    if size(img, 3) == 3, img = rgb2gray(img); end
    ratio = sum(img(:) == 0 | img(:) == 255) / numel(img);
    if ratio > 0.001, report = 'salt-and-pepper';
    else, report = 'none';
    end
end