function corrected_image = remove_noise(image_data, noise_type)
    if size(image_data, 3) == 3, image_data = rgb2gray(image_data); end
    switch noise_type
        case 'salt-and-pepper', corrected_image = medfilt2(image_data, [3 3]);
        case 'gaussian', corrected_image = wiener2(image_data, [5 5]);
        otherwise, corrected_image = image_data;
    end
end