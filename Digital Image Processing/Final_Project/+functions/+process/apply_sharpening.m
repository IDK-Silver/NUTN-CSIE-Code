function corrected_image = apply_sharpening(image_data)
    if size(image_data, 3) == 3, image_data = rgb2gray(image_data); end
    corrected_image = imsharpen(image_data, 'Radius', 1.5, 'Amount', 5);
end