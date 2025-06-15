classdef config
% CONFIG 專案固定設定檔
% 包含學號、姓名、檔案命名格式等不常變動的資訊。
    properties (Constant)
        

        STUDENT_ID = 'S11159005';
        STUDENT_NAME = '黃毓峰';
        PROCESSED_IMAGE_FILENAME_FORMAT = '%s%s.png';
        SUPPORTED_IMAGE_FORMATS = {"*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff", '所有支援的圖片檔案 (*.png, *.jpg, etc.)'};
        OUTPUT_FOLDER_NAME = './';

        APP_TITLE = 'Yu Fun 古文獻影像強化與分析系統';
    end
end