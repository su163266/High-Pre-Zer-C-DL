%   这个脚本主要把实验上拍得的780*780裁剪，重置为256*256
% 设置文件夹路径
close all ;
clear;clc;
sourceFolder1 = 'D:\Program Files\MATLAB\R2022a\mat_script\optical\real_data_process\real_data_dx';
outputFolder = 'D:\Program Files\MATLAB\R2022a\mat_script\optical\real_data_process\re_dx_256';
img_dx_0 = double(imread('diff_dx_0.png'));

for k = 1:20000
    sourceFile1 = fullfile(sourceFolder1, ['diff_dx_', num2str(k), '.png']);
    img_dx = double(imread(sourceFile1));
    img_dx = img_dx - img_dx_0; %实验图减去背景噪声
    img_dx = img_dx(25:770,22:750);
    img_dx = imresize(img_dx,[256,256]);
    disp(k);
    % imshow(img_dx,[])
    % 保存处理后的图像
        outputFileName = fullfile(outputFolder, ['re_dx_', num2str(k), '.png']);
        imwrite(uint8(img_dx), outputFileName); % 保存为 uint8 格式
end     