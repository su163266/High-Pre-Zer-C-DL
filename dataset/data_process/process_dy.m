%   这个脚本主要把实验上拍得的780*780裁剪，重置为256*256
% 设置文件夹路径
close all ;
clear;clc;
sourceFolder2 = 'D:\Program Files\MATLAB\R2022a\mat_script\optical\real_data_process\real_data_dy';
outputFolder2 = 'D:\Program Files\MATLAB\R2022a\mat_script\optical\real_data_process\re_dy_256';
img_dy_0 = double(imread('diff_dy_0.png'));

for k = 1:20000
    sourceFile2 = fullfile(sourceFolder2, ['diff_dy_', num2str(k), '.png']);
    img_dy = double(imread(sourceFile2));
    img_dy = img_dy - img_dy_0; %实验图减去背景噪声
    img_dy = img_dy(25:774,38:765);
    img_dy = imresize(img_dy,[256,256]);
    disp(k);
%     imshow(img_dy,[])
    % 保存处理后的图像
    outputFileName = fullfile(outputFolder2, ['re_dy_', num2str(k), '.png']);
    imwrite(uint8(img_dy), outputFileName); % 保存为 uint8 格式
end