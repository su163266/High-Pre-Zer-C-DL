%   这个脚本主要把实验上拍得的780*780裁剪，重置为256*256,写入bin文件
% 设置文件夹路径
close all ;
clear;clc;
% bin文件写入路径
fileFolderPath = 'D:\Program Files\MATLAB\R2022a\mat_script\optical\real_data_process\xy\';

sourceFolder1 = 'D:\Program Files\MATLAB\R2022a\mat_script\optical\real_data_process\real_data_dx';
img_dx_0 = double(imread('diff_dx_0.png'));
sourceFolder2 = 'D:\Program Files\MATLAB\R2022a\mat_script\optical\real_data_process\real_data_dy';
img_dy_0 = double(imread('diff_dy_0.png'));

N = 256;
for k = 1:19400
    sourceFile1 = fullfile(sourceFolder1, ['diff_dx_', num2str(k), '.png']);
    img_dx = double(imread(sourceFile1));
    img_dx = img_dx - img_dx_0; %实验图减去背景噪声
    img_dx = img_dx(25:770,22:750);
    img_dx = imresize(img_dx,[N,N]);

    sourceFile2 = fullfile(sourceFolder2, ['diff_dy_', num2str(k), '.png']);
    img_dy = double(imread(sourceFile2));
    img_dy = img_dy - img_dy_0; %实验图减去背景噪声
    img_dy = img_dy(25:774,38:765);
    img_dy = imresize(img_dy,[N,N]);
    disp(k);

    Ixy = zeros(N,N,2);
    Ixy(:,:,1) = img_dx;  Ixy(:,:,2) = img_dy;

    ext = '.bin';
    prefix2 = 'train33-xy_';
    destinationFile = fullfile(fileFolderPath, [prefix2, num2str(k), ext]);
    fileID = fopen(destinationFile, 'wb');
    if fileID == -1
        disp(['Failed to open file for writing: ', destinationFile]);
    else
        fwrite(fileID, Ixy, 'float');
        fclose(fileID);
    end

    % imshow(img_dx,[])
    % 保存处理后的图像
    % outputFileName = fullfile(outputFolder1, ['re_dx_', num2str(k), '.png']);
    % imwrite(uint8(img_dx), outputFileName); % 保存为 uint8 格式
end   