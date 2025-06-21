%% generate phase fuction
clear;clc;
n=25;
N = 256; 
tol = 10000;
label_co = zeros(25,tol);
mu = 0; % 均值
sigma = 0.3; % 标准差

for k = 1:tol
    %Normal distribution random number
  c1 = mu + sigma * randn(1, n); % 正态分布

    Z_all1 = zeros(size(N,N));
    for i = 1:n
        Z_all1 = Z_all1 + c1(i)*single_zer(i,1,N);
    end
    Z_min = min(Z_all1(:)); 
    Z_max = max(Z_all1(:));
    Z_all1 = (Z_all1 - Z_min) / (Z_max - Z_min); % 归一化公式
    Z_all1 = Z_all1 * (2 * pi)-pi;
    fit1 = fit_for_f(Z_all1,N,n);
    label_co(:,2*k-1) = fit1';

    % Uniformly distributed random numbers
    c2= -1 + 2 * rand(1, n); % 均匀分布在 [-1, 1]
    Z_all2 = zeros(size(N,N));
    for j = 1:n
        Z_all2 = Z_all2 + c2(j)*single_zer(j,1,N);
    end
    Z_min = min(Z_all2(:)); % find max
    Z_max = max(Z_all2(:)); 
    Z_all2 = (Z_all2 - Z_min) / (Z_max - Z_min); 
    Z_all2 = Z_all2 * (2 * pi)-pi;

    fit2 = fit_for_f(Z_all2,N,n);
    label_co(:,2*k) = fit2';
    
disp(k)

end
save('label_25.mat','label_co','-mat')
