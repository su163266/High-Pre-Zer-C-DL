% 将f拟合为j阶zernike多项式
% 输入函数f:二维矩阵 ,[N,N] ,double;  拟合阶数:j ,int
% 输出拟合后的多项式 c[j,1]
function c = fit_for_f(f,N,j)
z = zeros(N,N,j);
for i = 1:j
   z(:,:,i) = single_zer(i,1,N);
end
z_re = reshape(z,N*N,j);
f_re = reshape(f,N*N,1);
%%拟合后的 系数矩阵c
c = z_re\f_re; % '\' 运算表示 inv（z_re）*f_re 
end

