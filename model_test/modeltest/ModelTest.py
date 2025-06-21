import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.ndimage import zoom
import os
from net_new import ResNet
from scipy.io import loadmat
from scipy.io import savemat

# 创建模型实例
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ResNet().to(device)
# 加载模型权重
model.load_state_dict(torch.load('real_data_model_net_new_2.pth', map_location=device,weights_only=True))
test_line = 1
# mat_file_path = 'Zer_co.mat'
mat_file_path = 'ZerTest1.mat'
Zer_co = loadmat(mat_file_path)
labels = Zer_co['test_co']  #  (25, 3500)
# 转换形状
labels = labels.T  # 转换为 (3500, 25)
labels = labels[test_line-1:test_line,:]
# 输入数据
# folder_path = 'D:/Program Files/MATLAB/R2022a/mat_script/optical/quantified_phase/xy'
folder_path = 'D:/Program Files/MATLAB/R2022a/mat_script/optical/quantified_phase/test'
file_name = f'test-xy_{test_line}.bin'
file_path = os.path.join(folder_path, file_name)
# 使用 np.memmap 读取二进制文件
data = np.memmap(file_path, dtype=np.float32, mode='r', shape=(2, 256, 256))
tensor_data = torch.tensor(data, dtype=torch.float32)
tensor_data = tensor_data.unsqueeze(0)  # 转化(1, 2, 512, 512)
Input = tensor_data.to(device)
# 进行预测
model.eval()
with torch.no_grad():
    pre = model(Input)
pre_np = pre.squeeze().cpu().numpy()  # 转换为 numpy 数组并去掉多余的维度
label_np = labels.squeeze()  # labels 已经是 numpy 数组，直接去掉多余的维度
#   写入mat文件，方便matlab绘图
pre_mat = pre.cpu().detach().numpy()  # 转换为 NumPy 数组
# 创建字典以保存数据
mat_data = {'pre': pre_mat}
# 保存为 .mat 文件
savemat('prediction.mat', mat_data)

# 绘制
plt.figure(figsize=(10, 5))
plt.plot(pre_np[:], label='Prediction', marker='o', linestyle='-', color='blue')
plt.plot(label_np[:], label='Label', marker='x', linestyle='--', color='red')
plt.title('Model Prediction vs Actual Label')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()






