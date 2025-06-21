import torch
import torch.nn as nn
import os
from net_new import ResNet
# from Net256 import ResNet
from scipy.io import loadmat, savemat
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 设置设备（GPU 如果可用，否则使用 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 假设这是你的模型结构类（需要与训练时的模型结构相同）
model = ResNet().to(device)

# 加载模型权重
model.load_state_dict(torch.load('real_data_model_net_new_2.pth', map_location=device, weights_only=True))

# 读取标签文件
mat_file_path = 'ZerTest1.mat'
Zer_co = loadmat(mat_file_path)
test_labels = Zer_co['test_co']  # (25, 3500)
test_labels = test_labels.T  # 转换为 (3500, 25)
# test_labels = test_labels[0:400,]
print(test_labels.shape)

# 将 numpy 数组转换为 PyTorch 张量
label_tensor = torch.tensor(test_labels, dtype=torch.float32)

# 定义自定义数据集
class TestDataset(Dataset):
    def __init__(self, data_folder: str, labels: torch.Tensor):
        self.data_folder = data_folder
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file_name = f'test-xy_{idx+1}.bin'
        file_path = os.path.join(self.data_folder, file_name)
        # 使用 np.memmap 读取二进制文件
        data = np.memmap(file_path, dtype=np.float32, mode='r', shape=(2, 256, 256))
        # 将数据转换为 Tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)
        # 获取对应的标签
        label = self.labels[idx]
        return data_tensor, label

# 创建测试数据集
test_dataset = TestDataset(data_folder='D:/Program Files/MATLAB/R2022a/mat_script/optical/quantified_phase/test', labels=label_tensor)
#test_dataset = TestDataset(data_folder='D:/Program Files/MATLAB/R2022a/mat_script/optical/quantified_phase/test_dirty', labels=label_tensor)
Testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
print(f"Testloader: {len(Testloader)} samples")
# 评估测试集
criterion = nn.MSELoss()
def testdate(Testloader):
    model.eval()  # 切换到评估模式
    predictions = []  # 用于存储所有预测结果
    total_loss = 0.0  # 用于累计损失
    total_samples = 0  # 样本总数

    with torch.no_grad():  # 禁用梯度计算
        for j, (ft_images, rec) in tqdm(enumerate(Testloader), total=len(Testloader)):
            ft_images = ft_images.to(device)
            rec = rec.to(device)

            # 预测结果
            pred_img = model(ft_images)

            # 计算当前批次的损失
            test_loss_rec = criterion(pred_img, rec)
            total_loss += test_loss_rec.item()  # 累加损失

            # 更新样本数量
            total_samples += ft_images.size(0)  # 每个批次的样本数（这里是 1）

            # 将预测结果转换为 numpy 格式并存储
            predictions.append(pred_img.cpu().numpy())  # 将数据从 GPU 转移到 CPU，并转成 numpy 数组

    # 将预测结果转换为 numpy 数组，并调整形状
    predictions = np.concatenate(predictions, axis=0)  # (1000, 25)

    # 计算平均 MSE 损失
    avg_loss = total_loss / total_samples
    print(f"Average MSE Loss: {avg_loss:.4f}")

    return predictions, avg_loss


# 测试并保存预测结果
predictions, avg_loss = testdate(Testloader)

# 保存预测结果到 .mat 文件
savemat('Test_pre.mat', {'predictions': predictions})
print("Predictions saved to 'predictions.mat'.")

# 打印平均损失
print(f"Test MSE Loss: {avg_loss:.6f}")
# # 评估测试集
# def testdate(Testloader):
#     model.eval()
#     predictions = []  # 用于存储所有预测结果
#
#     # 禁用梯度计算，节省内存
#     with torch.no_grad():
#         for j, (ft_images, _) in tqdm(enumerate(Testloader), total=len(Testloader)):
#             ft_images = ft_images.to(device)
#             pred_img = model(ft_images)
#             predictions.append(pred_img.cpu().numpy())  # 将预测结果保存到 predictions 列表
#
#     # 将预测结果转换为 numpy 数组并调整形状
#     predictions = np.concatenate(predictions, axis=0)  # (1000, 25)
#     return predictions

# # 获取测试集的预测结果
# predictions = testdate(Testloader)
#
# # 保存预测结果为 .mat 文件
# savemat('predictions.mat', {'predictions': predictions})
#
# print("Predictions saved to 'predictions.mat'.")

