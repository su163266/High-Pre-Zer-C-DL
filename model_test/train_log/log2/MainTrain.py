import os
import csv
import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from net_new import ResNet

# =========================
# ———— 全局配置部分 ————
# =========================

# 定义参数
EPOCHS = 150             # 总训练轮数
NGPUS = 1                  # 使用的 GPU 数量
BATCH_SIZE = 48            # 每个批次的数据量
LR = 1e-5          # 学习率

# 输出目录
output_dir = 'train_log'
os.makedirs(output_dir, exist_ok=True)

# 日志文件路径
log_file = os.path.join(output_dir, 'train_2(150).log')

# 配置 logging
logger = logging.getLogger('WavefrontReconstruction')
logger.setLevel(logging.INFO)

# 日志格式
formatter = logging.Formatter(
    fmt='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 控制台 Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 文件 Handler
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info(f"GPUs: {NGPUS}, Batch size: {BATCH_SIZE}, Learning rate: {LR}")

# =========================
# ———— 数据准备部分 ————
# =========================

# 训练、测试、验证数据的数量
NLINES = 18000   # 训练数据的行数
NLTEST = 0       # 测试数据的行数（目前未使用）
N_TRAIN = NLINES
N_VALID = 3000   # 验证数据的数量

# 读取标签文件
mat_file_path = 'Label.mat'
Zer_co = loadmat(mat_file_path)
labels = Zer_co['label_co']  # (25, 3500)
labels = labels.T             # 转换为 (3500, 25)
labels = labels[0:NLINES,]    # 只取前 NLINES 行

# 将 numpy 数组转换为 PyTorch 张量
label_tensor = torch.tensor(labels, dtype=torch.float32)

class MyDataset(Dataset):
    def __init__(self, data_folder: str, labels: torch.Tensor, transform=None):
        self.data_folder = data_folder
        self.labels = labels
        self.transform = transform
        self.num_samples = NLINES  # 样本数

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        file_name = f'train-xy_{idx+1}.bin'
        file_path = os.path.join(self.data_folder, file_name)

        # 使用 np.memmap 读取二进制文件
        data = np.memmap(file_path, dtype=np.float32, mode='r', shape=(2, 256, 256))
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # 获取对应的标签
        label = self.labels[idx]

        # 数据增强
        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, label

# 定义数据增强操作
data_transforms = transforms.Compose([
    transforms.RandomRotation(25),  # 随机旋转 ±25 度
])

# 创建训练和验证数据集
train_dataset = MyDataset(data_folder='D:/ls/xy/', labels=label_tensor, transform=data_transforms)

# 将数据集分成训练集和验证集
train_data, valid_data = torch.utils.data.random_split(train_dataset, [N_TRAIN - N_VALID, N_VALID])

# 创建 DataLoader
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
validloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

logger.info(f"Trainloader: {len(trainloader.dataset)} samples, Validloader: {len(validloader.dataset)} samples")

# 检查 GPU 可用性
if torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("Using CUDA")
else:
    device = torch.device("cpu")
    logger.info("Using CPU")

# =========================
# ———— 模型与优化器部分 ————
# =========================

# 初始化模型
model = ResNet().to(device)

# 计算每个 epoch 的迭代次数，用于 CyclicLR 的步长设置
iterations_per_epoch = np.floor((N_TRAIN - N_VALID) / BATCH_SIZE) + 1
step_size = 3 * iterations_per_epoch  # 在 3 个 epoch 内完成一个上升/下降循环
logger.info(f"LR step size (iterations): {step_size} ≈ every {step_size/iterations_per_epoch:.1f} epochs")

# 损失函数与优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=LR / 10,
    max_lr=LR,
    step_size_up=step_size,
    cycle_momentum=False,
    mode='triangular2'
)

def update_saved_model(model, path):
    """保存模型到指定路径"""
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    save_path = os.path.join(path, 'real_data_model_net_new_2.pth')
    torch.save(state_dict, save_path)
    logger.info(f"Model saved to: {save_path}")

# =========================
# ———— 训练与验证函数 ————
# =========================

def train_one_epoch(trainloader, metrics, epoch_idx):
    """单个 epoch 的训练过程"""
    model.train()
    tot_loss = 0.0
    tot_rec_loss = 0.0

    for i, (ft_images, rec) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch_idx+1} [Train]")):
        ft_images = ft_images.to(device)
        rec = rec.to(device)

        pred_img = model(ft_images)  # 前向传递
        loss_rec = criterion(pred_img, rec)
        loss = loss_rec

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
        tot_rec_loss += loss_rec.item()

        # 更新学习率并记录
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        metrics['lrs'].append(current_lr)

    avg_loss = tot_loss / len(trainloader)
    avg_rec_loss = tot_rec_loss / len(trainloader)
    metrics['losses'].append([avg_loss, avg_rec_loss])

    logger.info(f"Epoch {epoch_idx+1} [Train] - Avg Total Loss: {avg_loss:.6f}, Avg Rec Loss: {avg_rec_loss:.6f}, LR: {current_lr:.6e}")

def validate_one_epoch(validloader, metrics, epoch_idx):
    """单个 epoch 的验证过程"""
    model.eval()
    tot_val_loss = 0.0
    tot_val_rec = 0.0

    with torch.no_grad():
        for j, (ft_images, rec) in enumerate(tqdm(validloader, desc=f"Epoch {epoch_idx+1} [Valid]")):
            ft_images = ft_images.to(device)
            rec = rec.to(device)

            pred_img = model(ft_images)
            val_rec_loss = criterion(pred_img, rec)
            val_loss = val_rec_loss

            tot_val_loss += val_loss.item()
            tot_val_rec += val_rec_loss.item()

    avg_val_loss = tot_val_loss / (j + 1)
    avg_val_rec = tot_val_rec / (j + 1)
    metrics['val_losses'].append([avg_val_loss, avg_val_rec])

    logger.info(f"Epoch {epoch_idx+1} [Valid] - Avg Total Loss: {avg_val_loss:.6f}, Avg Rec Loss: {avg_val_rec:.6f}")

    # 如果验证损失降低，则保存模型
    if avg_val_loss < metrics['best_val_loss']:
        logger.info(f"Validation loss improved from {metrics['best_val_loss']:.6f} to {avg_val_loss:.6f}. Saving model...")
        metrics['best_val_loss'] = avg_val_loss
        update_saved_model(model, output_dir)

# =========================
# ———— 主训练流程 ————
# =========================
if os.path.exists('real_data_model_net_new_1.pth'):
    model.load_state_dict(torch.load('real_data_model_net_new_1.pth'))
    # print("Loaded saved model")
if __name__ == '__main__':
    # 打开 CSV 文件并写入表头
    csv_path = 'loss_values_net_new_2.csv'
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train_Total_Loss', 'Train_Rec_Loss', 'Val_Total_Loss', 'Val_Rec_Loss'])
    logger.info(f"Initialized CSV for logging losses: {csv_path}")

    metrics = {
        'losses': [],        # 每个 epoch 的 [train_total_loss, train_rec_loss]
        'val_losses': [],    # 每个 epoch 的 [val_total_loss, val_rec_loss]
        'lrs': [],           # 每个 iteration 的学习率
        'best_val_loss': np.inf
    }

    # 打开交互式绘图模式（可选）
    plt.ion()
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    # 开始训练循环
    for epoch in range(EPOCHS):
        # 训练并记录
        train_one_epoch(trainloader, metrics, epoch)

        # 验证并记录
        validate_one_epoch(validloader, metrics, epoch)

        # 将该 Epoch 的损失追加到 CSV
        train_loss, train_rec = metrics['losses'][-1]
        val_loss, val_rec = metrics['val_losses'][-1]
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, train_rec, val_loss, val_rec])

        logger.info(f"Epoch {epoch+1} summary => Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # 绘制并保存整体损失曲线
    epochs = range(1, EPOCHS + 1)
    ax1.clear()
    ax1.plot(epochs, [x[0] for x in metrics['losses']], 'b-', label='Train Loss')
    ax1.plot(epochs, [x[0] for x in metrics['val_losses']], 'r-', label='Val Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    plt.show()

    curve_path = os.path.join(output_dir, 'loss_curve.png')
    fig.savefig(curve_path)
    logger.info(f"Saved loss curve figure to: {curve_path}")
