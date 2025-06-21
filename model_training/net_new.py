import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def dense_block(input_channels, out_channels, num_residuals,inplace=True):
    layers = []
    for i in range( num_residuals):
        layers.append(nn.Conv2d(in_channels=input_channels, out_channels=out_channels, kernel_size=3, padding=1))
        # layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace))
    return nn.Sequential(*layers)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False, strides=2):
    layers = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            layers.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=strides))
        else:
            layers.append(Residual(num_channels, num_channels))
    return nn.Sequential(*layers)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1) #(8,256,256)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #(8,128,128)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=5, stride=1, padding=2) #(32,128,128)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.dense1 = dense_block(32, 32,3)

        self.block2 = resnet_block(32, 64, 2) #(64,64,64)
        self.dense2 = dense_block(64, 64,3)
        self.block3 = resnet_block(64, 128, 2) #(128,32,32)
        self.dense3 = dense_block(128, 128,2)
        self.block4 = resnet_block(128, 256, 2)# (256,16,16)
        self.dense4 = dense_block(256, 256,2)
        self.block5 = resnet_block(256, 512, 2)# (512,8,8)
        self.dense5 = dense_block(512, 512,2)
        self.block6 = resnet_block(512, 1024, 2) # (1024,4,4)
        self.dense6 = dense_block(1024, 1024,2,inplace=False)
        self.fc1 = nn.Linear(1024 * 4 * 4, 256 * 1 * 1)  # 从 1024*4*4 到 512*2*2
        #self.fc2 = nn.Linear(512 * 2 * 2, 256 * 1 * 1)  # 从 512*2*2 到 256*1*1
        self.dropout = nn.Dropout(p=0.4)  # 50% 的 dropout
        self.fc2 = nn.Linear(256, 25)  # 最终输出到 25


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        p1 = self.pool2(x)

        d1 = self.dense1(p1)
        b2 = self.block2(p1+d1)
        d2 = self.dense2(b2)
        b3 = self.block3(d2+b2)
        d3 = self.dense3(b3)
        b4 = self.block4(b3+d3)
        d4 = self.dense4(b4)
        b5 = self.block5(b4+d4)
        d5 = self.dense5(b5)
        b6 = self.block6(d5+b5)
        d6 = self.dense6(b6)

        # 1. 调整形状为 (1, 512, 2, 2)
        d6_flat = d6.view(d6.size(0), -1)  # 变成 (1, 1024 * 4 * 4)
        d6_fc1 = self.fc1(d6_flat)  # (1, 512 * 2 * 2)

        # 2. 调整形状为 (1, 256, 1, 1)
        # d6_fc2 = self.fc2(d6_fc1)  # (1, 256 * 1 * 1)
        # d6_fc2 = d6_fc2.view(d6.size(0), 256, 1, 1)  # (1, 256, 1, 1)

        # 3. 使用 Dropout
        d6_dropout = self.dropout(d6_fc1)  # 应用 dropout

        # 4. 最后输出到 (1, 25, 1, 1)
        output = self.fc2(d6_dropout.view(d6.size(0), -1))  # 展平并输出到 25
        # 5. 去掉多余的维度
        output = output.squeeze(1)  # 或者使用 output = output.view(d6.size(0), 25)

        return output


net = ResNet()
X = torch.rand(size=(1, 2, 256, 256))
output = net(X)
print("Output shape:", output.shape)