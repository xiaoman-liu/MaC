import torch
import torch.nn as nn

# 定义输入数据
batch_size = 10
features = 3
length = 1

# 创建一个形状为 (batch_size, features, length) 的随机输入张量
input_data = torch.randn(batch_size, features, length)

# 定义一个一维卷积层
conv1d = nn.Conv1d(in_channels=features, out_channels=16, kernel_size=1)

# 进行卷积运算
output_data = conv1d(input_data)

# 输出结果的形状
print(output_data.shape)
