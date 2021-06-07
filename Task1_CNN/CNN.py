# -*- coding: utf-8 -*-
# @Time : 2021/6/7 20:21
# @Author : hangzhouwh
# @Email: hangzhouwh@gmail.com
# @File : CNN.py
# @Software: PyCharm


import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 选择模型训练再CPU还是GPU上进行 （GPU可以极大加快训练速度，但安装环境时需要安装对应的GPU版本）
learning_rate = 1e-4 # 学习速率
keep_prob_rate = 0.7 # probability of an element to be zeroed
max_epoch = 3 # 最大训练轮次
BATCH_SIZE = 50 # 每一个batch的样本数


# 如果通过代码直接下载数据集用这段代码
# mnist_path = './mnist/'
# DOWNLOAD_MNIST = False
# if not(os.path.exists(mnist_path)) or not os.listdir(mnist_path):
#     # not mnist dir or mnist is empyt dir
#     DOWNLOAD_MNIST = True

# 如果已经把数据集下载到本地了用这段代码，路径自己修改
mnist_path = 'D:/Workspace/Zucc_AI_Lab_dataset/mnist/'  # 数据集的路径

# ------------------------读取数据集------------------------ #
train_data = torchvision.datasets.MNIST(root=mnist_path, train=True, transform=torchvision.transforms.ToTensor(),
										download=True)  # download必须设置成True,不管之前是不是已经下载好数据集
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root=mnist_path, train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:500] / 255.
test_y = test_data.test_labels[:500].numpy() # 为简化，测试集只取前500个
# --------------------------------------------------------- #

# --------------------定义CNN模型结构----------------------- #
class CNN(nn.Module):
	def __init__(self): # 此处定义模型各层结构
		super(CNN, self).__init__()
		self.conv1 = nn.Sequential(
			# TODO(convolution fuction,
			#  params: 1 in channels; 32 out channels; patch 7 * 7; stride is 1;
			#  padding style is same(that means the convolution opration's input and output have the same size)
			# shape(1, 28, 28) -> shape(32, 28, 28)
			nn.Conv2d(
				in_channels=,
				out_channels=,
				kernel_size=,
				stride=,
				padding=,
			),
			nn.ReLU(),  # activation function
			nn.MaxPool2d(2),  # pooling operation, shape(32, 28, 28) -> shape(32, 14, 14)
		)

		self.conv2 = nn.Sequential(
			# TODO(convolution function,
			#  params: 32 in channels; 64 out channels; patch 5*5; stride is 1; padding style is same)
			# shape(32, 28, 28) -> shape(64, 14, 14)


			# TODO(choosing your activation function)


			# TODO(choosing your pooling operation function)
			# shape(64, 14, 14) -> shape(64, 7, 7)

		)

		# full connection layer: nn.Linear(in_features, out_features, bias=True)
		self.out1 = nn.Linear(64 * 7 * 7, 1024, bias=True)  # fc1
		self.dropout = nn.Dropout(keep_prob_rate)
		self.out2 = nn.Linear(1024, 10, bias=True)  # fc2

	def forward(self, x): # 此处数据根据定义的结构进行前向传播，return为模型输出
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view() # TODO(flatten the output of conv2 to (batch_size ,64 * 7 * 7))
		out1 = self.out1(x)
		out1 = F.relu(out1)
		out1 = self.dropout(out1)
		out2 = self.out2(out1)
		output = F.softmax(out2)
		return output
# --------------------------------------------------------- #


# -------------计算CNN模型在测试集上的分类准确率-------------- #
def test(cnn):
	global prediction
	y_pre = cnn(test_x) # 输出对不同标签的预测概率
	_, pre_index = torch.max(y_pre, 1) # 找到概率最大的标签
	pre_index = pre_index.view(-1)
	prediction = pre_index.data.numpy()
	correct = np.sum(prediction == test_y) # 计算预测值与真实值匹配的比例 即准确率
	return correct / 500.0
# --------------------------------------------------------- #

# ---------------------训练模型----------------------------- #
def train(cnn):
	optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate) # 优化器 此处采用Adam
	loss_func = nn.CrossEntropyLoss() # 损失函数 多分类问题选择交叉熵损失
	for epoch in range(max_epoch): # 共训练 max_epoch 轮
		for step, (x_, y_) in enumerate(train_loader):
			x, y = Variable(x_), Variable(y_)
			output = cnn(x) # 通过网络对输入x作出预测
			loss = loss_func(output, y) # 计算预测值与真实值的误差
			# 通过误差来更新模型权重 以达到训练模型的目的
			optimizer.zero_grad() # 清空梯度
			loss.backward() # 反向传播
			optimizer.step() # 更新权重
			# 每隔一定轮次输出模型在测试集上的准确率
			if step != 0 and step % 20 == 0:
				print("=" * 10, step, "=" * 5, "=" * 5, "test accuracy is ", test(cnn), "=" * 10)
# --------------------------------------------------------- #


if __name__ == '__main__':
	cnn = CNN()

	# print the structure of model
	print(cnn)

	# print params of model
	for name, parameter in cnn.named_parameters():
		print(name, ':', parameter.size())

	train(cnn)
