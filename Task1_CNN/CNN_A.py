# -*- coding: utf-8 -*-
# @Time : 2021/6/7 20:21
# @Author : hangzhouwh
# @Email: hangzhouwh@gmail.com
# @File : CNN_Q.py
# @Software: PyCharm


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
from torchsummary import summary

import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 1e-4
keep_prob_rate = 0.7
max_epoch = 3
BATCH_SIZE = 50

# 如果通过代码直接下载数据集用这段代码
# mnist_path = './mnist/'
# DOWNLOAD_MNIST = False
# if not(os.path.exists(mnist_path)) or not os.listdir(mnist_path):
#     # not mnist dir or mnist is empyt dir
#     DOWNLOAD_MNIST = True

# 如果已经把数据集下载到本地了用这段代码，路径自己修改
mnist_path = 'D:/Workspace/Zucc_AI_Lab_dataset/mnist/'  # 数据集的路径

train_data = torchvision.datasets.MNIST(root=mnist_path, train=True, transform=torchvision.transforms.ToTensor(),
										download=True)  # download必须设置成True,不管之前是不是已经下载好数据集
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root=mnist_path, train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:500] / 255.
test_y = test_data.test_labels[:500].numpy()


class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Sequential(
			# convolution fuction
			#  params: 1 in channels; 32 out channels; patch 7 * 7; stride is 1;
			#  padding style is same(that means the convolution opration's input and output have the same size)
			nn.Conv2d(
				in_channels=1,
				out_channels=32,
				kernel_size=7,
				stride=1,
				padding=3,
			),  # shape(1, 28, 28) -> shape(32, 28, 28)
			nn.ReLU(),  # activation function
			nn.MaxPool2d(2),  # pooling operation, shape(32, 28, 28) -> shape(32, 14, 14)
		)

		self.conv2 = nn.Sequential(
			# convolution function
			#  params: 32 in channels; 64 out channels; patch 5*5; stride is 1; padding style is same)
			nn.Conv2d(
				in_channels=32,
				out_channels=64,
				kernel_size=5,
				stride=1,
				padding=2,
			),  # shape(32, 14, 14) -> shape(64, 14, 14)
			nn.ReLU(),  # activation function
			nn.MaxPool2d(2),  # pooling operation, shape(64, 14, 14) -> shape(64, 7, 7)
		)

		# full connection layer: nn.Linear(in_features, out_features, bias=True)
		self.out1 = nn.Linear(64 * 7 * 7, 1024, bias=True)  # fc1
		self.dropout = nn.Dropout(keep_prob_rate)
		self.out2 = nn.Linear(1024, 10, bias=True)  # fc2

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size ,64 * 7 * 7)
		out1 = self.out1(x)
		out1 = F.relu(out1)
		out1 = self.dropout(out1)
		out2 = self.out2(out1)
		output = F.softmax(out2)
		return output


def test(cnn):
	global prediction
	y_pre = cnn(test_x)
	_, pre_index = torch.max(y_pre, 1)
	pre_index = pre_index.view(-1)
	prediction = pre_index.data.numpy()
	correct = np.sum(prediction == test_y)
	return correct / 500.0


def train(cnn):
	optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
	loss_func = nn.CrossEntropyLoss()
	for epoch in range(max_epoch):
		for step, (x_, y_) in enumerate(train_loader):
			x, y = Variable(x_), Variable(y_)
			output = cnn(x)
			loss = loss_func(output, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if step != 0 and step % 20 == 0:
				print("=" * 10, step, "=" * 5, "=" * 5, "test accuracy is ", test(cnn), "=" * 10)


if __name__ == '__main__':
	cnn = CNN()

	# print the structure of model
	print(cnn)

	# print the shape of output of each layer
	print(summary(cnn, (1, 28, 28)))

	# print params of model
	# for name, parameter in cnn.named_parameters():
	# 	print(name, ':', parameter.size())
	#
	# train(cnn)
