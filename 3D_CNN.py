import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import PIL
import matplotlib.pyplot as plt
import tqdm
import time


class C3D(torch.nn.Module):
    def __init__(self, depth, frames, coefficients):
        super(C3D, self).__init__()
        self.depth, self.frames, self.coefficients = depth, frames, coefficients
        self.conv1_1 = torch.nn.Conv3d(1, 16, kernel_size=(3, 1, 5), stride=(1, 1, 1))
        self.PReLu1_1 = torch.nn.PReLU()
        self.conv1_2 = torch.nn.Conv3d(16, 16, kernel_size=(3, 9, 1), stride=(1, 2, 1))
        self.PReLu1_2 = torch.nn.PReLU()
        self.pool1 = torch.nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))
        self.conv2_1 = torch.nn.Conv3d(16, 32, kernel_size=(3, 1, 4), stride=(1, 1, 1))
        self.PReLu2_1 = torch.nn.PReLU()
        self.conv2_2 = torch.nn.Conv3d(32, 32, kernel_size=(3, 8, 1), stride=(1, 2, 1))
        self.PReLu2_2 = torch.nn.PReLU()
        self.pool2 = torch.nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))
        self.conv3_1 = torch.nn.Conv3d(32, 64, kernel_size=(3, 1, 3), stride=(1, 1, 1))
        self.PReLu3_1 = torch.nn.PReLU()
        self.conv3_2 = torch.nn.Conv3d(64, 64, kernel_size=(3, 7, 1), stride=(1, 1, 1))
        self.PReLu3_2 = torch.nn.PReLU()
        self.conv4_1 = torch.nn.Conv3d(64, 128, kernel_size=(3, 1, 3), stride=(1, 1, 1))
        self.PReLu4_1 = torch.nn.PReLU()
        self.conv4_2 = torch.nn.Conv3d(128, 128, kernel_size=(3, 7, 1), stride=(1, 1, 1))
        self.PReLu4_2 = torch.nn.PReLU()
        self.FC5 = torch.nn.Linear(4 * 3 * 3 * 128, 128)
        self.PReLu5 = torch.nn.PReLU()
        self.FC6 = torch.nn.Linear(128, 511)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.PReLu1_1(x)
        x = self.conv1_2(x)
        x = self.PReLu1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.PReLu2_1(x)
        x = self.conv2_2(x)
        x = self.PReLu2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.PReLu3_1(x)
        x = self.conv3_2(x)
        x = self.PReLu3_2(x)
        x = self.conv4_1(x)
        x = self.PReLu4_1(x)
        x = self.conv4_2(x)
        x = self.PReLu4_2(x)
        x = x.view(-1, 4 * 3 * 3 * 128)
        x = self.FC5(x)
        x = self.PReLu5(x)
        x = F.softmax(self.FC6(x))
        return x


if __name__ == '__main__':
    # CNN = SimpleCNN()
    # trainNet(CNN, batch_size=32, n_epochs=5, learning_rate=0.001)
    toy_data = torch.randn(1, 1, 20, 80, 40)
    toy = C3D()
    output = toy(toy_data)
    print('hi')
