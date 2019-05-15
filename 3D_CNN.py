import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import copy
import PIL
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


class C3D(torch.nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        self.conv1_1 = torch.nn.Conv3d(1, 16, kernel_size=(3, 1, 5), stride=(1, 1, 1))
        self.batch_norm1_1 = torch.nn.BatchNorm3d(num_features=16)
        self.PReLu1_1 = torch.nn.PReLU()
        self.conv1_2 = torch.nn.Conv3d(16, 16, kernel_size=(3, 9, 1), stride=(1, 2, 1))
        self.batch_norm1_2 = torch.nn.BatchNorm3d(num_features=16)
        self.PReLu1_2 = torch.nn.PReLU()
        self.pool1 = torch.nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))
        self.conv2_1 = torch.nn.Conv3d(16, 32, kernel_size=(3, 1, 4), stride=(1, 1, 1))
        self.batch_norm2_1 = torch.nn.BatchNorm3d(num_features=32)
        self.PReLu2_1 = torch.nn.PReLU()
        self.conv2_2 = torch.nn.Conv3d(32, 32, kernel_size=(3, 8, 1), stride=(1, 2, 1))
        self.batch_norm2_2 = torch.nn.BatchNorm3d(num_features=32)
        self.PReLu2_2 = torch.nn.PReLU()
        self.pool2 = torch.nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))
        self.conv3_1 = torch.nn.Conv3d(32, 64, kernel_size=(3, 1, 3), stride=(1, 1, 1))
        self.batch_norm3_1 = torch.nn.BatchNorm3d(num_features=64)
        self.PReLu3_1 = torch.nn.PReLU()
        self.conv3_2 = torch.nn.Conv3d(64, 64, kernel_size=(3, 7, 1), stride=(1, 1, 1))
        self.batch_norm3_2 = torch.nn.BatchNorm3d(num_features=64)
        self.PReLu3_2 = torch.nn.PReLU()
        self.conv4_1 = torch.nn.Conv3d(64, 128, kernel_size=(3, 1, 3), stride=(1, 1, 1))
        self.batch_norm4_1 = torch.nn.BatchNorm3d(num_features=128)
        self.PReLu4_1 = torch.nn.PReLU()
        self.conv4_2 = torch.nn.Conv3d(128, 128, kernel_size=(3, 7, 1), stride=(1, 1, 1))
        self.batch_norm4_2 = torch.nn.BatchNorm3d(num_features=128)
        self.PReLu4_2 = torch.nn.PReLU()
        self.FC5 = torch.nn.Linear(4 * 3 * 3 * 128, 128)
        self.batch_normFC5 = torch.nn.BatchNorm1d(num_features=128)
        self.PReLu5 = torch.nn.PReLU()
        self.FC6 = torch.nn.Linear(128, 621)  # 511

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.batch_norm1_1(x)
        x = self.PReLu1_1(x)
        x = self.conv1_2(x)
        x = self.batch_norm1_2(x)
        x = self.PReLu1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.batch_norm2_1(x)
        x = self.PReLu2_1(x)
        x = self.conv2_2(x)
        x = self.batch_norm2_2(x)
        x = self.PReLu2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.batch_norm3_1(x)
        x = self.PReLu3_1(x)
        x = self.conv3_2(x)
        x = self.batch_norm3_2(x)
        x = self.PReLu3_2(x)
        x = self.conv4_1(x)
        x = self.batch_norm4_1(x)
        x = self.PReLu4_1(x)
        x = self.conv4_2(x)
        x = self.batch_norm4_2(x)
        x = self.PReLu4_2(x)
        x = x.view(-1, 4 * 3 * 3 * 128)
        x = self.FC5(x)
        x = self.batch_normFC5(x)
        x = self.PReLu5(x)
        x = self.FC6(x)
        x = F.softmax(x, dim=1)
        return x


def LossAndOptimizer(learning_rate, model):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return loss, optimizer


def Fit(train_set, val_set, model, learning_rate=.01, n_epochs=10, batch_size=10):
    loss, optimizer = LossAndOptimizer(learning_rate, model)
    train_data, train_labels = train_set
    N = train_data.shape[0]
    epoch = 0
    train_loss = []
    val_loss = []
    while epoch < n_epochs:
        running_loss = 0.0
        print_every = batch_size // 10
        start_time = time.time()
        total_train_loss = 0
        for j in range(int(N / batch_size)):
            j_start = j * batch_size
            j_end = (j + 1) * batch_size
            inds = range(j_start, j_end)
            X = train_data[inds]
            y = train_labels[inds]
            y = torch.from_numpy(y)
            y = y.long()
            X, y = Variable(X), Variable(y)
            optimizer.zero_grad()
            outputs = model(X)
            loss_size = loss(outputs, y)
            loss_size.backward()
            optimizer.step()
            running_loss += loss_size.data
            # total_train_loss += loss_size.data
            # if (j + 1) % (print_every + 1) == 0:
            #     print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
            #         epoch + 1, int(100 * (j + 1) / batch_size), running_loss / print_every,
            #         time.time() - start_time))
            #     running_loss = 0.0
            #     start_time = time.time()
            #     total_val_loss = 0
        train_loss.append(float(running_loss) / (N / batch_size))
        print("train_loss", float(running_loss) / (N / batch_size), "time", time.time() - start_time)
        X_val, y_val = val_set
        y_val = torch.from_numpy(y_val)
        X_val, y_val = Variable(X_val), Variable(y_val)
        y_val = y_val.long()
        Val_outputs = model(X_val)
        val_loss_size = loss(Val_outputs, y_val)
        total_val_loss = val_loss_size.data
        val_loss.append(float(total_val_loss))
        print("val_loss", float(total_val_loss))
        print('epoch', epoch + 1)
        epoch += 1
    plt.plot(train_loss, label='train', color='b')
    plt.plot(val_loss, label='validation')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    print("hi")
    transform = torchvision.transforms.ToTensor()
    data = torch.load(
        "C:/Users/anala/Desktop/coursecontent/Speech_and_Speaker/Project/Programming_part/dataset.pt")
    labels = torch.load(
        "C:/Users/anala/Desktop/coursecontent/Speech_and_Speaker/Project/Programming_part/labels.pt")
    total_speakers = np.unique(np.array(labels))
    # indices = np.arange(len(total_speakers))
    # ind_labels = np.array(labels.copy())
    # for index, speaker in enumerate(total_speakers):
    #     ind_labels[ind_labels == speaker] = index
    # ind_labels = np.array(list(map(int, ind_labels)))
    # torch.save(ind_labels, 'labels.pt')
    val_data = data[:10][:, None, :, :], labels[:10]
    train_data = data[10:][:, None, :, :], labels[10:]
    model = C3D()
    Fit(train_set=train_data, val_set=val_data, model=model)

    print('hi')
