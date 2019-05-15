import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch
import copy
import PIL
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from Utils import *
from load_data import AudioDataset
import subprocess
from sklearn.model_selection import train_test_split
import numpy as np


class C3D(torch.nn.Module):
    def __init__(self, n_speaker):
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
        self.FC6 = torch.nn.Linear(128, n_speaker)

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


def calculate_accuracy(model, X, Y):
    oupt = model(X)
    (max_vals, arg_maxs) = torch.max(oupt.data, dim=1)

    num_correct = torch.sum(Y == arg_maxs)
    acc = (num_correct * 100.0 / Y.shape[0])
    return acc.item()  #


def LossAndOptimizer(learning_rate, model):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return loss, optimizer


def train(train_set, val_set, model, learning_rate=.0001, n_epochs=10, batch_size=64, load_model=False):
    loss, optimizer = LossAndOptimizer(learning_rate, model)
    train_data, train_labels = train_set
    val_data, val_labels = val_set

    if load_model is True:
        if not os.path.exists(c.MODEL_DIR):
            print("No Such a dir")

        checkpoint = torch.load(c.MODEL_DIR + '/model_best.pt')
        best_accuracy = checkpoint['best_accuracy']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    N = train_data.shape[0]
    loops = int(N / batch_size)
    epochs_per_save = c.EPOCHS_PER_SAVE
    epoch = 0
    train_loss = []
    val_loss = []
    val_acc = []
    train_acc = []
    desc = 'Training 3D CNN'
    best_accuracy = 0.0
    with tqdm(desc=desc, total=n_epochs * loops) as progress:
        while epoch < n_epochs:
            running_loss = 0.0
            for j in range(loops):
                j_start = j * batch_size
                j_end = (j + 1) * batch_size
                inds = range(j_start, j_end)
                X = train_data[inds]
                y = train_labels[inds]
                y = y.long()
                X, y = Variable(X), Variable(y)

                optimizer.zero_grad()
                outputs = model(X)
                loss_size = loss(outputs, y)
                loss_size.backward()
                optimizer.step()
                running_loss += loss_size.data
                progress.update()

            Val_outputs = model(val_data)
            val_loss_size = loss(Val_outputs, val_labels)
            total_val_loss = val_loss_size.data

            val_temp_acc = calculate_accuracy(model, val_data, val_labels)
            train_temp_acc = calculate_accuracy(model, train_data, train_labels)

            if val_temp_acc > best_accuracy:
                best_accuracy = val_temp_acc

            val_acc.append(val_temp_acc)
            train_acc.append(train_temp_acc)
            val_loss.append(round(float(total_val_loss), 4))
            train_loss.append(round(float(running_loss) / (N / batch_size), 4))

            print(
                "\n\nEpoch {}/{}, Train_Loss: {:.3f}, Train_Accuracy: {:.3f}, Val_Loss: {:.3f}, Val_Accuracy: {:.3f}".format(
                    epoch + 1,
                    n_epochs,
                    round(float(running_loss) /
                          (N / batch_size), 5),
                    train_temp_acc,
                    round(float(total_val_loss), 5),
                    val_temp_acc))


            if int(epoch + 1) % epochs_per_save == 0:
                if not os.path.exists(c.MODEL_DIR):
                    os.mkdir(c.MODEL_DIR)

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_accuracy': best_accuracy,
                    'optimizer': optimizer.state_dict(),
                }, is_best=val_temp_acc == best_accuracy, filename=os.path.join(c.MODEL_DIR,
                                                                                'saved_' +
                                                                                  str(epoch + 1) + '_model.pt'))
            epoch += 1

    plot_loss_acc(train_loss, train_acc, val_loss, val_acc)


if __name__ == '__main__':
    dataset_file = c.ROOT + '/dataset'
    labels_file = c.ROOT + '/label'
    db_file = c.ROOT + '/db'

    if not os.path.exists(c.DATA_TEMP):
        dirs = CopyDataFiles(n_samples=10000)

        cube = FeatureCube((80, 40, 20))

        transform = transforms.Compose([CMVN(), cube, ToTensor()])

        db = AudioDataset(c.DATA_TEMP + 'samples_paths.txt', c.DATA_TEMP, transform=transform)

        file = np.genfromtxt(c.DATA_TEMP + 'samples_paths.txt', dtype='str')

        N = len(file)

        dataset = [db.__getitem__(idx)[0] for idx in range(N)]
        labels = [db.__getitem__(idx)[1] for idx in range(N)]

        unique_speakers = len(np.unique(labels))
        print('Unique speakers ', unique_speakers)
        X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.1,
                                                            random_state=56)

        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)

        X_test = torch.tensor(X_test)
        y_test = torch.tensor(y_test)

        save_file(c.ROOT + '/X_train', X_train)
        save_file(c.ROOT + '/y_train', y_train)
        save_file(c.ROOT + '/X_test', X_test)
        save_file(c.ROOT + '/y_test', y_test)

    else:
        X_train = load_file(c.ROOT + '/X_train')
        y_train = load_file(c.ROOT + '/y_train')
        X_test = load_file(c.ROOT + '/X_test')
        y_test = load_file(c.ROOT + '/y_test')

    unique_speakers = np.unique(torch.cat([y_test, y_train]))
    total_speakers = torch.cat([y_test, y_train])

    train_data = [X_train, y_train]
    val_data = [X_test, y_test]

    model = C3D(len(unique_speakers))

    train(train_set=train_data, val_set=val_data, model=model)
    #
    # args = ['nohup', 'python', '-u', '/Users/polaras/PycharmProjects/Speech_recognition_Project/3D_CNN.py',
    #         '>', 'log.txt', '&']
    #
    # p = subprocess.Popen(args)
    # ToDO Learning Rate policy scheduler = StepLR(optimizer, step_size=args.epochs_per_lr_drop, gamma=args.gamma)
    # ToDO configure the feature extraction
