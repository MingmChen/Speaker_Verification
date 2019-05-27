import torch
import torch.nn.functional as F
from torch import nn


class C3D(nn.Module):
    def __init__(self,n_labels):
        super(C3D, self).__init__()

        ################
        ### Method 1 ###
        ################
        self.conv11 = nn.Conv3d(1, 16, (4, 9, 9), stride=(1, 2, 1))
        self.conv11_bn = nn.BatchNorm3d(16)
        self.conv11_activation = torch.nn.PReLU()
        self.conv12 = nn.Conv3d(16, 16, (4, 9, 9), stride=(1, 1, 1))
        self.conv12_bn = nn.BatchNorm3d(16)
        self.conv12_activation = torch.nn.PReLU()
        self.conv21 = nn.Conv3d(16, 32, (3, 7, 7), stride=(1, 1, 1))
        self.conv21_bn = nn.BatchNorm3d(32)
        self.conv21_activation = torch.nn.PReLU()
        self.conv22 = nn.Conv3d(32, 32, (3, 7, 7), stride=(1, 1, 1))
        self.conv22_bn = nn.BatchNorm3d(32)
        self.conv22_activation = torch.nn.PReLU()
        self.conv31 = nn.Conv3d(32, 64, (3, 5, 5), stride=(1, 1, 1))
        self.conv31_bn = nn.BatchNorm3d(64)
        self.conv31_activation = torch.nn.PReLU()
        self.conv32 = nn.Conv3d(64, 64, (3, 5, 5), stride=(1, 1, 1))
        self.conv32_bn = nn.BatchNorm3d(64)
        self.conv32_activation = torch.nn.PReLU()
        self.conv41 = nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 1, 1))
        self.conv41_bn = nn.BatchNorm3d(128)
        self.conv41_activation = torch.nn.PReLU()



        # Fully-connected
        self.fc1 = nn.Linear(128 * 4 * 6 * 2, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc1_activation = torch.nn.PReLU()
        self.fc2 = nn.Linear(128, n_labels)


        # ################
        # ### Method 2 ###
        # ################
        # self.cnn = nn.Sequential(
        #      nn.Conv3d(1, 16, (4, 9, 9), stride=(1, 2, 1)),
        #      nn.BatchNorm3d(16),
        #      nn.ReLU(),
        #      nn.Conv3d(16, 16, (4, 9, 9), stride=(1, 1, 1)),
        #      nn.BatchNorm3d(16),
        #      nn.ReLU(),
        #      nn.Conv3d(16, 32, (3, 7, 7), stride=(1, 1, 1)),
        #      nn.BatchNorm3d(32),
        #      nn.ReLU(),
        #      nn.Conv3d(32, 32, (3, 7, 7), stride=(1, 1, 1)),
        #      nn.BatchNorm3d(32),
        #      nn.ReLU(),
        #      nn.Conv3d(32, 64, (3, 5, 5), stride=(1, 1, 1)),
        #      nn.BatchNorm3d(64),
        #      nn.ReLU(),
        #      nn.Conv3d(64, 64, (3, 5, 5), stride=(1, 1, 1)),
        #      nn.BatchNorm3d(64),
        #      nn.ReLU(),
        #      nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 1, 1)),
        #      nn.BatchNorm3d(128),
        #      nn.ReLU(),
        # )
        #
        # self.fc = nn.Sequential(
        #      nn.Linear(128 * 4 * 6 * 2, 512),
        #      nn.BatchNorm1d(512),
        #      nn.ReLU(),
        #      nn.Linear(512, 1211),
        # )

    def features(self, x):
        # Method-1
        x = self.conv11_activation(self.conv11_bn(self.conv11(x)))
        x = self.conv12_activation(self.conv12_bn(self.conv12(x)))
        x = self.conv21_activation(self.conv21_bn(self.conv21(x)))
        x = self.conv22_activation(self.conv22_bn(self.conv22(x)))
        x = self.conv31_activation(self.conv31_bn(self.conv31(x)))
        x = self.conv32_activation(self.conv32_bn(self.conv32(x)))
        x = self.conv41_activation(self.conv41_bn(self.conv41(x)))
        x = x.view(-1, 128 * 4 * 6 * 2)
        x = self.fc1_bn(self.fc1(x))
        x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-12)

        # # Method Sequential
        # x = self.cnn(x)
        # x = x.view(-1, 128 * 4 * 6 * 2)
        # x = self.fc(x)

        return x

    def forward(self, x):
        # Method-1
        x = self.features(x)
        x = self.fc1_activation(x)
        x = self.fc2(x)

        return x

class C3D2(torch.nn.Module):
    def __init__(self, n_labels):
        super(C3D2, self).__init__()
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
        self.FC6 = torch.nn.Linear(128, n_labels)

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


class CNN3D3(nn.Module):
    """
    input:  n * channels(3) * uttr(20) * frame(80) * freq(40)
    output: n * num_classes
    """

    def __init__(self, num_classes):
        super(CNN3D3, self).__init__()
        self.conv1_1 = nn.Conv3d(1, 16, kernel_size=(3, 1, 5), stride=(1, 1, 1), padding=0, bias=False)
        self.conv1_2 = nn.Conv3d(16, 16, kernel_size=(3, 9, 1), stride=(1, 2, 1), padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(16)

        self.conv2_1 = nn.Conv3d(16, 32, kernel_size=(3, 1, 4), stride=(1, 1, 1), padding=0, bias=False)
        self.conv2_2 = nn.Conv3d(32, 32, kernel_size=(3, 8, 1), stride=(1, 2, 1), padding=0, bias=False)
        self.bn2 = nn.BatchNorm3d(32)

        self.conv3_1 = nn.Conv3d(32, 64, kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=0, bias=False)
        self.conv3_2 = nn.Conv3d(64, 64, kernel_size=(3, 7, 1), stride=(1, 1, 1), padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(64)

        self.conv4_1 = nn.Conv3d(64, 128, kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=0, bias=False)
        self.conv4_2 = nn.Conv3d(128, 128, kernel_size=(3, 7, 1), stride=(1, 1, 1), padding=0, bias=False)
        self.bn4 = nn.BatchNorm3d(128)

        self.relu = nn.PReLU()
        self.avg_pool = nn.AvgPool3d([1, 1, 2])  # average on 'freq'

        self.fc1 = nn.Linear(128 * 4 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inp):
        out = self.conv1_1(inp)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1_2(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.avg_pool(out)

        out = self.conv2_1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2_2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avg_pool(out)

        out = self.conv3_1(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3_2(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4_1(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.conv4_2(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        # embed = out  # (n, 128)

        out = self.fc2(out)  # (n, classes)
        return out