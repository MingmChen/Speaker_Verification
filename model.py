import torch
import torch.nn.functional as F


class C3D(torch.nn.Module):
    def __init__(self, n_labels):
        super(C3D, self).__init__()
        self.conv1_1 = torch.nn.Conv3d(1, 16, kernel_size=(3, 1, 5), stride=(1, 1, 1))
        torch.nn.init.kaiming_normal_(self.conv1_1.weight, nonlinearity='leaky_relu')

        self.batch_norm1_1 = torch.nn.BatchNorm3d(num_features=16)
        self.PReLu1_1 = torch.nn.PReLU()

        self.conv1_2 = torch.nn.Conv3d(16, 16, kernel_size=(3, 9, 1), stride=(1, 2, 1))
        torch.nn.init.kaiming_normal_(self.conv1_2.weight, nonlinearity='leaky_relu')

        self.batch_norm1_2 = torch.nn.BatchNorm3d(num_features=16)
        self.PReLu1_2 = torch.nn.PReLU()

        self.pool1 = torch.nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))

        self.conv2_1 = torch.nn.Conv3d(16, 32, kernel_size=(3, 1, 4), stride=(1, 1, 1))
        torch.nn.init.kaiming_normal_(self.conv2_1.weight, nonlinearity='leaky_relu')

        self.batch_norm2_1 = torch.nn.BatchNorm3d(num_features=32)
        self.PReLu2_1 = torch.nn.PReLU()

        self.conv2_2 = torch.nn.Conv3d(32, 32, kernel_size=(3, 8, 1), stride=(1, 2, 1))
        torch.nn.init.kaiming_normal_(self.conv2_2.weight, nonlinearity='leaky_relu')

        self.batch_norm2_2 = torch.nn.BatchNorm3d(num_features=32)
        self.PReLu2_2 = torch.nn.PReLU()

        self.pool2 = torch.nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))

        self.conv3_1 = torch.nn.Conv3d(32, 64, kernel_size=(3, 1, 3), stride=(1, 1, 1))
        torch.nn.init.kaiming_normal_(self.conv3_1.weight, nonlinearity='leaky_relu')

        self.batch_norm3_1 = torch.nn.BatchNorm3d(num_features=64)
        self.PReLu3_1 = torch.nn.PReLU()

        self.conv3_2 = torch.nn.Conv3d(64, 64, kernel_size=(3, 7, 1), stride=(1, 1, 1))
        torch.nn.init.kaiming_normal_(self.conv3_2.weight, nonlinearity='leaky_relu')

        self.batch_norm3_2 = torch.nn.BatchNorm3d(num_features=64)
        self.PReLu3_2 = torch.nn.PReLU()

        self.conv4_1 = torch.nn.Conv3d(64, 128, kernel_size=(3, 1, 3), stride=(1, 1, 1))
        torch.nn.init.kaiming_normal_(self.conv4_1.weight, nonlinearity='leaky_relu')

        self.batch_norm4_1 = torch.nn.BatchNorm3d(num_features=128)
        self.PReLu4_1 = torch.nn.PReLU()

        self.conv4_2 = torch.nn.Conv3d(128, 128, kernel_size=(3, 7, 1), stride=(1, 1, 1))
        torch.nn.init.kaiming_normal_(self.conv4_2.weight, nonlinearity='leaky_relu')

        self.batch_norm4_2 = torch.nn.BatchNorm3d(num_features=128)
        self.PReLu4_2 = torch.nn.PReLU()

        self.FC5 = torch.nn.Linear(4 * 3 * 3 * 128, 128)
        torch.nn.init.kaiming_normal_(self.FC5.weight, nonlinearity='leaky_relu')

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


if __name__ == "__main__":
    model = C3D(50)
    model.eval()
    x = torch.rand((1, 1, 20, 80, 40))
    x = model(x)
