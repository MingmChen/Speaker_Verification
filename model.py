import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class C3D(nn.Module):
    def __init__(self, n_labels):
        super(C3D, self).__init__()
        print('their model')
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
    def __init__(self, n_labels, num_channels):
        super(C3D2, self).__init__()
        self.n_labels, self.num_channels = n_labels, num_channels
        print('tasos model')

        self.conv1_1 = torch.nn.Conv3d(num_channels, 16, kernel_size=(3, 1, 5), stride=(1, 1, 1))
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
        # self.batch_normFC5 = torch.nn.BatchNorm1d(num_features=128)
        self.PReLu5 = torch.nn.PReLU()
        self.FC6 = torch.nn.Linear(128, n_labels)

    def forward(self, x, development=True):
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
        if development:
            # x = self.batch_normFC5(x)
            x = self.PReLu5(x)
            x = self.FC6(x)
            x = F.softmax(x, dim=1)
        return x

    def load_checkpoint(self, checkpoint_dict):
        model = C3D2(n_labels=self.n_labels, num_channels=self.num_channels)
        if torch.cuda.is_available():
            model.cuda()
        model_dict = model.state_dict()
        pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict["state_dict"].items() if
                           k.replace('module.', '') in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
        return model

    def create_Speaker_Model(self, utterance):
        self.eval()
        Speaker_Model = self.forward(utterance, development=False)
        return Speaker_Model

class C3D3(torch.nn.Module):
    def __init__(self, n_labels, num_channels):
        super(C3D3, self).__init__()
        self.n_labels, self.num_channels = n_labels, num_channels
        print('short model')
        self.conv1_1 = torch.nn.Conv3d(num_channels, 16, kernel_size=(6, 2, 10), stride=(1, 1, 1))
        self.batch_norm1_1 = torch.nn.BatchNorm3d(num_features=16)
        self.PReLu1_1 = torch.nn.PReLU()
        self.conv1_2 = torch.nn.Conv3d(16, 16, kernel_size=(6, 9, 2), stride=(1, 2, 1))
        self.batch_norm1_2 = torch.nn.BatchNorm3d(num_features=16)
        self.PReLu1_2 = torch.nn.PReLU()
        self.pool1 = torch.nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2))
        self.conv2_1 = torch.nn.Conv3d(16, 32, kernel_size=(3, 16, 2), stride=(1, 2, 1))
        self.batch_norm2_1 = torch.nn.BatchNorm3d(num_features=32)
        self.PReLu2_1 = torch.nn.PReLU()
        self.conv2_2 = torch.nn.Conv3d(32, 32, kernel_size=(3, 2, 6), stride=(1, 2, 1))
        self.batch_norm2_2 = torch.nn.BatchNorm3d(num_features=32)
        self.PReLu2_2 = torch.nn.PReLU()
        self.FC3 = torch.nn.Linear(7680, 128)
        self.PReLu5 = torch.nn.PReLU()
        self.FC4 = torch.nn.Linear(128, n_labels)

    def forward(self, x, development=True):
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
        x = x.view(-1, 7680)
        x = self.FC3(x)
        if development:
            x = self.PReLu5(x)
            x = torch.nn.Softmax(self.FC4(x))
        return x

    def load_checkpoint(self, checkpoint_dict):
        model = C3D3(n_labels=self.n_labels, num_channels=self.num_channels)
        model_dict = model.state_dict()
        pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict["state_dict"].items() if
                           k.replace('module.', '') in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
        return model

    def create_Speaker_Model(self, utterance):
        self.eval()
        Speaker_Model = self.forward(utterance, development=False)
        return Speaker_Model

class C3D4(torch.nn.Module):
    def __init__(self, n_labels, num_channels):
        super(C3D4, self).__init__()
        self.n_labels, self.num_channels = n_labels, num_channels
        print('shorter model')
        self.conv1_1 = torch.nn.Conv3d(num_channels, 16, kernel_size=(3, 2, 5), stride=(1, 1, 1))
        self.batch_norm1_1 = torch.nn.BatchNorm3d(num_features=16)
        self.PReLu1_1 = torch.nn.PReLU()
        self.conv1_2 = torch.nn.Conv3d(16, 16, kernel_size=(3, 9, 1), stride=(1, 2, 1))
        self.batch_norm1_2 = torch.nn.BatchNorm3d(num_features=16)
        self.PReLu1_2 = torch.nn.PReLU()
        self.pool1 = torch.nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2))
        self.conv2_1 = torch.nn.Conv3d(16, 32, kernel_size=(3, 8, 2), stride=(2, 2, 2))
        self.batch_norm2_1 = torch.nn.BatchNorm3d(num_features=32)
        self.PReLu2_1 = torch.nn.PReLU()
        self.conv2_2 = torch.nn.Conv3d(32, 32, kernel_size=(3, 2, 6), stride=(1, 2, 1))
        self.batch_norm2_2 = torch.nn.BatchNorm3d(num_features=32)
        self.PReLu2_2 = torch.nn.PReLU()
        self.FC3 = torch.nn.Linear(32*5*7*3, 128)
        self.PReLu5 = torch.nn.PReLU()
        self.FC4 = torch.nn.Linear(128, n_labels)
        self.softmax =torch.nn.Softmax(dim=1)

    def forward(self, x, development=True):
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
        x = x.view(-1, 32*5*7*3)
        x = self.FC3(x)
        if development:
            x = self.PReLu5(x)
            x = self.FC4(x)
            x = self.softmax(x)
        return x

    def load_checkpoint(self, checkpoint_dict):
        model = C3D3(n_labels=self.n_labels, num_channels=self.num_channels)
        model_dict = model.state_dict()
        pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict["state_dict"].items() if
                           k.replace('module.', '') in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
        return model

    def create_Speaker_Model(self, utterance):
        self.eval()
        Speaker_Model = self.forward(utterance, development=False)
        return Speaker_Model

class C2D(torch.nn.Module):
    """
    input = (x,3,40,100)
    """

    def __init__(self):
        super(C2D, self).__init__()
        self.Relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(3, 32, (7, 7), stride=(2, 2))
        self.norm1 = torch.nn.BatchNorm2d(num_features=32)
        self.conv2 = torch.nn.Conv2d(32, 64, (5, 5), stride=(1, 1))
        self.norm2 = torch.nn.BatchNorm2d(num_features=64)
        self.conv3 = torch.nn.Conv2d(64, 128, (3, 3), stride=(1, 1))
        self.conv4 = torch.nn.Conv2d(128, 256, (3, 3), stride=(1, 1))
        self.conv5 = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1))
        self.FC1 = torch.nn.Linear(256 * 7 * 37, 1024)
        self.FC2 = torch.nn.Linear(1024, 256)
        self.FC3 = torch.nn.Linear(256, 1024)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.Relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.Relu(x)
        x = self.conv3(x)
        x = self.Relu(x)
        x = self.conv4(x)
        x = self.Relu(x)
        x = self.conv5(x)
        x = self.Relu(x)
        x = x.view(-1, 256 * 7 * 37)
        x = self.FC1(x)
        x = self.Relu(x)
        x = self.FC2(x)
        x = self.Relu(x)
        x = torch.nn.Softmax(self.FC3(x))

        return x


def create_speaker_models():
    import constants as c
    import numpy as np
    import os
    from utils import create_dataset

    model_path = os.path.join(c.ROOT, 'Models/model_14_percent_best_so_far.pt')
    save_speaker_models_path = os.path.join(c.ROOT, 'speaker_models')
    enrollment_set = os.path.join(c.ROOT, '50_first_ids.txt')
    indexed_labels = np.load(c.ROOT + '/50_first_ids.npy', allow_pickle=True).item()

    dataset = create_dataset(indexed_labels=indexed_labels, origin_file_path=enrollment_set)

    if not os.path.exists(save_speaker_models_path):
        os.mkdir(save_speaker_models_path)

    if not torch.cuda.is_available():
        model = C3D2(100, 1).load_checkpoint(torch.load(model_path, map_location=lambda storage,loc: storage))
        # model = C3D2(100, 1).load_checkpoint(torch.load(model_path))
    else:
        model = C3D2(100, 1).load_checkpoint(torch.load(model_path))

    model.eval()
    for i in range(len(dataset)):
        # get the inputs
        train_input, _ = dataset.__getitem__(i)
        [a, b, cc, d] = train_input.shape
        train_input = torch.from_numpy(train_input.reshape((1, a, b, cc, d)))

        if torch.cuda.is_available():
            train_input = Variable(train_input.cuda())
        else:
            train_input = Variable(train_input)

        current_id = dataset.sound_files[i][0:7]

        speaker_model = model(train_input, development=False)
        torch.save(speaker_model,'{}/{}.pt'.format(save_speaker_models_path, current_id))


if __name__ == '__main__':
    create_speaker_models()
