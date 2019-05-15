import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import dataset as data
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import time
from Utils import *

class C3D(torch.nn.Module):
    def __init__(self, data_shape, depth=3, frames=1, coefficients=5):
        super(C3D, self).__init__()
        self.data_shape, self.depth, self.frames, self.coefficients = data_shape, depth, frames, coefficients
        if self.data_shape[2] < 3:
            self.depth = self.data_shape[1]
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

    def LossAndOptimizer(self):
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.adadelta.Adadelta(C3D.parameters(self), lr=self.learning_rate)

    def Fit(self, batch_size, n_epochs, learning_rate=.001, train_data=None, validation_data=None, test_data=None,
            labels=None):
        self.batch_size, self.n_epochs, self.learning_rate = batch_size, n_epochs, learning_rate
        print("=" * 30)
        print("batch_size", batch_size)
        print("epochs", n_epochs)
        print("learning_rate", learning_rate)
        print("=" * 30)
        n_training_samples = 20000
        train_sampler = SubsetRandomSampler(np.arange(self.data_shape[0], dtype=np.int64))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler,
                                                   num_workers=2)
        n_val_samples = 5000
        val_sampler = SubsetRandomSampler(
            np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))
        n_test_samples = 5000
        test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, sampler=test_sampler, num_workers=2)
        val_loader = torch.utils.data.DataLoader(train_data, batch_size=128, sampler=val_sampler, num_workers=2)
        n_batches = len(train_loader)
        self.LossAndOptimizer()
        training_start_time = time.time()
        for epoch in tqdm(range(n_epochs)):
            start_time = time.time()
            running_loss = 0.0
            print_every = n_batches // 10
            total_train_loss = 0
            for index, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                self.optimizer.zero_grad()
                outputs = C3D(inputs)
                loss_size = self.loss(outputs, labels)
                loss_size.backward()
                self.optimizer.step()
                running_loss += loss_size.data
                total_train_loss += loss_size.data
                if (index + 1) % (print_every + 1) == 0:
                    print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch + 1, int(100 * (index + 1) / n_batches), running_loss / print_every,
                        time.time() - start_time))
                    running_loss = 0.0
                    start_time = time.time()
                total_val_loss = 0
                for inputs, labels in val_loader:
                    inputs, labels = Variable(inputs), Variable(labels)
                    val_outputs = C3D(inputs)
                    val_loss_size = self.loss(val_outputs, labels)
                    total_val_loss += val_loss_size.data
                print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
            print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


if __name__ == '__main__':
    # CNN = SimpleCNN()
    # trainNet(CNN, batch_size=32, n_epochs=5, learning_rate=0.001)
    #
    # dirs = Utils.CopyDataFiles(n_samples=1000)
    # cube = Feature_Cube(cube_shape=(20, 80, 40), augmentation=True)
    #
    # db = AudioDataset(c.DATA_TEMP + 'samples_paths.txt', c.DATA_TEMP, transform=transform)

    # trainloader = data.DataLoader(db, batch_size=64)
    #
    # N = len(np.genfromtxt(c.DATA_ORIGIN + 'train_paths.txt', dtype='str'))
    #
    # dataset = torch.cat([db.__getitem__(idx)[0] for idx in range(N)])
    # labels = [db.__getitem__(idx)[1] for idx in range(N)]


    cube = Feature_Cube(cube_shape=(20, 80, 40), augmentation=True)
    transform = transforms.Compose([CMVN(), cube, ToTensor()])
    trainset = AudioDataset(c.DATA_TEMP + 'samples_paths.txt', c.DATA_TEMP, transform=transform)

    trainloader = data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    for index, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)