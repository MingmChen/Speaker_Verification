import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch
import np as np
import PIL
import matplotlib.pyplot as plt
import tqdm
import time

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Training

n_training_samples = 20000
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

# Validation

n_val_samples = 5000
val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))

# Test

n_test_samples = 5000
test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))

class Try(torch.nn.Module):
    def __init__(self):
        super(Try, self).__init__()
        self.conv1 = torch.nn.Conv3d(1,16, kernel_size=(3, 1, 5), stride=(1, 1, 1))#, padding=(1, 0, 2))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return x

class SimpleCNN(torch.nn.Module):
    """
    input (3, 32, 32)
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()
        """
        3 Inputs, 18 outputs
        """

        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        """
        4608 input features, 64 output features
        """

        self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)

        """
        64 input features, 10 output features for the 10 classes
        """

        self.fc2 = torch.nn.Linear(64, 10)

    def outputSize(self, in_size, kernel_size, stride, padding):
        """
        :param in_size:
        :param kernel_size:
        :param stride:
        :param padding:
        :return:
        Automatically compute the output of a convolutional layer
        """
        output = int((in_size - kernel_size + 2 * (padding)) / stride) + 1
        return output

    def forward(self, x):
        """
        :param x:
        :return:
        Complete forward pass
        """
        """
        activation of the first convolution
        size from (3, 32, 32) -> (18, 32, 32)
        """
        x = F.relu(self.conv1(x))

        """
        from (18, 32, 32) -> (18, 16, 16)
        """

        x = self.pool(x)

        """
        fully connected reshape
        (18, 16, 16) -> (1, 4608)
        """

        x = x.view(-1, 18 * 16 * 16)

        """
        fully connected layer 1 activation
        """

        x = F.relu(self.fc1(x))

        """
        compute fully connected layer 2, no activation atm
        """

        x = self.fc2(x)

        return x


def get_train_loader(batch_size):
    """
    :param batch_size:
    :return:
    takes in a dataset and a sampler loading
    """
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=2)
    return (train_loader)


"""
Test and validation Loaders, always the same
"""

test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=2)
val_loader = torch.utils.data.DataLoader(train_set, batch_size=128, sampler=val_sampler, num_workers=2)


def createLossAndOptimizer(net, learning_rate=0.001):
    """
    :param net:
    :param learning_rate:
    :return:
    Loss and Optimizer
    """
    """
    Loss Function
    """
    loss = torch.nn.CrossEntropyLoss()
    """
    Optimizer
    """
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    return (loss, optimizer)


def trainNet(net, batch_size, n_epochs, learning_rate):
    """
    :param net:
    :param batch_size:
    :param n_epochs:
    :param learning_rate:
    :return:
    result of training
    """
    print("====>HYPERPARAMETERS<====")
    print("batch_size", batch_size)
    print("epochs", n_epochs)
    print("learning_rate", learning_rate)
    print("=" * 30)

    """
    Get train data
    """
    train_loader = get_train_loader(batch_size=batch_size)
    n_batches = len(train_loader)

    """
    Create our loss and optimizer functions
    """
    loss, optimizer = createLossAndOptimizer(net=net, learning_rate=learning_rate)

    """
    Time for printing
    """
    training_start_time = time.time()

    """
    training loop
    """
    for epoch in range(n_epochs):
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):
            """
            get inputs
            """
            inputs, labels = data
            """
            wrap them in variable object
            """
            inputs, labels = Variable(inputs), Variable(labels)
            """
            Set the parameter gradients to zero
            """
            optimizer.zero_grad()
            """
            forward,backward,optimize
            """
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            """
            print statistics
            """
            running_loss += loss_size.data
            total_train_loss += loss_size.data
            """
            print every 10th batch of an epoch
            """
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, time.time() - start_time))
                """
                reset running loss
                """
                running_loss = 0.0
                start_time = time.time()

            """
            at the end of an epoch, do a pass on the validation set
            """
            total_val_loss = 0
            for inputs, labels in val_loader:
                """
                wrap tensors in Variables
                """
                inputs, labels = Variable(inputs), Variable(labels)
                """
                Forward pass
                """
                val_outputs = net(inputs)
                val_loss_size = loss(val_outputs, labels)
                total_val_loss += val_loss_size.data
            print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


if __name__ == '__main__':
    # CNN = SimpleCNN()
    # trainNet(CNN, batch_size=32, n_epochs=5, learning_rate=0.001)
    toy_data = torch.randn(1,1,20,80,40)
    toy = Try()
    output = toy(toy_data)
    print('hi')
