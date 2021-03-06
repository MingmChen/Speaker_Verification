import torch.nn.init as init
from model import C3D, C3D2
import gcloud_wrappers
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import time
from utils import *
from load_data import AudioDataset
import numpy as np


def xavier(param):
    init.xavier_uniform_(param)


# Initializer function
def weights_init(m):
    """
    Different type of initialization have been used for conv and fc layers.
    :param m: layer
    :return: Initialized layer. Return occurs in-place.
    """
    if isinstance(m, torch.nn.Conv3d):
        xavier(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, torch.nn.Linear):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)


def train_with_loader(train_loader, n_labels, num_channels):
    model = C3D2(n_labels, num_channels)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.apply(weights_init)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.SGD(model.parameters(),
                          lr=c.LEARNING_RATE,
                          momentum=c.MOMENTUM,
                          weight_decay=c.WEIGHT_DECAY)

    loss_criterion = torch.nn.CrossEntropyLoss()
    # scheduler = CosineAnnealingLR(optimizer, T_max=4 * len(train_loader))
    # scheduler = StepLR(optimizer,
    #                    step_size=c.STEP_SIZE,
    #                    gamma=c.GAMMA)

    n_epochs = c.N_EPOCHS

    train_loss = []

    train_acc = []

    best_accuracy = 0.0
    for epoch in range(n_epochs):

        train_running_loss = 0.0
        # Step the lr scheduler each epoch!
        total_loss = 0
        start = time.time()

        for i, data in enumerate(train_loader, 1):

            # get the inputs
            train_input, train_labels = data
            if torch.cuda.is_available():
                train_input, train_labels = Variable(train_input.cuda()), Variable(train_labels.cuda())
            else:
                train_input, train_labels = Variable(train_input), Variable(train_labels)

            # zero the parameter gradients
            optimizer.zero_grad()
            # scheduler.step()

            # forward
            outputs = model(train_input)
            # Loss
            train_loss_ = loss_criterion(outputs, train_labels)
            # backward & optimization
            train_loss_.backward()
            optimizer.step()
            train_running_loss += train_loss_.item()

            if i % c.BATCH_PER_LOG == 0:  # print every 2000 mini-batches
                end = time.time()
                total_time = end - start
                print('[Epoch %d,Batch %d] loss: %.3f time: %.3f' %
                      (epoch + 1, i, train_running_loss / c.BATCH_PER_LOG, total_time))

                total_loss += train_running_loss / c.BATCH_PER_LOG
                train_running_loss = 0.0

        # scheduler = CosineAnnealingLR(optimizer, T_max=4 * len(train_loader))
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            for data in train_loader:

                images, labels = data
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            model.train()

        end = time.time()
        total_time = end - start
        print('Accuracy of the network: %d %%, loss: %.5f for epoch %d time %d \n' % (
            100 * correct / total, total_loss, epoch + 1, total_time))

        train_loss.append(total_loss)
        train_acc.append(100 * correct / total)

        if best_accuracy < 100 * correct / total:

            if not os.path.exists(c.MODEL_DIR):
                os.mkdir(c.MODEL_DIR)
            print("\nBEST MODEL SO FAR")
            best_accuracy = 100 * correct / total
            save_checkpoint(
                {

                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'accuracy': 100 * correct / total,
                    'optimizer': optimizer.state_dict(),
                },
                is_best=100 * correct / total == best_accuracy,
                filename=os.path.join(
                    c.MODEL_DIR,
                    'saved_' +
                    str(epoch + 1) + '_model.pt'
                )
            )

        if int(epoch + 1) % c.EPOCHS_PER_SAVE == 0:

            if not os.path.exists(c.MODEL_DIR):
                os.mkdir(c.MODEL_DIR)

            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'accuracy': 100 * correct / total,
                    'optimizer': optimizer.state_dict(),
                },
                is_best=100 * correct / total == best_accuracy,
                filename=os.path.join(
                    c.MODEL_DIR,
                    'saved_' +
                    str(epoch + 1) + '_model.pt'
                )
            )

    plot_loss_acc(train_loss, train_acc)


def check_files_missing(origin_file_path):
    content = np.genfromtxt(origin_file_path, dtype='str')

    counter = 0
    list = []
    for file in content:
        if not os.path.exists(os.path.join(c.DATA_ORIGIN, file)):
            counter += 1
            list.append(file)

    print('non existing files : {}'.format(counter))
    print('non existing files list : {}'.format(list))


def main():
    if not os.path.exists(c.ROOT + '/50_first_ids.npy'):
        indexed_labels = np.load(c.ROOT + '/labeled_indices.npy').item()
        origin_file_path = c.DATA_ORIGIN + 'train_paths.txt'
    else:
        indexed_labels = np.load(c.ROOT + '/50_first_ids.npy', allow_pickle=True).item()
        origin_file_path = c.ROOT + '/50_first_ids.txt'

    if c.DERIVATIVE:
        num_channels = 3
        cube_shape = (80, 40, 20, num_channels)
        cube = FeatureCube3C(cube_shape)

    else:
        num_channels = 1
        cube_shape = (80, 40, 20)
        cube = FeatureCube(cube_shape)

    transform = transforms.Compose([CMVN(), cube, ToTensor()])

    check_files_missing(origin_file_path)

    dataset = AudioDataset(
        origin_file_path,
        c.DATA_ORIGIN,
        indexed_labels=indexed_labels,
        transform=transform)

    # train_loader, validation_loader = split_sets(dataset)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=c.BATCH_SIZE,
        sampler=train_sampler
    )

    train_with_loader(
        train_loader,
        len(indexed_labels.keys()),
        num_channels=num_channels
    )

    gcloud_wrappers.stop_speech_vm()

if __name__ == '__main__':
    main()
