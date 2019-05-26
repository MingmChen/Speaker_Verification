import torch.nn.init as init
import gcloud_wrappers
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from model import C3D
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import time
from utils import *
from load_data import AudioDataset
import numpy as np


def xavier(param):
    init.xavier_uniform(param)


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


def train_with_loader(train_loader, n_labels, validation_loader=None):
    model = C3D(n_labels)

    model.apply(weights_init)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.SGD(model.parameters(),
                          lr=c.LEARNING_RATE,
                          momentum=c.MOMENTUM,
                          weight_decay=c.WEIGHT_DECAY)

    loss_criterion = torch.nn.CrossEntropyLoss()

    scheduler = StepLR(optimizer,
                       step_size=c.STEP_SIZE,
                       gamma=c.GAMMA)

    n_epochs = c.N_EPOCHS

    train_loss = []

    train_acc = []

    train_best_accuracy = 0.0

    for epoch in range(n_epochs):

        train_running_loss = 0.0
        train_running_accuracy = 0.0

        # Step the lr scheduler each epoch!
        scheduler.step()

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

            # forward
            outputs = model(train_input)

            # Loss
            train_loss_ = loss_criterion(outputs, train_labels)

            # train_accuracy = calculate_accuracy(model, train_input, train_labels)

            # backward & optimization
            train_loss_.backward()
            optimizer.step()
            train_running_loss += train_loss_.data.item()

            _, predictions = torch.max(outputs, dim=1)
            correct_count = (predictions == train_labels).double().sum().item()
            train_accuracy = float(correct_count) / c.BATCH_SIZE

            # best accuracy
            if train_accuracy > train_best_accuracy:
                train_best_accuracy = train_accuracy

            # adding to loss for the batch
            # train_running_loss += train_loss_.data.item()
            train_running_accuracy += train_accuracy

            # Print stats every 5 batches
            if i % c.BATCH_PER_LOG == 0:
                print((
                        'epoch {:2d} ' +
                        '|| batch {:2d} of {:2d} ||' +
                        ' Batch-Loss: {:.8f} ||' +
                        ' Batch-Accuracy: {:.4f}\n').format(
                    epoch + 1,
                    i,
                    len(train_loader),
                    train_running_loss / c.BATCH_PER_LOG,
                    train_accuracy  # train_accuracy
                ),
                    end='')
                train_running_loss = 0.0

        end = time.time()
        duration_estimate = end - start

        # progress.update()
        train_loss.append(train_running_loss / len(train_loader))  # c.BATCH_SIZE)
        train_acc.append(float(100.0 * train_running_accuracy / len(train_loader)))

        print('The averaged accuracy for each epoch: {:.4f}.\n'.format(
            100.0 * train_running_accuracy / len(train_loader)), end='')

        if int(epoch + 1) % c.EPOCHS_PER_SAVE == 0:

            if not os.path.exists(c.MODEL_DIR):
                os.mkdir(c.MODEL_DIR)

            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_accuracy': train_best_accuracy,
                    'optimizer': optimizer.state_dict(),
                },
                is_best=train_running_accuracy == train_best_accuracy,
                filename=os.path.join(
                    c.MODEL_DIR,
                    'saved_' +
                    str(epoch + 1) + '_model.pt'
                )
            )

        # print(
        #     "epoch {:2d}/{:2d}, Train Loss: {:.8f}, Train Accuracy: {:.4f}, "
        #     "Time: {:.4f}".format(
        #         epoch + 1,
        #         n_epochs,
        #         train_running_loss / len(train_loader),  # c.BATCH_SIZE,
        #         float(100.0 * train_running_accuracy / len(train_loader)),
        #         duration_estimate
        #     )
        # )

    plot_loss_acc(train_loss, train_acc)


def check_files_missing(origin_file_path):
    content = np.genfromtxt(origin_file_path, dtype='str')

    counter = 0
    list = []
    for file in content:
        if not os.path.exists(os.path.join(c.DATA_ORIGIN + 'wav', file)):
            counter += 1
            list.append(file)

    print('non existing files : {}'.format(counter))
    print('non existing files list : {}'.format(list))


def main():
    if not os.path.exists(c.ROOT + '/50_first_ids.npy'):
        indexed_labels = np.load(c.ROOT + '/labeled_indices.npy').item()
        origin_file_path = c.DATA_ORIGIN + 'train_paths.txt'
    else:
        indexed_labels = np.load(c.ROOT + '/50_first_ids.npy').item()
        origin_file_path = c.ROOT + '/50_first_ids.txt'

    cube = FeatureCube(c.CUBE_SHAPE)
    transform = transforms.Compose([CMVN(), cube, ToTensor()])

    check_files_missing(origin_file_path)
    # return

    # try:

    dataset = AudioDataset(
        origin_file_path,
        c.DATA_ORIGIN + 'wav/',
        indexed_labels=indexed_labels,
        transform=transform)

    # train_loader, validation_loader = split_sets(dataset)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=c.BATCH_SIZE,
                                               sampler=train_sampler)

    train_with_loader(train_loader, len(indexed_labels.keys()))

    # except:
    #     credentials = GoogleCredentials.get_application_default()
    #
    #     compute = discovery.build(
    #         'compute',
    #         'v1',
    #         credentials=credentials
    #     )
    #     gcloud_wrappers.stop_speech_vm(compute)
    #
    # credentials = GoogleCredentials.get_application_default()
    #
    # compute = discovery.build(
    #     'compute',
    #     'v1',
    #     credentials=credentials
    # )
    # gcloud_wrappers.stop_speech_vm(compute)


if __name__ == '__main__':
    main()
