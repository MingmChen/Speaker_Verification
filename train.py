import sys
from model import C3D
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import copy
import PIL
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from utils import *
from load_data import AudioDataset
import subprocess
from sklearn.model_selection import train_test_split
import numpy as np


# def train(train_set, val_set, model, learning_rate=.0001, n_epochs=10, batch_size=64, load_model=False):
#     loss, optimizer = LossAndOptimizer(learning_rate, model)
#     train_data, train_labels = train_set
#     val_data, val_labels = val_set
#
#     if load_model is True:
#         if not os.path.exists(c.MODEL_DIR):
#             print("No Such a dir: {}".format(c.MODEL_DIR))
#             sys.exit()
#         checkpoint = torch.load(c.MODEL_DIR + '/model_best.pt')
#         best_accuracy = checkpoint['best_accuracy']
#         model.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#
#     N = train_data.shape[0]
#     loops = int(N / batch_size)
#     epochs_per_save = c.EPOCHS_PER_SAVE
#     epoch = 0
#     train_loss = []
#     val_loss = []
#     val_acc = []
#     train_acc = []
#     desc = 'Training 3D CNN'
#     best_accuracy = 0.0
#     with tqdm(desc=desc, total=n_epochs * loops) as progress:
#         while epoch < n_epochs:
#             train_train_running_loss = 0.0
#             for j in range(loops):
#                 j_start = j * batch_size
#                 j_end = (j + 1) * batch_size
#                 inds = range(j_start, j_end)
#                 X = train_data[inds]
#                 y = train_labels[inds]
#                 y = y.long()
#                 X, y = Variable(X), Variable(y)
#
#                 optimizer.zero_grad()
#                 outputs = model(X)
#                 loss_size = loss(outputs, y)
#                 loss_size.backward()
#                 optimizer.step()
#                 train_train_running_loss += loss_size.data
#                 progress.update()
#
#             Val_outputs = model(val_data)
#
#             val_loss_size = loss(Val_outputs, val_labels)
#             total_val_loss = val_loss_size.data
#
#             val_temp_acc = calculate_accuracy(model, val_data, val_labels)
#             train_temp_acc = calculate_accuracy(model, train_data, train_labels)
#
#             if val_temp_acc > best_accuracy:
#                 best_accuracy = val_temp_acc
#
#             val_acc.append(val_temp_acc)
#             train_acc.append(train_temp_acc)
#             val_loss.append(round(float(total_val_loss), 4))
#             train_loss.append(round(float(train_train_running_loss) / (N / batch_size), 4))
#
#             print(
#                 "\n\nEpoch {}/{}, Train_Loss: {:.3f}, Train_Accuracy: {:.3f}, Val_Loss: {:.3f}, Val_Accuracy: {:.3f}".format(
#                     epoch + 1,
#                     n_epochs,
#                     round(float(train_train_running_loss) /
#                           (N / batch_size), 5),
#                     train_temp_acc,
#                     round(float(total_val_loss), 5),
#                     val_temp_acc))
#
#             if int(epoch + 1) % epochs_per_save == 0:
#                 if not os.path.exists(c.MODEL_DIR):
#                     os.mkdir(c.MODEL_DIR)
#
#                 save_checkpoint({
#                     'epoch': epoch + 1,
#                     'state_dict': model.state_dict(),
#                     'best_accuracy': best_accuracy,
#                     'optimizer': optimizer.state_dict(),
#                 }, is_best=val_temp_acc == best_accuracy, filename=os.path.join(c.MODEL_DIR,
#                                                                                 'saved_' +
#                                                                                 str(epoch + 1) + '_model.pt'))
#             epoch += 1
#
#     plot_loss_acc(train_loss, train_acc, val_loss, val_acc)
#

def train_with_loader(train_loader, validation_loader):
    model = C3D()

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
    val_loss = []
    val_acc = []
    train_acc = []

    train_best_accuracy = 0.0

    for epoch in range(n_epochs):

        train_running_loss = 0.0
        train_running_accuracy = 0.0
        val_running_loss = 0.0
        val_running_acc = 0.0

        # Step the lr scheduler each epoch!
        scheduler.step()
        for i, data in enumerate(train_loader, 1):

            t0 = time.time()
            # get the inputs
            train_input, train_labels = data
            if torch.cuda.is_available():
                train_input, train_labels = Variable(train_input.cuda()), Variable(train_labels.cuda())
            else:
                train_input, train_labels = Variable(train_input), Variable(train_labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(train_input)

            # Loss
            train_loss_ = loss_criterion(outputs, train_labels)

            # forward, backward & optimization
            train_loss_.backward()

            optimizer.step()

            # best accuracy
            train_accuracy = calculate_accuracy(model, train_input, train_labels)
            if train_accuracy > train_best_accuracy:
                train_best_accuracy = train_accuracy

            # adding to loss for the batch
            train_running_loss += train_loss_.data
            train_running_accuracy += train_accuracy

            # print statistics
            t1 = time.time()
            duration_estimate = t1 - t0

            # update the tqdm bar
            # progress.update()

        # Get the validation loss and accuracy
        for i, data in enumerate(validation_loader, 1):

            val_input, val_labels = data
            if torch.cuda.is_available():
                val_input, val_labels = Variable(val_input.cuda()), Variable(val_labels.cuda())
            else:
                val_input, val_labels = Variable(val_input),Variable(val_labels)

            val_outputs = model(val_input)

            val_loss_ = loss_criterion(val_outputs, val_labels)
            val_acc_ = calculate_accuracy(model, val_input, val_labels)

            val_running_loss += val_loss_.data
            val_running_acc += val_acc_

        train_loss.append(train_running_loss)
        train_acc.append(train_running_accuracy)

        val_loss.append(val_running_loss)
        val_acc.append(val_running_acc)

        if int(epoch + 1) % c.EPOCHS_PER_SAVE == 0:
            if not os.path.exists(c.MODEL_DIR):
                os.mkdir(c.MODEL_DIR)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_accuracy': train_best_accuracy,
                'optimizer': optimizer.state_dict(),
            },
                is_best=val_running_acc == train_best_accuracy,
                filename=os.path.join(c.MODEL_DIR,
                                      'saved_' +
                                      str(epoch + 1) + '_model.pt'))

        print(
            "epoch {:2d}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, "
            "Val Loss: {:.4f}, Val Accuracy: {:.4f}".format(
                epoch + 1, train_running_loss / c.BATCH_SIZE, train_running_accuracy,
                val_running_loss / c.BATCH_SIZE, val_running_acc))

        plot_loss_acc(train_loss, train_acc, val_loss, val_acc)


if __name__ == '__main__':
    cube = FeatureCube(c.CUBE_SHAPE)

    transform = transforms.Compose([CMVN(), cube, ToTensor()])

    dataset = AudioDataset(c.DATA_TEMP + 'samples_paths.txt', c.DATA_TEMP, transform=transform)

    train_loader, validation_loader = split_sets(dataset)

    train_with_loader(train_loader, validation_loader)
