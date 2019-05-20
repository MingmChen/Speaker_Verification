from model import C3D
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import time
from utils import *
from load_data import AudioDataset
import numpy as np



def train_with_loader(train_loader, validation_loader, n_labels):
    model = C3D(n_labels)

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

            _, predictions = torch.max(outputs, dim=1)

            correct_count = (predictions == train_labels).double().sum().item()
            train_accuracy = float(correct_count) / c.BATCH_SIZE
            # train_accuracy = calculate_accuracy(model, train_input, train_labels)

            # backward & optimization
            train_loss_.backward()
            optimizer.step()

            # best accuracy
            if train_accuracy > train_best_accuracy:
                train_best_accuracy = train_accuracy

            # adding to loss for the batch
            train_running_loss += train_loss_.data.item()
            train_running_accuracy += train_accuracy

            # Print stats every 5 batches
            if i % c.BATCH_PER_LOG == 0:
                print((
                        'epoch {:2d} ' +
                        '|| batch {:2d} of {:2d} ||' +
                        ' Batch-Loss: {:.4f} ||' +
                        ' Batch-Accuracy: {:.4f}\n').format(
                    epoch + 1,
                    i,
                    len(train_loader),
                    train_running_loss / c.BATCH_PER_LOG,
                    train_accuracy
                ),
                    end=' ')

        end = time.time()
        duration_estimate = end - start

        # Get the validation loss and accuracy
        for i, data in enumerate(validation_loader, 1):

            val_input, val_labels = data
            if torch.cuda.is_available():
                val_input, val_labels = Variable(val_input.cuda()), Variable(val_labels.cuda())
            else:
                val_input, val_labels = Variable(val_input), Variable(val_labels)

            val_outputs = model(val_input)

            val_loss_ = loss_criterion(val_outputs, val_labels)
            # val_acc_ = calculate_accuracy(model, val_input, val_labels)

            _, predictions = torch.max(val_outputs, dim=1)
            correct_count = (predictions == val_labels).double().sum().item()
            val_acc_ = float(correct_count) / c.BATCH_SIZE

            val_running_loss += val_loss_.data.item()
            val_running_acc += val_acc_


        train_loss.append(train_running_loss/c.BATCH_SIZE)
        train_acc.append(float(100.0 * train_running_accuracy/len(train_loader)))

        val_loss.append(val_running_loss/c.BATCH_SIZE)
        val_acc.append(float(100.0 * val_running_acc / len(validation_loader)))

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
                is_best=val_running_acc == train_best_accuracy,
                filename=os.path.join(
                    c.MODEL_DIR,
                    'saved_' +
                    str(epoch + 1) + '_model.pt'
                )
            )

        print(
            "epoch {:2d}/{:2d}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, "
            "Val Loss: {:.4f}, Val Accuracy: {:.4f}, Time: {:.4f}".format(
                epoch + 1,
                n_epochs,
                train_running_loss / c.BATCH_SIZE,
                float(100.0 * train_running_accuracy/len(train_loader)),
                val_running_loss / c.BATCH_SIZE,
                float(100.0 * val_running_acc / len(validation_loader)),
                duration_estimate
            )
        )

    plot_loss_acc(train_loss, train_acc, val_loss, val_acc)


if __name__ == '__main__':
    indexed_labels = np.load(c.ROOT + '/labeled_indices.npy').item()

    cube = FeatureCube(c.CUBE_SHAPE)

    transform = transforms.Compose([CMVN(), cube, ToTensor()])

    dataset = AudioDataset(
        c.DATA_ORIGIN + 'train_paths.txt',
        c.DATA_ORIGIN + 'wav/',
        indexed_labels=indexed_labels,
        transform=transform)

    train_loader, validation_loader = split_sets(dataset)

    train_with_loader(train_loader, validation_loader, len(indexed_labels.keys()))
