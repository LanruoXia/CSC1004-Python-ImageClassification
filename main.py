from __future__ import print_function
import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from utils.config_utils import read_args, load_config, Dict2Object


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    """
    tain the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :return:
    """
    file_train = open('files/training_record.txt', 'w')
    model.train()
    accuracies = []  # append accuracy value of each batch
    losses = []  # append loss value of each batch
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # TODO problem: for F.nll_loss, 0D or 1D target tensor expected, multi-target not supported?
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        '''Fill your code'''

        correct_count = 0
        for k in range(0, output.shape[0]):
            # print(output[k],target[k])
            if torch.argmax(output[k], -1).item() == target[k].item():  # check if yhat corresponds to y
                correct_count += 1
        batch_accuracy = correct_count / train_loader.batch_size
        accuracies.append(batch_accuracy)
        losses.append(loss.item())
        # record loss and accuracy of each batch in this epoch
        # file_train = open('files/training_record.txt', 'a')
        # file_train.write(f'batch {batch_idx + 1}/{len(train_loader)}: ' + "Loss: " + str(loss.item())
        #                  + "Accuracy: " + str(batch_accuracy) + "\n")
        print(f'batch {batch_idx + 1}/{len(train_loader)}: ' + "Loss: " + str(loss.item())
              + " Accuracy: " + str(batch_accuracy) + "\n")
        # file_train.close()

    # training_acc, training_loss = None, None  # replace this line
    # calculating average accuracy and loss of all batches in each epoch
    training_acc = sum(accuracies) / len(accuracies)
    training_loss = sum(losses) / len(losses)
    return training_acc, training_loss


def test(model, device, test_loader):
    """
    test the model and return the tesing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            '''Fill your code'''
            data, target = data.to(device), target.to(device)
            output = model(data) # torch.size([batch size, 10]
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            for k in range(0, output.shape[0]):
                # print(output[k],target[k])
                if torch.argmax(output[k], -1).item() == target[k].item():  # check if yhat corresponds to y
                    correct += 1

            # pass
    testing_acc = correct/(len(test_loader)*test_loader.batch_size)
    # TODO way to calculate training loss?
    testing_loss = test_loss/(len(test_loader)*test_loader.batch_size)
    print(testing_loss)
    return testing_acc, testing_loss


def plot(epoches, performance, type):
    """
    plot the model peformance
    :param epoches: recorded epoches
    :param performance: recorded performance
    :return:
    """
    """Fill your code"""
    plt.plot(epoches, performance)
    plt.xlabel("Epoch Number")
    plt.ylabel(type)
    plt.title(type + " (average per epoch)")
    plt.show()
    # pass


def run(config):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(config.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': config.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': config.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True, }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    """add random seed to the DataLoader, pls modify this function"""
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    """record the performance"""
    epoches = []
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    # for epoch in range(1, config.epochs + 1):
    for epoch in range(1, 5):
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch)
        """record training info, Fill your code"""
        training_loss.append(train_loss)
        training_accuracies.append(train_acc)
        file_train = open('files/training_record.txt', 'a')
        file_train.write(f'Epoch {epoch}/{config.epochs}: ' + "Loss: " + str(train_loss)
                         + "Accuracy: " + str(train_acc) + "\n")
        print(f'Epoch {epoch}/{config.epochs}: ' + "Loss: " + str(train_loss)
                         + "Accuracy: " + str(train_acc) + "\n")
        file_train.close()

        file_train_loss = open('files/training_loss_epoch.txt', 'a')
        file_train_loss.write(str(train_loss) + "\n")
        file_train_loss.close()

        file_train_acc = open('files/training_acc_epoch.txt', 'a')
        file_train_acc.write(str(train_acc) + "\n")
        file_train_acc.close()

        test_acc, test_loss = test(model, device, test_loader)
        """record testing info, Fill your code"""
        testing_accuracies.append(test_acc)
        testing_loss.append(test_loss)

        file_test_loss = open('files/testing_loss_epoch.txt', 'a')
        file_test_loss.write(str(test_loss) + "\n")
        file_test_loss.close()

        file_test_acc = open('files/testing_acc_epoch.txt', 'a')
        file_test_acc.write(str(test_acc) + "\n")
        file_test_acc.close()
        epoches.append(epoch)

        scheduler.step()
        """update the records, Fill your code"""

    """plotting training performance with the records"""
    plot(epoches, training_accuracies, "training_accuracies")
    plot(epoches, training_loss, "training_loss")

    """plotting testing performance with the records"""
    plot(epoches, testing_accuracies, "testing_accuracies")
    plot(epoches, testing_loss, "testing_loss")

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


def plot_mean():
    """
    Read the recorded results.
    Plot the mean results after three runs.
    :return:
    """
    """fill your code"""
    pass


if __name__ == '__main__':
    arg = read_args()

    """toad training settings"""
    config = load_config(arg)

    """train model and record results"""
    run(config)

    """plot the mean results"""
    # plot_mean()
