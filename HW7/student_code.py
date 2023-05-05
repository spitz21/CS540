# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # certain definitions
        self.c1 = nn.Conv2d(in_channels=3,out_channels=6, kernel_size=5, stride=(1, 1), padding=(0, 0))
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=(2, 2), padding=(0, 0))

        self.c2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=(1, 1), padding=(0, 0))
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(kernel=2, padding=(0, 0), stride=(2, 2))

        self.flatten = nn.Flatten()
        self.ll1 = nn.Linear(in_features=400, out_features=256)
        self.relu3 = nn.ReLU()
        self.ll2 = nn.Linear(in_features=256, out_features=128)
        self.relu4 = nn.ReLU()
        self.ll3 = nn.Linear(in_features=128, out_features=num_classes)
    def forward(self, x):
        shape_dict = {}
        # certain operations
        #convolute, maxpool,  add to dict
        x = nn.functional.relu(self.c1(x))
        x = self.mp1(x)
        shape_dict[1] = list(x.shape)

        #convulve, maxpool, add to dict
        x = nn.functional.relu(self.c2)
        x = self.mp2(x)
        shape_dict[2] = list(x.shape)

        #flatten, add to dict
        x = x.view(-1, 400)
        shape_dict[3] = list(x.shape)

        #full connect with, add to dict
        x = nn.functional.relu(self.ll1(x))
        shape_dict[4] = list(x.shape)

        #full connect with relu, add to dict
        x = nn.functional.relu(self.ll2(x))
        shape_dict[5] = list(x.shape)

        #full connect without relu, add to dict
        x = self.ll3(x)
        shape_dict[6] = list(x.shape)

        out = x

        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0

    for pname, p in model.named_parameters():
        model_params += p

    return (model_params / 1000000)


    return model_params


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
