"""Classification models in pytorch.

This module contains the following classification models:
    - Multi-class logistic regression.
    - MLP.
    - Lenet5 convolutional network.
"""

import time
import torch
from torch import nn
from torch import optim


class MultiLR(nn.Module):
    """Multi-class logistic regression.

    Contains the architecture for multi-class logistic regression.
    """
    def __init__(self, dimx, nlabels):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(dimx, nlabels),
            # Dim is the dimension along which Softmax will be computed
            # (so every slice along dim will sum to 1)
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.classifier(x)

        return x


class ExtendedMultiLR(MultiLR):
    """Extended Multi-class logistic regression.

    Extends Multi-class logistic regression and includes an optimization loop
    and method for evaluation. Also, is optimized to run in the GPU.
    """
    def __init__(self, dimx, nlabels, epochs=100, lr=0.001):
        super().__init__(dimx, nlabels)

        # Moving computations to the gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Training parameters
        self.lr = lr
        self.epochs = epochs
        self.optim = optim.Adam(self.parameters(), self.lr)
        self.criterion = nn.NLLLoss()

        # A list to store the loss evolution along training
        self.loss_during_training = []

        # List to store the loss evolution along the evaluation set
        self.valid_loss_during_training = []

    def trainloop(self, trainloader, validloader):
        # Optimization Loop
        for e in range(int(self.epochs)):
            start_time = time.time()

            running_loss = 0.

            # Random data permutation at each epoch
            for images, labels in trainloader:
                # Move input and label tensors to the default device
                images, labels = images.to(self.device), labels.to(self.device)

                # Needed to reset the gradients after each set of operations
                self.optim.zero_grad()

                out = self.forward(images.view(images.shape[0], -1))

                loss = self.criterion(out, labels)

                running_loss += loss.item()

                loss.backward()

                self.optim.step()

            self.loss_during_training.append(running_loss / len(trainloader))

            with torch.no_grad():
                running_loss = 0.

                for images, labels in validloader:
                    # Move input and label tensors to the default device
                    images, labels = images.to(self.device), labels.to(
                        self.device)

                    out = self.forward(images.view(images.shape[0], -1))

                    loss = self.criterion(out, labels)

                    running_loss += loss.item()

                self.valid_loss_during_training.append(
                    running_loss / len(validloader))

            if e % 1 == 0:  # Every 10 epochs
                print("Epoch %d. Training loss: %f, "
                      "Validation loss: %f, Time per epoch: %f seconds"
                      % (e, self.loss_during_training[-1],
                         self.valid_loss_during_training[-1],
                         (time.time() - start_time)))

    def eval_performance(self, dataloader):
        accuracy = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for images, labels in dataloader:
                # Move input and label tensors to the default device
                images, labels = images.to(self.device), labels.to(self.device)

                probs = self.forward(images.view(images.shape[0], -1))

                top_p, top_class = probs.topk(1, dim=1)
                equals = (top_class == labels.view(images.shape[0], 1))
                accuracy += torch.mean(equals.type(torch.FloatTensor))

            return accuracy / len(dataloader)


class MLP(nn.Module):
    """Multilayer neural network architecture

    Contains the architecture for a multilayer neural network with three
    customizable hidden layers. Also, it contains regularization features,
    dropout layers after the first three layers of the network.
    """
    def __init__(self, dimx, hidden1, hidden2, hidden3, nlabels):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(dimx, hidden1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden3, nlabels),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.features(x)
        x = self.classifier(x)

        return x


class ExtendedMLP(MLP):
    """Extended MLP.

    Extends MLP and includes an optimization loop and a method for evaluation.
    Also, is optimized to run in the GPU.
    """
    def __init__(self, dimx, hidden1, hidden2, hidden3, nlabels, epochs=100,
                 lr=0.001):
        super().__init__(dimx, hidden1, hidden2, hidden3, nlabels)

        # Moving computations to the gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Training parameters
        self.epochs = epochs
        self.lr = lr
        self.optim = optim.Adam(self.parameters(), self.lr)
        self.criterion = nn.NLLLoss()

        # A list to store the loss evolution along training
        self.loss_during_training = []

        # A list to store the loss evolution along the evaluation set
        self.valid_loss_during_training = []

    def trainloop(self, trainloader, validloader):

        # Optimization Loop
        for e in range(int(self.epochs)):
            start_time = time.time()

            running_loss = 0.

            # Random data permutation at each epoch
            for images, labels in trainloader:
                # Move input and label tensors to the default device
                images, labels = images.to(self.device), labels.to(self.device)

                # Needed to reset the gradients after each set of operations
                self.optim.zero_grad()

                out = self.forward(images.view(images.shape[0], -1))

                loss = self.criterion(out, labels)

                running_loss += loss.item()

                loss.backward()

                self.optim.step()

            self.loss_during_training.append(running_loss / len(trainloader))

            # Compute the loss in the validation set
            running_loss = 0.

            with torch.no_grad():
                # Set model to evaluation mode (as it contains dropout layers)
                self.eval()

                for images, labels in validloader:
                    # Move input and label tensors to the default device
                    images, labels = images.to(self.device), labels.to(
                        self.device)
                    out = self.forward(images.view(images.shape[0], -1))

                    loss = self.criterion(out, labels)

                    running_loss += loss.item()

            # Set model back to train mode
            self.train()

            self.valid_loss_during_training.append(
                running_loss / len(validloader))

            if e % 1 == 0:  # Every 10 epochs
                print("Epoch %d. Training loss: %f, "
                      "Validation loss: %f, Time per epoch: %f seconds"
                      % (e, self.loss_during_training[-1],
                         self.valid_loss_during_training[-1],
                         (time.time() - start_time)))

    def eval_performance(self, dataloader):
        accuracy = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            # Set model to evaluation mode
            self.eval()

            for images, labels in dataloader:
                # Move input and label tensors to the default device
                images, labels = images.to(self.device), labels.to(self.device)

                probs = self.forward(images.view(images.shape[0], -1))

                top_p, top_class = probs.topk(1, dim=1)
                equals = (top_class == labels.view(images.shape[0], 1))
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        # Set model back to train mode
        self.train()

        return accuracy / len(dataloader)


class Lenet5(nn.Module):
    """Lenet5 network.

    Contains the whole architecture of the Lenet5 network and the feedforward
    method. Also, it contains regularization features: dropout layers at the end
    of dense layers for better generalization and batch normalization after
    convolutional layers.
    """

    def __init__(self, dimx, nlabels, use_batch_norm):
        super().__init__()

        self.use_batch_norm = use_batch_norm
        self.nlabels = nlabels
        self.dimx = dimx

        # Spatial dimension of the Tensor at the output of the 2nd CNN
        self.final_dim = int(((self.dimx - 4) / 2 - 4) / 2)

        self.features = []

        self.features.append(
            # Convolutional layer (sees 28x28x1 image tensor)
            nn.Conv2d(in_channels=1, out_channels=6,
                      kernel_size=5, stride=1, padding=0)
        )

        if self.use_batch_norm:
            self.features.append(nn.BatchNorm2d(6))

        self.features.extend([
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ])

        self.features.append(
            nn.Conv2d(6, 16, 5, padding=0)
        )

        if self.use_batch_norm:
            self.features.append(nn.BatchNorm2d(16))

        self.features.extend([
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ])

        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Linear(16 * self.final_dim ** 2, 120),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(84, self.nlabels),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # Pass the input tensor through the CNN operations
        x = self.features(x)

        # Flatten the tensor into a vector
        x = x.view(-1, 16 * self.final_dim ** 2)

        # Pass the tensor through the Dense Layers
        x = self.classifier(x)

        return x


class ExtendedLenet5(Lenet5):
    """Extended Lenet5 network.

    Extends Lenet5 network and includes an optimization loop and a method for
    evaluation.
    """
    def __init__(self, dimx, nlabels, use_batch_norm, epochs=100, lr=0.001):
        super().__init__(dimx, nlabels, use_batch_norm)

        # Moving computations to the gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Training parameters
        self.epochs = epochs
        self.lr = lr
        self.optim = optim.Adam(self.parameters(), self.lr)
        self.criterion = nn.NLLLoss()

        # List to store the loss evolution along training
        self.loss_during_training = []

        # List to store the loss evolution along the evaluation set
        self.valid_loss_during_training = []

    def trainloop(self, trainloader, validloader):

        # Optimization Loop
        for e in range(int(self.epochs)):
            start_time = time.time()

            running_loss = 0.

            for images, labels in trainloader:
                # Move input and label tensors to the default device
                images, labels = images.to(self.device), labels.to(self.device)

                self.optim.zero_grad()

                out = self.forward(images)

                loss = self.criterion(out, labels)

                running_loss += loss.item()

                loss.backward()

                self.optim.step()

            self.loss_during_training.append(running_loss / len(trainloader))

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():

                # Set model to evaluation mode
                self.eval()

                running_loss = 0.

                for images, labels in validloader:
                    # Move input and label tensors to the default device
                    images, labels = images.to(self.device), labels.to(
                        self.device)

                    out = self.forward(images)

                    loss = self.criterion(out, labels)

                    running_loss += loss.item()

                self.valid_loss_during_training.append(
                    running_loss / len(validloader))

                # Set model back to train mode
                self.train()

            # Log message every 10 epochs
            if e % 1 == 0:
                print("Epoch %d. Training loss: %f, "
                      "Validation loss: %f, Time per epoch: %f seconds"
                      % (e, self.loss_during_training[-1],
                         self.valid_loss_during_training[-1],
                         (time.time() - start_time)))

    def eval_performance(self, dataloader):
        loss = 0
        accuracy = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            # Set model to evaluation mode
            self.eval()

            for images, labels in dataloader:
                # Move input and label tensors to the default device
                images, labels = images.to(self.device), labels.to(self.device)

                probs = self.forward(images)

                top_p, top_class = probs.topk(1, dim=1)
                equals = (top_class == labels.view(images.shape[0], 1))
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        # Set model back to train mode
        self.train()

        return accuracy / len(dataloader)

