import torch.nn as nn
import torch.nn.functional as F

##
# Should be copid from @ds_train/CNN_Training.py > Net()
##
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Convolutional layer(s) with 16 filters, kernel size of 3x3, and padding of 1
        # Batch normalization layer for each convolutional layer
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        # Max pooling layer with a kernel size of 2x2 and stride of 2
        self.pool1 = nn.MaxPool2d(2, 2)

        # Convolutional layer(s) with 32 filters, kernel size of 3x3, and padding of 1
        # Batch normalization layer for each convolutional layer
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        # Max pooling layer with a kernel size of 2x2 and stride of 2
        self.pool2 = nn.MaxPool2d(2, 2)

        # Convolutional layer(s) with 64 filters, kernel size of 3x3, and padding of 1
        # Batch normalization layer for each convolutional layer
        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        # Max pooling layer with a kernel size of 2x2 and stride of 2
        self.pool3 = nn.MaxPool2d(2, 2)


        # Fully connected layers with 1024 and 256 output neurons
        # Batch normalization layer for each fully connected layer
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn8 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 31)

    def forward(self, x):
        # Convolutional layer 1 with batch normalization, followed by ReLU activation
        x = F.relu(self.bn1(self.conv1(x)))

        # Max pooling layer 1 with a kernel size of 2x2 and stride of 2
        # Convolutional layer 2 with batch normalization, followed by ReLU activation
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))


        # Convolutional layer 3 with batch normalization, followed by ReLU activation
        x = F.relu(self.bn3(self.conv3(x)))

        # Max pooling layer 2 with a kernel size of 2x2 and stride of 2
        # Convolutional layer 4 with batch normalization, followed by ReLU activation
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))


        # Convolutional layer 5 with batch normalization, followed by ReLU activation
        x = F.relu(self.bn5(self.conv5(x)))

        # Max pooling layer 3 with a kernel size of 2x2 and stride of 2
        # Convolutional layer 6 with batch normalization, followed by ReLU activation
        x = self.pool3(F.relu(self.bn6(self.conv6(x))))


        # Flatten the output of the convolutional layers
        x = x.view(-1, 64 * 8 * 8)
        # Fully connected layers with batch normalization, followed by ReLU activation
        x = F.relu(self.bn7(self.fc1(x)))
        x = F.relu(self.bn8(self.fc2(x)))

        # Output layer with no activation function applied
        x = self.fc3(x)
        return x

net = Net()