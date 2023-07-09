from PIL import Image, ImageOps, ImageFilter
import numpy as np
from scipy.ndimage import center_of_mass
from torchvision import datasets, transforms
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
import matplotlib.font_manager as fm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

##
# Check if GPU is available, use it for computation if available else use CPU
##
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device - ", device)

##
# Define 'Process' class for image processing
##
class Process(object):
    def __call__(self, img):
        # Convert the image to grayscale
        convertedImg = img.convert("L")
        # Invert the image (convert white pixels to black and black pixels to white)
        #invertedImg = ImageOps.invert(convertedImg)
        # Apply Max filter to the inverted image (remove small noise and enhance the edges of the image)
        #filteredImg = invertedImg.filter(ImageFilter.MaxFilter(5))
        #filteredImg = convertedImg.filter(ImageFilter.MaxFilter(5))
        # Resize the image to 48x48 using Lanczos interpolation
        #resizeRatio = 48.0 / max(filteredImg.size)
        resizeRatio = 48.0 / max(convertedImg.size)
        #newSize = tuple([int(round(x * resizeRatio)) for x in filteredImg.size])
        newSize = tuple([int(round(x * resizeRatio)) for x in convertedImg.size])
        #resizeImg = filteredImg.resize(newSize, Image.LANCZOS)
        resizeImg = convertedImg.resize(newSize, Image.LANCZOS)

        # Convert the resized image to a numpy array
        resizeImgArray = np.asarray(resizeImg)
        # Find the center of mass of the image
        com = center_of_mass(resizeImgArray)
        # Create a new image of size 64x64
        result = Image.new("L", (64, 64))
        # Calculate the top-left corner of the resized image in the new image
        box = (int(round(32.0 - com[1])), int(round(32.0 - com[0])))
        # Paste the resized image in the new image with calculated box coordinates
        result.paste(resizeImg, box)
        return result

##
# Define a series of image transformations to be applied to the images
##
transform = transforms.Compose([Process(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

##
# Define the directory containing training, testing, validating images
##
trainDir = 'C:/Users/iitfypvmadmin/PycharmProjects/IIT-MSc-FYP-ML/ds_source/train'
validDir = 'C:/Users/iitfypvmadmin/PycharmProjects/IIT-MSc-FYP-ML/ds_source/valid'
testDir = 'C:/Users/iitfypvmadmin/PycharmProjects/IIT-MSc-FYP-ML/ds_source/test'

##
# Create a dataset objects('trainSet', 'validationSet', 'testSet') for training images
# Print the number of images in each set
##
#trainSet = datasets.ImageFolder(trainDir, transform)
#validationSet = datasets.ImageFolder(validDir, transform)
#testSet = datasets.ImageFolder(testDir, transform)
#print("Complete Training Set - ", len(trainSet))
#print("Complete Validation Set - ", len(validationSet))
#print("Complete Test Set - ", len(testSet))

##
# Load train, validation and test data using PyTorch DataLoader
##
#trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True)
#validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=64, shuffle=True)
#testLoader = torch.utils.data.DataLoader(testSet, batch_size=64, shuffle=True)



#####

trainingSet = datasets.ImageFolder(trainDir, transform)
print("Full Train Set - ", len(trainingSet))

trainsize = int(round(0.8 * len(trainingSet)))
trainSet, validationSet = torch.utils.data.random_split(trainingSet, [trainsize, len(trainingSet) - trainsize],
                                                        generator=torch.Generator().manual_seed(42))
print("Train Set - ", len(trainSet))
print("Validation Set - ", len(validationSet))
testSet = datasets.ImageFolder(testDir, transform)
print("Test Set - ", len(testSet))

trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True)
validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=64, shuffle=True)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=64, shuffle=True)

#####



##
# Load mapping of Unicode values to characters using pandas DataFrame
##
df = pd.read_csv('CharacterMapping.csv', header=0)

##
# Convert Unicode values to characters
##
unicodeList = df["Unicode"].tolist()
char_list = []

for element in unicodeList:
    codeList = element.split()
    charsTogether = ""
    for code in codeList:
        hex = "0x" + code
        charInt = int(hex, 16)
        character = chr(charInt)
        charsTogether += character
    char_list.append(charsTogether)

##
# Create a list of available classes (characters)
# Print the classes list
# !!! Update the count to all language classes once system is in an acceptable state
##
classes = []
for i in range(len(char_list)):
    index = int(testSet.classes[i])
    char = char_list[index]
    classes.append(char)

print("Available Classes", classes)

##
# Initialize the weights of the network
##
def initializeWeights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Initialize weights using Kaiming Normal initialization
        nn.init.kaiming_normal_(m.weight)
        # Initialize bias terms to zeros
        nn.init.zeros_(m.bias)

##
# Neural network architecture:
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
        #print('Input size:', x.size())
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
        #print('After flattening:', x.size())
        # Fully connected layers with batch normalization, followed by ReLU activation
        x = F.relu(self.bn7(self.fc1(x)))
        x = F.relu(self.bn8(self.fc2(x)))

        #if x.size(0) == 1 and x.size(1) == 1024:
        #    x = F.relu(self.fc1(x))
        #    x = F.relu(self.fc2(x))
        #else:
        #    x = F.relu(self.bn7(self.fc1(x)))
            #print('After fc1:', x.size())
        #    x = F.relu(self.bn8(self.fc2(x)))
            #print('After fc2:', x.size())

        # Output layer with no activation function applied
        x = self.fc3(x)
        #print('Output size:', x.size())
        return x

##
# Define a function to get all predictions of a given neural network on a given data loader
# @torch.no_grad() - This decorator indicates that gradients should not be computed during the function call
##
@torch.no_grad()
def get_all_preds(model, loader):
    # Create an empty tensor to store all predictions
    all_preds = torch.tensor([])
    for batch in loader:
        # Get input images and corresponding labels from the data loader
        images, labels = batch

        # Get predictions from model
        preds = model(images)
        all_preds = torch.cat(
            # Concatenate current batch's predictions with all previous predictions
            # Concatenate along the 0th dimension (batch dimension)
            (all_preds, preds)
            , dim=0
        )
    # Return tensor containing all predictions for all data in the loader
    return all_preds

##
# Function to plot a confusion matrix
##
def plot_confusion_matrix(cm, classes, title, normalize=False, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized Confusion Matrix")
    else:
        print('Non-normalized Confusion Matrix')

    print(title)
    print(cm)

    # Load a font for plotting tick labels in non-latin scripts
    prop = fm.FontProperties(fname='Nirmala.ttf')

    # Plot the confusion matrix as an image
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # Set tick labels for x-axis and y-axis
    plt.xticks(tick_marks, classes, rotation=45, fontproperties=prop)
    plt.yticks(tick_marks, classes, fontproperties=prop)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # Add text annotations to each cell of the confusion matrix
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

##
# Create a new neural network model, initialize its weights
# Move it to the specified device (e.g. GPU)
# Print its architecture
##
net = Net()
net.apply(initializeWeights)
net.to(device)
print("Net - ", net)

##
# Define the Cross Entropy loss function and optimizer
##
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), weight_decay=0.003, lr=0.001)

##
# Create lists for storing the data to be plotted
##
x = []
train_loss = []
val_loss = []
train_accuracy = []
val_accuracy = []
fig, axs = plt.subplots(3)
fig.suptitle('Accuracy')

##
# Create two lists for plotting the running losses
##
x_two = []
running_losses = []

##
# Loop through 30 epochs
# !!! For now, it loops through 5 for the code testing and till the system is in an acceptable state
##
#for epoch in range(30): 24
print('----------------------------------------------------')
for epoch in range(62):
    # Append current epoch number to the x list
    x.append(epoch)

    # Initialize training metrics for the current epoch
    curr_train_loss = 0.0
    train_total = 0
    train_correct = 0
    running_loss = 0.0

    # Loop through the training data
    for i, data in enumerate(trainLoader, 0):
        # Move input and labels to the device
        inputs, labels = data[0].to(device), data[1].to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)
        # Get the predicted class labels
        _, predicted = torch.max(outputs.data, 1)
        # Update training metrics
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # Calculate the loss
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()
        # Optimize the weights
        optimizer.step()
        # Update the current training loss
        curr_train_loss += loss.item()

        # Update the running loss for plotting
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
        running_losses.append(running_loss)
        x_two.append(epoch + i * 64 / len(trainSet))
        running_loss = 0.0

    # Calculate the average training loss and accuracy for the current epoch
    train_loss.append(curr_train_loss / len(trainSet) * 64)
    train_accuracy.append(100 * train_correct / train_total)

    # Initialize validation metrics for the current epoch
    val_correct = 0
    val_total = 0
    curr_val_loss = 0.0

    # Loop through the validation data
    with torch.no_grad():
        for data in validationLoader:
            # Move input and labels to the device
            images, labels = data[0].to(device), data[1].to(device)
            # Forward pass
            outputs = net(images)
            # Get the predicted class labels
            _, predicted = torch.max(outputs.data, 1)
            # Update validation metrics
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            # Calculate the loss
            curr_val_loss += criterion(outputs, labels).item()

    # Calculate the average validation loss and accuracy for the current epoch
    val_loss.append(curr_val_loss / len(validationSet) * 64)
    val_accuracy.append(100 * val_correct / val_total)

    # Print the training and validation metrics for the current epoch
    print('EPOCH              : ' + str(epoch + 1))
    print('Training loss      : ' + str(train_loss[-1]))
    print('Training accuracy  : ' + str(train_accuracy[-1]) + "%")
    print('Validation loss    : ' + str(val_loss[-1]))
    print('Validation accuracy: ' + str(val_accuracy[-1]) + "%")
    print('----------------------------------------------------')

    # Plot the training and validation losses over the epochs
    # Plot the training and validation accuracies over the epochs
    # Plot the running loss over the iterations
    axs[0].plot(x, train_loss, 'r-', val_loss, 'b-')
    axs[1].plot(x, train_accuracy, 'r-', val_accuracy, 'b-')
    axs[2].plot(x_two, running_losses)

# Display the plot
plt.show()

##
# Get an iterator for the test data loader
##
dataIteratorTest = iter(testLoader)
# get a batch of test data
data_thing = dataIteratorTest.__next__()
# extract the images and labels from the batch and move them to the device
images, labels = data_thing[0].to(device), data_thing[1].to(device)

##
# Move the images to the cpu
# Print the ground truth labels for the first 10 images
##
imgs = images.cpu()
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(10)))

##
# Make predictions on the first 10 images and print the predicted labels
##
outputs = net(images[:10])
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(10)))

##
# Initialize variables for tracking accuracy
##
lableList = []
predictedList = []
correct = 0
total = 0
# Turn off gradients and loop through the test data to make predictions
with torch.no_grad():
    for data in testLoader:
        # Get a batch of test data
        images, labels = data[0].to(device), data[1].to(device)
        # Append the labels to the label list
        lableList.extend(labels)
        # Make predictions and append the predicted labels to the predicted list
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        predictedList.extend(predicted)
        # Track accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %f %%' % (100 * correct / total))

##
# Turn off gradients and make predictions on the entire test set
##
with torch.no_grad():
    test_prediction_loader = torch.utils.data.DataLoader(testSet, batch_size=1240)
    test_preds = get_all_preds(net, test_prediction_loader)

##
# Calculate and plot the confusion matrix for the test set
##
cmTesting = confusion_matrix(testSet.targets, test_preds.argmax(dim=1))
plt.figure(figsize=(31, 31))
plot_confusion_matrix(cmTesting, classes, "Confusion Matrix for Test Set")

##
# Print the classification report for the test set
##
print("Classification report")
print(classification_report(lableList, predictedList))

##
# Function for getting the actual labels and class probabilities for a given class
##
def test_class_probabilities(which_class):
    actuals = []
    probabilities = []
    with torch.no_grad():
        for data, target in testLoader:
            # Get a batch of test data
            data, target = data.to(device), target.to(device)
            # Make predictions and append the actual labels and class probabilities
            output = net(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction) == which_class)
            probabilities.extend(np.exp(output[:, which_class]))
    return [i.item() for i in actuals], [i.item() for i in probabilities]

##
# Function for plotting the ROC curve for a list of classes
##
def plot_AUC_ROC_curve(classList):
    colors = ['blue', 'darkorange', 'red', 'yellow', 'green', 'pink']
    plt.figure()
    lw = 2

    j = 0
    for i in classList:
        # Get the actual labels and predicted class probabilities for the current class
        actuals, class_probabilities = test_class_probabilities(i)
        # Compute the false positive rate, true positive rate, and AUC for the ROC curve
        fpr, tpr, _ = roc_curve(actuals, class_probabilities)
        roc_auc = auc(fpr, tpr)
        # Plot the ROC curve for the current class
        plt.plot(fpr, tpr, color=colors[j], lw=lw, label='Class ' + str(i)
                                                         + 'ROC curve (area = %0.2f)' % roc_auc)
        j = j + 1

    # Plot the diagonal line for the ROC curve of a random classifier
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # Set the x and y axis limits for the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for ' + str(classList) + ' classes')
    # Add a legend to the plot to identify the ROC curve for each class
    plt.legend(loc="lower right")
    plt.show()

##
# Plot ROC curves for multiple sets of classes
# !!! Once system in an acceptable state, for final training, add below codes for all classes in both sinhala and tamil dataset classes
##
plot_AUC_ROC_curve([0, 1, 2, 3, 4])
plot_AUC_ROC_curve([5, 6, 7, 8, 9])
#plot_AUC_ROC_curve([11, 12, 13, 14, 15])
#plot_AUC_ROC_curve([16, 17, 18, 19, 20])
#plot_AUC_ROC_curve([21, 22, 23, 24, 25])
#plot_AUC_ROC_curve([26, 27, 28, 29, 30])

##
# Save the trained PyTorch model
##
torch.save(net.state_dict(), '../ds_trained/SinhalaTamil_CNN_Trained.pt')