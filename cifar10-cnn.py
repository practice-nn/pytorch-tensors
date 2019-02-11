#%%
import torch
import torchvision # includes datasets
import torchvision.transforms as transforms

#%%
trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transforms.ToTensor())

#%%
trainset

#%%
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2) # provides iterators of the dataset; shuffling helps prevent picking up arbitrary patterns due to order of data being inputed

#%%
testset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transforms.ToTensor())
testset

#%%
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

#%%
labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#%%
import matplotlib.pyplot as plt
import numpy as np
images_batch, labels_batch = iter(trainloader).next()
images_batch.shape

#%%
img = torchvision.utils.make_grid(images_batch) # use torchvision utility to make a grid of images in this batch
img.shape # 8 images side-by-side with 2 pixels padding: 32+2*2=36

#%%
# to display the images using matplotlib we need the channels to be the last dimension
np.transpose(img, (1, 2, 0)).shape # height, width, number of channels
plt.imshow(np.transpose(img, (1, 2, 0))) # temporary reshaping
plt.axis('off')
plt.show()

#%%
import torch.nn as nn
in_size = 3
hid1_size = 16 # no. channels output by the first convolution layer
hid2_size = 32 # feature maps
out_size = len(labels) # no. of categories in this classification, 10
k_conv_size = 5 # 5x5 convolution kernel for the convolutional layers

#%%
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # channels need only be specified in the input and output
        # the size of the inputs, including the batch size will be automatically inferred by the convolutional layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_size, hid1_size, k_conv_size), # 3 input features and produces 16 output features
            nn.BatchNorm2d(hid1_size), # normalize the outputs of this layer for one batch so they have 0 mean and unit variance; only need the no. of channels - the batch size, height and width of the input can be inferred
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)) # 2x2 kernel

        self.layer2 = nn.Sequential(
            nn.Conv2d(hid1_size, hid2_size, k_conv_size), # size of output from previous layer as input here
            nn.BatchNorm2d(hid2_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)) # pooling layers do not change the number of input features

        self.fc = nn.Linear(hid2_size * k_conv_size * k_conv_size, out_size) # 32 x 5 x 5 input is smaller because of the pooling;
        # number of feature maps or convolutional layers X
        # size of image after passing through 2 convolutional and 2 pooling layers

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1) # reshape the output so each image is represented as 1D vector to feed into the linear layer
        out = self.fc(out)
        return out


#%%
model = ConvNet()
learning_rate = 0.001
criterion = nn.CrossEntropyLoss() # the distance between probability distributions, usually the loss function used in classification models
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#%%
total_step = len(trainloader)
num_epochs = 5
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader): # one batch of images at a time
        outputs = model(images)
        loss = criterion(outputs, labels) # calculate loss using the cross entropy loss function
        optimizer.zero_grad() # zero out gradient of optimizer before backward pass
        loss.backward()
        optimizer.step() # to update the model parameters

        if (i + 1) % 2000 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

#%%
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # find the output with the highest probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the model on the 10000 test images: {}%'.format(100 * correct / total))

#%%
