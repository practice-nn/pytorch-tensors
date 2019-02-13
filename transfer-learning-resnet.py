#%%
import torch
from torchvision import datasets, models, transforms
#%%
mean = [0.485, 0.456, 0.406] # images fed to pre-trained models have to be normalized using these parameters https://pytorch.org/docs/stable/torchvision/models.html#id3
std = [0.229, 0.224, 0.225]

#%%
train_transform = transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)]) # perform arbitrary transforms and normalize the input images to be fed into the pretrained model

test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)]) # no random flipping or cropping

#%%
import zipfile
zip = zipfile.ZipFile('C:/Users/pcuci/Downloads/pytorch-building-deep-learning-models/datasets/flowers_.zip')
zip.extractall('datasets')
data_dir = 'datasets/flowers_'
image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(data_dir + '/train', train_transform) # applies a series of tranformations to an image folder path
image_datasets['test'] = datasets.ImageFolder(data_dir + '/test', test_transform)
print("Training data size: %d" % len(image_datasets['train']))
print("Test data size: %d" % len(image_datasets['test']))
#%%

class_names = image_datasets['train'].classes
class_names # 5 types of flowers

#%%
image_datasets # a dictionary with two keys: train, test

#%%
dataloaders = {} # used to iterate over the datasets
dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=8, shuffle=True, num_workers=4)
dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=8, shuffle=True, num_workers=4)
dataloaders

#%%
# input images to pre-trained models should be in the format [batch_size, num_channels, height, width]
inputs, labels = next(iter(dataloaders['train']))
inputs.shape # a 4D tensor
#%%
labels # numeric values of 0 to 4 corresponding to the 5 categories of flowers

#%%
import torchvision
inp = torchvision.utils.make_grid(inputs)
inp.shape # stacked all images side by side

#%%
inp.max() # however, Matplotlib requires floating point RGB values to be in the 0-1 range
#%%
import numpy as np
np.clip(inp, 0, 1).max()

#%%
inp.numpy().transpose((1, 2, 0)).shape # matplotlib expects channels in the last dimension
#%%
import matplotlib.pyplot as plt
plt.ion()
def img_show(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean # denormalize the image
    inp = np.clip(inp, 0, 1)

    plt.figure(figsize=(16, 4))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

#%%
img_show(inp, title=[class_names[x] for x in labels])

#%%
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
num_ftrs

#%%
import torch.nn as nn
model.fc = nn.Linear(num_ftrs, 5) # 512 features as input to classify into 5 categories; this replaces the existing linear layer in the model

import torch.optim as optim
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # use momentum to accelerate model convergence

#%%
from torch.optim import lr_scheduler # learning rate scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # the learning rate scheduler which decays the learning rate as we get close to convergence, reduce the learning rate by 0.1 every 7 epochs

#%%
def calculate_accuracy(phase, running_loss, running_corrects):
    epoch_loss = running_loss / len(image_datasets[phase])
    epoch_acc = running_corrects.double() / len(image_datasets[phase])
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    return (epoch_loss, epoch_acc)

#%%
def phase_train(model, criterion, optimizer, scheduler): # the training phase
    scheduler.step()
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders['train']:
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels) # calculate the cross entropy loss
            loss.backward() # calculate gradients
            optimizer.step() # update model parameters
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    calculate_accuracy('train', running_loss, running_corrects)

#%%
import copy
criterion = nn.CrossEntropyLoss()
best_acc = 0.0 # save only the best model parameters on test data
def phase_test(model, criterion, optimizer):
    model.eval() # to run the model in the test phase
    running_loss = 0.0
    running_corrects = 0
    global best_acc # keep track of the model weights which produce the best accuracy on the test data

    for inputs, labels in dataloaders['test']:
        optimizer.zero_grad()
        with torch.no_grad(): # don't calculate gradients in the test phase
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss, epoch_acc = calculate_accuracy('test', running_loss, running_corrects)
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
    return best_model_wts

#%%
def build_model(model, criterion, optimizer, scheduler, num_epochs=10): # train the model with the flowers dataset
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs -1))
        print('-' * 10)
        phase_train(model, criterion, optimizer, scheduler)
        best_model_wts = phase_test(model, criterion, optimizer)
        print()
    print('Best test Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts) # at the end of the training, load the model that has the best accuracy in test
    return model

#%%
model = build_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=1)


#%%
# run the model for predictions
with torch.no_grad():
    # retrieve one batch of test images
    inputs, labels = iter(dataloaders['test']).next()
    inp = torchvision.utils.make_grid(inputs) # turn them into a grid
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)

    for j in range(len(inputs)): # display the predicted label for each image
        inp = inputs.data[j]
        img_show(inp, 'predicted:' + class_names[preds[j]])

#%%
# no need to train on all the layers (more typical usecase)
frozen_model = models.resnet18(pretrained=True)
for param in frozen_model.parameters():
    param.requires_grad = False # freezes model weights so they don't get updated during training

frozen_model.fc = nn.Linear(num_ftrs, 5) # replace the last layer, which also
optimizer = optim.SGD(frozen_model.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()
best_acc = 0

#%%
frozen_model = build_model(frozen_model, criterion, optimizer, exp_lr_scheduler, num_epochs=1) # the accuracy is less because of the frozen layers

#%%
with torch.no_grad():
    inputs, labels = iter(dataloaders['test']).next()
    inp = torchvision.utils.make_grid(inputs)

    outputs = frozen_model(inputs)
    _, preds = torch.max(outputs, 1)

    for j in range(len(inputs)):
        inp = inputs.data[j]
        img_show(inp, 'predicted:' + class_names[preds[j]])

#%%
