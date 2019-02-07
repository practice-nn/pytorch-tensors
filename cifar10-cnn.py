#%%
import torch
import torchvision # includes datasets
import torchvision.transforms as transforms

#%%
trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transforms.ToTensor())

#%%
trainset

#%%
