#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
x_train = np.array([[3.3], [4.4], [6.71], [4.168], [9.779], [3.1]], dtype = np.float32)
y_train = np.array([[1.6], [3.5], [6.2], [23.2], [13.2], [5.3]], dtype = np.float32)

#%%
plt.plot(x_train, y_train, 'ro', label = 'Original data')
plt.show()

#%%
import torch
X_train = torch.from_numpy(x_train)
Y_train = torch.from_numpy(y_train)
print('requires_grad for X_train: ', X_train.requires_grad) # False, because it's not needed - only variables that are trained during the training phase need autograd on
print('requires_grad for Y_train:', Y_train.requires_grad)


#%%
input_size = 1 # each data point is a single feature i.e.: represented by a single value
hidden_size = 100 # handcraft a neural network with a single layer with 100 neurons and ReLU activation
output_size = 1 # only one output
learning_rate = 1e-6 # the factor by which we adjust our weights for every epoch; calculate gradients in backward pass to manually adjust the weights

w1 = torch.rand(input_size, hidden_size, requires_grad=True) # turn on gradients during training
w1.size()

#%%
w2 = torch.rand(hidden_size, output_size, requires_grad=True)
w2.size()


#%%
for iter in range(1, 301):
    y_pred = X_train.mm(w1).clamp(min=0).mm(w2) # mm: matrix multiply the activation function; clamp forces all negative values to 0 (ReLU); then matrix multiply with the second weight
    loss = (y_pred - Y_train).pow(2).sum() # mean square error loss, sum all the square of the differences between actual and predicted values
    if iter % 50 == 0:
        print(iter, loss.item())

    loss.backward() # use autograd for an automated way of implementing the backward pass through the neural network; computes gradient of loss with respect to all tensors which have requires_grad=True

    # tweak the nn to make better predictions
    with torch.no_grad(): # don't need gradients when we adjust the model
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()


#%%
print('w1: ', w1) # final model parameters
print('w1: ', w1)


#%%
x_train_tensor = torch.from_numpy(x_train)
x_train_tensor


#%%
predicted_in_tensor = x_train_tensor.mm(w1).clamp(min=0).mm(w2)
predicted_in_tensor

#%%
predicted = predicted_in_tensor.detach().numpy() # detach the tensor from the current graph so no gradients are computed on the new tensor
predicted

#%%
plt.plot(x_train, y_train, 'ro', label = 'Original data')
plt.plot(x_train, predicted, label = 'Fitted line ')
plt.legend()
plt.show()
