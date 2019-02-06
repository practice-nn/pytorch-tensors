#%%
import pandas as pd

#%%
titanic_data = pd.read_csv('C:/Users/pcuci/Downloads/pytorch-building-deep-learning-models/datasets/titanic_data/train.csv')
titanic_data.head()

#%%
unwanted_features = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Embarked'] # remove unwanted features
titanic_data = titanic_data.drop(unwanted_features, axis=1)
titanic_data.head()

#%%
titanic_data = titanic_data.dropna() # get rid also of missing information rows

#%%
from sklearn import preprocessing
le = preprocessing.LabelEncoder() # encode categorical values as numeric labels
titanic_data['Sex'] = le.fit_transform(titanic_data['Sex'])
titanic_data.head() # 0 for female, 1 male

#%%
features = ['Pclass', 'Sex', 'Age', 'Fare']
titanic_features = titanic_data[features]
titanic_features.head()

#%%
titanic_features = pd.get_dummies(titanic_features, columns=['Pclass']) # one-hot encode the passanger class
titanic_features.head()

#%%
titanic_target = titanic_data[['Survived']]
titanic_target.head() # 1 indicates survival, 0 did not survive the sinking

#%%
from sklearn.model_selection import train_test_split
X_train, x_test, Y_train, y_test = train_test_split(titanic_features, titanic_target, test_size=0.2, random_state=0) # keep 80% of data for training, the remaining 20% for testing
X_train.shape, Y_train.shape

#%%
# import training and test data into torch sensors
import torch
import numpy as np
Xtrain_ = torch.from_numpy(X_train.values).float()
Xtest_ = torch.from_numpy(x_test.values).float()
Xtrain_.shape, Xtest_.shape

#%%
# reshape our data to match the y-label format required by our loss function
# extract y-labels as a 1D tensor - one row containing all labels
Ytrain_ = torch.from_numpy(Y_train.values).view(1, -1)[0]
Ytest_ = torch.from_numpy(y_test.values).view(1, -1)[0]
Ytrain_.shape, Ytest_.shape

#%%
import torch.nn as nn
import torch.nn.functional as F # includes the log soft max function
input_size = 6 # 6 input features: age, sex, fare, and the one-hot encoded 3 pclasses
output_size = 2 # survived or not
hidden_size = 10 # one hidden layer with 10 neurons
input_size, hidden_size, output_size

#%%
# build our own custom nn modules by subclassing the nn.Module class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() # initialize the nn before we add in our layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x): # forward pass on input data x
        x = F.sigmoid(self.fc1(x)) # first read the input x into the fully connected linear layer, then apply the sigmoid function
        x = F.sigmoid(self.fc2(x)) # again
        x = self.fc3(x) # no activation function at the end
        return F.log_softmax(x, dim=-1) # better than log(softmax(x)) which is numerically unstable
        # dim=-1 is the dimension along which we want to compute softmax, -1 infers the right dimension on its own
model = Net() # instantiate our model
model.parameters

#%%
import torch.optim as optim
optimizer = optim.Adam(model.parameters()) # adam is a momentum based optimizer as a loss function
loss_fn = nn.NLLLoss() # set up the loss function

#%%
epoch_data = []
epochs = 1001

for epoch in range(1, epochs):
    optimizer.zero_grad() # zero out the gradients so it calculates fresh gradients in the forward pass
    Ypred = model(Xtrain_)

    # calculate the loss on the prediction and backpropagate to calculate new gradients
    loss = loss_fn(Ypred, Ytrain_)
    loss.backward()

    optimizer.step() # update the model parameters by applying gradients
    Ypred_test = model(Xtest_)
    loss_test = loss_fn(Ypred_test, Ytest_)
    _, pred = Ypred_test.data.max(1) # pick the highest probability, this is the predicted value
    accuracy = pred.eq(Ytest_.data).sum().item() / y_test.values.size # does predicted data match actual labels; divided by the total number of instances in our test dataset to get a percentage accuracy
    epoch_data.append([epoch, loss.data.item(), loss_test.data.item(), accuracy]) # append the info for each epoch
    if epoch % 100 == 0: # print out every 100 epochs
        print('epoch - %d (%d%%) train loss - %.2f test loss - %.2f accuracy - %.4f' % (epoch, epoch/150 * 10, loss.data.item(), loss_test.data.item(), accuracy))


#%%
df_epochs_data = pd.DataFrame(epoch_data, columns=["epoch", "train_loss", "test_loss", "accuracy"])
import matplotlib.pyplot as plt
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
df_epochs_data[["train_loss", "test_loss"]].plot(ax=ax1)
df_epochs_data[["accuracy"]].plot(ax=ax2)
plt.ylim(ymin=0.5)
plt.show()

#%%
