#%%
import pandas as pd

#%%
automobile_data = pd.read_csv('C:/Users/pcuci/Downloads/pytorch-building-deep-learning-models/datasets/Automobile_data.csv', sep=r'\s*,\s*', engine='python')

#%%
automobile_data.head()

#%%
automobile_data = automobile_data.replace('?', np.nan)
automobile_data.head()

#%%
automobile_data = automobile_data.dropna() # clean out the rows with missing data
automobile_data.head()

#%%
col = ['make', 'fuel-type', 'body-style', 'horsepower']
automobile_features = automobile_data[col]
automobile_features.head()

#%%
automobile_target = automobile_data[['price']]
automobile_target.head()

#%%
automobile_features['horsepower'].describe() # dtype: object (becuase strings)

#%%
pd.options.mode.chained_assignment = None # turn off the SettingWithCopyWarning which warns of unpredictable results with chained assignments, ref: http://pandas-docs.github.io/pandas-docs-travis/indexing.html#why-does-assignment-fail-when-using-chained-indexing


#%%
automobile_features['horsepower'] = pd.to_numeric(automobile_features['horsepower'])
automobile_features['horsepower'].describe() # now dtype: float64 with cool stats!

#%%
automobile_target = automobile_target.astype(float)
automobile_target['price'].describe()

#%%
automobile_features = pd.get_dummies(automobile_features, columns=['make', 'fuel-type', 'body-style']) # one-hot encoding for non-numeric values
automobile_features.head()

#%%
automobile_features.columns

#%%
from sklearn import preprocessing
automobile_features[['horsepower']] = preprocessing.scale(automobile_features[['horsepower']]) # standardize the numeric values: subtract mean, divide by standard deviation
automobile_features[['horsepower']].head() # all ML algorithms work better when the numeric values are standardized to be roughly in the same range

#%%
from sklearn.model_selection import train_test_split
X_train, x_test, Y_train, y_test = train_test_split(automobile_features, automobile_target, test_size=0.2, random_state=0) # use 80% of the data for training purposes

#%%
import torch
dtype = torch.float

#%%
X_train_tensor = torch.tensor(X_train.values, dtype = dtype)
x_test_tensor = torch.tensor(x_test.values, dtype = dtype)
X_train_tensor.shape

#%%
Y_train_tensor = torch.tensor(Y_train.values, dtype = dtype)
y_test_tensor = torch.tensor(y_test.values, dtype = dtype)
Y_train_tensor.shape

#%%
inp = 26 # the 26 features in X_train_tensor
out = 1 # a single value output, the price

hid = 100 # hidden layer neurons

loss_fn = torch.nn.MSELoss() # use the torch.nn library to not calculate the loss by hand as in the previous example

learning_rate = 0.0001

#%%
model = torch.nn.Sequential(torch.nn.Linear(inp, hid), torch.nn.Sigmoid(), torch.nn.Linear(hid, out)) # a sequential model holding NN layers in sequence; all neural network classes derive from the base torch.nn.Module class; any layer can contain a nested module; the Sigmoid is our choice for the activation function

#%%
for iter in range(10000): # 10K epochs, or passes through our network
    y_pred = model(X_train_tensor) # apply the model to the input training data
    loss = loss_fn(y_pred, Y_train_tensor)
    if iter % 1000 == 0:
        print(iter, loss.item())
    model.zero_grad() # zero out the model gradients before backpropagation
    loss.backward()
    with torch.no_grad(): # don't calculate gradients while we're updating our model's parameters
        for param in model.parameters(): # accesses all params from the nn
            param -= learning_rate * param.grad


#%%
# take a sample from our test data and perform a prediction
sample = x_test.iloc[23]
sample

#%%
sample_tensor = torch.tensor(sample.values, dtype = dtype) # convert to tensor
sample_tensor


#%%
y_pred = model(sample_tensor) # pass the sample tensor through our nn model
print("Predicted price of automible is: ", int(y_pred.item()))
print("Actual price of automible is: ", int(y_test.iloc[23]))

#%%
# now run predictions on the entire test dataset
y_pred_tensor = model(x_test_tensor)
y_pred = y_pred_tensor.detach().numpy() # for visualisation
plt.scatter(y_pred, y_test.values)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

#%%
# serialize the model to disk
torch.save(model, 'my_model')
saved_model = torch.load('my_model')
y_pred_tensor = saved_model(x_test_tensor)
y_pred = y_pred_tensor.detach().numpy()
plt.figure(figsize=(15, 6))
plt.plot(y_pred, label='Predicted Price')
plt.plot(y_test.values, label='Actual Price')
plt.legend()
plt.show()

#%%
