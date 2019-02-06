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
