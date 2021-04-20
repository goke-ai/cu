# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().system('pip3 install -q xlrd xlwt')
get_ipython().system('pip3 install -q openpyxl')
get_ipython().system('pip3 install -q seaborn')


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


# %%
rawDataset = pd.read_excel('vehicle_emission_1.xls', index_col=0)
rawDataset


# %%
rawDataset.columns


# %%
dataset = rawDataset.copy()
dataset.tail()


# %%
dataset.isna().sum()


# %%
dataset = dataset.dropna()


# %%
# prefix = ['Make', 'FuelType', 'VehicleUse', 'Study', 'OGEPA', 'VehicleType', 'EUROII', 'NESREAI', 'NESREAII', 'EUROIII']
boolCols = ['Study', 'OGEPA', 'EUROII', 'NESREAI', 'NESREAII', 'EUROIII']

for col in boolCols:
    dict_colData = { x:i for i,x in enumerate(rawDataset[col].unique())}
    print(dict_colData)
    dataset[col + '_N'] = dataset[col].map(dict_colData)
    
# dataset = pd.get_dummies(dataset, prefix=prefix, prefix_sep='_')
dataset


# %%
dataset = dataset.drop(boolCols, axis=1)


# %%
prefix = ['Make', 'FuelType', 'VehicleUse', 'VehicleType']
dataset = pd.get_dummies(dataset, prefix=prefix, prefix_sep='_')
dataset


# %%
dataset.columns


# %%
favCols = ['CO', 'CO2', 'O2', 'HC', 'Age', 'lamda', 'AFR']


# %%
sns.pairplot(dataset[favCols], diag_kind='kde')


# %%
dataset.describe().transpose()


# %%
h = 15
X = dataset.copy()
X = X.dropna()
X = X.loc[::h]
sns.pairplot(X[favCols], diag_kind='kde')


# %%
X.columns


# %%
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

print(tf.__version__)


# %%
outputCols = ['CO', 'CO2', 'O2', 'HC', 'lamda', 'AFR', 'Study_N',
       'OGEPA_N', 'EUROII_N', 'NESREAI_N', 'NESREAII_N', 'EUROIII_N' ]
outputCols


# %%
inputCols = ['Age', 'Make_Honda', 'Make_Mazda', 'Make_Mercedez B', 'Make_Mitsubushi',
       'Make_Nissan', 'Make_Peugeot', 'Make_Toyota', 'Make_Volkswagen',
       'FuelType_Petrol', 'VehicleUse_Commercial', 'VehicleUse_Official',
       'VehicleUse_Private', 'VehicleType_Bullion V', 'VehicleType_Bus',
       'VehicleType_CABSTAR', 'VehicleType_Car', 'VehicleType_Jeep',
       'VehicleType_PICK-UP', 'VehicleType_TRUCK']

inputCols


# %%
train_dataset = X.sample(frac=0.8, random_state=0)
test_dataset = X.drop(train_dataset.index)


# %%
sns.pairplot(train_dataset[favCols], diag_kind='kde')


# %%
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features[outputCols]
test_labels = test_features[outputCols]

train_features = train_features.drop(outputCols, axis=1)
test_features = test_features.drop(outputCols, axis=1)

print(train_features.columns)
print(train_labels.columns)


# %%
train_dataset.describe().transpose()


# %%
normalizer = preprocessing.Normalization()


# %%
normalizer.adapt(np.array(train_features))


# %%
print(normalizer.mean.numpy())


# %%
first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())


# %%
# Single Input and single target
sglInput = 'Age'
sglOutput = 'CO2'


# %%
age = np.array(train_features[sglInput])

age_normalizer = preprocessing.Normalization(input_shape=[1,])
age_normalizer.adapt(age)


# %%
age_model = tf.keras.Sequential([
    age_normalizer,
    layers.Dense(units=1)
])

age_model.summary()


# %%
age_model.predict(age[:10])


# %%
age_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')


# %%
get_ipython().run_cell_magic('time', '', 'history = age_model.fit(\n    train_features[sglInput], train_labels[sglOutput],\n    epochs=100,\n    # suppress logging\n    verbose=0,\n    # Calculate validation results on 20% of the training data\n    validation_split = 0.2)')


# %%
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# %%
def plot_loss(history, ylabel='Error [Y]'):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
#   plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel(ylabel)
  plt.legend()
  plt.grid(True)


# %%
plot_loss(history, f'Error [{sglOutput}]')


# %%
test_results = {}

test_results['age_model'] = age_model.evaluate(
    test_features[sglInput],
    test_labels, verbose=0)


# %%
x = tf.linspace(0.0, 50, 251)
y = age_model.predict(x)


# %%
def plot_age(x, y):
  plt.scatter(train_features['Age'], train_labels['CO2'], label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Age')
  plt.ylabel('CO2')
  plt.legend()


# %%
plot_age(x,y)


# %%



