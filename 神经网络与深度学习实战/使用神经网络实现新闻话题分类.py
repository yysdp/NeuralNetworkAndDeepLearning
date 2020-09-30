import pandas as pd
import numpy as np

data_path = 'housing.csv'
housing = pd.read_csv(data_path)


#将ocean_proximity转换为数值

housing['ocean_proximity'] = housing['ocean_proximity'].astype('category')

housing['ocean_proximity'] = housing['ocean_proximity'].cat.codes

data = housing.values
train_data = data[:, [0,1,2,3,4,5,6,7,9]]
train_value = data[:,[8]]

print(np.isnan(train_data).any())
print(np.argwhere(np.isnan(train_data)))
train_data[np.isnan(train_data)] = 0
print(np.isnan(train_data).any())

print(train_data[0:5])
mean = train_data.mean(axis=0)
print(mean)
train_data -= mean
print(train_data[0:5])
std = train_data.std(axis = 0)
print(std)
train_data /= std
print(train_data[0:5])
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

history = model.fit(train_data, train_value, epochs=300,
                    validation_split=0.2,
                    batch_size=32)

val_mae_history = history.history['metrics']
plt.plot(range(1, len(val_mae_history) + 1), val_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()