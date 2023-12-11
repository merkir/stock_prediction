import math
import numpy as np

def createXyData(lookbackdays, data):
  x_data=[] #lookbackdays-day perdiction data
  y_data=[] #actual opening price

  for i in range(lookbackdays, len(data)):
    x_data.append(data[i-lookbackdays:i,0])
    y_data.append(data[i])

  return x_data, y_data

def createDataSet(all_data, param, plt):
  start_index, end_index = int(param['start_index']), int(param['end_index'])
  train_len = math.ceil((end_index - start_index)*.8)
  # train data
  train_data = all_data[start_index:start_index+train_len,:]
  x_train, y_train = createXyData(LOOKBACK_DAYS, train_data)
  x_train, y_train = np.array(x_train), np.array(y_train)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
  print(x_train.shape, y_train.shape)

  # test data
  test_data = all_data[start_index+train_len-LOOKBACK_DAYS:start_index+(end_index - start_index),:]
  x_test, y_test = createXyData(LOOKBACK_DAYS, test_data)
  x_test, y_test = np.array(x_test), np.array(y_test)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
  print(x_test.shape, y_test.shape)

  plt.plot(open[start_index:start_index+train_len], color='yellow') # train
  plt.plot(open[start_index+train_len:end_index], color='green') # test

  param['x_train'] = x_train
  param['y_train'] = y_train
  param['x_test'] = x_test
  param['y_test'] = y_test

  return param, plt
