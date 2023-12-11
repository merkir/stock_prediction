import math
import time
import numpy as np
import matplotlib.pyplot as plt

def lstmExperiment(param, lstm, batch_size=32,epochs=5):
  start_index, end_index, x_train, y_train, x_test, y_test = param['start_index'], param['end_index'], param['x_train'], param['y_train'], param['x_test'], param['y_test']
  lstm, exec_time = lstmTrain(lstm, x_train, y_train, batch_size, epochs)
  mae, mse, rmse, predictions = lstmTest(lstm, x_test, y_test)

  print("execuation time:", exec_time, "mae:", mae, "mse:", mse, "rmse:", rmse)

  # visualize
  train_len = math.ceil((end_index - start_index)*.8)
  train = open[start_index:start_index+train_len]
  valid = open[start_index+train_len:end_index]
  valid['predictions'] = predictions
  plt = lstmVisualize(train, valid)
  return lstm, exec_time, predictions, plt


def lstmTrain(lstm, x_train, y_train, batch_size=32,epochs=5):
  lstm.compile(optimizer='adam', loss='mean_squared_error')
  start = time.time()
  lstm.fit(x_train, y_train, batch_size=batch_size,epochs=epochs)
  exec_time = time.time() - start
  return lstm, exec_time

def lstmTest(lstm, x_test, y_test):
  predictions=lstm.predict(x_test)
  mae=np.mean(np.abs(predictions-y_test))
  mse=np.mean((predictions-y_test)**2)
  rmse=np.sqrt(mse)
  print("STANDARLIZED", "mae:", mae, "mse:", mse, "rmse:", rmse)

  predictions=SCALER.inverse_transform(predictions)
  y_test=SCALER.inverse_transform(y_test)
  mae=np.mean(np.abs(predictions-y_test))
  mse=np.mean((predictions-y_test)**2)
  rmse=np.sqrt(mse)
  return mae, mse,rmse, predictions

def lstmVisualize(train, valid):
  #Visualize the data
  plt.figure(figsize=(15,7))
  plt.xlabel('Date', fontsize=18)
  plt.ylabel('Open Price USD($)', fontsize=18)
  plt.plot(train , color='red')
  plt.plot(valid['Open'] , color='blue')
  plt.plot(valid['predictions'] , color='green')
  plt.legend(['Validation', 'predictions'], loc='lower right')
  return plt