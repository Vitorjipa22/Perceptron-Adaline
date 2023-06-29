from random import seed
from random import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class perceptron:
  def __init__(self, n_features, lr = 0.01):
    self.__n_features = n_features
    self.__lr = lr
    self.__epoch = 0
    self.__erro = True
    self.__W = list()
    self.__sys = list()

    for i in range(self.__n_features + 1):
      self.__W.append(random())

  def degree(self, num):
    return (1 if num >= 0 else 0)

  def bipolar_degree(self, num):
    if num > 0:
      num = 1

    elif num < 0:
      num = -1

    return num

  def train(self, X_train, y_train, activation_func = 'bipolar'):
    self.__epoch = 0
    self.__sys.append(np.array(self.W))
    while(self.__erro):
      k = 0

      self.__erro = False
      for amostra in X_train:
        u = sum(self.__W * amostra.T)

        if activation_func == 'bipolar':
          y_hat = self.bipolar_degree(u)

        if activation_func == 'degree':
          y_hat = self.degree(u)

        if y_hat != y_train[k]:
          self.__W = self.__W + (self.__lr*(y_train[k] - y_hat)*amostra)
          self.__sys.append(self.__W)
          self.__erro = True

        k += 1

      self.__epoch+=1

    return self.__W, self.__epoch

  def plot_sys(self):
    columns = ['W'+str(i) for i in range(self.__n_features + 1)]
    sys = pd.DataFrame(self.__sys, columns = columns)

    plt.figure(figsize = (15,7))
    plt.title('Valor dos pesos ao decorrer das modificações - Peceptron.')

    for i in range(self.__n_features + 1):
      plt.plot(np.arange(0,len(self.__sys)), sys[columns[i]], label=columns[i])

    plt.legend(loc = "upper right")
    plt.show()

  def predict(self, X_val, activation_func = 'bipolar'):
    pred = list()
    for amostra in X_val:
      amostra = np.array(amostra)
      u = sum(self.W * amostra.T)

      if activation_func == 'bipolar':
        y_hat = self.bipolar_degree(u)

      if activation_func == 'degree':
        y_hat = self.degree(u)

      pred.append(y_hat)

    return pred

  @property
  def W(self):
    return self.__W
  

class adaline(perceptron):
  def __init__(self, n_features, lr = 0.01, epsilon = 0.000001):
    self.epsilon = epsilon
    self.__n_features = n_features
    self.__lr = lr
    self.__epoch = 0
    self.__erro = True
    self.__W = list()
    self.__sys = list()

    for i in range(self.__n_features + 1):
      self.__W.append(random())

  def train(self, X_train, y_train, activation_func = 'bipolar'):
    self.__epoch = 0
    self.__sys.append(np.array(self.__W))
    EQM_anterior = 0
    EQM = 0
    self.p = X_train.shape[0]

    while(True):
      k = 0
      EQM_anterior = EQM
      EQM = 0

      for amostra in X_train:
        u = sum(self.__W * amostra.T)
        EQM += ((y_train[k] - u) ** 2)

        self.__W = self.__W + (self.__lr*(y_train[k] - u)*amostra)
        self.__sys.append(self.__W)
        self.__erro = True

        k += 1

      self.__epoch+=1

      EQM /= self.p

      if abs(EQM - EQM_anterior) <= self.epsilon or (self.__epoch > 1000):
        break

    return self.__W, self.__epoch

  def plot_sys(self):
    columns = ['W'+str(i) for i in range(self.__n_features + 1)]
    sys = pd.DataFrame(self.__sys, columns = columns)

    plt.figure(figsize = (15,7))
    plt.title('Valor dos pesos ao decorrer das modificações - Adaline.')

    for i in range(self.__n_features + 1):
      plt.plot(np.arange(0,len(self.__sys)), sys[columns[i]], label=columns[i])

    plt.legend(loc = "upper right")
    plt.show()

  def predict(self, X_val, activation_func = 'bipolar'):
    pred = list()
    for amostra in X_val:
      amostra = np.array(amostra)
      u = sum(self.W * amostra.T)

      if activation_func == 'bipolar':
        y_hat = self.bipolar_degree(u)

      if activation_func == 'degree':
        y_hat = self.degree(u)

      pred.append(y_hat)

    return pred

  @property
  def W(self):
    return self.__W
  