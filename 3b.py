import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error
import matplotlib.pyplot as plt

def sigmoid(z):
        return 1/(1 + np.exp(-z))

columns = [i for i in range(1,11)]
columns.append("label")
df = pd.read_csv("diabetes.txt",delimiter = "\t",names = columns,header = None)
df = df[1:]
print(df)
df = df.astype(float)

# for i in range(len(df)):
#     df.iloc[i] = df.iloc[i].astype(float)
    
# print(df.plot())
# df.show()
# x.show()

# print(df.describe())

X_train, X_test, y_train, y_test = train_test_split(df[columns[:-1]], df[columns[-1]], test_size=0.33, random_state=42)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# print(X_train.shape)

W = np.random.rand(10,1)
print(W)
# W = np.ones((10,1), dtype=int)
# print(W)
# print(W)
# print(X_train)
b = np.random.rand()
# b = 1
# print("B:",b)

X_train = X_train.T

numOfTrainSamples = X_train.shape[1]
numOfFeatures = X_train.shape[0]
Z = np.zeros(numOfTrainSamples)

# print(y_train)
# for i in range(len(y_train)):
#     y_train[i] = sigmoid(y_train[i])
#     y_train[i] = np.where(y_train[i] < 0.5, 0, 1)

y_train = np.expand_dims(y_train,axis =0)

# print("X_train: ",X_train)s
y_train = y_train.astype(int)
for i in range(1):
    Z = np.dot(W.T,X_train,) + b


    # def sigmoid(z):
    #     return 1/(1 + np.exp(-z))

    # A = sigmoid(Z)

    # A = np.where(A < 0.5, 0, 1)

    # print(y_train.shape)
    print(y_train)
    print(Z)
    J = mean_squared_error(y_train,Z)
    print(J)
    dz = Z - y_train

    dw = np.dot(X_train,dz.T)/numOfTrainSamples

    # print(dw.shape)

    db = np.sum(dz,axis =1)/numOfTrainSamples
    alpha = 0.1
    W = W - alpha * dw
    b = b - alpha *db

    # print(b)
    # print(W)

print(J)