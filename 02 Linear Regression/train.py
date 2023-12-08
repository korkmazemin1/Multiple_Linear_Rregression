import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
import pandas as pd

facebook_data = pd.read_csv("dataset_Facebook.csv", sep=";")# dataset okundu

X=facebook_data[["Category","Page total likes","Post Month","Post Hour","Post Weekday","Paid"]].values# bağımsız değişkenler X ile belirtildi

Y=facebook_data[["Total Interactions"]].values# bağımlı değişken belirlendi
print("y: ",Y)
print("X:",X)
print(type(X))


"""

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)#veriler test ve eğitim verileri olarak ayrıldı

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], Y, color = "b", marker = "o", s = 30)
plt.show()

reg = LinearRegression(lr=0.01)
reg.fit(X_train,y_train)
predictions = reg.predict(X_test)

def mse(y_test, predictions):
    return np.mean((y_test-predictions)**2)

mse = mse(y_test, predictions)
print(mse)

y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
plt.show()
"""