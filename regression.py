import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

facebook_data = pd.read_csv("dataset_Facebook.csv", sep=";")# dataset okundu

X=facebook_data[["Category","Page total likes","Post Month","Post Hour","Post Weekday","Paid"]]# bağımsız değişkenler X ile belirtildi

Y=facebook_data[["Total Interactions"]]# bağımlı değişken belirlendi

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)#veriler test ve eğitim verileri olarak ayrıldı



# gradiant descent uygulanacak elindeki kaynağı dikkatli oku yeni kaynaklara bak en kısa yoldan uygulamaya geç