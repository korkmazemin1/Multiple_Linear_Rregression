import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
veriseti = pd.read_csv("dataset_Facebook.csv", sep=";")# veri seti yüklendi

veriseti.fillna(0, inplace=True)# veri seti içinde az sayıda bulunan kayıp verilere 0 atandı

X = veriseti[["Category", "Page total likes", "Post Month", "Post Hour", "Post Weekday", "Paid"]]# bağımsız değişkenler

Y = veriseti["Total Interactions"].values# bağımlı değişkenler-- tek sütün olduğundan daha rahat çağırmak adına tek boyutlu dizi haline getirildi

print(X.iloc[0])
print(Y[0])

scaler = StandardScaler() # aykırı değerler yerine aralığı daha sabit verilere geçmek adına bağımsız değişkenler normalize edilir
#scaler= MinMaxScaler# kodun işleyişine göre diğer bir normalize işlemi uygulanacak
X_scaled = scaler.fit_transform(X)



X_egitim, X_test, y_egitim, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("\n",y_egitim[15])
print("\n",X_egitim.iloc[15])