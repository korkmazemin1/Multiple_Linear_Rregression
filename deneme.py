import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# def En_Kucuk_Kare(y_egitim, tahmin_dizi):
#     kayip_egitim_toplam = np.sum((y_egitim - tahmin_dizi) ** 2)
#     kayip_egitim = kayip_egitim_toplam / len(y_egitim)
#     return kayip_egitim_toplam, kayip_egitim

# def gradyan_inisi(agirlik, ogrenme_katsayisi, tahmin_dizi, X_egitim, y_egitim, bias):
#     kayip = y_egitim - tahmin_dizi
#     agirlik_turev = -(2 / len(X_egitim)) * X_egitim.T.dot(kayip)
#     bias_maliyet = -(2 / len(X_egitim)) * np.sum(kayip)

#     agirlik = agirlik - ogrenme_katsayisi * agirlik_turev
#     bias = bias - ogrenme_katsayisi * bias_maliyet

#     return agirlik, bias

# def lineer_regresyon(X_egitim, y_egitim, ogrenme_katsayisi, iterasyon):
#     agirliklar = np.random.randn(X_egitim.shape[1])
#     bias = 0

#     for i in range(1, iterasyon + 1):
#         tahmin_dizi = X_egitim.dot(agirliklar) + bias
#         kayip_egitim_toplam, kayip_egitim = En_Kucuk_Kare(y_egitim, tahmin_dizi)

#         agirliklar, bias = gradyan_inisi(agirlik=agirliklar, ogrenme_katsayisi=ogrenme_katsayisi,
#                                          tahmin_dizi=tahmin_dizi, X_egitim=X_egitim, y_egitim=y_egitim, bias=bias)

#         if i % 100 == 0:
#             print(f"Iterasyon: {i}, Kayıp: {kayip_egitim}")

#     return agirliklar, bias

# # Veri setini yükle
# veriseti = pd.read_csv("dataset_Facebook.csv", sep=";")
# veriseti.fillna(0, inplace=True)

# # Bağımsız değişkenleri ve bağımlı değişkeni seç
# X = veriseti[["Category", "Page total likes", "Post Month", "Post Hour", "Post Weekday", "Paid"]]
# y = veriseti["Total Interactions"].values

# # Veriyi ölçeklendir
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Veriyi eğitim ve test setlerine ayır
# X_egitim, X_test, y_egitim, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Modeli eğit
# agirliklar, bias = lineer_regresyon(X_egitim=X_egitim, y_egitim=y_egitim, ogrenme_katsayisi=0.001, iterasyon=3000)

# # Modeli değerlendir
# tahmin_dizi_test = X_test.dot(agirliklar) + bias
# kayip_test_toplam, kayip_test = En_Kucuk_Kare(y_test, tahmin_dizi_test)
# print("tahmin_dizi_test= ",tahmin_dizi_test)
# print(f"\nTest Seti Kayıp: {kayip_test}")



class LinearRegressionFromScratch:
    def __init__(self, learning_rate, num_iterations):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for i in range(self.num_iterations):
            predictions = np.dot(X, self.weights) + self.bias
            errors = predictions - y

            # Gradient descent ile güncelleme
            self.weights -= (self.learning_rate / m) * np.dot(X.T, errors)
            print(X)
            self.bias -= (self.learning_rate / m) * np.sum(errors)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

def benzerlik_yuzdesi(gercek, tahmin):
    benzerlik = 1 - np.abs(gercek - tahmin).mean() / gercek.mean()
    return benzerlik * 100


veriseti = pd.read_csv("dataset_Facebook.csv", sep=";")
veriseti.fillna(0, inplace=True)
# Veri setini oluştur
X = veriseti[["Category", "Page total likes", "Post Month", "Post Hour", "Post Weekday", "Paid"]]
y = veriseti["Total Interactions"].values

#Veriyi ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veriyi eğitim ve test setlerine ayır
X_egitim, X_test, y_egitim, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modeli oluştur
model = LinearRegressionFromScratch(learning_rate=0.0001, num_iterations=8000)

# Veriyi modele uyum sağla
model.fit(X_egitim, y_egitim)

# Eğitilmiş modelin ağırlıklarını ve bias'ını göster
#print("Eğitilmiş Ağırlıklar:", model.weights)
#print("Eğitilmiş Bias:", model.bias)

# Test verisi üzerinde modeli değerlendir
tahmin_test = model.predict(X_test)
#print("Tahminler:", tahmin_test)

yuzde_benzerlik = benzerlik_yuzdesi(y_test, tahmin_test)
print(f"Yüzde Benzerlik: {yuzde_benzerlik}%")

mse = ((y_test - tahmin_test) ** 2).mean()
print(f"Ortalama Kare Hata (MSE): {mse}")

mae = np.abs(y_test - tahmin_test).mean()
print(f"Ortalama Mutlak Hata (MAE): {mae}")

