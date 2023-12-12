import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def kayip_hesaplayici(anlik_ornek_sayisi, tahmin, gercek_deger):
    kayip = 0
    for k in range(anlik_ornek_sayisi):
        kayip += abs(tahmin[k] - gercek_deger[k])
    return kayip

def En_Kucuk_Kare(y_egitim, tahmin, sira, kayip_egitim_toplam=0):
    kayip_egitim_toplam += (y_egitim[sira] - tahmin) ** 2
    kayip_egitim_toplam += kayip_egitim_toplam
    kayip_egitim = kayip_egitim_toplam / (sira + 1)
    return kayip_egitim_toplam, kayip_egitim

def gradyan_inisi(agirlik, ogrenme_katsayisi, tahmin, anlik_ornek_sayisi, gercek_deger, bias, toplam_ornek_sayisi, kayip_dizi):
    agirlik_turev = (1 / toplam_ornek_sayisi) * np.dot(X_egitim.T, kayip_dizi)
    bias_maliyet = (1 / toplam_ornek_sayisi) * kayip_dizi
    agirlik = agirlik - (ogrenme_katsayisi * agirlik_turev)
    bias = bias - (ogrenme_katsayisi * bias_maliyet)
    return agirlik, bias

def lineer_regresyon(X_egitim, y_egitim, ogrenme_katsayisi, iterasyon, bagimsiz_sayi):
    agirliklar = np.array([0,0,0,0,0,0,0])
    bias = 20

    for i in tqdm(range(1, iterasyon + 1)):
        tahmin_dizi = np.array([])
        kayip = np.array([])

        for k in range(0, toplam_ornek_sayisi_egitim):
            tahmin = np.dot(X_egitim[k], agirliklar) + bias
            tahmin_dizi = np.append(tahmin_dizi, tahmin)
            kayip_deger = tahmin_dizi[k] - y_egitim[k]
            kayip = np.append(kayip, kayip_deger)

        agirliklar, bias = gradyan_inisi(agirlik=agirliklar, ogrenme_katsayisi=ogrenme_katsayisi, tahmin=tahmin_dizi,
                                         toplam_ornek_sayisi=toplam_ornek_sayisi_egitim, anlik_ornek_sayisi=k,
                                         gercek_deger=y_egitim, bias=bias, kayip_dizi=kayip)

    return agirliklar, bias


def plot_errors(X_train, Y_train, X_test, Y_test, agirliklar, bias):
    train_predictions = np.dot(X_train, agirliklar) + bias
    test_predictions = np.dot(X_test, agirliklar) + bias

    train_errors = Y_train - train_predictions
    test_errors = Y_test - test_predictions

    train_mse = np.mean(train_errors ** 2)
    test_mse = np.mean(test_errors ** 2)

    # Eğitim ve test hatalarını çiz
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(Y_train, train_predictions, color='blue')
    plt.title('Eğitim Seti: Gerçek Değerler vs. Tahminler')
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahminler')

    plt.subplot(1, 2, 2)
    plt.scatter(Y_test, test_predictions, color='red')
    plt.title('Test Seti: Gerçek Değerler vs. Tahminler')
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahminler')

    plt.tight_layout()
    plt.show()

    return train_mse, test_mse

# Rest of your code remains unchanged


veriseti = pd.read_csv("dataset_Facebook.csv", sep=";")

label_encoder = LabelEncoder()
veriseti['Type'] = label_encoder.fit_transform(veriseti['Type'])

veriseti.fillna(0, inplace=True)

X = veriseti[["Category", "Type", "Page total likes", "Post Month", "Post Hour", "Post Weekday", "Paid"]]
Y = veriseti["Total Interactions"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

bias = 0
kayip_egitim_toplam = 0

X_egitim, X_test, y_egitim, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=41)

bagimsiz_sayi = len(X.iloc[0])
toplam_ornek_sayisi_egitim = 400

agirliklar, bias = lineer_regresyon(X_egitim=X_egitim, y_egitim=y_egitim, ogrenme_katsayisi=0.0001, iterasyon=1000,
                                   bagimsiz_sayi=bagimsiz_sayi)

train_mse, test_mse = plot_errors(X_train=X_egitim, Y_train=y_egitim, X_test=X_test, Y_test=y_test,
                                  agirliklar=agirliklar, bias=bias)

print("Eğitim Hatası (MSE):", train_mse)
print("Test Hatası (MSE):", test_mse)
print("Eğitilmiş Ağırlık Katsayıları:", agirliklar)
print("Bias:", bias)
