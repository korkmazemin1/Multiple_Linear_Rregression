import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# Veri setini yükle
veriseti = pd.read_csv("dataset_Facebook.csv", sep=";")

label_encoder = LabelEncoder()
veriseti['Type'] = label_encoder.fit_transform(veriseti['Type'])

veriseti.fillna(0, inplace=True)

# Bağımsız ve bağımlı değişkenleri seç
X = veriseti[["Category","Type", "Page total likes", "Post Month", "Post Hour", "Post Weekday", "Paid"]]
Y = veriseti["Total Interactions"].values

# Verileri normalize et
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)


scaler = StandardScaler()
X_olceklenmemis = X[["Page total likes", "Post Month", "Post Hour", "Post Weekday", "Paid"]]
X_olceklenmis = scaler.fit_transform(X_olceklenmemis)
X.loc[:, ["Page total likes", "Post Month", "Post Hour", "Post Weekday", "Paid"]] = X_olceklenmis
X[["Page total likes", "Post Month", "Post Hour", "Post Weekday", "Paid"]] = X_olceklenmis

# Veriyi eğitim ve test setine bölelim
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Gradyan İnişi için gerekli fonksiyonları tanımlayalım

def maliyet_hesapla(X, Y, agirliklar):
    m = len(Y)
    tahminler = np.dot(X, agirliklar)
    maliyet = (1 / (2 * m)) * np.sum(np.square(tahminler - Y))
    return maliyet

def gradyan_inisi(X, Y, agirliklar, learning_rate, iterasyon_sayisi):
    m = len(Y)
    maliyet_gecmisi = []

    for iterasyon in range(iterasyon_sayisi):
        tahminler = np.dot(X, agirliklar)
        hatalar = tahminler - Y
        gradyan = (1 / m) * np.dot(X.T, hatalar)
        agirliklar = agirliklar - learning_rate * gradyan

        maliyet = maliyet_hesapla(X, Y, agirliklar)
        maliyet_gecmisi.append(maliyet)

    return agirliklar, maliyet_gecmisi

# İlk ağırlık katsayılarını tanımla
initial_agirliklar = np.zeros(X_train.shape[1])

# Gradyan İnişi ile modeli eğit
learning_rate = 0.1
iterasyon_sayisi = 1000
egitilmis_agirliklar, maliyet_gecmisi = gradyan_inisi(X_train, Y_train, initial_agirliklar, learning_rate, iterasyon_sayisi)

# Eğitim ve test verileri için Toplam Kare Hatayı (Sum Squared Error) hesapla
egitim_hatasi = maliyet_hesapla(X_train, Y_train, egitilmis_agirliklar)
test_hatasi = maliyet_hesapla(X_test, Y_test, egitilmis_agirliklar)

# Toplam Kare Hata (Sum Squared Error) grafiğini çiz
plt.plot(range(1, iterasyon_sayisi+1), maliyet_gecmisi, color='blue')
plt.title('Toplam Kare Hata - Gradyan İnişi')
plt.xlabel('iterasyon_sayisi')
plt.ylabel('maliyet')
plt.show()
# Eğitim ve test hatalarını çiz
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(Y_train,np.dot(X_train, egitilmis_agirliklar), color='blue')
plt.title('Eğitim Seti: Gerçek Değerler vs. Tahminler')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahminler')

plt.subplot(1, 2, 2)
plt.scatter(Y_test,np.dot(X_test, egitilmis_agirliklar), color='red')
plt.title('Test Seti: Gerçek Değerler vs. Tahminler')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahminler')

plt.tight_layout()
plt.show()

def yuzde_hata_hesaplama(Y, tahminler):

    return (100 / np.sum(np.square(Y))) * np.sum(np.square(tahminler - Y))
    

yuzde_egitim_hatasi = yuzde_hata_hesaplama(Y_train, np.dot(X_train, egitilmis_agirliklar))
yüzde_test_hatasi = yuzde_hata_hesaplama(Y_test, np.dot(X_test, egitilmis_agirliklar))


print("Eğitim Hatası (Yüzde Hata):", yuzde_egitim_hatasi)
print("Test Hatası (Yüzde Hata):", yüzde_test_hatasi)

# Sonuçları yazdır
print("Eğitim Hatası (MSE):", egitim_hatasi)
print("Test Hatası (MSE):", test_hatasi)
print("Eğitilmiş Ağırlık Katsayıları:", egitilmis_agirliklar)

