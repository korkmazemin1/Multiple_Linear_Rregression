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



X_egitim, X_test, y_egitim, y_egitim = train_test_split(X, Y, test_size=0.2, random_state=42)

bagimsiz_sayi=X.iloc[0].size # bağımsız değişkenlerin sayısı elde edildi

def En_Kucuk_Kare(y_egitim,tahmin,sıra,kayip_egitim_toplam=0):
    kayip_egitim_toplam=(y_egitim[sıra]-tahmin)**2# en küçük kareler kayıp fonksiyonu hesaplandı
    kayip_egitim_toplam+=kayip_egitim_toplam# toplamları alındı
    kayip_egitim=kayip_egitim_toplam/sıra # gradyan düşüşünde uygulanmak için ortalaması hesaplandı
    
    return kayip_egitim_toplam,kayip_egitim

def gradyan_düsüsü(agirliklar,kayip_egitim,ogrenme_katsayısı):
        """agirlik_türev = -(2/bagimsiz_sayi) * (X (y-y_predicted))
        bias_türev = -(2/bagimsiz_sayi) *(y-y_predicted)
         
        # Updating weights and bias
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)

        Buraya gradyan düşüşü kodları yazılacak 
        for ile tek tek çağırmak yerine dizi olarak işlemlere bakılacak 

        ayrıca gradyan düşüşü formülü uygulanmasına bakılacak


"""


def lineer_regresyon(X_egitim,y_egitim,ogrenme_katsayısı,iterasyon,bagimsiz_sayi):
    agirliklar=np.array([0,0,0,0,0,0])# ağırlıkla için dizi oluşturuldu
    for i in range (0,bagimsiz_sayi):
        agirliklar[i]=np.random.randint(30)# rastgele ağırlıklar atandı
        print(agirliklar)##############################################################################################
        bias=50#bias değeri 50 olarak başlatıldı

    for i in range (1,iterasyon+1):
        for k in range(0,len(X_egitim)):
            tahmin=np.dot(X_egitim.iloc[0],agirliklar)+bias # regresyon formülü uygulanır bağımsız değişkenler ağırlık ile çarpılır
            kayip_egitim,kayip_egitim_toplam=En_Kucuk_Kare(kayip_egitim_toplam=kayip_egitim_toplam)
            if k==len(X_egitim):# iterasyon biter ve kare hata kaydedilir
                kayip_egitim_kayit=np.array([])
                kayip_egitim_kayit[i]=kayip_egitim# her bir iterasyonda dizinin indisi ile aynı sayı olacak şekilde kaydedilir
                agirliklar=gradyan_düsüsü(kayip_egitim)









