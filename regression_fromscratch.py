import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
veriseti = pd.read_csv("dataset_Facebook.csv", sep=";")# veri seti yüklendi

veriseti.fillna(0, inplace=True)# veri seti içinde az sayıda bulunan kayıp verilere 0 atandı

X = veriseti[["Category", "Page total likes", "Post Month", "Post Hour", "Post Weekday", "Paid"]]# bağımsız değişkenler

Y = veriseti["Total Interactions"].values# bağımlı değişkenler-- tek sütün olduğundan daha rahat çağırmak adına tek boyutlu dizi haline getirildi



scaler = StandardScaler() # aykırı değerler yerine aralığı daha sabit verilere geçmek adına bağımsız değişkenler normalize edilir
#scaler= MinMaxScaler# kodun işleyişine göre diğer bir normalize işlemi uygulanacak
X_scaled = scaler.fit_transform(X)

bias=0
kayip_egitim_toplam=0

X_egitim, X_test, y_egitim, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)


bagimsiz_sayi=len(X.iloc[0]) # bağımsız değişkenlerin sayısı elde edildi
toplam_ornek_sayisi_egitim=400
toplam_ornek_sayisi_test=100


def En_Kucuk_Kare(y_egitim,tahmin,sira,kayip_egitim_toplam=0):
    kayip_egitim_toplam=(y_egitim[sira]-tahmin)**2# en küçük kareler kayıp fonksiyonu hesaplandı
    kayip_egitim_toplam+=kayip_egitim_toplam# toplamları alındı
    kayip_egitim=kayip_egitim_toplam/sira # gradyan düşüşünde uygulanmak için ortalaması hesaplandı
    
    return kayip_egitim_toplam,kayip_egitim

def gradyan_inisi(agirlik,ogrenme_katsayisi,tahmin,anlik_ornek_sayisi,gercek_deger,bias,toplam_ornek_sayisi):
        kayip=gercek_deger-tahmin
        agirlik_turev = -(2/toplam_ornek_sayisi) * X_egitim[anlik_ornek_sayisi]*kayip
        bias_maliyet = -(2/toplam_ornek_sayisi) *kayip
    
        
        
        agirlik = agirlik - (ogrenme_katsayisi * agirlik_turev)# stokastik gradyan inisi formülü
        bias = bias - (ogrenme_katsayisi * bias_maliyet)
        #print(f"kayip:{kayip}")
        return agirlik,bias 



def lineer_regresyon(X_egitim,y_egitim,ogrenme_katsayisi,iterasyon,bagimsiz_sayi):
    agirliklar=np.array([])# ağırlıkla için dizi oluşturuldu
    for i in range (0,bagimsiz_sayi):
        agirliklar=np.append(agirliklar,np.random.randint(30))# rastgele ağırlıklar atandı
        
        bias=50#bias değeri 50 olarak başlatıldı

    for i in range (1,iterasyon+1):
        tahmin_dizi=np.array([])
        
        for k in range(0,toplam_ornek_sayisi_egitim):
            tahmin=np.dot(X_egitim[k],agirliklar)+bias # regresyon formülü uygulanır bağımsız değişkenler ağırlık ile çarpılır
            
                
                 
            
            tahmin_dizi=np.append(tahmin_dizi,tahmin)
            #kayip_egitim,kayip_egitim_toplam=En_Kucuk_Kare(kayip_egitim_toplam=kayip_egitim_toplam)
            
            #kayip_egitim_kayit=np.array([])
            #kayip_egitim_kayit[i]=kayip_egitim# her bir iterasyonda dizinin indisi ile aynı sayı olacak şekilde kaydedilir
            agirliklar,bias=gradyan_inisi(agirlik=agirliklar,ogrenme_katsayisi=ogrenme_katsayisi,tahmin=tahmin,toplam_ornek_sayisi=toplam_ornek_sayisi_egitim,anlik_ornek_sayisi=k,gercek_deger=y_egitim[k],bias=bias)
        print(f"tahmin={tahmin}\n bagimsiz degiskenler{X_egitim[k]}\n agirliklar={agirliklar} \n\n\n\n gercek={y_egitim[k]}")


    return agirliklar,bias    


"""def dizi_cikarim(dizi1,dizi2):
    sonuc=np.array([])
    for i in range(0,len(dizi1)-1):
          sonuc[i]=dizi1[i]-dizi2[i]
          
    return sonuc"""

lineer_regresyon(X_egitim=X_egitim,y_egitim=y_egitim,ogrenme_katsayisi=0.01,iterasyon=1000,bagimsiz_sayi=bagimsiz_sayi)

