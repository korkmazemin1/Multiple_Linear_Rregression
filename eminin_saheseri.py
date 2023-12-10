import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tqdm import tqdm
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

def kayip_hesaplayici(anlik_ornek_sayisi,tahmin,gercek_deger):
    anlik_ornek_sayisi
    for k in range(anlik_ornek_sayisi):
        kayip=abs(tahmin[k]-gercek_deger[k])

          
     
     

def En_Kucuk_Kare(y_egitim,tahmin,sıra,kayip_egitim_toplam=0):
    kayip_egitim_toplam=(y_egitim[sıra]-tahmin)**2# en küçük kareler kayıp fonksiyonu hesaplandı
    kayip_egitim_toplam+=kayip_egitim_toplam# toplamları alındı
    kayip_egitim=kayip_egitim_toplam/sıra # gradyan düşüşünde uygulanmak için ortalaması hesaplandı
    
    return kayip_egitim_toplam,kayip_egitim

def gradyan_inisi(agirlik,ogrenme_katsayisi,tahmin,anlik_ornek_sayisi,gercek_deger,bias,toplam_ornek_sayisi,kayip_dizi):
        
        agirlik_turev = -(2/toplam_ornek_sayisi) * X_egitim*kayip_dizi
        bias_maliyet = -(2/toplam_ornek_sayisi) *kayip_dizi
        
        
        agirlik = agirlik - (ogrenme_katsayisi * agirlik_turev)# stokastik gradyan inisi formülü
        bias = bias - (ogrenme_katsayisi * bias_maliyet)
        #print(f"kayip:{kayip}")
        return agirlik,bias 



def lineer_regresyon(X_egitim,y_egitim,ogrenme_katsayisi,iterasyon,bagimsiz_sayi):
    agirliklar=np.array([10,10,10,10,10,10])# ağırlıkla için dizi oluşturuldu
    bias=20
    """for i in range (0,bagimsiz_sayi):
        agirliklar=np.append(agirliklar,np.random.randint(30))# rastgele ağırlıklar atandı
        
        bias=50#bias değeri 50 olarak başlatıldı"""

    for i in tqdm(range (1,iterasyon+1)):
        
        tahmin_dizi=np.array([])
        kayip=np.array([])
        
        for k in range(0,toplam_ornek_sayisi_egitim):
            tahmin=np.dot(X_egitim[k],agirliklar)+bias # regresyon formülü uygulanır bağımsız değişkenler ağırlık ile çarpılır
            
            tahmin_dizi=np.append(tahmin_dizi,tahmin)
            kayip_deger=tahmin_dizi[k]-y_egitim[k]
            kayip=np.append(kayip,kayip_deger)
            
            
        agirliklar,bias=gradyan_inisi(agirlik=agirliklar,ogrenme_katsayisi=ogrenme_katsayisi,tahmin=tahmin_dizi,toplam_ornek_sayisi=toplam_ornek_sayisi_egitim,anlik_ornek_sayisi=k,gercek_deger=y_egitim,bias=bias,kayip_dizi=kayip)
        

    return agirliklar,bias    


"""def dizi_cikarim(dizi1,dizi2):
    sonuc=np.array([])
    for i in range(0,len(dizi1)-1):
          sonuc[i]=dizi1[i]-dizi2[i]
          
    return sonuc"""
agirliklar,bias=lineer_regresyon(X_egitim=X_egitim,y_egitim=y_egitim,ogrenme_katsayisi=0.0001,iterasyon=10000,bagimsiz_sayi=bagimsiz_sayi)


def test(y_test,X_test,agirlik,bias):
    total_hata=0
    tahmin=np.array([])
    hata=np.array([],dtype=int)
    a=0
    for k in range(len(y_test)):
        
        tahmin=np.append(tahmin,np.dot(X_test[k],agirlik)+bias)
        
             
        if y_test[k]==0:
             y_test[k]=1     
        hata_deger=abs((tahmin[k]-y_test[k])/y_test[k])*100
        print

        hata=np.append(hata,hata_deger)
        
        total_hata=hata_deger+total_hata
    print(f"tahmin:{tahmin[55]}\n gercek deger:{y_test[55]}")
    print(hata[55])
    print(hata)
    
    ortalama=total_hata/100
    print(f"ortalama:{ortalama}")

    
              
    return hata
hata=test(y_test=y_test,X_test=X_test,agirlik=agirliklar,bias=bias)  

#print(f"hata_ortalama:{hata_ortalama}")
