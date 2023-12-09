import numpy as np

def dizi_cikarim(dizi1,dizi2):
    sonuc=np.array([])
    for i in range(0,len(dizi1)-1):
        deger=dizi1[i]-dizi2[i]
        sonuc.append(sonuc,deger)          
    return sonuc



def gradyan_inisi(agirliklar,ogrenme_katsayısı,tahmin_dizi,bagimsiz_sayi,iterasyon_sayisi,gercek_deger,X_egitim):
        sonuc=dizi_cikarim(gercek_deger,tahmin_dizi)
        agirlik_türev = (1/bagimsiz_sayi) * np.dot(X_egitim,sonuc)
        bias_türev = (1/bagimsiz_sayi) *np.sum(sonuc)
         
        # Updating weights and bias
        agirliklar = agirliklar - (ogrenme_katsayısı * agirlik_türev)
        bias = bias - (ogrenme_katsayısı * bias_türev)