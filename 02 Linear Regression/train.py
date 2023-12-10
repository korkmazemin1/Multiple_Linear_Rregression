import numpy as np 
X_egitim=np.array([1,2,3])
kayip_dizi=np.array([1,2,3])
agirlik=np.array([1,2,3])
bias=5
tahmin=np.array([1,2,3])
gercek_deger=np.array([1,2,3])
ogrenme_katsayisi=1
toplam_ornek_sayisi=3
def gradyan_inisi(agirlik,ogrenme_katsayisi,tahmin,gercek_deger,bias,toplam_ornek_sayisi,kayip_dizi):
        
        agirlik_turev = -(2/toplam_ornek_sayisi) * X_egitim*kayip_dizi
        bias_maliyet = -(2/toplam_ornek_sayisi) *kayip_dizi
        print(f"agirlik_türev:{agirlik_turev}\n agirlik:{agirlik}")
        
        
        
        
        agirlik = agirlik - (ogrenme_katsayisi * agirlik_turev)# stokastik gradyan inisi formülü
        print(f"agirlik_degisim:{agirlik}")
        bias = bias - (ogrenme_katsayisi * bias_maliyet)
        #print(f"kayip:{kayip}")
        return agirlik,bias 

agirlik,bias=gradyan_inisi(agirlik=agirlik,ogrenme_katsayisi=ogrenme_katsayisi,tahmin=tahmin,gercek_deger=gercek_deger,bias=bias,toplam_ornek_sayisi=toplam_ornek_sayisi,kayip_dizi=kayip_dizi)

print(agirlik,bias)