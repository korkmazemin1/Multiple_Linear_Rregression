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
        
def gradyan_decent(baslangic_noktasi, iterasyon_sayisi, durdurma_kriteri,ogrenme_orani,turev):   

    for iterasyon in range(iterasyon_sayisi):
        gradyan = turev(baslangic_noktasi)
        adim = -ogrenme_orani * gradyan
        baslangic_noktasi = baslangic_noktasi + adim

        if durdurma_kriteri is not None and np.linalg.norm(gradyan) < durdurma_kriteri:
            print(f"Iterasyon {iterasyon + 1}: Gradyan normu, durdurma kriterinden küçük.")
            break

        print(f"Iterasyon {iterasyon + 1}: Nokta {baslangic_noktasi}, Gradyan {gradyan}")

    return baslangic_noktasi


import pandas as pd
import numpy as np


def gradyan_inisi(agirlik, bias, X_egitim, y_egitim):
    tahmin = np.dot(X_egitim, agirlik) + bias
    gradyan_agirlik = -(2/len(X_egitim)) * np.dot(X_egitim.T, (y_egitim - tahmin))
    gradyan_bias = -(2/len(X_egitim)) * np.sum(y_egitim - tahmin)
    
    return gradyan_agirlik, gradyan_bias

def gradient_descent(turev, baslangic_noktasi, ogrenme_orani, iterasyon_sayisi, durdurma_kriteri=None):
    current_nokta = baslangic_noktasi

    for iterasyon in range(iterasyon_sayisi):
        gradyan_agirlik, gradyan_bias = turev(current_nokta)
        
        adim_agirlik = -ogrenme_orani * gradyan_agirlik
        adim_bias = -ogrenme_orani * gradyan_bias
        
        current_nokta[0] = current_nokta[0] + adim_agirlik
        current_nokta[1] = current_nokta[1] + adim_bias

        if durdurma_kriteri is not None and np.linalg.norm(gradyan_agirlik) < durdurma_kriteri:
            print(f"Iterasyon {iterasyon + 1}: Gradyan normu, durdurma kriterinden küçük.")
            break

        print(f"Iterasyon {iterasyon + 1}: Nokta {current_nokta}, Gradyan Agirlik {gradyan_agirlik}, Gradyan Bias {gradyan_bias}")

    return current_nokta

# Ağırlıkları ve bias'ı başlat
baslangic_noktasi = np.array([0, 0])
ogrenme_orani = 0.001
iterasyon_sayisi = 1000

# Gradient inişi algoritmasını çalıştır
min_nokta = gradient_descent(turev=lambda x: gradyan_inisi(x[0], x[1], X_egitim, y_egitim),baslangic_noktasi=baslangic_noktasi,ogrenme_orani=ogrenme_orani,iterasyon_sayisi=iterasyon_sayisi)
