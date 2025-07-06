# 🐜 Karınca Kolonisi Tabanlı CNN Optimizasyonu

Bu proje, Karınca Kolonisi Optimizasyonu (ACO) algoritması ile bir Convolutional Neural Network (CNN) mimarisinin hiperparametrelerini optimize eder. Model, hastalıklı ve sağlıklı yaprak görüntülerini sınıflandırmak için eğitilmiştir.

## 🚀 Özellikler

- TensorFlow tabanlı özelleştirilebilir CNN mimarisi
- ACO algoritması ile otomatik hiperparametre seçimi
- Eğitim, doğrulama ve test metriklerinin detaylı raporu
- Otomatik öğrenme oranı, filtre sayısı, dropout, kernel boyutu, dense units, optimizer ve batch size seçimi

---

## 🧠 Kullanılan Hiperparametre Alanı

```python
param_space = {
    'filters': [128, 256, 512],
    'dropout': [0.2, 0.3, 0.4],
    'dense_units': [256, 512, 1024],
    'kernel_size': [(2,2), (3,3), (5,5)],
    'activation': ['relu'],
    'optimizer': ['adam'],
    'batch_size': [24, 32, 48]
}
****
