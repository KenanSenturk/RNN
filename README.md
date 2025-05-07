# DUYGU ANALİZİ: Tekrarlayan Sinir Ağları (RNN) ile Karşılaştırmalı Çalışma

Bu çalışma, **duygu analizi** problemi üzerinde iki farklı RNN modelinin karşılaştırılmasını amaçlamaktadır:

1. NumPy kullanılarak açık olarak RNN modeli
2. PyTorch kütüphanesi kullanılarak yazılmış RNN modeli

Çalışma **tekrarlanabilir** nitelikte olup, kullanılan yöntemler teorik olarak açıklanmış, sonuçlar grafik ve tablolarla gösterilmiş, modellerin üstünlükleri detaylı olarak tartışılmıştır.

---

## GİRİŞ

Doğal Dil İşleme (NLP) alanında **duygu analizi**, bir metnin olumlu mu olumsuz mu olduğunu belirlemeyi hedefler. Tekrarlayan Sinir Ağları (RNN'ler), dizisel verilerdeki bağıntıları modelleyebilme özellikleri sayesinde bu görev için popülerdir.

Bu çalışmada, iki farklı RNN yaklaşımı incelenmiştir:

* Sıfırdan matris işlemleriyle açık bir şekilde yazılmış RNN
* PyTorch kütüphanesinin hazır RNN katmanlarını kullanan bir model

Amaç, bu iki modelin **başarım**, **eğitim süresi** ve **tahmin doğruluğu** açısından karşılaştırılmasıdır.

---

## YÖNTEMLER

### Veriseti

Küçük bir eğitim ve test veriseti, `data.py` dosyasında tanımlıdır. Cümleler anahtar (key), etiketler (0 = olumsuz, 1 = olumlu) değer olarak verilir.

### Model 1: Açık RNN Modeli

* **Matris Çarpımları** ile ileri yayılım yapılır.
* **Tanh** aktivasyonu gizli katmanda, **Sigmoid** aktivasyonu çıkışta kullanılır.
* **Binary Cross Entropy** kaybı minimize edilir.
* Geriye yayılım işlemi manuel olarak zincir kuralı ile kodlanmıştır.


### Model 2: PyTorch ile RNN

* PyTorch'un `nn.RNN` katmanı kullanılmıştır.
* **BCELoss** kayıp fonksiyonu ve **Adam** optimizasyonu ile eğitilmiştir.
* İleri ve geri yayılım PyTorch’un otomatik farklılaştırma sistemi ile yapılır.


### Deneysel Ayarlar

* Eğitim adımları: Sıfırdan RNN için 2000 epoch, PyTorch modeli için 200 epoch.
* Gizli katman boyutu: 128
* Eğitim cümleleri: 10 civarı kısa cümle.

Çalıştırmak için ana script:

```bash
python main.py
```

Bu script iki modeli eğitir, test eder ve karşılaştırır.

---

## SONUÇLAR
Sonuçlar açık RNN modeli için 1000 epoch, Pytorch modeli için 200 epoch baz alınarak hesaplanmıştır. 2 model için de learning rate 0.01'dir
### 1. Doğruluk Sonuçları

| Model              | Test Doğruluğu | Eğitim Süresi (saniye) |
| ------------------ | -------------- | ---------------------- |
| Açık RNN modeli    | 1              | 7.94                   |
| PyTorch RNN        | 1              | 96.42                  |

### 2. Tahmin Karşılaştırması

| Cümle                     | Sıfırdan RNN   | PyTorch RNN    |
| ------------------------- | -------------- | -------------- |
| i am happy                | Olumlu (0.97)  | Olumlu (0.85)  |
| this is bad               | Olumsuz (0.01) | Olumsuz (0.01) |
| this is not bad           | Olumlu (0.97)  | Olumlu (0.84)  |
| i am not at all happy     | Olumsuz (0.02) | Olumsuz (0.43) |
| this is good right now    | Olumlu (0.99)  | Olumlu (0.97)  |
| this was very sad earlier | Olumsuz (0.00) | Olumsuz (0.07) |


## TARTIŞMA

Sonuçlar göstermektedir ki:

* Her iki model de uygun epoch değerleri verildiğinde 1 doğruluğa ulaşabilmektedir. Ancak Pytorch modeli 200 epoch ve sonrasında 1 e ulaşırken açık modelde 800. epochta 1 e ulaşmaktadır.
* Verilen epoch değerlerine göre açık model 8 kat daha hızlı çalışmaktadır.

Karmaşıklık matrisi, modellerin genel olarak dengeli tahmin yaptığını, fakat sıfırdan modelin birkaç hata yaptığını göstermektedir.

**Sonuç olarak**, gerçek dünya uygulamalarında PyTorch gibi kütüphaneler önerilirken, algoritmik öğrenim için manuel RNN kodlama yararlıdır.

---

## KAYNAKLAR

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. PyTorch Documentation: [https://pytorch.org/docs/stable/generated/torch.nn.RNN.html](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
3. numpy Documentation: [https://numpy.org/doc/](https://numpy.org/doc/)
4. [Stanford CS224N - NLP with Deep Learning](https://web.stanford.edu/class/cs224n/)

