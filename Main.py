"""
Tekrarlayan Sinir Ağları (RNN) ile Duygu Analizi Ödev Çözümü
Bu script hem sıfırdan yazılmış RNN modelini hem de PyTorch kullanarak
yazılmış RNN modelini çalıştırır ve sonuçlarını karşılaştırır.
"""


import time
import numpy as np
from data import train_data, test_data
from RNN import run_rnn_from_scratch, predict_sentiment
from PytorchRNN import run_pytorch_rnn, predict_pytorch_sentiment


def main():
    """Her iki modeli çalıştır ve sonuçları karşılaştır"""
    print("\n" + "=" * 50)
    print("DUYGU ANALİZİ - RNN MODELLERİ KARŞILAŞTIRMASI")
    print("=" * 50)

    # 1. Sıfırdan yazılmış RNN modeli
    print("\n" + "-" * 50)
    print("1. AÇIK RNN MODELİ")
    print("-" * 50)
    start_time = time.time()
    rnn_model, rnn_vocab = run_rnn_from_scratch()
    rnn_time = time.time() - start_time
    print(f"Açık RNN modeli çalışma süresi: {rnn_time:.2f} saniye")

    # 2. PyTorch RNN modeli
    print("\n" + "-" * 50)
    print("2. PYTORCH İLE YAZILMIŞ RNN MODELİ")
    print("-" * 50)
    start_time = time.time()
    pytorch_model, pytorch_vocab = run_pytorch_rnn()
    pytorch_time = time.time() - start_time
    print(f"PyTorch RNN çalışma süresi: {pytorch_time:.2f} saniye")

    # Karşılaştırma tablosu
    print("\n" + "=" * 50)
    print("MODEL KARŞILAŞTIRMASI")
    print("=" * 50)

    # Test örnekleri
    examples = [
        "i am happy",
        "this is bad",
        "this is not bad",
        "i am not at all happy",
        "this is good right now",
        "this was very sad earlier",
        "this is happy",
        "i am not sad",
        "this is very good"
    ]

    print("\nDUYGU ANALİZİ TAHMİNLERİ:")
    print("-" * 70)
    print(f"{'Cümle':<25} {'Sıfırdan RNN':<22} {'PyTorch RNN':<22}")
    print("-" * 70)

    for example in examples:
        # Sıfırdan RNN tahmini
        rnn_score = predict_sentiment(rnn_model, example, rnn_vocab)
        rnn_sentiment = "Olumlu" if rnn_score > 0.5 else "Olumsuz"

        # PyTorch RNN tahmini
        pytorch_score = predict_pytorch_sentiment(pytorch_model, example, pytorch_vocab)
        pytorch_sentiment = "Olumlu" if pytorch_score > 0.5 else "Olumsuz"

        print(f"{example:<25} {rnn_sentiment} ({rnn_score:.4f}) {pytorch_sentiment} ({pytorch_score:.4f})")

    print("\n" + "=" * 50)
    print("SONUÇ")
    print("=" * 50)
    print(f"Açık RNN modeli çalışma süresi: {rnn_time:.2f} saniye")
    print(f"PyTorch RNN çalışma süresi: {pytorch_time:.2f} saniye")
    print(f"Hız farkı: {pytorch_time/rnn_time:.2f}x")



if __name__ == "__main__":
    main()