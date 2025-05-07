import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Ağırlıklar
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Giriş -> gizli katman
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Gizli -> gizli katman
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Gizli -> çıkış katman

        # Bias terimleri
        self.bh = np.zeros((hidden_size, 1))  # Gizli katman bias
        self.by = np.zeros((output_size, 1))  # Çıkış katman bias

        # Hiperparametreler
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size

    def forward(self, inputs):

        h = np.zeros((self.hidden_size, 1))  # Gizli durum başlangıcı
        self.inputs = inputs
        self.hs = {}  # Bu adımdaki gizli durumları sakla
        self.hs[-1] = np.copy(h)

        # İleri yayılım adımları
        for t in range(len(inputs)):
            h = np.tanh(np.dot(self.Wxh, inputs[t]) + np.dot(self.Whh, h) + self.bh)
            self.hs[t] = h

        # Çıktı hesaplama (sigmoid aktivasyon ile)
        y = 1 / (1 + np.exp(-np.dot(self.Why, h) + self.by))

        return y, h

    def backward(self, y_pred, y_true):
        # Gradyanları sıfırla
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)

        # Çıkış katmanı hatası (dL/dy)
        dy = y_pred - y_true

        # Çıkış katmanının gradyanları
        dWhy += np.dot(dy, self.hs[len(self.inputs) - 1].T)
        dby += dy

        # Gizli katmanlardan geriye yayılım
        dh_next = np.dot(self.Why.T, dy)

        for t in reversed(range(len(self.inputs))):
            # Mevcut adımdaki gizli durumun gradyanı
            dh = dh_next

            # tanh'ın türevi
            dtanh = (1 - self.hs[t]**2) * dh

            # Bias ve ağırlık gradyanları
            dbh += dtanh
            dWxh += np.dot(dtanh, self.inputs[t].T)
            dWhh += np.dot(dtanh, self.hs[t-1].T)

            # Bir önceki zaman adımına geçmek için hatayı güncelle
            if t > 0:
                dh_next = np.dot(self.Whh.T, dtanh)

        # kaybolan gradyanı kırp
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        # Ağırlıkları güncelle
        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby

        # Kayıp hesaplama (binary cross entropy loss)
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

def create_vocab(train_data):
    """Veri setinden kelime dağarcığı oluştur"""
    vocab = set()
    for sentence in train_data.keys():
        for word in sentence.split():
            vocab.add(word)
    return {word: i for i, word in enumerate(sorted(list(vocab)))}

def sentence_to_vector(sentence, vocab):
    """Cümleyi kelime vektörlerine dönüştür"""
    words = sentence.split()
    vectors = []

    for word in words:
        if word in vocab:
            # One-hot encoding
            vec = np.zeros((len(vocab), 1))
            vec[vocab[word]] = 1
            vectors.append(vec)

    return vectors

def train_rnn_model(rnn, train_data, vocab, num_epochs=1000):
    """RNN modelini eğit"""
    losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0

        for sentence, label in train_data.items():
            # Cümleyi vektörlere dönüştür
            input_vectors = sentence_to_vector(sentence, vocab)
            if not input_vectors:  # Boş cümleleri atla
                continue

            # İleri yayılım
            y_pred, _ = rnn.forward(input_vectors)

            # Label'ı numpy array'e dönüştür (True=1, False=0)
            y_true = np.array([[1.0 if label else 0.0]])

            # Geriye yayılım
            loss = rnn.backward(y_pred, y_true)
            total_loss += loss[0][0]

            # Doğruluğu hesapla
            prediction = y_pred > 0.5
            if (prediction[0][0] and label) or (not prediction[0][0] and not label):
                correct_predictions += 1

        # Her 100 adımda bir ilerlemeyi göster
        if epoch % 200 == 0:
            avg_loss = total_loss / len(train_data)
            accuracy = correct_predictions / len(train_data)
            losses.append(avg_loss)
            print(f"Epoch {epoch}, Kayıp: {avg_loss:.4f}, Doğruluk: {accuracy:.4f}")

    # Son durumu da kaydet
    avg_loss = total_loss / len(train_data)
    accuracy = correct_predictions / len(train_data)
    losses.append(avg_loss)
    print(f"Son Epoch {num_epochs}, Kayıp: {avg_loss:.4f}, Doğruluk: {accuracy:.4f}")

    return losses

def test_rnn_model(rnn, test_data, vocab):
    """Test veri setiyle modeli değerlendir"""
    correct_predictions = 0

    for sentence, label in test_data.items():
        # Cümleyi vektörlere dönüştür
        input_vectors = sentence_to_vector(sentence, vocab)
        if not input_vectors:  # Boş cümleleri atla
            continue

        # İleri yayılım
        y_pred, _ = rnn.forward(input_vectors)

        # Tahmini değerlendir
        prediction = y_pred > 0.5
        if (prediction[0][0] and label) or (not prediction[0][0] and not label):
            correct_predictions += 1

    # Test doğruluğunu hesapla
    accuracy = correct_predictions / len(test_data)
    print(f"Test Doğruluğu: {accuracy:.4f}")
    return accuracy

def predict_sentiment(rnn, sentence, vocab):
    input_vectors = sentence_to_vector(sentence, vocab)
    if not input_vectors:  # Boş cümle kontrolü
        return 0.5  # Kararsız durum

    # İleri yayılım
    y_pred, _ = rnn.forward(input_vectors)
    return y_pred[0][0]



def run_rnn_from_scratch():
    """Ana fonksiyon: RNN modelini oluştur, eğit ve test et"""
    from data import train_data, test_data

    # Kelime dağarcığı oluştur
    vocab = create_vocab(train_data)
    print(f"Kelime dağarcığı boyutu: {len(vocab)}")

    # Model parametreleri
    input_size = len(vocab)
    hidden_size = 128
    output_size = 1
    learning_rate = 0.005

    rnn = RNN(input_size, hidden_size, output_size, learning_rate)

    losses = train_rnn_model(rnn, train_data, vocab, num_epochs=1000)

    print("\nTest veri seti üzerinde değerlendiriliyor...")
    test_accuracy = test_rnn_model(rnn, test_data, vocab)

    # Örnek tahminler yap
    print("\nBazı örnek tahminler:")
    examples = [
        "i am happy",
        "this is bad",
        "this is not bad",
        "i am not at all happy",
        "this is good right now",
        "this was very sad earlier"
    ]

    for example in examples:
        score = predict_sentiment(rnn, example, vocab)
        sentiment = "Olumlu" if score > 0.5 else "Olumsuz"
        print(f"'{example}' => {sentiment} ({score:.4f})")

    return rnn, vocab

if __name__ == "__main__":
    run_rnn_from_scratch()