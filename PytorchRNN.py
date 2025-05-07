import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PyTorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(PyTorchRNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN katmanı
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Çıkış katmanı
        self.fc = nn.Linear(hidden_size, output_size)

        # Sigmoid aktivasyon
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden=None):
        # Başlangıç gizli durumu
        if hidden is None:
            hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # RNN ileri yayılım
        out, hidden = self.rnn(x, hidden)

        # Son adımdaki çıktıyı al
        out = self.fc(out[:, -1, :])

        # Sigmoid aktivasyon ile [0,1] aralığına getir
        out = self.sigmoid(out)

        return out, hidden


def create_vocab(train_data):
    vocab = set()
    for sentence in train_data.keys():
        for word in sentence.split():
            vocab.add(word)
    return {word: i for i, word in enumerate(sorted(list(vocab)))}


def sentence_to_tensor(sentence, vocab):
    """Cümleyi tensor'a dönüştür"""
    words = sentence.split()
    tensor = torch.zeros(len(words), len(vocab))

    for i, word in enumerate(words):
        if word in vocab:
            tensor[i, vocab[word]] = 1

    return tensor.unsqueeze(0)  # Batch boyutu ekle (1, seq_len, vocab_size)


def train_pytorch_model(model, train_data, vocab, num_epochs=200, learning_rate=0.01):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for sentence, label in train_data.items():
            # Giriş tensörünü oluştur
            input_tensor = sentence_to_tensor(sentence, vocab)

            # Hedef tensörünü oluştur (True=1, False=0)
            target = torch.tensor([1.0 if label else 0.0]).unsqueeze(0)

            # İleri yayılım
            optimizer.zero_grad()
            output, _ = model(input_tensor)

            # Kayıp hesaplama
            loss = criterion(output, target)

            # Geriye yayılım
            loss.backward()
            optimizer.step()

            # İstatistikler
            running_loss += loss.item()
            predicted = (output > 0.5).float()
            total += 1
            correct += (predicted == target).sum().item()

        # Epoch sonundaki istatistikler
        epoch_loss = running_loss / len(train_data)
        epoch_acc = correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Kayıp: {epoch_loss:.4f}, Doğruluk: {epoch_acc:.4f}")

    print(f"Son Epoch {num_epochs}, Kayıp: {epoch_loss:.4f}, Doğruluk: {epoch_acc:.4f}")

    return losses, accuracies


def test_pytorch_model(model, test_data, vocab):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for sentence, label in test_data.items():
            # Giriş tensörünü oluştur
            input_tensor = sentence_to_tensor(sentence, vocab)

            # İleri yayılım
            output, _ = model(input_tensor)

            # Tahmin
            predicted = (output > 0.5).float()
            target = torch.tensor([1.0 if label else 0.0]).unsqueeze(0)

            # Doğruluğu güncelle
            correct += (predicted == target).sum().item()
            total += 1

    accuracy = correct / total
    print(f"PyTorch RNN Test Doğruluğu: {accuracy:.4f}")
    return accuracy

def predict_pytorch_sentiment(model, sentence, vocab):
    model.eval()

    # Giriş tensörünü oluştur
    input_tensor = sentence_to_tensor(sentence, vocab)

    # İleri yayılım
    with torch.no_grad():
        output, _ = model(input_tensor)

    return output.item()

def run_pytorch_rnn():
    from data import train_data, test_data

    # Kelime dağarcığı oluştur
    vocab = create_vocab(train_data)
    print(f"Kelime dağarcığı boyutu: {len(vocab)}")

    # Model parametreleri
    input_size = len(vocab)
    hidden_size = 256
    output_size = 1

    model = PyTorchRNN(input_size, hidden_size, output_size, num_layers=2)

    losses, accuracies = train_pytorch_model(model, train_data, vocab, num_epochs=200, learning_rate=0.001)

    print("\nTest veri seti üzerinde değerlendiriliyor...")
    test_accuracy = test_pytorch_model(model, test_data, vocab)

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
        score = predict_pytorch_sentiment(model, example, vocab)
        sentiment = "Olumlu" if score > 0.5 else "Olumsuz"
        print(f"'{example}' => {sentiment} ({score:.4f})")

    return model, vocab


if __name__ == "__main__":
    run_pytorch_rnn()