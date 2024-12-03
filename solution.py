import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import Adam, AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import evaluate

# Subset of categories we will use
category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'}


def get_data(categories=None, portion=1.0):
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # Train
    train_len = int(portion * len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # Test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test


# Helper function for training
def train_model(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# Helper function for evaluation
def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# Single-layer MLP
def single_layer_mlp_classification(portion):
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    vectorizer = TfidfVectorizer(max_features=2000)
    x_train = vectorizer.fit_transform(x_train).toarray()
    x_test = vectorizer.transform(x_test).toarray()

    train_data = list(zip(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)))
    test_data = list(zip(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)))

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.Sequential(nn.Linear(2000, len(category_dict))).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, val_accuracies = [], []
    for epoch in range(20):
        train_loss = train_model(model, train_loader, optimizer, loss_fn, device)
        train_losses.append(train_loss)
        val_accuracy = evaluate_model(model, test_loader, device)
        val_accuracies.append(val_accuracy)

    # Plotting
    plt.figure()
    plt.plot(range(1, 21), train_losses, label="Train Loss")
    plt.plot(range(1, 21), val_accuracies, label="Validation Accuracy")
    plt.title(f"Single Layer MLP (Portion={portion})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


# Multi-layer MLP
def multi_layer_mlp_classification(portion):
    # Similar to single-layer but with an additional hidden layer
    pass  # Add hidden layer logic and reuse the helper functions above


# Transformer-based classification
def transformer_classification(portion):
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    train_encodings = tokenizer(x_train, truncation=True, padding=True, return_tensors='pt')
    test_encodings = tokenizer(x_test, truncation=True, padding=True, return_tensors='pt')

    class TextDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

    train_dataset = TextDataset(train_encodings, y_train)
    test_dataset = TextDataset(test_encodings, y_test)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=len(category_dict))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        train_loss = train_model(model, train_loader, optimizer, nn.CrossEntropyLoss(), device)
        val_accuracy = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch + 1}: Loss={train_loss:.4f}, Accuracy={val_accuracy:.4f}")


if __name__ == "__main__":
    portions = [0.1, 0.2, 0.5, 1.0]
    for portion in portions:
        single_layer_mlp_classification(portion)
        # multi_layer_mlp_classification(portion)
        # transformer_classification(portion)  # Uncomment to run
