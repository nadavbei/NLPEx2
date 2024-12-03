

###################################################
# Exercise 2 - Natural Language Processing 67658  #
###################################################

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
import matplotlib as plt


# subset of categories that we will use
category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }

def get_data(categories=None, portion=1.):
    """
    Get data for given categories and portion
    :param portion: portion of the data to use
    :return:
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # train
    train_len = int(portion*len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


# Q1,2
def MLP_classification(portion, model, device):
    """
    Perform linear classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    ########### add your code here ###########
    # print(y_train)
    vectorizer = TfidfVectorizer(max_features=2000)
    x_train = vectorizer.fit_transform(x_train).toarray()
    x_test = vectorizer.transform(x_test).toarray()

    train_data = list(zip(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)))
    test_data = list(zip(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)))

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16)

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

    return model


# Q3
def transformer_classification(portion=1.):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import DataLoader
    import evaluate
    from tqdm import tqdm

    class Dataset(torch.utils.data.Dataset):
        """
        Dataset for loading data
        """
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    def train_epoch(model, data_loader, optimizer, dev='cpu'):
        """
        Perform an epoch of training of the model with the optimizer
        :param model:
        :param data_loader:
        :param optimizer:
        :param dev:
        :return: Average loss over the epoch
        """
        model.train()
        total_loss = 0.
        # iterate over batches
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(data_loader)

    def evaluate_model(model, data_loader, dev='cpu', metric=None):
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(data_loader):
                input_ids = batch['input_ids'].to(dev)
                attention_mask = batch['attention_mask'].to(dev)
                labels = batch['labels'].to(dev)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)

                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        return correct / total

    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Parameters
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = len(category_dict)
    epochs = 3
    batch_size = 16
    learning_rate = 5e-5

    # Model, tokenizer, and metric
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=num_labels).to(dev)
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    metric = evaluate.load("accuracy")

    # Datasets and DataLoaders
    train_dataset = Dataset(tokenizer(x_train, truncation=True, padding=True), y_train)
    val_dataset = Dataset(tokenizer(x_test, truncation=True, padding=True), y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training and evaluation
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, dev)
        val_accuracy = evaluate_model(model, val_loader, dev)

        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')

    plt.tight_layout()
    plt.show()

    print(print("model 3 numbers of parameters =", count_trainable_parameters(model)))

    return model


if __name__ == "__main__":
    portions = [0.1, 0.2, 0.5, 1.]
    # Q1 - single layer MLP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.Sequential(nn.Linear(2000, len(category_dict))).to(device)
    MLP_classification(0.1, model, device)
    print("model 1 numbers of parameters =", count_trainable_parameters(model))

    # Q2 - multi-layer MLP
    model = nn.Sequential(
        nn.Linear(2000, 500),  # Input layer to hidden layer
        nn.ReLU(),  # Activation function for non-linearity
        nn.Linear(500, len(category_dict))  # Hidden layer to output layer
    ).to(device)
    MLP_classification(0.1, model, device)
    print("model 2 numbers of parameters =", count_trainable_parameters(model))

    # Q3 - Transformer
    print("\nTransformer results:")
    for p in portions[:2]:
        print(f"Portion: {p}")
        transformer_classification(portion=p)
