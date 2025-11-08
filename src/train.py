import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score
from model import SimpleCNN


# fake test dataset for testing,  I assume size 64x64 for spectrogram, but we can change it quickly later
class FakeSpectrogramDataset(Dataset):
    def __init__(self, n_samples=200):
        self.data = np.random.randn(n_samples, 1, 64, 64).astype(np.float32)
        self.labels = np.random.randint(0, 2, n_samples)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


def train():
    dataset = FakeSpectrogramDataset()
    trainloader = DataLoader(dataset, batch_size=8, shuffle=True)

    net = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(3):  # loop over the dataset multiple times, this should be adjusted later

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        print(f"Epoch {epoch+1}, loss: {running_loss/len(trainloader):.3f}")

    # Compute fake F1-score on training data
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in trainloader:
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Training F1-score: {f1:.3f}")

    #TO DO
    #torch.save(net.state_dict(), "../models/model_name.pth")
    #print("Model saved to ../models/model_name.pth")


if __name__ == "__main__":
    train()
