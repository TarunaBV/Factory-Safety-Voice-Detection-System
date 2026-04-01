import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

from program.feature_extraction import extract_features_from_file


# Simple DS-CNN
class DSCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def load_dataset(dataset_root):
    X = []
    y = []

    for label, folder in enumerate(["background_noise", "stop"]):
        path = Path(dataset_root) / folder

        for file in path.rglob("*.wav"):
            features = extract_features_from_file(file)
            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # reshape for CNN
    X = np.expand_dims(X, axis=1)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y)


import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

from program.feature_extraction import extract_features_from_file


class AudioDataset(Dataset):
    def __init__(self, dataset_root):
        self.files = []
        self.labels = []

        for label, folder in enumerate(["background_noise", "stop"]):
            path = Path(dataset_root) / folder
            for file in path.rglob("*.wav"):
                self.files.append(file)
                self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        features = extract_features_from_file(self.files[idx])
        features = np.expand_dims(features, axis=0)  # (1, mel, time)
        return torch.tensor(features, dtype=torch.float32), self.labels[idx]


def train():
    print("🔥 Loading dataset...")

    dataset = AudioDataset("dataset/final")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DSCNN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("🚀 Training...")

    for epoch in range(10):
        total_loss = 0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/ds_cnn_model.pth")

    print("✅ Model saved!")


if __name__ == "__main__":
    train()