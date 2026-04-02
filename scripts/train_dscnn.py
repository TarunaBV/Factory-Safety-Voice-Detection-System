import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from program.feature_extraction import extract_features_from_file


# ---------------- MODEL ----------------
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

        self.fc = nn.Linear(32, 3)  # 3 classes

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------- DATASET ----------------
class AudioDataset(Dataset):
    def __init__(self, dataset_root):
        self.files = []
        self.labels = []

        folders = ["background_noise", "other_speech", "stop"]

        for label, folder in enumerate(folders):
            path = Path(dataset_root) / folder
            all_files = list(path.rglob("*.wav"))

            # 🔥 Balance dataset manually
            if folder == "stop":
                files = all_files[:10000]   # more STOP
            elif folder == "other_speech":
                files = all_files[:7000]   # reduce speech
            else:
                files = all_files[:5000]   # noise

            for file in files:
                self.files.append(file)
                self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        features = extract_features_from_file(self.files[idx])
        features = np.expand_dims(features, axis=0)
        return torch.tensor(features, dtype=torch.float32), self.labels[idx]


# ---------------- TRAIN ----------------
def train():
    print("🔥 Loading dataset...")

    dataset = AudioDataset("dataset/final")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DSCNN()

    # 🔥 Give more importance to STOP class
    class_weights = torch.tensor([1.0, 1.0, 2.0])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🚀 Training...")

    for epoch in range(20):
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