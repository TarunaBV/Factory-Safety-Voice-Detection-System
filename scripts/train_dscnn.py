import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from program.feature_extraction import FeatureConfig, extract_features_from_file


# ---------------- MODEL ----------------
class DSCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, padding=1),   # 🔥 NEW LAYER
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


# ---------------- DATASET ----------------
class AudioDataset(Dataset):
    def __init__(self, dataset_root):
        self.files = []
        self.labels = []

        folders = ["background_noise", "other_speech", "stop"]
        self.config = FeatureConfig(pre_emphasis=0.97)

        for label, folder in enumerate(folders):
            path = Path(dataset_root) / folder
            all_files = list(path.rglob("*.wav"))

            # 🔥 Balanced + prioritized
            if folder == "stop":
                files = all_files[:10000]
            elif folder == "other_speech":
                files = all_files[:8000]
            else:
                files = all_files[:6000]

            for file in files:
                self.files.append(file)
                self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        features = extract_features_from_file(self.files[idx], config=self.config)

        # 🔥 Light augmentation
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.01, features.shape)
            features = features + noise

        features = np.expand_dims(features, axis=0)

        return torch.tensor(features, dtype=torch.float32), self.labels[idx]


# ---------------- TRAIN ----------------
def train():
    print("🔥 Loading dataset...")

    dataset = AudioDataset("dataset/final")
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = DSCNN()

    # 🔥 prioritize STOP
    class_weights = torch.tensor([1.0, 1.5, 2.0])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🚀 Training...")

    for epoch in range(15):   # ⚡ fast + enough
        total_loss = 0.0

        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/ds_cnn_model.pth")

    print("✅ Model saved!")


if __name__ == "__main__":
    train()