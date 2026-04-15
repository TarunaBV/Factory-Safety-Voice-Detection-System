import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import random
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from program.feature_extraction import FeatureConfig, extract_features_from_file


LABELS = ["background_noise", "other_speech", "stop", "fire", "help"]


# 🔥 MODEL
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

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, len(LABELS))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


# 🔥 DATASET
class AudioDataset(Dataset):
    def __init__(self, dataset_root):
        self.files = []
        self.labels = []

        self.config = FeatureConfig(pre_emphasis=0.97)

        MAX_SAMPLES = {
            "background_noise": 4000,
            "other_speech": 4000,
            "stop": 4000,
            "fire": 4000,
            "help": 4000
        }

        for label_idx, folder in enumerate(LABELS):
            path = Path(dataset_root) / folder

            if not path.exists():
                print(f"{folder} not found, skipping...")
                continue

            all_files = list(path.rglob("*.wav"))

            if len(all_files) > MAX_SAMPLES[folder]:
                files = random.sample(all_files, MAX_SAMPLES[folder])
            else:
                files = all_files

            for file in files:
                self.files.append(file)
                self.labels.append(label_idx)

        print(f"✅ Total samples loaded: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        features = extract_features_from_file(self.files[idx], config=self.config)

        # 🔥 augmentation
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.01, features.shape)
            features = features + noise

        features = np.expand_dims(features, axis=0)

        return torch.tensor(features, dtype=torch.float32), self.labels[idx]


# 🔥 TRAINING WITH EARLY STOPPING
def train():
    print("🚀 Loading dataset...")

    dataset = AudioDataset("dataset/final")

    # 🔥 Train / Validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    model = DSCNN()

    class_weights = torch.tensor([
        1.0,
        1.0,
        1.0,
        1.0,
        1.0
    ])

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🔥 Training started...")

    best_val_loss = float("inf")
    patience = 6
    counter = 0

    for epoch in range(60):
        # ---- TRAIN ----
        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 🔥 EARLY STOPPING
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0

            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "models/ds_cnn_model.pth")
            print("✅ Model improved & saved!")

        else:
            counter += 1
            print(f"⚠️ No improvement ({counter}/{patience})")

            if counter >= patience:
                print("🛑 Early stopping triggered!")
                break


if __name__ == "__main__":
    train()