import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from program.feature_extraction import FeatureConfig, extract_features, extract_features_from_file, load_audio


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

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 3)

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
        target_per_class = 10000

        for label, folder in enumerate(folders):
            path = Path(dataset_root) / folder
            all_files = list(path.rglob("*.wav"))

            if folder in {"stop", "other_speech"}:
                files = all_files[:target_per_class]
            else:
                files = all_files[:target_per_class]

            if folder == "background_noise" and files:
                if len(files) < target_per_class:
                    extra = np.random.choice(files, size=target_per_class - len(files), replace=True)
                    files = list(files) + list(extra)

            for file in files:
                self.files.append(file)
                self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        label_name = ["background_noise", "other_speech", "stop"][label]

        if label_name == "background_noise":
            audio = load_audio(file_path, target_sr=self.config.sample_rate)
            gain = np.random.uniform(0.5, 1.5)
            audio = audio * gain
            if audio.size > 1:
                max_shift = max(1, int(0.1 * audio.size))
                shift = np.random.randint(0, max_shift)
                audio = np.roll(audio, shift)
            features = extract_features(audio, config=self.config)
        else:
            features = extract_features_from_file(file_path, config=self.config)

        if np.random.rand() < 0.4:
            time_width = np.random.randint(1, 4)
            time_start = np.random.randint(0, max(1, features.shape[1] - time_width))
            features[:, time_start : time_start + time_width] = 0.0
        if np.random.rand() < 0.3:
            freq_width = np.random.randint(1, 4)
            freq_start = np.random.randint(0, max(1, features.shape[0] - freq_width))
            features[freq_start : freq_start + freq_width, :] = 0.0

        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.01, features.shape)
            features = features + noise

        features = np.expand_dims(features, axis=0)

        return torch.tensor(features, dtype=torch.float32), self.labels[idx]


# ---------------- TRAIN ----------------
def train(dataset_root: str, epochs: int, batch_size: int):
    print("Loading dataset...")

    dataset = AudioDataset(dataset_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DSCNN()

    class_weights = torch.tensor([1.0, 1.5, 2.0])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    print("Training...")

    for epoch in range(epochs):
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

    print("Model saved!")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DS-CNN model.")
    parser.add_argument("--dataset-root", default="dataset/final", help="Dataset folder.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args.dataset_root, args.epochs, args.batch_size)
