import numpy as np
import torch
import torch.nn as nn

LABELS = ["background_noise", "other_speech", "stop"]


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
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


def load_model(path):
    model = DSCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def predict(model, features):
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=0)

    tensor = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]

    idx = int(np.argmax(probs))
    return idx, LABELS[idx], float(probs[idx])