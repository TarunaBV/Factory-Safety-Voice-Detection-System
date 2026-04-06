import numpy as np
import torch
import torch.nn as nn


LABELS = ["background_noise", "other_speech", "stop"]


# ---------------- MODEL (MATCH TRAINING EXACTLY) ----------------
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
        self.fc = nn.Linear(64, len(LABELS))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


# ---------------- LOAD MODEL ----------------
def load_model(model_path, device="cpu"):
    model = DSCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ---------------- PREDICT ----------------
def predict(model, features, device="cpu"):
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=0)

    tensor = torch.tensor(features, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    label_index = int(np.argmax(probs))
    confidence = float(probs[label_index])
    label_name = LABELS[label_index]

    return label_index, label_name, confidence
