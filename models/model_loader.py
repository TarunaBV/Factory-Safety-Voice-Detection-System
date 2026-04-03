import numpy as np
import torch
import torch.nn as nn


LABELS = ["background_noise", "other_speech", "stop"]

class DSCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(32, len(LABELS))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def load_model(model_path, device="cpu"):
    model = DSCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict(model, features):
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=0)

    tensor = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]

    label_index = int(np.argmax(probs))
    confidence = float(probs[label_index])
    label_name = LABELS[label_index]

    return label_index, label_name, confidence
