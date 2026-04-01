import torch
import torch.nn as nn
import numpy as np


class DSCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def load_model(model_path, device="cpu"):
    model = DSCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict(model, features):
    # features shape: (mel, time)
    features = np.expand_dims(features, axis=0)  # batch
    features = np.expand_dims(features, axis=0)  # channel

    tensor = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]

    label = np.argmax(probs)
    confidence = probs[label]

    return label, confidence