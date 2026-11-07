import torch.nn as nn
from torchvision import models


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    # 5 blocks: 224 -> 112 -> 56 -> 28 -> 14 -> 7, then global average pool
    def __init__(self, channels=(32, 64, 128, 256, 512), num_classes=2, dropout=0.3):
        super().__init__()
        layers, c_in = [], 3
        for c in channels:
            layers.append(conv_block(c_in, c))
            c_in = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.dropout(x))


def build_model(name, num_classes=2, pretrained=True):
    if name == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "convnext_tiny":
        m = models.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    raise ValueError(f"unknown model {name}")


def gradcam_layer(model, name):
    # last conv stage, 7x7 output for 224 input
    if name == "resnet50":
        return model.layer4
    if name == "convnext_tiny":
        return model.features[-1]
    if name == "simplecnn":
        return model.features[-1][0]
    raise ValueError(name)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
