# scripts/model.py

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, output_dim=64, freeze=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # <-- this line
            nn.Flatten(),
        )
        self.fc = nn.Linear(64 * 4 * 4, output_dim)  # 1024 -> 64

    def forward(self, x):
        return self.fc(self.encoder(x))


class PretrainedEncoder(nn.Module):
    def __init__(self, in_channels=3, output_dim=64, freeze=True):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)

        if in_channels != 3:
            # Replace first conv layer to handle grayscale or other channel configs
            old_conv = base.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            nn.init.kaiming_normal_(
                new_conv.weight, mode="fan_out", nonlinearity="relu"
            )
            base.conv1 = new_conv

        if freeze:
            for p in base.parameters():
                p.requires_grad = False

        self.features = nn.Sequential(*list(base.children())[:-1])  # no FC
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.features(x)  # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 512]
        return self.fc(x)  # [B, output_dim]


class LocomotionClassifier(nn.Module):
    def __init__(
        self,
        input_mode="both",  # "rgb", "depth", or "both"
        use_lstm=True,
        use_pretrained=False,
        freeze_encoder=True,
        image_feature_dim=64,
        hidden_dim=128,
        num_classes=3,
    ):
        super().__init__()
        assert input_mode in {"rgb", "depth", "both"}

        EncoderCls = PretrainedEncoder if use_pretrained else ConvEncoder

        if input_mode == "rgb":
            self.rgb_encoder = EncoderCls(
                in_channels=3, output_dim=image_feature_dim, freeze=freeze_encoder
            )
            self.depth_encoder = None
            input_dim = image_feature_dim
        elif input_mode == "depth":
            self.rgb_encoder = None
            self.depth_encoder = EncoderCls(
                in_channels=1, output_dim=image_feature_dim, freeze=freeze_encoder
            )
            input_dim = image_feature_dim
        else:  # both
            self.rgb_encoder = EncoderCls(
                in_channels=3, output_dim=image_feature_dim, freeze=freeze_encoder
            )
            self.depth_encoder = EncoderCls(
                in_channels=1, output_dim=image_feature_dim, freeze=freeze_encoder
            )
            input_dim = image_feature_dim * 2

        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, rgb=None, depth=None):
        """
        Inputs (B, T, C, H, W)
        """
        features = []
        if self.rgb_encoder and rgb is not None:
            # print(">> RGB input received:", rgb.shape)
            if self.use_lstm:
                B, T, C, H, W = rgb.shape
                rgb_flat = rgb.view(B * T, C, H, W)
                rgb_feat = self.rgb_encoder(rgb_flat).view(B, T, -1)
            else:
                B, C, H, W = rgb.shape
                rgb_feat = self.rgb_encoder(rgb).view(B, -1)
            # print(">> RGB feat shape:", rgb_feat.shape)
            features.append(rgb_feat)
        # else:
        # print(">> No RGB input or encoder")
        if self.depth_encoder and depth is not None:
            # print(">> Depth input received:", depth.shape)
            if self.use_lstm:
                B, T, C, H, W = depth.shape
                depth_flat = depth.view(B * T, C, H, W)
                depth_feat = self.depth_encoder(depth_flat).view(B, T, -1)
            else:
                B, C, H, W = depth.shape
                depth_feat = self.depth_encoder(depth).view(B, -1)
            # print(">> Depth feat shape:", depth_feat.shape)
            features.append(depth_feat)
        # else:
        # print(">> No Depth input or encoder")

        x = torch.cat(features, dim=-1) if len(features) > 1 else features[0]

        if self.use_lstm:
            if self.use_lstm:
                assert x.shape[-1] == self.lstm.input_size, (
                    f"Expected input size {self.lstm.input_size}, got {x.shape[-1]}"
                )
            # print(f"rgb_feat: {rgb_feat.shape if 'rgb_feat' in locals() else 'N/A'}")
            # print(
            # f"depth_feat: {depth_feat.shape if 'depth_feat' in locals() else 'N/A'}"
            # )
            # print(
            # f"Concat feat shape: {x.shape}, LSTM input size: {self.lstm.input_size}"
            # )
            x, _ = self.lstm(x)
            x = x[:, -1]  # use final hidden state

        out = self.classifier(x)
        return out
