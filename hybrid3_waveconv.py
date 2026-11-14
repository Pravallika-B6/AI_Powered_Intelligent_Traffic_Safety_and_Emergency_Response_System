"""
HYBRID MODEL 3: WaveConv-DualDomain
Spatial-Frequency Fusion Hybrid:
Decomposes CCTV frames via a 2D Haar Discrete Wavelet Transform (DWT) in PyTorch.
High-frequency sub-bands (fractured metal, shattered glass, collision debris)
are fused with deep spatial CNN representations via Channel Attention (CBAM).
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet34_Weights
import config
from dataset import get_dataloaders
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class DWT2D(nn.Module):
    """
    Native Differentiable 2D Haar Wavelet Transform in PyTorch.
    Zero external dependencies, GPU/CPU accelerated.
    Produces 4 sub-bands: LL (Approximation), LH (Horizontal), HL (Vertical), HH (Diagonal).
    """
    def __init__(self):
        super().__init__()
        # 2x2 Haar wavelet kernels
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])

        # Shape: [4, 1, 2, 2]
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filters)

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        # Depthwise convolution per channel
        weight = self.filters.repeat(c, 1, 1, 1)  # [4*C, 1, 2, 2]
        subbands = F.conv2d(x, weight, stride=2, padding=0, groups=c)  # [B, 4*C, H/2, W/2]
        subbands = subbands.view(b, c, 4, h // 2, w // 2)

        ll = subbands[:, :, 0, :, :]  # Low-freq approximation: [B, C, H/2, W/2]
        high_freq = subbands[:, :, 1:, :, :].reshape(b, c * 3, h // 2, w // 2)  # [B, 3*C, H/2, W/2]
        return ll, high_freq

class FrequencyFeatureExtractor(nn.Module):
    """Processes high-frequency wavelet sub-bands (LH, HL, HH) to detect crash fractures."""
    def __init__(self, in_channels=9, out_channels=128):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8))
        )

    def forward(self, x):
        return self.conv_block(x)

class ChannelAttentionFusion(nn.Module):
    """CBAM-inspired Channel Attention mechanism fusing Spatial and Wavelet features."""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        scale = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale

class WaveConvNet(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        self.dwt = DWT2D()
        self.freq_branch = FrequencyFeatureExtractor(in_channels=9, out_channels=128)

        # Spatial Branch: ResNet-34
        weights = ResNet34_Weights.DEFAULT
        resnet = models.resnet34(weights=weights)
        self.spatial_backbone = nn.Sequential(*list(resnet.children())[:-2])  # Output: 512 channels, [B, 512, 8, 8]

        # Fusion: 512 (spatial) + 128 (wavelet freq) = 640 channels
        self.fusion_attention = ChannelAttentionFusion(channels=640)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(640, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 1. Frequency Domain Wavelet Decomposition
        _, high_freq = self.dwt(x)  # [B, 9, 128, 128]
        freq_feat = self.freq_branch(high_freq)  # [B, 128, 8, 8]

        # 2. Spatial Domain Deep Feature Extraction
        spatial_feat = self.spatial_backbone(x)  # [B, 512, 8, 8]

        # 3. Dual-Domain Feature Concat & Attention
        fused = torch.cat([spatial_feat, freq_feat], dim=1)  # [B, 640, 8, 8]
        fused = self.fusion_attention(fused)

        # 4. Global Pooling & Classification
        vector = self.pool(fused).flatten(1)  # [B, 640]
        return self.classifier(vector)

def train_and_eval_hybrid3(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE):
    print("=" * 60)
    print("HYBRID 3: WaveConv-DualDomain (Spatial-Frequency Fusion)")
    print("=" * 60)
    device = config.DEVICE
    print(f"Using compute device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)
    model = WaveConvNet(num_classes=config.NUM_CLASSES).to(device)

    # Freeze initial spatial layers
    for param in list(model.spatial_backbone.parameters())[:25]:
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_weights_path = config.OUTPUT_DIR / "hybrid3_waveconv_best.pth"

    start_train_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)

    training_time = time.time() - start_train_time
    print(f"\nTraining completed in {training_time:.1f}s. Evaluating on Test Split...")

    if best_weights_path.exists():
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    inference_times = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            t0 = time.time()
            outputs = model(images)
            inference_times.append((time.time() - t0) / images.size(0))
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    avg_latency_ms = (sum(inference_times) / len(inference_times)) * 1000

    metrics = {
        "model_name": "Hybrid 3: WaveConv",
        "test_acc": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency_ms": avg_latency_ms,
        "training_time_s": training_time
    }
    print(f"-> Test Acc: {test_acc*100:.2f}% | Precision: {precision*100:.2f}% | "
          f"Recall: {recall*100:.2f}% | F1: {f1*100:.2f}% | Latency: {avg_latency_ms:.1f} ms/frame\n")
    return metrics

if __name__ == "__main__":
    train_and_eval_hybrid3(epochs=5)
