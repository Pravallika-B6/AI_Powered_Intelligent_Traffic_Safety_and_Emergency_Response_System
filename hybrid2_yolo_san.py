"""
HYBRID MODEL 2: YOLO-SAN (YOLO-Guided Spatial Attention Network)
Object-Centric Hybrid: YOLOv11 vehicle detection generates an Object Proximity Map (OPM)
which serves as a spatial attention gate on deep CNN features.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights
from ultralytics import YOLO
import config
from dataset import get_dataloaders
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SpatialAttentionGate(nn.Module):
    """Gates CNN feature maps using an external Object Proximity Heatmap."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv_mask = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feature_map, heatmap):
        # Resize heatmap to match feature map resolution [B, 1, H', W']
        downsampled_mask = F.interpolate(heatmap, size=feature_map.shape[2:], mode="bilinear", align_corners=False)
        gate = self.conv_mask(downsampled_mask)
        # Residual spatial gating: base features + gated amplification
        return feature_map * (1.0 + gate)

class YOLOSANNet(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT
        mobilenet = models.mobilenet_v3_large(weights=weights)
        self.backbone = mobilenet.features  # Output channels: 960
        self.spatial_gate = SpatialAttentionGate(in_channels=960)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(960, 256),
            nn.Hardswish(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, heatmap):
        # x: [B, 3, 256, 256], heatmap: [B, 1, 256, 256]
        feat = self.backbone(x)  # [B, 960, 8, 8]
        gated_feat = self.spatial_gate(feat, heatmap)
        pooled = self.pool(gated_feat).flatten(1)
        return self.classifier(pooled)

class HeatmapGenerator:
    """Uses YOLOv11 to extract vehicle bounding boxes and render proximity density heatmaps."""
    def __init__(self, weights_path=config.YOLO_WEIGHTS_PATH):
        # Vehicle COCO classes: 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = {2, 3, 5, 7}
        if weights_path.exists():
            self.yolo = YOLO(str(weights_path))
        else:
            print("[WARN] YOLO weights not found. Falling back to downloading yolo11n.pt")
            self.yolo = YOLO("yolo11n.pt")

    def generate_batch_heatmaps(self, img_paths, target_size=(256, 256)):
        batch_heatmaps = []
        # Batch inference with YOLO
        results = self.yolo.predict(img_paths, verbose=False, conf=0.25)
        for r in results:
            hm = np.zeros(target_size, dtype=np.float32)
            orig_h, orig_w = r.orig_shape
            scale_y = target_size[0] / orig_h
            scale_x = target_size[1] / orig_w

            boxes = []
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id in self.target_classes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        boxes.append(xyxy)

            # Render Gaussian blobs for each vehicle
            for i, b in enumerate(boxes):
                x1, y1, x2, y2 = b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                bw, bh = max(int((x2 - x1) / 2), 5), max(int((y2 - y1) / 2), 5)
                
                # Boost intensity if boxes overlap (indicates proximity / impact zone)
                intensity = 1.0
                for j, b_other in enumerate(boxes):
                    if i != j:
                        # Check bounding box distance
                        other_cx = (b_other[0] + b_other[2]) * scale_x / 2
                        other_cy = (b_other[1] + b_other[3]) * scale_y / 2
                        dist = np.hypot(cx - other_cx, cy - other_cy)
                        if dist < (bw + bh):
                            intensity += 1.0

                # Draw Gaussian circle
                y_grid, x_grid = np.ogrid[:target_size[0], :target_size[1]]
                dist_sq = ((x_grid - cx) / bw)**2 + ((y_grid - cy) / bh)**2
                gaussian = intensity * np.exp(-dist_sq)
                hm = np.maximum(hm, gaussian.astype(np.float32))

            # Normalize heatmap
            if hm.max() > 0:
                hm = hm / hm.max()
            batch_heatmaps.append(hm)

        # Return tensor [B, 1, H, W]
        tensor_hm = torch.from_numpy(np.stack(batch_heatmaps)).unsqueeze(1)
        return tensor_hm

def train_and_eval_hybrid2(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE):
    print("=" * 60)
    print("HYBRID 2: YOLO-SAN (YOLO-Guided Spatial Attention Network)")
    print("=" * 60)
    device = config.DEVICE
    print(f"Using compute device: {device}")

    # DataLoaders that return image paths for YOLO heatmap extraction
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size, return_path=True)
    heatmap_gen = HeatmapGenerator()
    model = YOLOSANNet(num_classes=config.NUM_CLASSES).to(device)

    # Freeze lower layers
    for param in model.backbone[:8].parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_weights_path = config.OUTPUT_DIR / "hybrid2_yolosan_best.pth"

    start_train_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels, paths in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Generate heatmaps with YOLO
            heatmaps = heatmap_gen.generate_batch_heatmaps(paths).to(device)

            optimizer.zero_grad()
            outputs = model(images, heatmaps)
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
            for images, labels, paths in val_loader:
                images, labels = images.to(device), labels.to(device)
                heatmaps = heatmap_gen.generate_batch_heatmaps(paths).to(device)
                outputs = model(images, heatmaps)
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
        for images, labels, paths in test_loader:
            images = images.to(device)
            t0 = time.time()
            heatmaps = heatmap_gen.generate_batch_heatmaps(paths).to(device)
            outputs = model(images, heatmaps)
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
        "model_name": "Hybrid 2: YOLO-SAN",
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
    train_and_eval_hybrid2(epochs=5)
