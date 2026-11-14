"""
HYBRID MODEL 1: TransCNN-AccidentNet
Dual-Stream Cross-Attention Hybrid (EfficientNetV2-S + Spatial Transformer)
Combines local convolutional deformation features with global multi-vehicle relational attention.
"""

import time
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_V2_S_Weights
import config
from dataset import get_dataloaders
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional Cross-Attention Fusion (CAFF):
    - Stream A (Local CNN) queries Stream B (Global Attention)
    - Stream B (Global Attention) queries Stream A (Local CNN)
    """
    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        self.mha_l2g = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.mha_g2l = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, d_model)
        )

    def forward(self, f_local, f_global):
        # f_local: [B, 1, d_model], f_global: [B, 1, d_model]
        attn_l2g, _ = self.mha_l2g(query=f_local, key=f_global, value=f_global)
        out_local = self.norm1(f_local + attn_l2g)

        attn_g2l, _ = self.mha_g2l(query=f_global, key=f_local, value=f_local)
        out_global = self.norm2(f_global + attn_g2l)

        merged = torch.cat([out_local, out_global], dim=-1)
        fused = self.fusion_mlp(merged).squeeze(1)  # [B, d_model]
        return fused

class TransCNNAccidentNet(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, d_model=256):
        super().__init__()
        # Branch 1: Pre-trained EfficientNetV2-S
        weights = EfficientNet_V2_S_Weights.DEFAULT
        effnet = models.efficientnet_v2_s(weights=weights)
        self.cnn_features = effnet.features
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_proj = nn.Linear(1280, d_model)

        # Branch 2: Spatial Patch Transformer Encoder
        self.patch_proj = nn.Conv2d(1280, d_model, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=512, batch_first=True, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Fusion Module
        self.cross_attention = BidirectionalCrossAttention(d_model=d_model, n_heads=4)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat_map = self.cnn_features(x)  # [B, 1280, H', W']

        # 1. Local Branch
        local_vec = self.cnn_pool(feat_map).flatten(1)  # [B, 1280]
        local_emb = self.cnn_proj(local_vec).unsqueeze(1)  # [B, 1, d_model]

        # 2. Global Spatial Attention Branch
        proj_map = self.patch_proj(feat_map)  # [B, d_model, H', W']
        spatial_tokens = proj_map.flatten(2).permute(0, 2, 1)  # [B, H'*W', d_model]
        attended = self.transformer_encoder(spatial_tokens)  # [B, H'*W', d_model]
        global_emb = attended.mean(dim=1, keepdim=True)  # [B, 1, d_model]

        # 3. Cross-Attention Fusion
        fused = self.cross_attention(local_emb, global_emb)  # [B, d_model]

        # 4. Final Logits
        return self.classifier(fused)

def train_and_eval_hybrid1(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE):
    print("=" * 60)
    print("HYBRID 1: TransCNN-AccidentNet (EfficientNetV2 + Cross-Attention)")
    print("=" * 60)
    device = config.DEVICE
    print(f"Using compute device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)
    model = TransCNNAccidentNet(num_classes=config.NUM_CLASSES).to(device)

    # Freeze base CNN layers initially, train head and attention
    for param in model.cnn_features[:6].parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_weights_path = config.OUTPUT_DIR / "hybrid1_transcnn_best.pth"

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

    # Load best model for test evaluation
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
        "model_name": "Hybrid 1: TransCNN",
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
    train_and_eval_hybrid1(epochs=5)
