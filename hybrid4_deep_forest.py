"""
HYBRID MODEL 4: Deep-Forest Ensemble
Deep Feature-Stacking Hybrid:
Extracts high-dimensional deep semantic representations using dual pre-trained backbones
(EfficientNetV2-S + ConvNeXt-Tiny) and trains a Gradient Boosted Decision Forest (HistGBDT / Random Forest)
meta-learner for optimal generalization bounds on sample-constrained data.
"""

import time
import joblib
import numpy as np
import torch
import torchvision.models as models
from torchvision.models import EfficientNet_V2_S_Weights, ConvNeXt_Tiny_Weights
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import config
from dataset import get_dataloaders

class DeepFeatureExtractor:
    """Extracts and concatenates 2048-dim composite embeddings using dual deep backbones."""
    def __init__(self, device=config.DEVICE):
        self.device = device

        # Backbone 1: EfficientNetV2-S (1280 features)
        effnet = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        self.effnet_features = effnet.features.to(device).eval()
        self.effnet_pool = nn_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # Backbone 2: ConvNeXt-Tiny (768 features)
        convnext = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.convnext_features = convnext.features.to(device).eval()
        self.convnext_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    @torch.no_grad()
    def extract_features(self, dataloader):
        all_features = []
        all_labels = []

        for images, labels in dataloader:
            images = images.to(self.device)

            # EffNet embedding: [B, 1280]
            f1 = self.effnet_pool(self.effnet_features(images)).flatten(1)
            # ConvNeXt embedding: [B, 768]
            f2 = self.convnext_pool(self.convnext_features(images)).flatten(1)

            # Concatenated embedding: [B, 2048]
            composite = torch.cat([f1, f2], dim=1)
            all_features.append(composite.cpu().numpy())
            all_labels.append(labels.numpy())

        features_matrix = np.concatenate(all_features, axis=0)
        labels_vector = np.concatenate(all_labels, axis=0)
        return features_matrix, labels_vector

def train_and_eval_hybrid4():
    print("=" * 60)
    print("HYBRID 4: Deep-Forest Ensemble (Dual Backbone + GBDT Meta-Learner)")
    print("=" * 60)
    device = config.DEVICE
    print(f"Using compute device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=config.BATCH_SIZE)
    extractor = DeepFeatureExtractor(device=device)

    start_train_time = time.time()
    print("Extracting deep embeddings across Train, Val, and Test splits...")
    X_train, y_train = extractor.extract_features(train_loader)
    X_val, y_val = extractor.extract_features(val_loader)
    X_test, y_test = extractor.extract_features(test_loader)

    print(f"Feature matrix shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

    # Standardize feature representations
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Combine Train + Val for final meta-learner training
    X_combined = np.vstack([X_train_scaled, X_val_scaled])
    y_combined = np.concatenate([y_train, y_val])

    print("Training Level-2 Gradient-Boosted Forest Ensemble...")
    # Base learners: HistGradientBoosting (LightGBM equivalent) + Random Forest
    gbdt = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=1.0,
        random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )

    ensemble = VotingClassifier(
        estimators=[("gbdt", gbdt), ("rf", rf)],
        voting="soft"
    )

    ensemble.fit(X_combined, y_combined)
    training_time = time.time() - start_train_time

    # Save trained ensemble and scaler
    joblib.dump(ensemble, config.OUTPUT_DIR / "hybrid4_ensemble_meta.joblib")
    joblib.dump(scaler, config.OUTPUT_DIR / "hybrid4_scaler.joblib")
    print(f"Ensemble training completed in {training_time:.1f}s.")

    # Evaluate on Test Split
    t0 = time.time()
    y_pred = ensemble.predict(X_test_scaled)
    total_infer_time = time.time() - t0
    avg_latency_ms = (total_infer_time / len(X_test)) * 1000

    test_acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "model_name": "Hybrid 4: Deep-Forest",
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
    train_and_eval_hybrid4()
