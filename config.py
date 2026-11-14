"""
Configuration file for Hybrid Models on Accident Detection Dataset.
Target Dataset: data1 (989 frames: 791 train, 98 val, 100 test).
"""

from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

# Dataset paths
DATASET_DIR = PROJECT_ROOT / "datasets" / "Accident Detection" / "data1"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

# Weights and outputs
YOLO_WEIGHTS_PATH = PROJECT_ROOT / "Baseline_Model" / "yolo11n.pt"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters
IMG_SIZE = (256, 256)
BATCH_SIZE = 16
NUM_WORKERS = 0  # 0 for Windows multiprocessing safety
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Class labels
CLASS_NAMES = ["Non Accident", "Accident"]
NUM_CLASSES = 2

# Hardware device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
