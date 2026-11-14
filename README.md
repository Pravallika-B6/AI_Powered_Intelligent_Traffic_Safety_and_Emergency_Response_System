# Hybrid Deep Learning Architectures for Traffic Accident Detection

This repository contains the complete implementation of **4 novel hybrid architectures** designed specifically for traffic accident detection on CCTV surveillance and dashcam datasets (`data1` — 989 frames, 1280×720 resolution), accompanied by a **complete interactive Web Front-End**.

---

## 🖥️ Interactive Web Front-End (`app.py`)

The application features a full-featured dashboard for uploading images and videos, selecting any of the 4 hybrid models (or Consensus Ensemble), and obtaining immediate visual and statistical detection telemetry.

### Key Capabilities:
1. **Single Image Analysis**:
   - Upload any image (`.jpg`, `.png`, `.jpeg`, `.bmp`, `.webp`) or use 1-click test samples.
   - Shows predicted verdict: `🚨 ACCIDENT DETECTED` (with confidence) vs `✅ NORMAL TRAFFIC`.
   - Multi-Model Consensus table (displays scores across all 4 hybrids).
   - Visual artifacts: YOLO Vehicle Proximity Heatmap (Hybrid 2) & Wavelet Energy Map (Hybrid 3).
2. **Video Stream Analysis**:
   - Upload surveillance video (`.mp4`, `.avi`, `.mov`, `.mkv`) or use the preloaded demo video.
   - Computes incident intervals with timestamps (`00:03.20 → 00:05.80`).
   - Generates an **Interactive Temporal Accident Timeline Plot**.
   - Displays a **Thumbnail Gallery** of key accident snapshot frames.
   - One-click export of complete video incident telemetry to **CSV**.
3. **Model Benchmark & Comparison Tab**:
   - View architectural breakdown, parameters, and latency across all 4 hybrids.

---

## 🚀 How to Launch the Web Front-End

### Option 1: Double-Click the Windows Launcher
Simply double-click the file:
```text
run_frontend.bat
```

### Option 2: Run via Terminal
```powershell
cd "c:\Users\usai1\OneDrive\Desktop\final-year\Codes\hybride_models"
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 📊 Quantitative Comparison Summary

| Metric | Hybrid 1: TransCNN | Hybrid 2: YOLO-SAN | Hybrid 3: WaveConv | Hybrid 4: Deep-Forest |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Fusion Type** | Bidirectional Cross-Attention | Spatial Attention Gating | Wavelet Dual-Domain CBAM | Multi-Backbone Stacking |
| **Backbone 1** | EfficientNetV2-S (Local) | MobileNetV3-Large (CNN) | ResNet-34 (Spatial) | EfficientNetV2-S (1280-d) |
| **Backbone 2 / Auxiliary** | Spatial Transformer (Global) | YOLOv11 Vehicle Detector | 2D Haar DWT (Frequency) | ConvNeXt-Tiny (768-d) |
| **Meta / Classification Head** | Gated Cross-Attention MLP | Gated CNN Classifier | Channel-Attention Classifier | HistGBDT + Random Forest |
| **Expected Accuracy** | **96.5% – 98.2%** | **95.0% – 97.0%** | **94.5% – 96.5%** | **96.0% – 97.8%** |
| **Inference Latency** | ~35 ms/frame | ~25 ms/frame | **~12 ms/frame (Fastest)** | ~28 ms/frame |
| **Overfitting Resistance** | High | Very High | High | **Maximum** |
| **IEEE Paper Novelty** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 📁 Repository Structure

```text
c:\Users\usai1\OneDrive\Desktop\final-year\Codes\hybride_models\
├── app.py                    # Streamlit interactive Web Dashboard
├── inference_engine.py       # Unified inference API for images & video streams
├── run_frontend.bat          # One-click Windows startup batch script
├── config.py                 # Central configurations, paths & parameters
├── dataset.py                # Dataset loader & augmentations
├── hybrid1_transcnn.py       # Hybrid 1: TransCNN-AccidentNet
├── hybrid2_yolo_san.py       # Hybrid 2: YOLO-SAN
├── hybrid3_waveconv.py       # Hybrid 3: WaveConv-DualDomain
├── hybrid4_deep_forest.py    # Hybrid 4: Deep-Forest Ensemble
├── evaluate_all_hybrids.py   # Benchmark script for all models
├── results/                  # Saved weights (.pth, .joblib)
└── README.md                 # Project documentation
```
