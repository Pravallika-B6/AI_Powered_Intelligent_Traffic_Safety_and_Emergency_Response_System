# AI-Powered Intelligent Traffic Safety and Emergency Response System

## Hybrid Deep Learning Architectures for Traffic Accident Detection

This project focuses on intelligent traffic monitoring and emergency response using AI, deep learning, and computer vision. It supports wrong-way vehicle detection, accident detection, automated alerts, and emergency vehicle priority management.

This repository contains the implementation of four hybrid architectures designed for traffic accident detection using CCTV surveillance and dashcam datasets, accompanied by an interactive Streamlit web application.

---

## Interactive Web Front-End (`app.py`)

The application provides a dashboard for uploading images and videos, selecting any of the four hybrid models (or Consensus Ensemble), and viewing detection results.

### Key Capabilities

1. **Single Image Analysis**
   - Upload JPG, PNG, JPEG, BMP, and WEBP images.
   - Predict accident or normal traffic.
   - View confidence scores and multi-model comparison.
   - Visualize vehicle proximity heatmaps and wavelet energy maps.

2. **Video Stream Analysis**
   - Upload MP4, AVI, MOV, and MKV videos.
   - Identify accident intervals with timestamps.
   - Generate an interactive accident timeline.
   - Display accident snapshot frames.
   - Export incident information to CSV.

3. **Model Benchmark and Comparison**
   - Compare the four hybrid architectures.
   - View model architecture, parameters, and inference latency.

---

## How to Launch the Web Application

### Option 1: Windows Launcher

Double-click:

`run_frontend.bat`

### Option 2: Terminal

```powershell
streamlit run app.py
```

Open: http://localhost:8501

---

## Hybrid Model Architectures

1. TransCNN-AccidentNet
2. YOLO-SAN
3. WaveConv-DualDomain
4. Deep-Forest Ensemble

---

## Repository Structure

```text
hybride_models/
├── app.py
├── inference_engine.py
├── run_frontend.bat
├── config.py
├── dataset.py
├── hybrid1_transcnn.py
├── hybrid2_yolo_san.py
├── hybrid3_waveconv.py
├── hybrid4_deep_forest.py
├── evaluate_all_hybrids.py
├── results/
└── README.md
```

**Note:** Model accuracy and inference latency should be reported only after actual experimental evaluation.
