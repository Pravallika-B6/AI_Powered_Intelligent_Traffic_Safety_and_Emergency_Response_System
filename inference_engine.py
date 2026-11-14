"""
Inference Engine for Hybrid Accident Detection Models.
Provides unified inference APIs for single images and video streams.
Supports all 4 Hybrids plus an Ensemble Voting mode.

Updates:
  - process_video_stream: reads ENTIRE video from first to last frame (no early exit)
  - Saves accident frame images to disk (PNG) in results/accident_frames/<video_stem>/
  - Returns accident_images_dir so the UI can build CSV + PDF report
"""

import time
import os
import io
import tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import joblib

import config
from hybrid1_transcnn import TransCNNAccidentNet
from hybrid2_yolo_san import YOLOSANNet, HeatmapGenerator
from hybrid3_waveconv import WaveConvNet, DWT2D
from hybrid4_deep_forest import DeepFeatureExtractor

# Shared standard ImageNet transform
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

MODEL_NAMES = {
    "ensemble": "🏆 Multi-Hybrid Consensus Ensemble (Recommended)",
    "hybrid1": "Hybrid 1: TransCNN-AccidentNet (Cross-Attention)",
    "hybrid2": "Hybrid 2: YOLO-SAN (Spatial Attention Gating)",
    "hybrid3": "Hybrid 3: WaveConv-DualDomain (Wavelet Frequency)",
    "hybrid4": "Hybrid 4: Deep-Forest (Dual Backbones + GBDT)"
}


class HybridInferenceEngine:
    def __init__(self, device=None):
        self.device = device if device else config.DEVICE
        self.results_dir = config.OUTPUT_DIR
        self.models_loaded = {}
        self._load_models()

    def _load_models(self):
        """Loads all available hybrid model weights from results directory."""
        # 1. Hybrid 1: TransCNN
        h1_path = self.results_dir / "hybrid1_transcnn_best.pth"
        if h1_path.exists():
            try:
                m1 = TransCNNAccidentNet(num_classes=2).to(self.device)
                m1.load_state_dict(torch.load(h1_path, map_location=self.device))
                m1.eval()
                self.models_loaded["hybrid1"] = m1
            except Exception as e:
                print(f"[WARN] Failed loading Hybrid 1: {e}")

        # 2. Hybrid 2: YOLO-SAN
        h2_path = self.results_dir / "hybrid2_yolosan_best.pth"
        if h2_path.exists():
            try:
                m2 = YOLOSANNet(num_classes=2).to(self.device)
                m2.load_state_dict(torch.load(h2_path, map_location=self.device))
                m2.eval()
                self.models_loaded["hybrid2"] = m2
            except Exception as e:
                print(f"[WARN] Failed loading Hybrid 2: {e}")

        # YOLO Heatmap Generator (used for Hybrid 2 and visual overlays)
        try:
            self.heatmap_gen = HeatmapGenerator()
        except Exception as e:
            print(f"[WARN] Failed initializing Heatmap Generator: {e}")
            self.heatmap_gen = None

        # 3. Hybrid 3: WaveConv
        h3_path = self.results_dir / "hybrid3_waveconv_best.pth"
        if h3_path.exists():
            try:
                m3 = WaveConvNet(num_classes=2).to(self.device)
                m3.load_state_dict(torch.load(h3_path, map_location=self.device))
                m3.eval()
                self.models_loaded["hybrid3"] = m3
            except Exception as e:
                print(f"[WARN] Failed loading Hybrid 3: {e}")

        # 4. Hybrid 4: Deep-Forest
        h4_ens_path = self.results_dir / "hybrid4_ensemble_meta.joblib"
        h4_scl_path = self.results_dir / "hybrid4_scaler.joblib"
        if h4_ens_path.exists() and h4_scl_path.exists():
            try:
                self.ensemble4 = joblib.load(h4_ens_path)
                self.scaler4 = joblib.load(h4_scl_path)
                self.extractor4 = DeepFeatureExtractor(device=self.device)
                self.models_loaded["hybrid4"] = True
            except Exception as e:
                print(f"[WARN] Failed loading Hybrid 4: {e}")

        # DWT wavelets for visual feature inspection
        self.dwt = DWT2D().to(self.device)

    def is_model_available(self, model_key):
        if model_key == "ensemble":
            return len(self.models_loaded) > 0
        return model_key in self.models_loaded

    def predict_single_frame(self, pil_image, model_key="ensemble", conf_threshold=0.50):
        """
        Runs inference on a single PIL image.
        Returns prediction dict with probabilities, latency, and visual artifacts.
        """
        t0 = time.time()
        img_rgb = pil_image.convert("RGB")
        tensor_img = IMAGE_TRANSFORM(img_rgb).unsqueeze(0).to(self.device)

        probs = {}
        visual_artifacts = {}

        # 1. Hybrid 1
        if "hybrid1" in self.models_loaded and model_key in ["hybrid1", "ensemble"]:
            with torch.no_grad():
                out1 = self.models_loaded["hybrid1"](tensor_img)
                p1 = F.softmax(out1, dim=1)[0].cpu().numpy()
                probs["hybrid1"] = {"non_accident": float(p1[0]), "accident": float(p1[1])}

        # 2. Hybrid 2
        if "hybrid2" in self.models_loaded and model_key in ["hybrid2", "ensemble"]:
            if self.heatmap_gen:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    img_rgb.save(tmp.name)
                    tmp_path = tmp.name
                try:
                    hm_tensor = self.heatmap_gen.generate_batch_heatmaps([tmp_path]).to(self.device)
                    with torch.no_grad():
                        out2 = self.models_loaded["hybrid2"](tensor_img, hm_tensor)
                        p2 = F.softmax(out2, dim=1)[0].cpu().numpy()
                        probs["hybrid2"] = {"non_accident": float(p2[0]), "accident": float(p2[1])}
                    if model_key == "hybrid2":
                        visual_artifacts["heatmap"] = hm_tensor[0, 0].cpu().numpy()
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

        # 3. Hybrid 3
        if "hybrid3" in self.models_loaded and model_key in ["hybrid3", "ensemble"]:
            with torch.no_grad():
                out3 = self.models_loaded["hybrid3"](tensor_img)
                p3 = F.softmax(out3, dim=1)[0].cpu().numpy()
                probs["hybrid3"] = {"non_accident": float(p3[0]), "accident": float(p3[1])}
            if model_key == "hybrid3":
                with torch.no_grad():
                    _, high_freq = self.dwt(tensor_img)
                    hf_energy = torch.mean(torch.abs(high_freq[0]), dim=0).cpu().numpy()
                    visual_artifacts["wavelet_energy"] = hf_energy

        # 4. Hybrid 4
        if "hybrid4" in self.models_loaded and model_key in ["hybrid4", "ensemble"]:
            with torch.no_grad():
                f1 = self.extractor4.effnet_pool(self.extractor4.effnet_features(tensor_img)).flatten(1)
                f2 = self.extractor4.convnext_pool(self.extractor4.convnext_features(tensor_img)).flatten(1)
                comp = torch.cat([f1, f2], dim=1).cpu().numpy()
                comp_scaled = self.scaler4.transform(comp)
                p4 = self.ensemble4.predict_proba(comp_scaled)[0]
                probs["hybrid4"] = {"non_accident": float(p4[0]), "accident": float(p4[1])}

        # Calculate final consensus probability
        if model_key == "ensemble":
            acc_list = [v["accident"] for v in probs.values()]
            accident_prob = float(np.mean(acc_list)) if acc_list else 0.0
            non_accident_prob = 1.0 - accident_prob
        elif model_key in probs:
            accident_prob = probs[model_key]["accident"]
            non_accident_prob = probs[model_key]["non_accident"]
        else:
            accident_prob = 0.0
            non_accident_prob = 1.0

        latency_ms = (time.time() - t0) * 1000
        is_accident = accident_prob >= conf_threshold

        # Annotate image with alert banner
        annotated_img = self._create_annotated_frame(img_rgb, is_accident, accident_prob)

        return {
            "is_accident": is_accident,
            "accident_prob": accident_prob,
            "non_accident_prob": non_accident_prob,
            "latency_ms": latency_ms,
            "all_model_probs": probs,
            "annotated_image": annotated_img,
            "visual_artifacts": visual_artifacts
        }

    def _create_annotated_frame(self, pil_img, is_accident, accident_prob):
        """Draws professional bounding notification overlay on image."""
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]

        color = (0, 0, 220) if is_accident else (0, 180, 0)
        label = f"ACCIDENT DETECTED ({accident_prob*100:.1f}%)" if is_accident else f"NORMAL TRAFFIC ({(1-accident_prob)*100:.1f}%)"
        tag = "ALERT [HIGH RISK]" if is_accident else "STATUS [NORMAL]"

        # Top banner overlay
        banner_h = int(h * 0.08)
        overlay = img_cv.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, img_cv, 0.25, 0, img_cv)

        # Border outline
        border_thickness = max(int(w * 0.005), 3)
        cv2.rectangle(img_cv, (0, 0), (w, h), color, border_thickness)

        # Text labels
        font_scale = max(w / 1200.0, 0.6)
        cv2.putText(img_cv, tag, (20, int(banner_h * 0.42)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, color, 2)
        cv2.putText(img_cv, label, (20, int(banner_h * 0.82)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    def process_video_stream(
        self,
        video_path,
        model_key="ensemble",
        conf_threshold=0.50,
        frame_skip=5,
        progress_callback=None,
        max_saved_accident_frames=20
    ):
        """
        Processes ENTIRE video from first frame to last frame.
        - Analyses every Kth frame (frame_skip), skipping but NOT stopping.
        - Saves accident vehicle images as PNG files to results/accident_frames/<video_stem>/
        - Returns full timeline_data, accident_frames list (with disk paths), intervals, etc.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s = total_frames / fps if total_frames > 0 else 0.0

        video_stem = Path(video_path).stem
        analysis_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── Directory to persist accident frame images on disk ──────────────
        accident_imgs_dir = self.results_dir / "accident_frames" / f"{video_stem}_{analysis_ts}"
        accident_imgs_dir.mkdir(parents=True, exist_ok=True)

        # Output annotated video writer
        out_video_path = Path(tempfile.gettempdir()) / f"annotated_{video_stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(
            str(out_video_path), fourcc,
            max(fps / max(frame_skip, 1), 5.0),
            (width, height)
        )

        timeline_data = []
        accident_frames = []          # list of dicts with image (PIL) + disk path
        processed_count = 0
        current_frame = 0

        t_start = time.time()

        # ── Read ALL frames; analyse every frame_skip-th ────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                break                 # End of video – natural exit only here

            if current_frame % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)

                pred = self.predict_single_frame(
                    pil_frame,
                    model_key=model_key,
                    conf_threshold=conf_threshold
                )
                timestamp_s = current_frame / fps
                mins = int(timestamp_s // 60)
                secs = timestamp_s % 60
                timestamp_str = f"{mins:02d}:{secs:05.2f}"

                timeline_data.append({
                    "frame_idx":     current_frame,
                    "timestamp_s":   round(timestamp_s, 2),
                    "timestamp_str": timestamp_str,
                    "accident_prob": round(pred["accident_prob"] * 100, 2),
                    "is_accident":   pred["is_accident"],
                    "model_key":     model_key,
                })

                # Write annotated frame to output video
                annotated_cv = cv2.cvtColor(np.array(pred["annotated_image"]), cv2.COLOR_RGB2BGR)
                out_writer.write(annotated_cv)

                # ── Save accident frame to disk (PNG) ────────────────────
                if pred["is_accident"]:
                    img_filename = f"accident_frame{current_frame:06d}_conf{pred['accident_prob']*100:.1f}.png"
                    img_path = accident_imgs_dir / img_filename
                    pred["annotated_image"].save(str(img_path))

                    if len(accident_frames) < max_saved_accident_frames:
                        accident_frames.append({
                            "frame_idx":     current_frame,
                            "timestamp_str": timestamp_str,
                            "confidence":    pred["accident_prob"] * 100,
                            "image":         pred["annotated_image"],   # PIL – for in-app gallery
                            "image_path":    str(img_path),             # disk path – for PDF/CSV
                            "image_filename": img_filename,
                        })

                processed_count += 1
                if progress_callback and total_frames > 0:
                    progress_callback(min(current_frame / total_frames, 1.0))

            current_frame += 1

        # Signal 100 % to caller
        if progress_callback:
            progress_callback(1.0)

        cap.release()
        out_writer.release()
        elapsed_time = time.time() - t_start

        # ── Detect contiguous accident intervals ─────────────────────────────
        intervals = []
        in_accident = False
        start_ts = None
        for item in timeline_data:
            if item["is_accident"] and not in_accident:
                in_accident = True
                start_ts = item["timestamp_str"]
            elif not item["is_accident"] and in_accident:
                in_accident = False
                intervals.append((start_ts, item["timestamp_str"]))
        if in_accident and timeline_data:
            intervals.append((start_ts, timeline_data[-1]["timestamp_str"]))

        return {
            "total_frames":        total_frames,
            "processed_frames":    processed_count,
            "fps":                 fps,
            "duration_s":          duration_s,
            "elapsed_time_s":      elapsed_time,
            "fps_speed":           processed_count / elapsed_time if elapsed_time > 0 else 0,
            "timeline_data":       timeline_data,
            "accident_frames":     accident_frames,
            "accident_intervals":  intervals,
            "output_video_path":   str(out_video_path),
            "accident_images_dir": str(accident_imgs_dir),
            "analysis_timestamp":  analysis_ts,
            "video_stem":          video_stem,
            "model_key":           model_key,
            "conf_threshold":      conf_threshold,
            "frame_skip":          frame_skip,
        }
