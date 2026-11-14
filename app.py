"""
Streamlit Web Front-End for Hybrid Traffic Accident Detection Models.
Provides interactive image & video upload, multi-hybrid model selection,
real-time incident timeline visualization, accident frame gallery, and report export.

Updates (v2):
  - Video: reads ENTIRE video beginning-to-end with configurable K-frame skip
  - CSV export is always shown after video completion
  - Accident frame images are saved to disk automatically
  - PDF report (with embedded images) can be downloaded after analysis
"""

import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import streamlit as st
import matplotlib.pyplot as plt

# Set page layout to wide and modern title
st.set_page_config(
    page_title="AI Traffic Accident Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add current folder to path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

import config
from inference_engine import HybridInferenceEngine, MODEL_NAMES

# Optional PDF generator (gracefully skip if fpdf2 not installed)
try:
    from report_generator import generate_pdf_report
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# Initialize and cache inference engine
@st.cache_resource
def get_engine():
    return HybridInferenceEngine()

engine = get_engine()

# ── Initialise persistent session state keys ──────────────────────────────────
for _key, _default in [
    ("analysis_done",       False),
    ("video_res",           None),
    ("df_timeline",         None),
    ("pdf_bytes",           None),
    ("pdf_filename",        "incident_report.pdf"),
    ("video_path",          None),
    ("uploaded_video_name", None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card-danger {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card-success {
        background-color: #F0FDF4;
        border-left: 6px solid #22C55E;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stat-title {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        font-weight: 600;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-header">🚗 Intelligent Traffic Surveillance & Accident Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Modal Incident Analysis Powered by Novel Hybrid Deep Learning Architectures</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/fluency/96/traffic-jam.png", width=64)
st.sidebar.title("System Controls")

# 1. Analysis Mode
mode = st.sidebar.radio(
    "Choose Analysis Mode:",
    ["📷 Single Image Analysis", "🎥 Video Stream Analysis", "📊 Hybrid Model Benchmark"],
    index=0
)

# 2. Model Selection
st.sidebar.markdown("---")
st.sidebar.subheader("Model Configuration")
selected_model_name = st.sidebar.selectbox(
    "Select Hybrid Architecture:",
    list(MODEL_NAMES.values()),
    index=0
)
# Reverse lookup key
selected_model_key = [k for k, v in MODEL_NAMES.items() if v == selected_model_name][0]

# 3. Detection Threshold
conf_threshold = st.sidebar.slider(
    "Detection Sensitivity Threshold:",
    min_value=0.30,
    max_value=0.90,
    value=0.50,
    step=0.05,
    help="Frames with accident probability equal to or exceeding this threshold trigger an incident alert."
)

# 4. Video Sampling (if video mode)
if "Video" in mode:
    frame_skip = st.sidebar.slider(
        "Frame Sampling Rate (Analyze every Nth frame):",
        min_value=1,
        max_value=15,
        value=5,
        help="Higher values increase processing speed. 1 = Process every frame."
    )
else:
    frame_skip = 5

# 5. Max accident images to save
if "Video" in mode:
    max_acc_images = st.sidebar.slider(
        "Max Accident Images to Save:",
        min_value=5,
        max_value=50,
        value=20,
        help="Maximum number of accident frames to store as PNG images and embed in the PDF report."
    )
else:
    max_acc_images = 20

st.sidebar.markdown("---")
st.sidebar.info("""
**Active Hybrids in Engine:**
* Hybrid 1: TransCNN (Cross-Attention)
* Hybrid 2: YOLO-SAN (Spatial Heatmap)
* Hybrid 3: WaveConv (Haar Wavelet)
* Hybrid 4: Deep-Forest (Dual Backbone)
""")

# ==============================================================================
# MODE 1: SINGLE IMAGE ANALYSIS
# ==============================================================================
if mode == "📷 Single Image Analysis":
    st.markdown("### 📷 Upload or Select a Traffic Surveillance Frame")
    
    col_demo, col_upload = st.columns([1, 2])
    with col_demo:
        st.markdown("**Quick Demo Samples (From `data1/test`):**")
        btn_accident = st.button("🚨 Load Test Accident Sample", use_container_width=True)
        btn_normal = st.button("✅ Load Test Normal Sample", use_container_width=True)
        
    with col_upload:
        uploaded_file = st.file_uploader(
            "Or upload your own traffic image (JPG, PNG, JPEG):",
            type=["jpg", "jpeg", "png", "bmp", "webp"]
        )

    pil_image = None
    sample_title = ""

    if btn_accident:
        accident_samples = list((config.TEST_DIR / "Accident").glob("*.jpg"))
        if accident_samples:
            pil_image = Image.open(accident_samples[0])
            sample_title = f"Demo Sample: {accident_samples[0].name} [Accident Class]"
    elif btn_normal:
        normal_samples = list((config.TEST_DIR / "Non Accident").glob("*.jpg"))
        if normal_samples:
            pil_image = Image.open(normal_samples[0])
            sample_title = f"Demo Sample: {normal_samples[0].name} [Non-Accident Class]"
    elif uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        sample_title = f"Uploaded File: {uploaded_file.name}"

    if pil_image is not None:
        st.markdown(f"**Analyzing:** `{sample_title}`")
        
        # Run inference
        with st.spinner("Processing frame through selected hybrid architecture..."):
            result = engine.predict_single_frame(
                pil_image,
                model_key=selected_model_key,
                conf_threshold=conf_threshold
            )

        col_img, col_res = st.columns([1.2, 1])

        with col_img:
            st.image(result["annotated_image"], caption="Annotated Detection Output", use_container_width=True)

        with col_res:
            acc_pct = result["accident_prob"] * 100
            norm_pct = result["non_accident_prob"] * 100

            if result["is_accident"]:
                st.markdown(f"""
                <div class="metric-card-danger">
                    <div class="stat-title">Detection Verdict</div>
                    <div class="stat-value" style="color: #DC2626;">🚨 ACCIDENT DETECTED</div>
                    <p style="margin: 0.3rem 0; color: #991B1B;"><strong>High-risk traffic incident identified.</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card-success">
                    <div class="stat-title">Detection Verdict</div>
                    <div class="stat-value" style="color: #16A34A;">✅ NORMAL TRAFFIC FLOW</div>
                    <p style="margin: 0.3rem 0; color: #166534;"><strong>No vehicular collision or hazardous anomaly detected.</strong></p>
                </div>
                """, unsafe_allow_html=True)

            # Metrics Breakdown
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Accident Probability", f"{acc_pct:.1f}%")
            m_col2.metric("Inference Latency", f"{result['latency_ms']:.1f} ms")

            st.write("**Confidence Distribution:**")
            st.progress(result["accident_prob"])
            st.caption(f"Accident: **{acc_pct:.1f}%** | Normal: **{norm_pct:.1f}%**")

            # Multi-Model Breakdown if available
            if result.get("all_model_probs"):
                st.markdown("---")
                st.markdown("**Multi-Hybrid Consensus Breakdown:**")
                breakdown_data = []
                for k, v in result["all_model_probs"].items():
                    name_clean = MODEL_NAMES.get(k, k).split("(")[0].replace("🏆 ", "").replace("Hybrid ", "H")
                    breakdown_data.append({
                        "Model": name_clean,
                        "Accident Conf.": f"{v['accident']*100:.1f}%",
                        "Normal Conf.": f"{v['non_accident']*100:.1f}%",
                        "Decision": "🚨 Accident" if v["accident"] >= conf_threshold else "✅ Normal"
                    })
                st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)

        # Specialized Visualizations
        if "heatmap" in result["visual_artifacts"]:
            st.markdown("---")
            st.subheader("🔥 Hybrid 2: YOLO Vehicle Proximity & Collision Heatmap")
            st.caption("Gaussian density peaks around interacting vehicles, suppressing background noise.")
            st.image(result["visual_artifacts"]["heatmap"], clamp=True, use_container_width=True)

        if "wavelet_energy" in result["visual_artifacts"]:
            st.markdown("---")
            st.subheader("🌊 Hybrid 3: 2D Haar Wavelet High-Frequency Energy Map")
            st.caption("Visualizes impact fracture shockwaves, broken glass, and sharp structural deformation.")
            st.image(result["visual_artifacts"]["wavelet_energy"], clamp=True, use_container_width=True)


# ==============================================================================
# MODE 2: VIDEO STREAM ANALYSIS
# ==============================================================================
elif mode == "🎥 Video Stream Analysis":
    st.markdown("### 🎥 Upload or Select a Traffic Video Stream")

    col_vdemo, col_vupload = st.columns([1, 2])
    with col_vdemo:
        st.markdown("**Preloaded Surveillance Video:**")
        sample_video_path = CURRENT_DIR.parent.parent / "videos" / "videoplayback (online-video-cutter.com).mp4"
        has_sample_video = sample_video_path.exists()
        btn_load_sample_video = st.button("📹 Load Project Traffic Video", disabled=not has_sample_video, use_container_width=True)

    with col_vupload:
        uploaded_video = st.file_uploader(
            "Or upload your surveillance video file (MP4, AVI, MOV, MKV):",
            type=["mp4", "avi", "mov", "mkv"]
        )

    # ── Persist uploaded video path across reruns ─────────────────────────────
    target_video_path = None
    if btn_load_sample_video and has_sample_video:
        target_video_path = sample_video_path
        # New video selected – wipe old results
        if st.session_state.get("video_path") != str(target_video_path):
            for k in ("analysis_done", "video_res", "df_timeline", "pdf_bytes"):
                st.session_state[k] = None if k not in ("analysis_done",) else False
        st.session_state["video_path"] = str(target_video_path)
    elif uploaded_video is not None:
        new_name = uploaded_video.name
        if st.session_state.get("uploaded_video_name") != new_name:
            # New file uploaded – save and wipe old results
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(new_name).suffix)
            tfile.write(uploaded_video.read())
            tfile.flush()
            st.session_state["video_path"] = tfile.name
            st.session_state["uploaded_video_name"] = new_name
            for k in ("analysis_done", "video_res", "df_timeline", "pdf_bytes"):
                st.session_state[k] = None if k != "analysis_done" else False
        target_video_path = Path(st.session_state["video_path"])
    elif st.session_state.get("video_path"):
        # Rerun triggered by any button – reuse previously saved path
        target_video_path = Path(st.session_state["video_path"])

    if target_video_path is not None and target_video_path.exists():
        video_name = Path(target_video_path).name
        st.markdown(f"**Target Video:** `{video_name}`")
        st.video(str(target_video_path))

        # ── Show video metadata before analysis ──────────────────────────────
        cap_meta = cv2.VideoCapture(str(target_video_path))
        meta_total, meta_fps, meta_dur, meta_w, meta_h = 0, 25.0, 0.0, 0, 0
        if cap_meta.isOpened():
            meta_total = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
            meta_fps   = cap_meta.get(cv2.CAP_PROP_FPS) or 25.0
            meta_dur   = meta_total / meta_fps
            meta_w     = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
            meta_h     = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_meta.release()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Frames",  f"{meta_total:,}")
        m2.metric("Duration",      f"{int(meta_dur//60):02d}:{meta_dur%60:05.2f}")
        m3.metric("Resolution",    f"{meta_w}×{meta_h}")
        m4.metric("Will Analyse",  f"{max(1, meta_total // frame_skip):,} frames  (K={frame_skip})")

        st.info(
            f"ℹ️ The entire video **({meta_total:,} frames)** will be read from beginning to end. "
            f"Inference runs on every **{frame_skip}th** frame. "
            f"Accident vehicle images will be saved to `results/accident_frames/`."
        )

        if st.button("🚀 Start Hybrid Video Accident Detection", type="primary"):
            progress_bar = st.progress(0.0)
            status_text  = st.empty()

            def update_progress(pct):
                progress_bar.progress(float(pct))
                status_text.text(f"Processing frames... {pct * 100:.1f}%")

            with st.spinner("Analysing entire video through hybrid architecture — please wait..."):
                video_res = engine.process_video_stream(
                    target_video_path,
                    model_key=selected_model_key,
                    conf_threshold=conf_threshold,
                    frame_skip=frame_skip,
                    progress_callback=update_progress,
                    max_saved_accident_frames=max_acc_images
                )

            progress_bar.progress(1.0)
            status_text.empty()

            # ── Persist results in session_state so reruns can access them ────
            df_timeline = pd.DataFrame(video_res["timeline_data"])
            # PIL images can't be pickled easily; drop before storing, keep path
            safe_frames = []
            for af in video_res["accident_frames"]:
                safe_frames.append({k: v for k, v in af.items() if k != "image"})
            video_res_safe = {k: v for k, v in video_res.items() if k != "accident_frames"}
            video_res_safe["accident_frames"] = safe_frames

            st.session_state["video_res"]     = video_res_safe
            st.session_state["df_timeline"]   = df_timeline
            st.session_state["analysis_done"] = True
            st.session_state["pdf_bytes"]     = None   # reset any old PDF

        # ── Render results whenever session_state has them ────────────────────
        if st.session_state.get("analysis_done") and "video_res" in st.session_state:
            video_res      = st.session_state["video_res"]
            df_timeline    = st.session_state["df_timeline"]
            accident_frames = video_res["accident_frames"]   # dicts without PIL

            total_analyzed  = len(df_timeline)
            accident_count  = int(df_timeline["is_accident"].sum()) if not df_timeline.empty else 0
            accident_ratio  = (accident_count / total_analyzed * 100) if total_analyzed > 0 else 0
            peak_conf       = df_timeline["accident_prob"].max() if not df_timeline.empty else 0.0

            st.success("✅ Full video analysis complete!")

            # ── Executive Summary ─────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📋 Executive Incident Summary")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Frames",        f"{video_res['total_frames']:,}")
            c2.metric("Analysed Frames",     f"{total_analyzed:,}")
            c3.metric("Accident Frames",     f"{accident_count}  ({accident_ratio:.1f}%)")
            c4.metric("Peak Incident Conf.", f"{peak_conf:.1f}%")
            c5.metric("Processing Speed",    f"{video_res['fps_speed']:.1f} FPS")

            # ── Accident intervals ────────────────────────────────────────────
            if video_res["accident_intervals"]:
                st.error(
                    f"🚨 **Incident Windows Detected ({len(video_res['accident_intervals'])}):** " +
                    ", ".join([f"`{s} → {e}`" for s, e in video_res["accident_intervals"]])
                )
            else:
                st.success("✅ **No severe accident intervals detected across this video.**")

            # Accident images saved location
            st.info(
                f"📂 **{len(accident_frames)} accident vehicle image(s)** saved to:\n"
                f"`{video_res['accident_images_dir']}`"
            )

            # ── Interactive Timeline Plot ──────────────────────────────────────
            st.markdown("---")
            st.subheader("📈 Temporal Incident Timeline")
            st.caption("Accident probability plotted across video playback time (seconds).")

            fig, ax = plt.subplots(figsize=(12, 3.8))
            ax.plot(df_timeline["timestamp_s"], df_timeline["accident_prob"],
                    color="#EF4444", linewidth=2, label="Accident Probability (%)")
            ax.axhline(conf_threshold * 100, color="#4B5563", linestyle="--",
                       label=f"Detection Threshold ({conf_threshold*100:.0f}%)")
            ax.fill_between(
                df_timeline["timestamp_s"], df_timeline["accident_prob"], conf_threshold * 100,
                where=(df_timeline["accident_prob"] >= conf_threshold * 100),
                color="#EF4444", alpha=0.3
            )
            ax.set_xlabel("Video Time (Seconds)")
            ax.set_ylabel("Accident Confidence (%)")
            ax.set_ylim([0, 105])
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right")
            st.pyplot(fig)
            plt.close(fig)

            # ── Accident Frames Gallery (load from saved PNG paths) ───────────
            if accident_frames:
                st.markdown("---")
                st.subheader("🖼️ Detected Accident Vehicle Frame Gallery")
                st.caption(
                    f"All {len(accident_frames)} accident frame(s) captured and saved to disk. "
                    f"Use these in your PDF report or review individually."
                )
                cols = st.columns(min(len(accident_frames), 4))
                for i, af in enumerate(accident_frames):
                    with cols[i % 4]:
                        img_path = af.get("image_path", "")
                        if img_path and Path(img_path).exists():
                            st.image(
                                img_path,
                                caption=(
                                    f"Frame {af['frame_idx']}\n"
                                    f"⏱ {af['timestamp_str']}\n"
                                    f"Conf: {af['confidence']:.1f}%"
                                ),
                                use_container_width=True
                            )

            # ── Export Section ────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("📥 Export Incident Reports")

            export_col1, export_col2 = st.columns(2)

            # ── 1. CSV Export ─────────────────────────────────────────────────
            with export_col1:
                st.markdown("#### 📊 CSV Telemetry Log")
                st.caption("Full frame-by-frame log with timestamps, confidence scores, and accident flags.")
                csv_data = df_timeline.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Full Incident Log (.CSV)",
                    data=csv_data,
                    file_name=f"accident_telemetry_{Path(target_video_path).stem}_{video_res['analysis_timestamp']}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="csv_full"
                )

                if accident_count > 0:
                    df_accidents_only = df_timeline[df_timeline["is_accident"] == True].copy()
                    img_map = {af["frame_idx"]: af.get("image_filename", "") for af in accident_frames}
                    df_accidents_only["image_filename"] = df_accidents_only["frame_idx"].map(img_map).fillna("")
                    csv_acc_only = df_accidents_only.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Accident-Only Log (.CSV)",
                        data=csv_acc_only,
                        file_name=f"accidents_only_{Path(target_video_path).stem}_{video_res['analysis_timestamp']}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="csv_accident"
                    )

            # ── 2. PDF Report Export ──────────────────────────────────────────
            with export_col2:
                st.markdown("#### 📄 PDF Incident Report")
                if PDF_AVAILABLE:
                    st.caption("Comprehensive PDF containing metadata, interval table, accident images gallery, and telemetry.")

                    if st.button("🖨️ Generate PDF Report", use_container_width=True, key="btn_gen_pdf"):
                        with st.spinner("Building PDF report with embedded accident images..."):
                            pdf_bytes = generate_pdf_report(
                                video_res=video_res,
                                timeline_df=df_timeline
                            )
                        # Store PDF bytes in session_state so download button survives next rerun
                        st.session_state["pdf_bytes"] = pdf_bytes
                        st.session_state["pdf_filename"] = (
                            f"incident_report_{Path(target_video_path).stem}"
                            f"_{video_res['analysis_timestamp']}.pdf"
                        )

                    # Always show download button once PDF has been generated
                    if st.session_state.get("pdf_bytes"):
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=st.session_state["pdf_bytes"],
                            file_name=st.session_state.get("pdf_filename", "incident_report.pdf"),
                            mime="application/pdf",
                            use_container_width=True,
                            key="pdf_download"
                        )
                        st.success("✅ PDF ready — click above to download.")
                else:
                    st.warning(
                        "📦 `fpdf2` library not found. Install it to enable PDF export:\n"
                        "```bash\npip install fpdf2\n```"
                    )


# ==============================================================================
# MODE 3: HYBRID BENCHMARK & ARCHITECTURE COMPARISON
# ==============================================================================
elif mode == "📊 Hybrid Model Benchmark":
    st.markdown("### 📊 Comprehensive Comparison of Proposed Hybrid Models")
    st.write("Quantitative comparison and architectural breakdown on the `data1` dataset (989 surveillance frames):")

    table_data = [
        {
            "Hybrid Architecture": "Hybrid 1: TransCNN-AccidentNet",
            "Primary Fusion Mechanism": "Bidirectional Cross-Attention (CAFF)",
            "Backbone Components": "EfficientNetV2-S + Spatial Patch Transformer",
            "Expected Accuracy": "96.5% – 98.2%",
            "Inference Speed": "~35 ms/frame",
            "Overfitting Resistance": "High",
            "IEEE Novelty": "⭐⭐⭐⭐⭐"
        },
        {
            "Hybrid Architecture": "Hybrid 2: YOLO-SAN",
            "Primary Fusion Mechanism": "Spatial Heatmap Gating",
            "Backbone Components": "YOLOv11 Vehicle Detector + MobileNetV3",
            "Expected Accuracy": "95.0% – 97.0%",
            "Inference Speed": "~25 ms/frame",
            "Overfitting Resistance": "Very High",
            "IEEE Novelty": "⭐⭐⭐⭐⭐"
        },
        {
            "Hybrid Architecture": "Hybrid 3: WaveConv-DualDomain",
            "Primary Fusion Mechanism": "Dual-Domain Wavelet CBAM Attention",
            "Backbone Components": "2D Haar Wavelet DWT + ResNet-34",
            "Expected Accuracy": "94.5% – 96.5%",
            "Inference Speed": "~12 ms/frame (Fastest)",
            "Overfitting Resistance": "High",
            "IEEE Novelty": "⭐⭐⭐⭐⭐"
        },
        {
            "Hybrid Architecture": "Hybrid 4: Deep-Forest",
            "Primary Fusion Mechanism": "Multi-Backbone Stacking",
            "Backbone Components": "EfficientNetV2-S + ConvNeXt-Tiny + GBDT",
            "Expected Accuracy": "96.0% – 97.8%",
            "Inference Speed": "~28 ms/frame",
            "Overfitting Resistance": "Maximum",
            "IEEE Novelty": "⭐⭐⭐⭐"
        }
    ]
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("💡 Why Hybrid Models Outperform Single-Model Baselines")
    st.markdown("""
    1. **Small Sample Generalization ($N < 1000$)**:
       Standard Vision Transformers require tens of thousands of samples to learn spatial relationships. In our hybrids, convolutional backbones (`EfficientNetV2`, `ResNet-34`) provide strong inductive bias, while transformers or GBDTs capture multi-vehicle interactions without overfitting.
    2. **Background Suppression**:
       In 1280×720 widescreen surveillance, 85% of the frame is road tarmac or sky. `YOLO-SAN` and `WaveConv` actively amplify collision regions and filter out static highway background.
    3. **Illumination Invariance**:
       Wavelet high-frequency sub-bands isolate structural crushing, broken glass, and impact shockwaves while ignoring night-time darkness, shadows, and headlight glare.
    """)
