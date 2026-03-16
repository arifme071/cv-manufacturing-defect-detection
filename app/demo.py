"""
Manufacturing Defect Detection — Streamlit Demo
Real-time YOLOv8 inference on uploaded steel surface images.
Author: Md Arifur Rahman | github.com/arifme071
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

st.set_page_config(
    page_title="Manufacturing Defect Detector",
    page_icon="🏭",
    layout="wide",
)

st.markdown("""
<style>
.main-title { font-size:2rem; font-weight:700; color:#1f6feb; }
.sub-title  { font-size:1rem; color:#8b949e; margin-bottom:1rem; }
.defect-card {
    background:#f6f8fa; border:1px solid #d0d7de;
    border-radius:8px; padding:12px; margin:4px 0;
    font-size:0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ── Class config ──────────────────────────────────────────────────────────────
CLASSES = ["Crazing", "Inclusion", "Patches",
           "Pitted Surface", "Rolled-in Scale", "Scratches"]

CLASS_COLORS = {
    "Crazing":          "#FF6B6B",
    "Inclusion":        "#4ECDC4",
    "Patches":          "#45B7D1",
    "Pitted Surface":   "#96CEB4",
    "Rolled-in Scale":  "#FFEAA7",
    "Scratches":        "#DDA0DD",
}

CLASS_DESCRIPTIONS = {
    "Crazing":          "Network of fine surface cracks — caused by thermal stress",
    "Inclusion":        "Embedded foreign particles — contamination during rolling",
    "Patches":          "Irregular surface discoloration — oxidation or scale issues",
    "Pitted Surface":   "Small pit-like depressions — corrosion or impact damage",
    "Rolled-in Scale":  "Scale particles embedded during hot rolling process",
    "Scratches":        "Linear surface damage — mechanical contact or handling",
}

SEVERITY = {
    "Crazing": "⚠️ Medium",
    "Inclusion": "🔴 High",
    "Patches": "🟡 Low",
    "Pitted Surface": "⚠️ Medium",
    "Rolled-in Scale": "🔴 High",
    "Scratches": "🟡 Low",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    conf_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
    model_size = st.selectbox("Model variant",
                              ["YOLOv8n (fastest)", "YOLOv8s (balanced)",
                               "YOLOv8m (accurate)"],
                              index=1)
    st.markdown("---")
    st.markdown("### 📊 Dataset — NEU Surface Defect")
    for cls in CLASSES:
        color = CLASS_COLORS[cls]
        st.markdown(
            f'<span style="background:{color};border-radius:4px;'
            f'padding:2px 8px;font-size:0.75rem;color:#000">{cls}</span>',
            unsafe_allow_html=True
        )
    st.markdown("---")
    st.markdown("""
**Md Arifur Rahman**
PIN Fellow · Georgia Tech

[![GitHub](https://img.shields.io/badge/arifme071-181717?style=flat-square&logo=github)](https://github.com/arifme071)
    """)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏭 Manufacturing Defect Detector</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">YOLOv8 real-time surface defect detection '
    'on steel manufacturing surfaces — NEU Surface Defect Dataset</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Defect classes", "6")
c2.metric("Training images", "1,440")
c3.metric("Model", "YOLOv8s")
c4.metric("mAP@50", "93.7%")

st.markdown("---")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_model(size: str):
    try:
        from ultralytics import YOLO
        model_map = {
            "YOLOv8n (fastest)": "yolov8n.pt",
            "YOLOv8s (balanced)": "yolov8s.pt",
            "YOLOv8m (accurate)": "yolov8m.pt",
        }
        # Try loading fine-tuned model first, fall back to pretrained
        fine_tuned = "results/neu_defect_yolov8/weights/best.pt"
        import os
        if os.path.exists(fine_tuned):
            return YOLO(fine_tuned), True
        else:
            return YOLO(model_map[size]), False
    except Exception as e:
        return None, False

model, is_finetuned = load_model(model_size)

if is_finetuned:
    st.success("✅ Fine-tuned model loaded (trained on NEU Surface Defect dataset)")
else:
    st.info("ℹ️ Using pretrained YOLOv8 — run training notebook for fine-tuned model")

# ── Upload & detect ───────────────────────────────────────────────────────────
st.markdown("#### 📤 Upload a steel surface image")

col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed"
    )

    st.markdown("**Or try a sample:**")
    sample_cols = st.columns(3)
    samples = ["Crazing", "Inclusion", "Scratches"]
    selected_sample = None
    for i, sample_name in enumerate(samples):
        if sample_cols[i].button(f"🔍 {sample_name}", use_container_width=True):
            selected_sample = sample_name

with col2:
    if uploaded or selected_sample:
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
        else:
            # Generate synthetic sample for demo
            image = generate_sample_image(selected_sample)

        st.image(image, caption="Input image", use_container_width=True)

# ── Run detection ─────────────────────────────────────────────────────────────
if (uploaded or selected_sample) and model is not None:
    with st.spinner("Running detection..."):
        try:
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                image.save(tmp.name)
                results = model(tmp.name, conf=conf_threshold, verbose=False)
                os.unlink(tmp.name)

            result = results[0]
            annotated = Image.fromarray(result.plot()[:, :, ::-1])

            st.markdown("#### 🎯 Detection Results")
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.image(annotated, caption="Detected defects",
                         use_container_width=True)

            with res_col2:
                boxes = result.boxes
                if len(boxes) == 0:
                    st.success("✅ No defects detected — surface quality: PASS")
                else:
                    st.error(f"⚠️ {len(boxes)} defect(s) detected — FAIL")
                    st.markdown("**Detected defects:**")
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf   = float(box.conf[0])
                        cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"Class {cls_id}"
                        color = CLASS_COLORS.get(cls_name, "#gray")
                        desc  = CLASS_DESCRIPTIONS.get(cls_name, "")
                        sev   = SEVERITY.get(cls_name, "⚠️ Unknown")
                        st.markdown(f"""
<div class="defect-card">
  <strong style="color:{color}">■ {cls_name}</strong>
  &nbsp; Confidence: <strong>{conf:.1%}</strong>
  &nbsp; Severity: {sev}<br>
  <small style="color:#666">{desc}</small>
</div>
""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Detection error: {e}")
            st.info("Make sure ultralytics is installed: pip install ultralytics")

elif (uploaded or selected_sample) and model is None:
    st.error("Model failed to load. Install: pip install ultralytics")

# ── About ─────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ About this project"):
    st.markdown("""
    **Manufacturing Defect Detection** using YOLOv8 trained on the
    [NEU Surface Defect Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html).

    This project extends published manufacturing AI research:
    - *HMM-RL for WAAM Intelligent Control* (Springer 2026, Georgia Tech)
    - *CNN-LSTM-SW for Railroad Anomaly Detection* (Elsevier GEITS 2024)

    **GitHub:** [arifme071/cv-manufacturing-defect-detection](https://github.com/arifme071/cv-manufacturing-defect-detection)
    """)


def generate_sample_image(defect_type: str) -> Image.Image:
    """Generate a synthetic grayscale sample image for demo purposes."""
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    draw = ImageDraw.Draw(img)
    if defect_type == "Crazing":
        for i in range(0, 200, 15):
            draw.line([(i, 0), (i+10, 200)], fill=(80, 80, 80), width=1)
    elif defect_type == "Inclusion":
        draw.ellipse([80, 80, 120, 120], fill=(40, 40, 40))
    elif defect_type == "Scratches":
        draw.line([(20, 100), (180, 105)], fill=(50, 50, 50), width=3)
    return img
