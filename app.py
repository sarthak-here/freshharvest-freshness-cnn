from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference import DEFAULT_CHECKPOINT, load_transfer_model, predict


st.set_page_config(
    page_title="FreshHarvest Inspector",
    page_icon="FH",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #133b2b 0%, #081711 42%, #050b08 100%);
        color: #edf7f0;
    }
    .block-container {max-width: 880px; padding-top: 2.2rem;}
    .hero {
        padding: 1.8rem 2rem;
        border: 1px solid rgba(123, 231, 168, .22);
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(24, 83, 57, .55), rgba(7, 25, 18, .78));
        box-shadow: 0 24px 70px rgba(0, 0, 0, .28);
        margin-bottom: 1.5rem;
    }
    .eyebrow {color: #7be7a8; letter-spacing: .12em; font-weight: 700; font-size: .78rem;}
    .hero h1 {font-size: 2.5rem; margin: .35rem 0 .45rem; color: #f5fff8;}
    .hero p {color: #bed3c5; margin: 0; font-size: 1.02rem;}
    .result-card {
        border-radius: 20px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
        background: rgba(12, 40, 28, .86);
        border: 1px solid rgba(123, 231, 168, .25);
    }
    .result-label {color: #a9c7b3; font-size: .84rem; text-transform: uppercase; letter-spacing: .08em;}
    .result-title {font-size: 2rem; font-weight: 750; color: #ffffff; margin: .15rem 0;}
    .fresh {color: #73e69e; font-weight: 700;}
    .spoiled {color: #ff8f7d; font-weight: 700;}
    div[data-testid="stFileUploader"] {
        border: 1px dashed rgba(123, 231, 168, .42);
        border-radius: 18px;
        padding: .5rem;
        background: rgba(8, 28, 19, .5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the inspection model...")
def get_model():
    return load_transfer_model(DEFAULT_CHECKPOINT)


st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">AI-ASSISTED QUALITY INSPECTION</div>
      <h1>FreshHarvest Inspector</h1>
      <p>Upload a produce image to identify its type and assess whether it appears fresh or spoiled.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

if not Path(DEFAULT_CHECKPOINT).exists():
    st.error("The trained model checkpoint is missing. Expected: artifacts/transfer/best_resnet50_model.pt")
    st.stop()

try:
    model, classes = get_model()
except Exception as error:
    st.error(f"The model could not be loaded: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Drag and drop a fruit or vegetable image",
    type=["jpg", "jpeg", "png", "webp"],
    help="For best results, use one clearly visible produce item under good lighting.",
)

if uploaded_file is None:
    st.info("Supported produce: banana, lemon, lulo, mango, orange, strawberry, tamarillo and tomato.")
else:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("This file could not be read as an image. Please upload a JPG, PNG or WebP file.")
        st.stop()

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.image(image, caption="Uploaded inspection image", use_container_width=True)

    with st.spinner("Inspecting produce..."):
        predictions = predict(image, model, classes, top_k=3)
    best = predictions[0]
    status_class = "fresh" if best["freshness"] == "Fresh" else "spoiled"

    with right:
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-label">Inspection result</div>
              <div class="result-title">{best['produce']}</div>
              <div class="{status_class}">{best['freshness']}</div>
              <div style="color:#bed3c5;margin-top:.55rem;">Confidence: {best['confidence']:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("Top model predictions")
        for item in predictions:
            label = f"{item['produce']} - {item['freshness']}"
            st.progress(item["confidence"], text=f"{label}: {item['confidence']:.1%}")

    if best["confidence"] < 0.70:
        st.warning("Confidence is below 70%. Send this item for manual inspection.")
    else:
        st.caption("Prototype decision support only. Production use requires validation with warehouse-camera images.")

