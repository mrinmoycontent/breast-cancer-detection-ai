import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoModelForImageClassification

st.set_page_config(page_title="BreastCheck Research Demo", page_icon="🩺", layout="centered")

MODEL_ID = "ghostsas001/busi-breast-ultrasound-siglip"

@st.cache_resource(show_spinner="Loading the BreastCheck model (first start may take several minutes)...")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    return model, device

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

st.title("BreastCheck")
st.subheader("Experimental breast-ultrasound image-classification demo")
st.warning(
    "Educational prototype only — not a medical device. Image type is not independently "
    "verified. Do not upload identifiable patient images. Do not use outputs for medical decisions."
)

uploaded = st.file_uploader("Upload a de-identified image", type=["png", "jpg", "jpeg", "webp"])

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Analyzing image..."):
            model, device = load_model()
            x = transform(image).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits = model(pixel_values=x).logits
                scores = torch.softmax(logits.float(), dim=-1)[0].cpu().tolist()

        labels = model.config.id2label
        ranked = sorted(
            [
                (str(labels.get(i, labels.get(str(i), f"Class {i}"))), float(score) * 100)
                for i, score in enumerate(scores)
            ],
            key=lambda item: item[1],
            reverse=True,
        )

        st.markdown("### Model output report")
        st.metric("Predicted class", ranked[0][0].title())
        st.markdown("**Scores for all model classes**")
        for label, score in ranked:
            st.write(f"**{label.title()}** — {score:.2f}%")
            st.progress(min(max(score / 100, 0.0), 1.0))

        st.info(
            "These are raw model scores after softmax, not verified or calibrated diagnostic "
            "probabilities. The app does not verify image type, and this prototype cannot "
            "diagnose or rule out cancer."
        )
    except Exception as exc:
        st.error(f"Could not analyze this image: {type(exc).__name__}: {exc}")
        st.caption("If the model fails to load, check the deployment logs and available memory.")
else:
    st.caption("Upload an image to view the model's predicted class and all class scores.")

st.divider()
st.caption("Portfolio prototype • Research and education only • Not for clinical use")
