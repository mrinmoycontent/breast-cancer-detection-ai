
import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoModelForImageClassification

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Detection AI | BreastCheck",
    page_icon="🎗️",
    layout="centered"
)

# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------
MODEL_ID = "ghostsas001/busi-breast-ultrasound-siglip"

@st.cache_resource(show_spinner="Loading the pretrained model...")
def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID
    )

    model.to(device)
    model.eval()

    return model, device


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    ),
])


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------
st.title("🎗️ BreastCheck AI")

st.subheader(
    "AI-Based Breast Cancer Detection Using Ultrasound Images"
)

st.markdown("""
BreastCheck AI is an experimental deep learning application
that classifies uploaded images into three model-defined
categories:

- **Benign**
- **Malignant**
- **Normal**

The application uses a pretrained image-classification model
to generate a predicted category and display scores for all
available model classes.

**Technology stack:** Python, PyTorch, Transformers,
Torchvision, Pillow and Streamlit.
""")

st.warning("""
**Important medical disclaimer**

This is an educational AI/ML prototype, not a clinically
validated diagnostic tool. Predictions cannot confirm or
rule out breast cancer and must not guide medical decisions.
""")


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
st.markdown("### Upload an Image")

st.caption(
    "Upload a PNG, JPG, JPEG or WEBP image. "
    "Use only de-identified images you have permission to use."
)

uploaded = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "webp"]
)


# --------------------------------------------------
# IMAGE CLASSIFICATION
# --------------------------------------------------
if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

        with st.spinner("Running model inference..."):
            model, device = load_model()

            input_tensor = transform(image)
            input_tensor = input_tensor.unsqueeze(0).to(device)

            with torch.inference_mode():
                outputs = model(pixel_values=input_tensor)
                logits = outputs.logits

                scores = torch.softmax(
                    logits.float(),
                    dim=-1
                )[0].cpu().tolist()

        # Read class names from the pretrained model.
        id2label = model.config.id2label

        results = []

        for index, score in enumerate(scores):
            label = id2label.get(
                index,
                id2label.get(str(index), f"Class {index}")
            )

            results.append({
                "label": str(label),
                "score": float(score) * 100
            })

        results.sort(
            key=lambda result: result["score"],
            reverse=True
        )

        # Display results.
        st.divider()
        st.markdown("### Model Output Report")

        st.metric(
            label="Predicted Class",
            value=results[0]["label"].title()
        )

        st.markdown("#### Scores for All Model Classes")

        for result in results:
            st.write(
                f"**{result['label'].title()}** — "
                f"{result['score']:.2f}%"
            )

            st.progress(
                min(max(result["score"] / 100, 0.0), 1.0)
            )

        st.info("""
        These values are raw model scores after softmax, not
        verified or calibrated diagnostic probabilities.

        The application does not independently verify image type.
        Results may be incorrect and must not be used to diagnose
        or rule out cancer.
        """)

    except Exception as exc:
        st.error(
            f"Unable to process the image: "
            f"{type(exc).__name__}: {exc}"
        )

        st.caption(
            "If the model fails to load, check the Streamlit "
            "deployment logs and available memory."
        )

else:
    st.info(
        "Upload an image to view the model's predicted class "
        "and scores for all classes."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Developed by Mrinmoy Choudhury | "
    "Aspiring AI/ML Engineer | Data Science Enthusiast"
)

st.caption(
    "Educational portfolio project • Not for clinical use"
)

st.markdown(
    "[View Source Code on GitHub]"
    "(https://github.com/mrinmoycontent/breast-cancer-detection-ai)"
)
