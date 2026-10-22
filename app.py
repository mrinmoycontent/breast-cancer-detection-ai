
import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoModelForImageClassification

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Breast Cancer Detection AI | BreastCheck",
    page_icon="🎗️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==================================================
# MODEL CONFIGURATION
# ==================================================
MODEL_ID = "ghostsas001/busi-breast-ultrasound-siglip"


@st.cache_resource(show_spinner="Loading the pretrained AI model...")
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


# ==================================================
# IMAGE PREPROCESSING
# ==================================================
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    ),
])


# ==================================================
# MAIN PAGE: BREAST CANCER DETECTION
# ==================================================
st.title("🎗️ BREAST CANCER DETECTION AI")

st.subheader(
    "BreastCheck AI | Ultrasound Image Classification"
)

st.markdown("""
Welcome to **BreastCheck AI**, an experimental AI/ML portfolio
project that demonstrates how deep learning can be used to
classify breast ultrasound images.

### What does this application do?

Upload an image to see the pretrained model's predicted category
and its scores for three model-defined classes:

- **Benign**
- **Malignant**
- **Normal**

### Technology used

Python · PyTorch · Hugging Face Transformers · Torchvision ·
Pillow · Streamlit
""")

st.info("""
**Project by Mrinmoy Choudhury**

Aspiring AI/ML Engineer | Data Science Enthusiast
""")


# ==================================================
# IMPORTANT MEDICAL DISCLAIMER
# ==================================================
st.warning("""
### Important Medical Disclaimer

This application is an educational and experimental prototype.
It is **not a clinically validated diagnostic tool**.

The model may produce incorrect predictions. Its scores are not
verified or calibrated diagnostic probabilities. This application
cannot confirm or rule out breast cancer and must not be used to
make medical, treatment, or screening decisions.

Always consult a qualified healthcare professional for medical
evaluation. Do not upload identifiable patient images.
""")


# ==================================================
# IMAGE UPLOAD
# ==================================================
st.divider()

st.header("📤 Upload a Breast Ultrasound Image")

st.write(
    "Select a PNG, JPG, JPEG, or WEBP image to run the "
    "experimental classification model."
)

uploaded = st.file_uploader(
    "Choose an image file",
    type=["png", "jpg", "jpeg", "webp"],
    help="Use only images you have permission to use."
)


# ==================================================
# IMAGE CLASSIFICATION
# ==================================================
if uploaded is not None:

    try:
        image = Image.open(uploaded).convert("RGB")

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption="Image submitted for experimental classification",
            use_container_width=True
        )

        with st.spinner(
            "Loading the model and processing the image..."
        ):
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

        # Obtain class labels from the model configuration.
        id2label = model.config.id2label

        results = []

        for index, score in enumerate(scores):

            label = id2label.get(
                index,
                id2label.get(
                    str(index),
                    f"Class {index}"
                )
            )

            results.append({
                "label": str(label),
                "score": float(score) * 100
            })

        # Display highest-scoring category first.
        results.sort(
            key=lambda result: result["score"],
            reverse=True
        )

        # ------------------------------------------
        # MODEL OUTPUT REPORT
        # ------------------------------------------
        st.divider()

        st.header("📊 Model Output Report")

        st.metric(
            label="Predicted Class",
            value=results[0]["label"].title()
        )

        st.caption(
            "This is the category with the highest model score, "
            "not a confirmed medical finding."
        )

        st.subheader("Scores for All Model Classes")

        for result in results:

            st.write(
                f"**{result['label'].title()}** — "
                f"{result['score']:.2f}%"
            )

            st.progress(
                min(max(result["score"] / 100, 0.0), 1.0)
            )

        # ------------------------------------------
        # RESULT LIMITATIONS
        # ------------------------------------------
        st.warning("""
        **How to interpret these results**

        These percentages are raw softmax scores produced by the
        model. They are not verified or calibrated probabilities
        of having or not having cancer.

        The application does not verify that an uploaded file is
        a breast ultrasound image. Predictions may be incorrect
        and must not be used for diagnosis or treatment decisions.
        """)

    except Exception as exc:

        st.error(
            f"Unable to process the image: "
            f"{type(exc).__name__}: {exc}"
        )

        st.caption(
            "If this problem persists, check the Streamlit "
            "deployment logs and available memory."
        )

else:

    st.info(
        "No image uploaded yet. Upload an image above to view "
        "the model's predicted class and class scores."
    )


# ==================================================
# FOOTER
# ==================================================
st.divider()

st.markdown("""
### About the Developer

**Mrinmoy Choudhury**  
Aspiring AI/ML Engineer | Data Science Enthusiast

This project demonstrates pretrained model integration,
image preprocessing, deep learning inference, and deployment
of an interactive AI application.
""")

st.markdown(
    "[View the source code on GitHub](https://github.com/"
    "mrinmoycontent/breast-cancer-detection-ai)"
)

st.caption(
    "BreastCheck AI | Educational portfolio project | "
    "Not intended for clinical use."
)
