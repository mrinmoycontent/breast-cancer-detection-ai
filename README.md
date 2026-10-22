# 🩺 BreastCheck AI — Breast Ultrasound Image Classification

### An AI/ML Portfolio Project by Mrinmoy Choudhury

**Aspiring AI/ML Engineer | Data Science Enthusiast**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://breastcheck-ultrasound-demo.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Model-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)

## 🚀 Project Overview

**BreastCheck AI** is a web-based machine learning prototype that classifies breast ultrasound images into three model-defined categories: **Benign, Malignant, and Normal**.

The project demonstrates how a pretrained image-classification model can be integrated into an interactive web application, allowing users to upload an image and view the predicted class alongside the model's scores for all three categories.

The application was developed as a hands-on project to explore deep learning, image preprocessing, model inference, and the deployment of AI applications using Streamlit.

🔗 **Try the live application:**  
https://breastcheck-ultrasound-demo.streamlit.app/

## ✨ Key Features

- **Image upload:** Submit an image through the web interface.
- **Deep learning inference:** Run an image through a pretrained classification model.
- **Three-class output:** Display the predicted category: Benign, Malignant, or Normal.
- **Class score reporting:** View the model's softmax scores for each class.
- **Interactive interface:** Access the application through a browser without installing the model locally.
- **Cloud deployment:** Run the application using Streamlit Community Cloud.
- **Responsible AI messaging:** Display limitations and warnings against using predictions for medical decisions.

## 🧠 How It Works

The application follows a straightforward image-classification pipeline:

1. **Image input:** The user uploads an image.
2. **Preprocessing:** The image is resized to 224 × 224 pixels, converted into a tensor, and normalized.
3. **Model inference:** The processed image is passed to the pretrained image-classification model.
4. **Score calculation:** Softmax is applied to the model's output scores.
5. **Result presentation:** The application displays the predicted class and the scores for all three classes.

### Model Information

| Component | Details |
|---|---|
| Model type | Pretrained image-classification model |
| Model repository | `ghostsas001/busi-breast-ultrasound-siglip` |
| Model loading | Hugging Face Transformers |
| Input processing | Resize, tensor conversion, normalization |
| Input size | 224 × 224 pixels |
| Output classes | Benign, Malignant, Normal |
| Inference framework | PyTorch |
| Web interface | Streamlit |

*Note: The application uses a pretrained model. This project should not be described as training a new model from scratch or as a clinically validated diagnostic system.*

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and inference |
| PyTorch | Tensor operations and model inference |
| Hugging Face Transformers | Loading the pretrained model |
| Torchvision | Image transformations and preprocessing |
| Pillow (PIL) | Image handling |
| Streamlit | Interactive web application |
| Git and GitHub | Version control and project hosting |
| Streamlit Community Cloud | Application deployment |

## 🖥️ Application Preview

### Live Demo

Open the deployed application to explore the image upload workflow and prediction report.

**[Launch BreastCheck AI →](https://breastcheck-ultrasound-demo.streamlit.app/)**

The interface presents the predicted category, scores for each class, key features, and a disclaimer explaining the limitations of the prototype.

## ⚙️ Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/mrinmoycontent/breastcheck-demo.git
cd breastcheck-demo
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run app.py
```

Streamlit will provide a local address, typically:

`http://localhost:8501`

The pretrained model may need to download the first time the application runs. An internet connection is therefore required for initial model loading unless the model is already cached locally.

## 🔒 Restrictions and Responsible AI

BreastCheck AI is an **educational and experimental portfolio project**. It is not a certified medical device and has not been established as clinically safe or effective.

- Model scores are raw softmax outputs and are not verified or calibrated diagnostic probabilities.
- A high score must not be interpreted as the probability that a patient does or does not have cancer.
- The application does not independently verify whether an uploaded image is a breast ultrasound image.
- Model predictions may be incorrect, including on unfamiliar images or images that differ from the model's training data.
- Clinical accuracy, sensitivity, specificity, calibration, and generalization have not been independently established through this project.
- The application must not be used to diagnose, rule out, or guide treatment for any medical condition.
- Medical evaluation and decisions must be made by qualified healthcare professionals.

## 🔮 Future Improvements

Potential development areas include:

- Evaluate the model using a properly separated, labelled test dataset.
- Measure classification performance using a confusion matrix, precision, recall, F1-score, and per-class sensitivity.
- Investigate probability calibration and model confidence reliability.
- Add automated tests for image preprocessing and inference.
- Improve error handling and accessibility in the user interface.
- Document dataset provenance, licensing, and model limitations.

These are proposed improvements, not features or results already demonstrated by the current application.

## 👨‍💻 Author

**Mrinmoy Choudhury**  
*Aspiring AI/ML Engineer | Data Science Enthusiast*

I am building practical projects to strengthen my understanding of machine learning, data science, deep learning, and AI application development. BreastCheck AI represents my work in connecting a pretrained model with an interactive application and deploying it for demonstration.

### Project Links

- **Live Application:** https://breastcheck-ultrasound-demo.streamlit.app/
- **GitHub Repository:** https://github.com/mrinmoycontent/breastcheck-demo

---

*Built as an AI/ML learning project to explore image classification, model inference, and web application deployment. Not intended for clinical use.*
