# BreastCheck — Streamlit Research Demo

A small portfolio demo that loads one pretrained image-classification model and displays its predicted class plus the scores for all model classes.

## Run locally

Python 3.10 or 3.11 is recommended for compatibility with PyTorch and Transformers.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the model weights from Hugging Face and may take several minutes. The model is cached once per Streamlit process using `st.cache_resource`.

## Deploy later on Streamlit Community Cloud

Upload `app.py`, `requirements.txt`, and `README.md` to a GitHub repository, then select that repository and `app.py` in Streamlit Community Cloud.

## Limitations

- Experimental portfolio prototype only; not a medical device.
- Image type is not independently verified.
- Model scores are not verified or calibrated diagnostic probabilities.
- No clinical validation is claimed.
- Do not upload identifiable patient images or use results for medical decisions.
