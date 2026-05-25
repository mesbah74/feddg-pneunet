# FedDG-PneuNet Streamlit App

Pure Python Streamlit version of FedDG-PneuNet for Streamlit Community Cloud. This version removes PHP and keeps the same AI workflow, medical dashboard UI, Grad-CAM output, PDF report download, JSON prediction save, and healthcare awareness pages.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Push this folder to GitHub and choose `app.py` as the main file on Streamlit Community Cloud.

Use Git LFS for `.h5` and `.npy` model files.

Detailed steps are in `DEPLOY_STREAMLIT.md`.
