# FedDG-PneuNet Streamlit Deployment Guide

This folder is the PHP-free Streamlit version of the app. It keeps the same core features:

- Chest X-ray upload
- FedDG-GATNet graph inference
- Grad-CAM heatmap
- PDF report download
- Prediction JSON save
- Print report button
- Research and awareness pages

## Required files

Keep these files in the root of the GitHub repository:

```text
app.py
requirements.txt
packages.txt
.gitattributes
model/
uploads/.gitkeep
reports/.gitkeep
```

The `model/` folder must contain:

```text
feature_extractor.h5
feddg_gatnet_model.h5
reference_features.npy
reference_labels.npy
inference.py
__init__.py
```

## GitHub upload commands

Open Command Prompt or PowerShell:

```bash
cd C:\xampp\htdocs\pnemunia\streamlit-app
git init
git lfs install
git add .gitattributes
git add .
git commit -m "Deploy FedDG-PneuNet Streamlit app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

Use Git LFS because `.h5` and `.npy` files are large model artifacts.

## Streamlit Community Cloud steps

1. Go to `https://streamlit.io/cloud`.
2. Sign in with GitHub.
3. Click `Create app`.
4. Select your GitHub repository.
5. Branch: `main`.
6. Main file path: `app.py`.
7. Open `Advanced settings`.
8. Select a Python version compatible with TensorFlow, preferably Python `3.12` or `3.13`.
9. Click `Deploy`.

## Important notes

- Streamlit reads Python packages from `requirements.txt`.
- Linux system packages are installed from `packages.txt`.
- Cloud-generated files inside `uploads/` and `reports/` are temporary runtime files. Users can download reports immediately after prediction.
- If deployment fails on TensorFlow, try changing `tensorflow-cpu==2.20.0` in `requirements.txt` to `tensorflow==2.20.0`, then commit and push again.
