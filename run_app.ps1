# FedDG-PneuNet Streamlit launcher (uses local .venv)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$BasePython = "C:\Users\ratul\AppData\Local\Programs\Python\Python311\python.exe"
if (-not (Test-Path $BasePython)) {
    $BasePython = "python"
}

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment..."
    & $BasePython -m venv .venv
}

$Python = ".\.venv\Scripts\python.exe"

Write-Host "Installing dependencies (TensorFlow + Keras 3 + NumPy 1.x)..."
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt

Write-Host "Starting Streamlit..."
& $Python -m streamlit run app.py
