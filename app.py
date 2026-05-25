from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable
from html import escape

import numpy as np
import streamlit as st
from PIL import Image

# Lazy imports keep the UI alive when TensorFlow/NumPy versions mismatch.
load_artifacts = None
run_prediction = None
INFERENCE_IMPORT_ERROR = ""
_cv2_module = None


def _get_cv2():
    global _cv2_module
    if _cv2_module is None:
        import cv2 as cv2_module

        _cv2_module = cv2_module
    return _cv2_module


def _ensure_inference_imported() -> None:
    global load_artifacts, run_prediction, INFERENCE_IMPORT_ERROR
    if load_artifacts is not None and run_prediction is not None:
        return
    if INFERENCE_IMPORT_ERROR:
        return
    try:
        from model.inference import load_artifacts as _load_artifacts
        from model.inference import run_prediction as _run_prediction

        load_artifacts = _load_artifacts
        run_prediction = _run_prediction
    except Exception as exc:
        load_artifacts = None
        run_prediction = None
        INFERENCE_IMPORT_ERROR = str(exc)

APP_NAME = "FedDG-PneuNet"
APP_ROOT = Path(__file__).resolve().parent
NAV_PAGES = ("Home", "Result", "Research", "Awareness")
MODEL_DIR = APP_ROOT / "model"
REQUIRED_MODEL_FILES = (
    "feature_extractor.h5",
    "feddg_gatnet_model.h5",
    "reference_features.npy",
    "reference_labels.npy",
)
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
INVALID_XRAY_MESSAGE = "Please upload a valid Chest X-ray image."
EMPTY_PATIENT_NAME_MESSAGE = "Please enter the patient's name before running prediction."
# Analysis size keeps validation fast while preserving image structure.
XRAY_VALIDATION_MAX_DIM = 512

MEDICAL_ICONS = {
    "thermometer": '<path d="M14 14.8V5a4 4 0 0 0-8 0v9.8a6 6 0 1 0 8 0ZM10 20a3 3 0 0 1-2-5.24V5a2 2 0 1 1 4 0v9.76A3 3 0 0 1 10 20Z"/>',
    "cough": '<path d="M5 9a4 4 0 0 1 4-4h2v2H9a2 2 0 0 0 0 4h2a5 5 0 0 1 0 10H7v-2h4a3 3 0 0 0 0-6H9a4 4 0 0 1-4-4Zm10-4h6v2h-6V5Zm2 5h4v2h-4v-2Zm-1 5h5v2h-5v-2Z"/>',
    "chest": '<path d="M12 3c-2.5 0-4 2.1-4 5.7V21H6V8.7C6 3.9 8.4 1 12 1s6 2.9 6 7.7V21h-2V8.7C16 5.1 14.5 3 12 3Zm-4 9h8v2H8v-2Zm1 4h6v2H9v-2Z"/>',
    "lungs": '<path d="M11 3h2v7.1c1.4-2.2 3.1-3.4 5-3.4 2.3 0 4 2.1 4 5.2V20h-2v-8.1c0-2-1-3.2-2.3-3.2-1.7 0-3.5 2.1-4.7 5.4V21h-2v-6.9C9.8 10.8 8 8.7 6.3 8.7 5 8.7 4 9.9 4 11.9V20H2v-8.1c0-3.1 1.7-5.2 4-5.2 1.9 0 3.6 1.2 5 3.4V3Z"/>',
    "energy": '<path d="M13 2 4 13h7l-1 9 9-12h-7l1-8Z"/>',
    "pulse": '<path d="M2 12h4l2-6 4 12 3-8 2 4h5v2h-6.2l-.6-1.2L12 23 8 11l-.6 3H2v-2Z"/>',
    "heart": '<path d="M12 21s-8-4.9-8-11a5 5 0 0 1 8-4 5 5 0 0 1 8 4c0 6.1-8 11-8 11Zm0-2.4c2.4-1.6 6-5 6-8.6a3 3 0 0 0-5.1-2.1l-.9.9-.9-.9A3 3 0 0 0 6 10c0 3.6 3.6 7 6 8.6Z"/>',
    "oxygen": '<path d="M8 13a5 5 0 1 1 0-10 5 5 0 0 1 0 10Zm0-2a3 3 0 1 0 0-6 3 3 0 0 0 0 6Zm8 10a6 6 0 1 1 0-12 6 6 0 0 1 0 12Zm0-2a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z"/>',
    "shield": '<path d="M12 2 20 5v6c0 5-3.4 9.7-8 11-4.6-1.3-8-6-8-11V5l8-3Zm0 2.2L6 6.5V11c0 3.9 2.5 7.5 6 8.8 3.5-1.3 6-4.9 6-8.8V6.5l-6-2.3Z"/>',
    "mask": '<path d="M4 8.5 12 5l8 3.5V13c0 3.3-3.6 6-8 6s-8-2.7-8-6V8.5Zm2 1.3V13c0 2.1 2.7 4 6 4s6-1.9 6-4V9.8l-6-2.6-6 2.6Zm2 1.2h8v2H8v-2Z"/>',
    "water": '<path d="M12 2s7 7.2 7 12a7 7 0 1 1-14 0c0-4.8 7-12 7-12Zm0 3.1C9.8 7.7 7 11.5 7 14a5 5 0 1 0 10 0c0-2.5-2.8-6.3-5-8.9Z"/>',
    "vaccine": '<path d="m18.4 3 2.6 2.6-1.4 1.4-.6-.6-3.2 3.2 2 2-1.4 1.4-.7-.7-7.2 7.2H5v-3.5l7.2-7.2-.7-.7L13 6.7l2 2 3.2-3.2-.6-.6L18.4 3ZM7 17.6h.7l6.6-6.6-.7-.7L7 16.9v.7Z"/>',
    "warning": '<path d="M12 2 22 20H2L12 2Zm0 4.1L5.4 18h13.2L12 6.1ZM11 10h2v4h-2v-4Zm0 5h2v2h-2v-2Z"/>',
    "default": '<path d="M11 3h2v7h7v2h-7v7h-2v-7H4v-2h7V3Z"/>',
}


st.set_page_config(
    page_title="FedDG-PneuNet | Pneumonia Detection Framework",
    page_icon="FP",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f2f7fb;
            --surface: rgba(255,255,255,.96);
            --surface-strong: #ecf6ff;
            --text: #111c2f;
            --muted: #5a6d85;
            --line: #d7e4f1;
            --blue: #1467e8;
            --blue-dark: #0b3f93;
            --cyan: #16a7c7;
            --mint: #12a37c;
            --danger: #c5303c;
            --amber: #9a6b12;
            --shadow: 0 22px 60px rgba(17,28,47,.13);
            --shadow-soft: 0 12px 34px rgba(20,103,232,.12);
            --radius: 10px;
            --space-xs: 8px;
            --space-sm: 12px;
            --space-md: 18px;
            --space-lg: 28px;
            --space-xl: 40px;
        }
        .stApp {
            background:
                linear-gradient(120deg, rgba(255,255,255,.96), rgba(232,244,255,.9) 42%, rgba(244,249,252,.98)),
                repeating-linear-gradient(90deg, rgba(20,103,232,.04) 0, rgba(20,103,232,.04) 1px, transparent 1px, transparent 78px),
                repeating-linear-gradient(0deg, rgba(22,167,199,.035) 0, rgba(22,167,199,.035) 1px, transparent 1px, transparent 78px),
                var(--bg);
            color: var(--text);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        header[data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"], #MainMenu, footer[data-testid="stFooter"] { visibility: hidden; }
        .block-container {
            width: min(1180px, calc(100% - 32px));
            max-width: 1180px;
            padding: var(--space-md) 0 var(--space-xl);
        }
        .block-container p, .block-container li, .block-container label,
        .block-container span, .block-container h1, .block-container h2, .block-container h3 {
            color: var(--text);
            overflow-wrap: anywhere;
            letter-spacing: 0;
        }
        [data-testid="stAppViewContainer"] { background: transparent; }
        [data-testid="stVerticalBlock"] { gap: 0.65rem; }
        [data-testid="stColumn"] { min-width: 0; }
        [data-testid="stHorizontalBlock"]:has(.nav-shell) {
            margin-bottom: var(--space-lg);
            align-items: center !important;
            gap: 1rem;
        }
        [data-testid="stHorizontalBlock"]:has(.nav-shell) [data-testid="stColumn"] {
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-width: 0;
        }
        .brand { display: inline-flex; align-items: center; gap: 12px; min-width: 0; }
        .brand-mark {
            display: inline-grid; place-items: center; width: 44px; height: 44px; border-radius: var(--radius);
            color: white; background: linear-gradient(135deg, var(--blue), var(--cyan) 58%, var(--mint));
            box-shadow: 0 12px 28px rgba(20,103,232,.24); font-weight: 850; animation: softPulse 3.8s ease-in-out infinite;
        }
        .brand strong, .brand small { display:block; line-height:1.15; color: var(--text); }
        .brand small { margin-top: 2px; color: var(--muted); font-size: .8rem; }
        .nav-shell [data-testid="stRadio"] { width: 100%; }
        .nav-shell [data-testid="stRadio"] > label { display: none; }
        .nav-shell div[role="radiogroup"] {
            display: flex !important; flex-wrap: wrap; justify-content: flex-end; gap: 6px;
            width: 100%; margin: 0; padding: 6px;
            border: 1px solid var(--line); border-radius: var(--radius);
            background: rgba(255,255,255,.92); box-shadow: var(--shadow-soft);
        }
        .nav-shell div[role="radiogroup"] label {
            flex: 1 1 auto; min-width: 88px; min-height: 42px; justify-content: center;
            border-radius: 6px; color: var(--muted) !important; font-weight: 800;
            transition: background .18s ease, color .18s ease;
        }
        .nav-shell div[role="radiogroup"] label:has(input:checked) {
            color: var(--blue-dark) !important; background: var(--surface-strong);
        }
        .hero-grid {
            display: grid; grid-template-columns: minmax(0,1fr) minmax(320px,470px); align-items: start;
            gap: var(--space-lg); min-height: auto;
        }
        .hero-copy { animation: fadeUp .68s .08s ease both; }
        .eyebrow { margin: 0 0 10px; color: var(--cyan); font-size: .78rem; font-weight: 900; text-transform: uppercase; }
        .hero-title { max-width: 800px; margin: 0; line-height: .96; font-weight: 900; }
        .hero-title .main {
            display:block; width:fit-content; margin-bottom: 12px; color: transparent;
            background: linear-gradient(105deg, var(--blue-dark), var(--blue), var(--cyan), var(--mint));
            -webkit-background-clip: text; background-clip: text;
            font-size: clamp(2.55rem, 5.5vw, 4.8rem);
        }
        .hero-title .sub {
            display:block; max-width:760px; color: var(--text); font-size: clamp(2rem, 3.4vw, 3rem);
            line-height: 1.08; font-weight: 850;
        }
        .lead { max-width: 680px; margin: 22px 0 0; color: rgba(17,28,47,.72); font-size: 1.07rem; line-height: 1.7; }
        .status-row { display:flex; flex-wrap:wrap; gap:10px; margin-top:28px; }
        .status-row span {
            display:inline-flex; align-items:center; min-height:36px; padding:8px 12px; border:1px solid var(--line);
            border-radius:6px; color:var(--muted); background:rgba(255,255,255,.88); box-shadow:0 10px 24px rgba(17,28,47,.05);
        }
        .status-row b { color: var(--text); margin-right: 6px; }
        .signal-board {
            position: relative; width:min(540px,100%); height:126px; margin-top:28px; overflow:hidden;
            border:1px solid rgba(20,103,232,.14); border-radius:8px;
            background: linear-gradient(90deg, rgba(255,255,255,.96), rgba(239,248,255,.88));
            box-shadow: var(--shadow-soft);
        }
        .signal-board:before {
            position:absolute; inset:0; content:""; background:linear-gradient(100deg, transparent 0 20%, rgba(20,103,232,.16) 42%, transparent 66%);
            transform: translateX(-100%); animation: scanMove 4.8s ease-in-out infinite;
        }
        .signal-line { position:absolute; left:24px; right:24px; top:62px; height:2px; background:linear-gradient(90deg,transparent,var(--blue),var(--cyan),var(--mint),transparent); animation: lineGlow 2.8s ease-in-out infinite; }
        .signal-node { position:absolute; top:55px; width:14px; height:14px; border:3px solid white; border-radius:50%; background:var(--blue); box-shadow:0 0 0 8px rgba(20,103,232,.12); animation:nodePulse 2.4s ease-in-out infinite; }
        .node-a { left:18%; } .node-b { left:48%; background:var(--cyan); animation-delay:.42s; } .node-c { right:18%; background:var(--mint); animation-delay:.84s; }
        .signal-bar { position:absolute; bottom:20px; width:9px; border-radius:999px; background:linear-gradient(180deg,var(--cyan),var(--blue)); opacity:.72; animation:barRise 1.8s ease-in-out infinite; }
        .bar-a { left:31%; height:30px; } .bar-b { left:35%; height:46px; animation-delay:.26s; } .bar-c { left:39%; height:24px; animation-delay:.52s; }
        .glass-card, .result-summary, .image-panel, .health-card, .research-card {
            border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface);
            box-shadow: var(--shadow); backdrop-filter: blur(16px);
        }
        [data-testid="stColumn"]:has(.upload-panel-anchor) > [data-testid="stVerticalBlock"] {
            position: relative; overflow: hidden;
            padding: var(--space-md); border: 1px solid var(--line); border-radius: var(--radius);
            background: var(--surface); box-shadow: var(--shadow);
            animation: cardFloatIn .72s .16s ease both;
        }
        [data-testid="stColumn"]:has(.upload-panel-anchor) > [data-testid="stVerticalBlock"]::before {
            position: absolute; inset: 0 0 auto; height: 4px; content: "";
            background: linear-gradient(90deg, var(--blue), var(--cyan), var(--mint));
            border-radius: var(--radius) var(--radius) 0 0;
            pointer-events: none;
        }
        .upload-panel-anchor { display: none !important; height: 0; margin: 0; padding: 0; }
        .panel-heading { display:flex; justify-content:space-between; gap:16px; align-items:flex-start; margin-bottom:var(--space-sm); }
        .panel-heading h2 { margin:0; font-size:1.45rem; line-height:1.15; color:var(--text); }
        .secure-badge { min-height:36px; padding:8px 12px; border:1px solid rgba(18,166,122,.22); border-radius:6px; color:#0c7556; background:rgba(18,166,122,.08); font-weight:800; }
        .upload-hint { margin: 0 0 var(--space-xs); color: var(--muted); font-size: .9rem; line-height: 1.5; }
        .health-icon svg, .warning-icon svg, .footer-email svg { width:26px; height:26px; fill:currentColor; }
        div[data-testid="stTextInput"] label p { color: var(--text) !important; font-weight: 700; }
        div[data-testid="stTextInput"] input {
            min-height: 46px; border-radius: 6px; border-color: var(--line);
            background: #fafcff; color: var(--text); font-weight: 600;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: var(--blue); box-shadow: 0 0 0 2px rgba(20,103,232,.12);
        }
        [data-testid="stColumn"]:has(.upload-panel-anchor) [data-testid="stFileUploader"] {
            padding: 0; margin: var(--space-xs) 0;
            background: transparent !important;
        }
        [data-testid="stColumn"]:has(.upload-panel-anchor) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] {
            min-height: 148px; padding: var(--space-md);
            border: 2px dashed #a9c7e9 !important; border-radius: var(--radius) !important;
            background: linear-gradient(180deg, rgba(239,248,255,.9), rgba(255,255,255,.98)) !important;
        }
        [data-testid="stColumn"]:has(.upload-panel-anchor) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"]:hover {
            border-color: var(--blue) !important;
            background: #eef6ff !important;
        }
        [data-testid="stColumn"]:has(.upload-panel-anchor) [data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] > div,
        [data-testid="stColumn"]:has(.upload-panel-anchor) [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stColumn"]:has(.upload-panel-anchor) [data-testid="stFileUploader"] small {
            background: transparent !important;
            color: var(--muted) !important;
        }
        [data-testid="stColumn"]:has(.upload-panel-anchor) [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] div,
        [data-testid="stColumn"]:has(.upload-panel-anchor) [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span {
            color: var(--text) !important;
        }
        [data-testid="stColumn"]:has(.upload-panel-anchor) [data-testid="stFileUploader"] button {
            color: var(--blue-dark) !important; background: #fff !important;
            border: 1px solid var(--line) !important; border-radius: 6px !important; font-weight: 700;
        }
        [data-testid="stFileUploader"] [data-testid="stFileUploaderBackdrop"] { display: none !important; }
        [data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] {
            color: var(--text);
        }
        [data-testid="stAlert"] {
            border-radius: var(--radius); color: var(--text) !important;
        }
        [data-testid="stAlert"] p, [data-testid="stAlert"] div { color: inherit !important; }
        div[data-testid="stNotification"] { color: var(--text); }
        [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
        [data-testid="stImage"] figcaption { color: var(--muted) !important; }
        .workflow-band { display:grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap:14px; margin-top:32px; }
        .workflow-step, .health-card, .research-card {
            min-height:160px; padding:18px; box-shadow:0 10px 28px rgba(17,28,47,.06);
            transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease; animation:fadeUp .64s ease both;
        }
        .workflow-step:hover, .health-card:hover, .research-card:hover, .image-panel:hover {
            border-color:rgba(20,103,232,.24); box-shadow:0 18px 44px rgba(17,28,47,.10); transform:translateY(-3px);
        }
        .workflow-step .index { display:inline-flex; align-items:center; justify-content:center; min-width:38px; height:28px; margin-bottom:16px; border-radius:6px; color:var(--blue-dark); background:linear-gradient(135deg,var(--surface-strong),white); font-weight:850; }
        .card-index { display:inline-flex; align-items:center; justify-content:center; min-width:38px; height:28px; margin-bottom:16px; border-radius:6px; color:var(--blue-dark); background:linear-gradient(135deg,var(--surface-strong),white); font-weight:850; }
        .workflow-step strong { display:block; margin-bottom:8px; }
        .workflow-step p, .health-card p, .research-card p { margin:0; color:var(--muted); line-height:1.58; }
        .page-heading { max-width:760px; margin-bottom:26px; animation:fadeUp .62s ease both; }
        .page-heading h1, .section-title { margin:0; font-size:clamp(2.2rem,5vw,4rem); line-height:1.02; font-weight:900; }
        .page-heading p:not(.eyebrow) { color:var(--muted); line-height:1.7; font-size:1.04rem; }
        .result-grid { display:grid; grid-template-columns:minmax(280px,.92fr) minmax(0,1.08fr); gap:18px; align-items:start; }
        .result-summary { position:relative; overflow:hidden; min-height:360px; padding:24px; }
        .result-summary:before { position:absolute; inset:0 0 auto; height:4px; content:""; background:linear-gradient(90deg,var(--blue),var(--cyan),var(--mint)); }
        .result-summary.risk:before { background:linear-gradient(90deg,var(--danger),#ff9c9c); }
        .result-summary.clear:before { background:linear-gradient(90deg,var(--mint),var(--cyan)); }
        .result-title { font-size:clamp(2.4rem,5vw,4rem); font-weight:900; margin:0 0 18px; line-height:1; }
        .risk { color: var(--danger); } .clear { color: var(--mint); }
        .confidence-meter { margin:18px 0 20px; }
        .meter-header { display:flex; justify-content:space-between; gap:12px; margin-bottom:10px; color:var(--muted); font-weight:800; }
        .meter-track { position:relative; overflow:hidden; height:12px; border-radius:999px; background:#eaf3fb; box-shadow:inset 0 0 0 1px rgba(20,103,232,.08); }
        .meter-fill { display:block; height:100%; border-radius:999px; background:linear-gradient(90deg,var(--blue),var(--cyan),var(--mint)); animation:meterFill .72s ease both; }
        .risk .meter-fill { background:linear-gradient(90deg,var(--danger),#ff9c9c); }
        .result-metrics { display:grid; gap:10px; margin:0; }
        .result-metrics div { padding:12px; border:1px solid var(--line); border-radius:8px; background:#fafcff; }
        .result-metrics dt { color:var(--muted); font-size:.86rem; }
        .result-metrics dd { margin:4px 0 0; color:var(--text); font-weight:850; }
        .report-tools { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
        .metric-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:16px 0; }
        .metric-card { padding:14px; border:1px solid var(--line); border-radius:8px; background:rgba(255,255,255,.9); }
        .metric-card span { display:block; color:var(--muted); font-size:.88rem; }
        .metric-card strong { display:block; margin-top:4px; font-size:1.1rem; }
        .image-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; margin-top:18px; }
        .image-panel { padding:16px; transition:transform .18s ease, box-shadow .18s ease; }
        .awareness-hero, .emergency-section, .medical-disclaimer-card {
            display:grid; gap:22px; margin-top:34px; border:1px solid var(--line); border-radius:8px; background:rgba(255,255,255,.88);
            box-shadow:var(--shadow-soft); animation:fadeUp .64s ease both;
        }
        .awareness-hero { grid-template-columns:minmax(0,1fr) 360px; align-items:center; padding:26px; }
        .awareness-hero h2, .section-heading h2, .emergency-copy h2 { margin:0; font-size:clamp(1.75rem,3vw,2.55rem); line-height:1.08; }
        .awareness-hero p:not(.eyebrow), .emergency-copy p { margin:14px 0 0; color:var(--muted); line-height:1.7; }
        .medical-illustration { position:relative; min-height:260px; border-radius:8px; background:linear-gradient(135deg,#eaf5ff,#ffffff); overflow:hidden; border:1px solid rgba(20,103,232,.12); }
        .illustration-screen { position:absolute; inset:28px; border:1px solid rgba(20,103,232,.16); border-radius:8px; background:rgba(255,255,255,.72); box-shadow:inset 0 0 0 8px rgba(236,246,255,.65); }
        .illustration-lung { position:absolute; top:76px; width:86px; height:120px; border-radius:48px 48px 58px 58px; background:linear-gradient(180deg,rgba(20,103,232,.84),rgba(22,167,199,.72)); opacity:.9; animation:nodePulse 3.2s ease-in-out infinite; }
        .illustration-lung.left { left:94px; transform:rotate(-9deg); } .illustration-lung.right { right:94px; transform:rotate(9deg); animation-delay:.28s; }
        .illustration-pulse { position:absolute; left:48px; right:48px; bottom:52px; height:3px; background:linear-gradient(90deg,transparent,var(--danger),var(--amber),var(--mint),transparent); animation:lineGlow 2.2s ease-in-out infinite; }
        .awareness-section { margin-top:42px; }
        .section-heading { display:flex; align-items:end; justify-content:space-between; gap:20px; margin-bottom:18px; }
        .health-card-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:14px; }
        .health-card { min-height:214px; }
        .health-card h3, .research-card h3, .workflow-step strong { color: var(--text); }
        .workflow-band { margin-top: var(--space-lg); }
        .health-icon, .warning-icon { display:inline-grid; place-items:center; width:44px; height:44px; margin-bottom:14px; border-radius:8px; color:var(--blue); background:linear-gradient(135deg,#eef6ff,white); box-shadow:0 10px 24px rgba(20,103,232,.12); font-weight:900; }
        .infographic-card { display:flex; flex-wrap:wrap; gap:8px; justify-content:flex-end; }
        .infographic-card span { padding:8px 11px; border:1px solid rgba(20,103,232,.12); border-radius:6px; background:#f7fbff; color:var(--blue-dark); font-weight:850; }
        .emergency-section { grid-template-columns:minmax(260px,.9fr) minmax(0,1.1fr); padding:24px; border-color:rgba(214,69,80,.28); background:linear-gradient(135deg,rgba(255,249,249,.92),rgba(255,255,255,.92)); }
        .warning-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }
        .warning-card { display:flex; gap:12px; align-items:center; min-height:92px; padding:16px; border:1px solid rgba(214,69,80,.25); border-radius:8px; background:rgba(255,249,249,.92); font-weight:850; transition:transform .18s ease, box-shadow .18s ease; }
        .warning-card:hover { transform:translateY(-3px); box-shadow:0 18px 42px rgba(214,69,80,.10); }
        .warning-icon { flex:0 0 auto; margin:0; color:var(--danger); background:#fff; }
        .medical-disclaimer-card { grid-template-columns:auto minmax(0,1fr); align-items:start; padding:22px; border-color:rgba(201,141,29,.34); background:rgba(255,249,237,.88); color:#6e4a08; line-height:1.7; }
        .medical-disclaimer-card .health-icon { color:var(--amber); margin:0; }
        .footer { position:relative; overflow:hidden; margin-top:42px; border:1px solid rgba(255,255,255,.13); border-radius:8px; color:#eaf4ff; background:linear-gradient(135deg, rgba(12,29,51,.98), rgba(12,50,80,.96)); box-shadow:0 24px 70px rgba(17,28,47,.22); }
        .footer:before { position:absolute; inset:0; content:""; background:linear-gradient(115deg,rgba(22,167,199,.20),transparent 36%,rgba(18,163,124,.14)); opacity:.82; pointer-events:none; }
        .footer-brand { position:relative; display:grid; grid-template-columns:auto minmax(0,1fr); gap:14px; align-items:center; padding:28px; border-bottom:1px solid rgba(255,255,255,.10); }
        .footer-mark { display:inline-grid; place-items:center; width:48px; height:48px; border-radius:8px; color:#fff; background:linear-gradient(135deg,var(--blue),var(--cyan),var(--mint)); font-weight:850; box-shadow:0 16px 34px rgba(22,167,199,.26); }
        .footer-brand strong { display:block; margin-bottom:6px; font-size:1.22rem; }
        .footer-brand p { margin:0; max-width:680px; color:#b8cce0; line-height:1.6; }
        .footer-grid { position:relative; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; padding:22px 28px 28px; }
        .footer-panel { min-height:190px; padding:18px; border:1px solid rgba(255,255,255,.10); border-radius:8px; background:linear-gradient(180deg, rgba(255,255,255,.075), rgba(255,255,255,.045)); backdrop-filter:blur(14px); transition:transform .18s ease, border-color .18s ease, background .18s ease; }
        .footer-panel:hover { border-color:rgba(22,167,199,.28); background:rgba(255,255,255,.085); transform:translateY(-3px); }
        .footer-panel h3 { margin-top:0; color:#fff; font-size:1rem; }
        .footer-panel p { color:#b8cce0; line-height:1.8; margin:0; }
        .footer-links { display:grid; gap:8px; }
        .footer-link, .footer-email { display:flex; align-items:center; gap:8px; min-height:32px; color:#b8cce0; transition:color .16s ease, transform .16s ease; overflow-wrap:anywhere; }
        .footer-link:before { display:inline-block; width:7px; height:7px; border-radius:50%; background:linear-gradient(135deg,var(--cyan),var(--mint)); content:""; opacity:.78; }
        .footer-link:hover, .footer-email:hover { color:#fff; transform:translateX(3px); }
        .social-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:16px; }
        .social-row span { display:grid; place-items:center; width:36px; height:36px; border:1px solid rgba(255,255,255,.12); border-radius:8px; color:#fff; background:rgba(255,255,255,.08); font-weight:850; text-transform:uppercase; transition:transform .18s ease, background .18s ease; }
        .social-row span:hover { transform:translateY(-2px); background:linear-gradient(135deg,var(--blue),var(--cyan)); }
        .footer-bottom { position:relative; display:flex; justify-content:space-between; gap:16px; padding:16px 28px 20px; border-top:1px solid rgba(255,255,255,.10); color:#9fb5ca; font-size:.9rem; }
        .stButton button, .stDownloadButton button {
            min-height:46px; border-radius:6px; font-weight:850;
            color: #fff !important; transition:transform .18s ease, box-shadow .18s ease;
        }
        .stButton button:hover, .stDownloadButton button:hover {
            transform:translateY(-1px); box-shadow:0 14px 30px rgba(23,105,224,.18);
        }
        [data-testid="stSpinner"] { color: var(--blue-dark) !important; }
        [data-testid="stTable"] { color: var(--text); }
        [data-testid="stTable"] th { color: var(--blue-dark); background: var(--surface-strong); }
        [data-testid="stTable"] td { color: var(--text); }
        @keyframes fadeUp { from { opacity:0; transform:translateY(18px); } to { opacity:1; transform:translateY(0); } }
        @keyframes cardFloatIn { from { opacity:0; transform:translateY(20px) scale(.985); } to { opacity:1; transform:translateY(0) scale(1); } }
        @keyframes scanMove { 0%,42% { transform:translateX(-110%); } 72%,100% { transform:translateX(110%); } }
        @keyframes lineGlow { 0%,100% { opacity:.66; filter:saturate(1); } 50% { opacity:1; filter:saturate(1.25); } }
        @keyframes nodePulse { 0%,100% { transform:scale(1); } 50% { transform:scale(1.18); } }
        @keyframes softPulse { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-2px); } }
        @keyframes iconFloat { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-5px); } }
        @keyframes barRise { 0%,100% { transform:scaleY(.72); opacity:.56; } 50% { transform:scaleY(1.08); opacity:.95; } }
        @keyframes meterFill { from { transform:scaleX(.82); transform-origin:left; opacity:.76; } to { transform:scaleX(1); opacity:1; } }
        @media (max-width: 980px) {
            [data-testid="stHorizontalBlock"]:has(.nav-shell) {
                flex-wrap: wrap;
            }
            [data-testid="stHorizontalBlock"]:has(.nav-shell) [data-testid="stColumn"] {
                width: 100% !important;
                flex: 1 1 100%;
            }
            .nav-shell div[role="radiogroup"] { justify-content: stretch; }
            .hero-grid, .image-grid, .result-grid, .awareness-hero, .emergency-section {
                grid-template-columns:1fr; min-height:auto;
            }
            .workflow-band, .health-card-grid, .footer-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
            .metric-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
            .medical-illustration { min-height:220px; }
            .result-title { font-size: clamp(1.8rem, 6vw, 3rem); }
        }
        @media (max-width: 720px) {
            .block-container { width:min(100% - 20px,1180px); padding: var(--space-sm) 0 var(--space-lg); }
            [data-testid="stHorizontalBlock"]:has(.nav-shell) { margin-bottom: var(--space-md); }
            .nav-shell div[role="radiogroup"] {
                display:grid !important; grid-template-columns:repeat(2,minmax(0,1fr)); width:100%;
            }
            .nav-shell div[role="radiogroup"] label { min-width:0; }
            .hero-grid { gap:var(--space-md); }
            .hero-title .main { font-size:2.2rem; margin-bottom:8px; }
            .hero-title .sub { font-size:1.55rem; }
            .signal-board { display:none; }
            .status-row span { width:100%; }
            .workflow-band, .health-card-grid, .warning-grid, .footer-grid, .metric-grid, .report-tools {
                grid-template-columns:1fr;
            }
            .awareness-hero, .emergency-section, .medical-disclaimer-card { padding:var(--space-md); }
            .section-heading { align-items:stretch; flex-direction:column; }
            .infographic-card { justify-content:flex-start; }
            .medical-disclaimer-card { grid-template-columns:1fr; }
            .footer-brand { padding:22px 18px; }
            .footer-grid { padding:18px; }
            .footer-bottom { flex-direction:column; align-items:flex-start; }
            [data-testid="stColumn"]:has(.upload-panel-anchor) > [data-testid="stVerticalBlock"] {
                padding: var(--space-sm);
            }
        }
        @media print {
            [data-testid="stHorizontalBlock"]:has(.nav-shell), .footer, .stButton, .stDownloadButton { display:none !important; }
            .block-container { width:100%; max-width:none; padding:0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def check_model_files() -> tuple[bool, str]:
    """Verify required artifacts exist before TensorFlow load."""
    _ensure_inference_imported()
    if load_artifacts is None:
        message = INFERENCE_IMPORT_ERROR or "Inference module could not be loaded."
        if "numpy" in message.lower() or "_ARRAY_API" in message:
            message += " Fix: pip install \"numpy>=1.26,<2.0\" then restart Streamlit."
        if "as_list" in message.lower() or "dtypepolicy" in message.lower() or "batch_shape" in message.lower():
            message += " Fix: pip install \"keras>=3.5\" then restart Streamlit."
        return False, f"Inference module unavailable: {message}"
    missing = [name for name in REQUIRED_MODEL_FILES if not (MODEL_DIR / name).is_file()]
    if missing:
        return (
            False,
            "Missing model files in `model/`: " + ", ".join(missing),
        )
    return True, ""


@st.cache_resource(show_spinner="Loading FedDG-PneuNet model artifacts...")
def get_artifacts():
    ready, message = check_model_files()
    if not ready:
        raise RuntimeError(message)
    return load_artifacts(MODEL_DIR)


def uploaded_file_size(uploaded_file) -> int:
    """Streamlit uploads may not populate `.size`; fall back to byte length."""
    size = getattr(uploaded_file, "size", None)
    if isinstance(size, int) and size >= 0:
        return size
    uploaded_file.seek(0)
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    uploaded_file.seek(0)
    return len(data)


def show_upload_preview(uploaded_file) -> None:
    """Render preview from PIL to avoid NoneType errors with raw upload handles."""
    try:
        uploaded_file.seek(0)
        preview = Image.open(uploaded_file).convert("RGB")
        uploaded_file.seek(0)
        st.image(preview, caption="Selected chest X-ray preview", use_container_width=True)
    except Exception:
        st.error("The uploaded image could not be previewed.")


def html_brand() -> None:
    st.markdown(
        """
        <div class="brand">
            <span class="brand-mark">FP</span>
            <span><strong>FedDG-PneuNet</strong><small>Medical Graph AI</small></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        """
        <div class="footer">
            <div class="footer-brand">
                <span class="footer-mark">FP</span>
                <div>
                    <strong>FedDG-PneuNet</strong>
                    <p>AI-Powered Pneumonia Detection Framework using Federated Dynamic Graph Neural Networks.</p>
                </div>
            </div>
            <div class="footer-grid">
                <div class="footer-panel">
                    <h3>Quick Links</h3>
                    <div class="footer-links">
                        <span class="footer-link">Home</span>
                        <span class="footer-link">Result</span>
                        <span class="footer-link">Awareness</span>
                        <span class="footer-link">Research</span>
                    </div>
                </div>
                <div class="footer-panel">
                    <h3>Research Information</h3>
                    <div class="footer-links">
                        <span class="footer-link">FedDG-GATNet</span>
                        <span class="footer-link">Chest X-ray Analysis</span>
                        <span class="footer-link">AI Healthcare Research</span>
                        <span class="footer-link">Medical Disclaimer</span>
                    </div>
                </div>
                <div class="footer-panel">
                    <h3>Contact</h3>
                    <span class="footer-email">
                        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Zm0 3.2V17h16V8.2l-8 5.3-8-5.3Zm1.2-1.2 6.8 4.5L18.8 7H5.2Z"/></svg>
                        ratulmesbah@gmail.com
                    </span>
                    <div class="social-row" aria-label="Social media placeholders">
                        <span>in</span><span>rg</span><span>gh</span>
                    </div>
                </div>
            </div>
            <div class="footer-bottom">
                <span>&copy; 2026 FedDG-PneuNet. All Rights Reserved.</span>
                <span>Research and educational medical AI platform.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def validate_upload(uploaded_file) -> tuple[bool, str]:
    if uploaded_file is None:
        return False, "Please choose a chest X-ray image."
    name = getattr(uploaded_file, "name", None) or "upload"
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return False, "Only JPG, JPEG, and PNG images are allowed."
    size = uploaded_file_size(uploaded_file)
    if size <= 0 or size > MAX_UPLOAD_BYTES:
        return False, "Image size must be greater than 0 and no larger than 8 MB."
    try:
        uploaded_file.seek(0)
        raw = uploaded_file.read()
        uploaded_file.seek(0)
        if not raw:
            return False, "The uploaded image could not be read."
        buffer = io.BytesIO(raw)
        with Image.open(buffer) as image:
            image.verify()
        buffer.seek(0)
        with Image.open(buffer) as image:
            width, height = image.size
        if width < 32 or height < 32:
            return False, "The uploaded image is too small to analyze."
    except Exception:
        return False, "The uploaded image could not be read."
    return True, ""


def _load_rgb_array(uploaded_file) -> np.ndarray | None:
    """Decode upload to RGB NumPy array for OpenCV-based checks."""
    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
        uploaded_file.seek(0)
        return np.asarray(image, dtype=np.uint8)
    except Exception:
        return None


def _resize_for_validation(image: np.ndarray) -> np.ndarray:
    """Downscale large images so validation stays fast and stable."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= XRAY_VALIDATION_MAX_DIM:
        return image
    scale = XRAY_VALIDATION_MAX_DIM / longest
    new_size = (int(width * scale), int(height * scale))
    cv2 = _get_cv2()
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def validate_chest_xray(uploaded_file) -> tuple[bool, str]:
    """
    Heuristic chest X-ray gate using grayscale similarity, color cues, and texture.

    Natural photos, cartoons, and colorful scenes fail before model inference runs.
    """
    rgb = _load_rgb_array(uploaded_file)
    if rgb is None or rgb.size == 0:
        return False, INVALID_XRAY_MESSAGE

    rgb = _resize_for_validation(rgb)
    if rgb.shape[0] < 8 or rgb.shape[1] < 8:
        return False, INVALID_XRAY_MESSAGE
    red = rgb[:, :, 0].astype(np.float32)
    green = rgb[:, :, 1].astype(np.float32)
    blue = rgb[:, :, 2].astype(np.float32)

    # Grayscale similarity: real X-rays keep R/G/B channels nearly aligned.
    channel_mean_diff = float(
        np.mean(np.abs(red - green) + np.abs(red - blue) + np.abs(green - blue)) / 3.0
    )
    if channel_mean_diff > 12.0:
        return False, INVALID_XRAY_MESSAGE

    cv2 = _get_cv2()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_from_mean = (red + green + blue) / 3.0
    grayscale_similarity = float(np.mean(np.abs(gray.astype(np.float32) - gray_from_mean)))
    if grayscale_similarity > 8.0:
        return False, INVALID_XRAY_MESSAGE

    # Color rejection: selfies, landscapes, and cartoons carry higher saturation/colorfulness.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # cv2 set above
    saturation = hsv[:, :, 1]
    sat_mean = float(np.mean(saturation))
    sat_std = float(np.std(saturation))
    if sat_mean > 45.0 or (sat_std > 35.0 and sat_mean > 25.0):
        return False, INVALID_XRAY_MESSAGE

    rg = red - green
    yb = 0.5 * (red + green) - blue
    colorfulness = float(
        np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    )
    if colorfulness > 25.0:
        return False, INVALID_XRAY_MESSAGE

    # Edge/texture analysis: radiographs have moderate detail, not flat cartoons or noisy photos.
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var < 20.0 or laplacian_var > 8000.0:
        return False, INVALID_XRAY_MESSAGE

    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = int(edges.size)
    if edge_pixels == 0:
        return False, INVALID_XRAY_MESSAGE
    edge_density = float(np.count_nonzero(edges) / edge_pixels)
    if edge_density < 0.008 or edge_density > 0.35:
        return False, INVALID_XRAY_MESSAGE

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    texture_energy = float(np.mean(np.hypot(sobel_x, sobel_y)))
    if texture_energy < 6.0 or texture_energy > 95.0:
        return False, INVALID_XRAY_MESSAGE

    # Intensity distribution: chest X-rays use a usable grayscale range with structured contrast.
    intensity_range = int(gray.max()) - int(gray.min())
    if intensity_range < 40:
        return False, INVALID_XRAY_MESSAGE

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (hist.sum() + 1e-8)
    entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-10)))
    if entropy < 3.5 or entropy > 7.8:
        return False, INVALID_XRAY_MESSAGE

    return True, ""


def read_bytes(path: str | Path) -> bytes:
    file_path = Path(path)
    if not file_path.is_file():
        return b""
    return file_path.read_bytes()


def svg_icon(name: str) -> str:
    return f'<svg viewBox="0 0 24 24" aria-hidden="true">{MEDICAL_ICONS.get(name, MEDICAL_ICONS["default"])}</svg>'


def page_home() -> None:
    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        st.markdown(
            """
            <div class="hero-copy">
                <p class="eyebrow">Chest X-ray diagnostic intelligence</p>
                <h1 class="hero-title">
                    <span class="main">FedDG-PneuNet</span>
                    <span class="sub">A Federated Dynamic Graph-Based</span>
                    <span class="sub">Pneumonia Detection Framework</span>
                </h1>
                <p class="lead">
                    Upload a frontal chest radiograph and run the FedDG-PneuNet hybrid pipeline with feature extraction,
                    dynamic graph construction, graph attention inference, and Grad-CAM visualization.
                </p>
                <div class="status-row">
                    <span><b>Model</b> FedDG-PneuNet</span>
                    <span><b>Input</b> 224 x 224 RGB</span>
                    <span><b>Runtime</b> Python + Streamlit</span>
                </div>
                <div class="signal-board">
                    <span class="signal-line"></span>
                    <span class="signal-node node-a"></span>
                    <span class="signal-node node-b"></span>
                    <span class="signal-node node-c"></span>
                    <span class="signal-bar bar-a"></span>
                    <span class="signal-bar bar-b"></span>
                    <span class="signal-bar bar-c"></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="upload-panel-anchor" aria-hidden="true"></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="panel-heading">
                <div><p class="eyebrow">New analysis</p><h2>Upload X-ray</h2></div>
                <span class="secure-badge">Secure</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        patient_name = st.text_input(
            "Patient name",
            placeholder="Enter patient name",
            help="Required before running pneumonia prediction.",
            key="patient_name_input",
        )
        st.markdown('<p class="upload-hint">Chest X-ray file (JPG, JPEG, or PNG, max 8 MB)</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Chest X-ray image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="chest_xray_uploader",
        )
        if uploaded is not None:
            ok, message = validate_upload(uploaded)
            if ok:
                xray_ok, xray_message = validate_chest_xray(uploaded)
                if xray_ok:
                    show_upload_preview(uploaded)
                else:
                    st.error(xray_message)
            else:
                st.error(message)
        models_ready, _ = check_model_files()
        run = st.button(
            "Run prediction",
            type="primary",
            use_container_width=True,
            disabled=not models_ready,
        )

    if run:
        patient_name_value = (patient_name or "").strip()
        if not patient_name_value:
            st.error(EMPTY_PATIENT_NAME_MESSAGE)
            return
        ok, message = validate_upload(uploaded)
        if not ok:
            st.error(message)
            return
        xray_ok, xray_message = validate_chest_xray(uploaded)
        if not xray_ok:
            st.error(xray_message)
            return
        models_ok, models_message = check_model_files()
        if not models_ok:
            st.error(models_message)
            return
        try:
            with st.spinner("Running FedDG-PneuNet inference, graph generation, Grad-CAM, and report creation..."):
                uploaded.seek(0)
                st.session_state.result = run_prediction(
                    uploaded,
                    get_artifacts(),
                    uploads_dir=APP_ROOT / "uploads",
                    reports_dir=APP_ROOT / "reports",
                )
                st.session_state.patient_name = patient_name_value
                st.session_state.navigate_to = "Result"
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return
        st.rerun()

    st.markdown(
        """
        <div class="workflow-band">
            <div class="workflow-step"><span class="index">01</span><strong>Feature extraction</strong><p>Fine-tuned EfficientNetV2 produces a 512-dimensional image embedding.</p></div>
            <div class="workflow-step"><span class="index">02</span><strong>Dynamic graph</strong><p>Nearest reference embeddings form an adaptive graph for the query image.</p></div>
            <div class="workflow-step"><span class="index">03</span><strong>GAT inference</strong><p>Edge-aware graph attention returns the Normal or Pneumonia probability.</p></div>
            <div class="workflow-step"><span class="index">04</span><strong>Grad-CAM</strong><p>A heatmap highlights image regions that influenced the prediction.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_result() -> None:
    result = st.session_state.get("result")
    st.markdown('<div class="page-heading"><p class="eyebrow">Diagnostic output</p><h1>Prediction Result</h1><p>Review the graph-based classification, confidence score, uploaded image, Grad-CAM heatmap, and generated report.</p></div>', unsafe_allow_html=True)
    if not result:
        st.info("No result yet. Upload a chest X-ray from Home to run a FedDG-PneuNet prediction.")
        if st.button("Start analysis", type="primary"):
            st.session_state.navigate_to = "Home"
            st.rerun()
        return

    prediction_label = str(result.get("prediction", "Unknown"))
    result_class = "risk" if prediction_label.lower() == "pneumonia" else "clear"
    confidence = min(max(float(result.get("confidence", 0.0)), 0.0), 100.0)
    patient_name = escape(str(st.session_state.get("patient_name", "—")))
    st.markdown(
        f"""
        <section class="result-grid">
            <article class="result-summary {result_class}">
                <p class="eyebrow">Patient · {patient_name}</p>
                <h2 class="result-title {result_class}">{patient_name}: {escape(prediction_label)}</h2>
                <div class="confidence-meter">
                    <div class="meter-header"><span>Confidence</span><strong>{confidence:.2f}%</strong></div>
                    <div class="meter-track"><span class="meter-fill" style="width:{confidence:.2f}%"></span></div>
                </div>
                <dl class="result-metrics">
                    <div><dt>Patient name</dt><dd>{patient_name}</dd></div>
                    <div><dt>Prediction result</dt><dd>{escape(prediction_label)}</dd></div>
                    <div><dt>Confidence score</dt><dd>{confidence:.2f}%</dd></div>
                    <div><dt>Pneumonia probability</dt><dd>{float(result.get('probability', 0.0)) * 100:.2f}%</dd></div>
                    <div><dt>Decision threshold</dt><dd>{float(result.get('threshold', 0.0)):.2f}</dd></div>
                    <div><dt>Graph nodes</dt><dd>{result.get('graph_nodes', '—')}</dd></div>
                    <div><dt>Prediction timestamp</dt><dd>{escape(str(result.get('timestamp', '—')))}</dd></div>
                </dl>
            </article>
            <article class="glass-card" style="padding:24px;">
                <p class="eyebrow">Graph context</p>
                <h3 style="margin-top:0;">Dynamic reference neighborhood</h3>
                <p style="color:#63748d;line-height:1.7;margin-top:0;">
                    FedDG-PneuNet selected the nearest reference embeddings, generated an adaptive adjacency matrix,
                    and classified the uploaded image through EdgeAwareGATv2 inference.
                </p>
                <div class="metric-grid">
                    <div class="metric-card"><span>Normal neighbors</span><strong>{result.get('normal_neighbors', '—')}</strong></div>
                    <div class="metric-card"><span>Pneumonia neighbors</span><strong>{result.get('pneumonia_neighbors', '—')}</strong></div>
                    <div class="metric-card"><span>References used</span><strong>{result.get('neighbors', '—')}</strong></div>
                    <div class="metric-card"><span>Preprocessing</span><strong>{escape(str(result.get('normalization', '—')))}</strong></div>
                </div>
            </article>
        </section>
        """,
        unsafe_allow_html=True,
    )

    image_path = Path(str(result.get("image_path", "")))
    heatmap_path = Path(str(result["heatmap_path"])) if result.get("heatmap_path") else None
    report_path = Path(str(result.get("report_path", "")))
    saved_prediction_path = Path(str(result.get("saved_prediction_path", "")))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="image-panel"><h3>Uploaded X-ray</h3>', unsafe_allow_html=True)
        if image_path.is_file():
            st.image(str(image_path), use_container_width=True)
        else:
            st.warning("Uploaded X-ray image is not available.")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="image-panel"><h3>Grad-CAM</h3>', unsafe_allow_html=True)
        if heatmap_path and heatmap_path.is_file():
            st.image(str(heatmap_path), use_container_width=True)
        else:
            st.warning("Grad-CAM was unavailable for this run.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.download_button(
            "Download PDF Report",
            read_bytes(report_path),
            file_name=report_path.name or "report.pdf",
            mime="application/pdf",
            use_container_width=True,
            disabled=not report_path.is_file(),
        )
    with b2:
        st.download_button(
            "Save Prediction",
            read_bytes(saved_prediction_path),
            file_name=saved_prediction_path.name or "prediction.json",
            mime="application/json",
            use_container_width=True,
            disabled=not saved_prediction_path.is_file(),
        )
    with b3:
        st.markdown('<button onclick="window.print()" style="width:100%;min-height:46px;border:0;border-radius:6px;background:#ecf6ff;color:#0b3f93;font-weight:850;cursor:pointer;">Print Report</button>', unsafe_allow_html=True)


def card_grid(items: Iterable[tuple[str, str, str]]) -> None:
    cards = ""
    for icon, title, text in items:
        cards += (
            f'<div class="health-card"><span class="health-icon">{svg_icon(icon)}</span>'
            f'<h3>{escape(title)}</h3><p>{escape(text)}</p></div>'
        )
    st.markdown(f'<div class="health-card-grid">{cards}</div>', unsafe_allow_html=True)


def page_awareness() -> None:
    st.markdown('<div class="page-heading"><p class="eyebrow">Healthcare awareness</p><h1>Pneumonia care guidance</h1><p>Symptoms, prevention, emergency signs, and medical disclaimer in one clear healthcare dashboard.</p></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <section class="awareness-hero" aria-label="Healthcare awareness overview">
            <div>
                <p class="eyebrow">Patient safety layer</p>
                <h2>Understand symptoms early and know when urgent care is needed.</h2>
                <p>
                    FedDG-PneuNet provides AI-assisted research output, while this page keeps essential pneumonia
                    awareness information easy to scan across desktop, tablet, and mobile devices.
                </p>
            </div>
            <div class="medical-illustration" aria-hidden="true">
                <span class="illustration-screen"></span>
                <span class="illustration-lung left"></span>
                <span class="illustration-lung right"></span>
                <span class="illustration-pulse"></span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    symptoms = [
        ("thermometer", "Fever", "Elevated body temperature may indicate infection and should be monitored carefully."),
        ("cough", "Persistent cough", "A continuous cough, sometimes with mucus, can be associated with lung inflammation."),
        ("chest", "Chest pain", "Sharp or heavy chest discomfort may worsen during coughing or deep breathing."),
        ("lungs", "Difficulty breathing", "Shortness of breath can signal reduced lung function and needs medical attention."),
        ("energy", "Fatigue and weakness", "Unusual tiredness may occur as the body responds to respiratory infection."),
        ("thermometer", "Chills and sweating", "Repeated chills or sweating episodes can accompany fever and infection."),
        ("heart", "Fast heartbeat", "A rapid pulse may appear when breathing is strained or oxygen demand increases."),
        ("oxygen", "Low oxygen level", "Low oxygen saturation is a serious sign that should be assessed promptly."),
        ("pulse", "Bluish lips or fingertips", "Bluish skin tone may reflect oxygen shortage and requires urgent evaluation."),
        ("energy", "Loss of appetite", "Reduced appetite can occur during infection and dehydration risk may increase."),
    ]
    preventions = [
        ("shield", "Wash hands regularly", "Use soap and clean water or sanitizer to reduce respiratory infection spread."),
        ("mask", "Wear masks in crowded places", "Masks can help reduce exposure in high-risk indoor or crowded environments."),
        ("lungs", "Avoid smoking", "Smoking damages lung defense mechanisms and increases respiratory risk."),
        ("heart", "Maintain healthy diet", "Balanced nutrition supports immunity and recovery from illness."),
        ("water", "Stay hydrated", "Adequate fluids help maintain airway moisture and overall health."),
        ("pulse", "Regular exercise", "Consistent activity supports cardiopulmonary fitness and immune resilience."),
        ("vaccine", "Vaccination", "Recommended vaccines can lower risk of severe respiratory disease."),
        ("shield", "Maintain hygiene", "Clean living spaces and safe coughing habits reduce transmission."),
        ("default", "Seek early medical consultation", "Early assessment can help prevent complications when symptoms worsen."),
        ("heart", "Protect elderly people and children", "High-risk groups benefit from extra prevention and early monitoring."),
    ]
    warnings = ["Severe breathing difficulty", "Persistent high fever", "Chest tightness", "Oxygen shortage", "Confusion or dizziness", "Bluish lips or fingertips"]
    st.markdown('<section class="awareness-section"><div class="section-heading"><div><p class="eyebrow">Pneumonia symptoms</p><h2>Common signs that may need medical attention</h2></div></div></section>', unsafe_allow_html=True)
    card_grid(symptoms)
    st.markdown('<section class="awareness-section"><div class="section-heading split"><div><p class="eyebrow">Prevention and safety</p><h2>Daily habits that reduce pneumonia risk</h2></div><div class="infographic-card" aria-hidden="true"><span>Wash</span><span>Protect</span><span>Vaccinate</span></div></div></section>', unsafe_allow_html=True)
    card_grid(preventions)
    warning_html = "".join(f'<article class="warning-card"><span class="warning-icon">{svg_icon("warning")}</span><strong>{escape(item)}</strong></article>' for item in warnings)
    st.markdown(
        f"""
        <section class="emergency-section">
            <div class="emergency-copy">
                <p class="eyebrow">Emergency warning signs</p>
                <h2>Seek urgent medical help if these symptoms appear.</h2>
                <p>Emergency symptoms can progress quickly. Immediate clinical evaluation is important when breathing, oxygen level, or mental clarity is affected.</p>
            </div>
            <div class="warning-grid">{warning_html}</div>
        </section>
        <section class="medical-disclaimer-card">
            <span class="health-icon">{svg_icon("shield")}</span>
            <div>
                <p class="eyebrow">Medical disclaimer</p>
                <p>"This system is intended for research and educational purposes only. It does not replace professional medical diagnosis or treatment. Please consult a licensed healthcare professional for clinical evaluation."</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def page_research() -> None:
    st.markdown('<div class="page-heading"><p class="eyebrow">Research model</p><h1>FedDG-PneuNet Research Framework</h1><p>FedDG-PneuNet combines fine-tuned EfficientNetV2 feature extraction with dynamic graph generation and edge-aware graph attention inference for chest X-ray classification.</p></div>', unsafe_allow_html=True)
    cards = [
        ("Fine-tuned EfficientNetV2", "Maps every uploaded X-ray into a multi-scale 512-dimensional embedding using feature_extractor.h5."),
        ("Dynamic Adaptive Graph", "Compares query embedding with reference_features.npy and creates an adaptive adjacency matrix."),
        ("EdgeAwareGATv2", "Custom graph attention layer incorporates edge weights for graph-based medical evidence."),
        ("Grad-CAM Visualization", "Backpropagates the graph prediction to produce a visual heatmap."),
    ]
    html = "".join(
        f'<div class="research-card"><span class="card-index">{index:02d}</span><h3>{escape(title)}</h3><p>{escape(text)}</p></div>'
        for index, (title, text) in enumerate(cards, start=1)
    )
    st.markdown(f'<div class="workflow-band">{html}</div>', unsafe_allow_html=True)
    st.markdown("### Model files")
    st.table(
        {
            "Artifact": ["feature_extractor.h5", "feddg_gatnet_model.h5", "reference_features.npy", "reference_labels.npy"],
            "Purpose": ["Extracts uploaded X-ray embedding", "Runs graph attention inference", "Stores reference embeddings", "Provides reference class context"],
        }
    )


def render_model_status() -> None:
    ready, message = check_model_files()
    if ready:
        return
    st.error(message)
    st.info(
        "Place `feature_extractor.h5`, `feddg_gatnet_model.h5`, "
        "`reference_features.npy`, and `reference_labels.npy` inside the `model/` folder."
    )


def _apply_pending_navigation() -> None:
    """Set nav page before the radio widget renders (Streamlit session-state rule)."""
    pending = st.session_state.pop("navigate_to", None)
    if pending in NAV_PAGES:
        st.session_state.page = pending
    elif "page" not in st.session_state or st.session_state.page not in NAV_PAGES:
        st.session_state.page = "Home"


def _run_app() -> None:
    css()
    _apply_pending_navigation()
    render_model_status()
    brand_col, nav_col = st.columns([1.35, 1], gap="large")
    with brand_col:
        html_brand()
    with nav_col:
        st.markdown('<div class="nav-shell">', unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
            list(NAV_PAGES),
            horizontal=True,
            label_visibility="collapsed",
            key="page",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if page == "Home":
        page_home()
    elif page == "Result":
        page_result()
    elif page == "Research":
        page_research()
    else:
        page_awareness()
    footer()


def main() -> None:
    try:
        _run_app()
    except Exception as exc:
        st.error("The application hit an unexpected error.")
        st.exception(exc)


if __name__ == "__main__":
    main()
