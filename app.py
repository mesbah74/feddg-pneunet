from __future__ import annotations

from pathlib import Path
from typing import Iterable
from html import escape

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from model.inference import load_artifacts, run_prediction

APP_NAME = "FedDG-PneuNet"
MODEL_DIR = Path("model")
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
            --surface: rgba(255,255,255,.92);
            --surface-strong: #ecf6ff;
            --text: #111c2f;
            --muted: #63748d;
            --line: #d7e4f1;
            --blue: #1467e8;
            --blue-dark: #0b3f93;
            --cyan: #16a7c7;
            --mint: #12a37c;
            --danger: #d64550;
            --amber: #c98d1d;
            --shadow: 0 22px 60px rgba(17,28,47,.13);
            --shadow-soft: 0 12px 34px rgba(20,103,232,.12);
            --radius: 8px;
        }
        .stApp {
            background:
                linear-gradient(120deg, rgba(255,255,255,.94), rgba(232,244,255,.88) 42%, rgba(244,249,252,.98)),
                repeating-linear-gradient(90deg, rgba(20,103,232,.045) 0, rgba(20,103,232,.045) 1px, transparent 1px, transparent 78px),
                repeating-linear-gradient(0deg, rgba(22,167,199,.04) 0, rgba(22,167,199,.04) 1px, transparent 1px, transparent 78px),
                var(--bg);
            color: var(--text);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        header[data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer[data-testid="stFooter"] { visibility: hidden; }
        .block-container {
            width: min(1180px, calc(100% - 32px));
            max-width: 1180px;
            padding: 24px 0 40px;
        }
        h1, h2, h3, p { overflow-wrap: anywhere; letter-spacing: 0; }
        .topbar {
            display: flex; align-items: center; justify-content: space-between; gap: 16px;
            min-height: 72px; margin-bottom: 28px; animation: fadeUp .52s ease both;
        }
        .brand { display: inline-flex; align-items: center; gap: 12px; min-width: 0; }
        .brand-mark {
            display: inline-grid; place-items: center; width: 44px; height: 44px; border-radius: var(--radius);
            color: white; background: linear-gradient(135deg, var(--blue), var(--cyan) 58%, var(--mint));
            box-shadow: 0 12px 28px rgba(20,103,232,.24); font-weight: 850; animation: softPulse 3.8s ease-in-out infinite;
        }
        .brand strong, .brand small { display:block; line-height:1.15; }
        .brand small { margin-top: 2px; color: var(--muted); font-size: .8rem; }
        div[data-testid="stRadio"] > label { display: none; }
        div[role="radiogroup"] {
            display: flex !important; gap: 6px; padding: 6px; border: 1px solid var(--line); border-radius: 8px;
            background: rgba(255,255,255,.82); box-shadow: 0 10px 30px rgba(17,28,47,.07); backdrop-filter: blur(14px);
            width: fit-content; margin-left: auto; margin-top: -86px; margin-bottom: 38px;
        }
        div[role="radiogroup"] label {
            min-width: 92px; min-height: 42px; justify-content: center; border-radius: 6px;
            color: var(--muted); font-weight: 800; transition: background .18s ease, color .18s ease;
        }
        div[role="radiogroup"] label:has(input:checked) { color: var(--blue-dark); background: var(--surface-strong); }
        .hero-grid {
            display: grid; grid-template-columns: minmax(0,1fr) minmax(360px,470px); align-items: center;
            gap: 36px; min-height: 610px;
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
        .upload-panel, .glass-card, .result-summary, .image-panel, .health-card, .research-card {
            border: 1px solid var(--line); border-radius: 8px; background: var(--surface); box-shadow: var(--shadow); backdrop-filter: blur(16px);
        }
        .upload-panel { position:relative; overflow:hidden; padding:22px; animation:cardFloatIn .72s .16s ease both; }
        .upload-panel:before { position:absolute; inset:0 0 auto; height:4px; content:""; background:linear-gradient(90deg,var(--blue),var(--cyan),var(--mint)); }
        .panel-heading { display:flex; justify-content:space-between; gap:16px; align-items:flex-start; margin-bottom:18px; }
        .panel-heading h2 { margin:0; font-size:1.45rem; line-height:1.15; }
        .secure-badge { min-height:36px; padding:8px 12px; border:1px solid rgba(18,166,122,.22); border-radius:6px; color:#0c7556; background:rgba(18,166,122,.08); font-weight:800; }
        .dropzone-note { color: var(--muted); font-size: .94rem; margin-top: -8px; margin-bottom: 12px; }
        .dropzone-copy {
            display:grid; place-items:center; min-height:150px; padding:22px 18px; margin-bottom:12px;
            border:2px dashed #a9c7e9; border-radius:8px; background:linear-gradient(180deg,rgba(239,248,255,.82),rgba(255,255,255,.95));
            text-align:center; transition:border-color .18s ease, background .18s ease, transform .18s ease, box-shadow .18s ease;
        }
        .dropzone-copy:hover { border-color:var(--blue); background:#eef6ff; transform:translateY(-2px); box-shadow:inset 0 0 0 1px rgba(20,103,232,.08), 0 18px 34px rgba(20,103,232,.12); }
        .drop-icon {
            display:grid; place-items:center; width:76px; height:76px; margin-bottom:14px; border-radius:8px; color:var(--blue);
            background:white; box-shadow:0 12px 28px rgba(23,105,224,.16); animation:iconFloat 3.4s ease-in-out infinite;
        }
        .drop-icon svg, .health-icon svg, .warning-icon svg, .footer-email svg { width:26px; height:26px; fill:currentColor; }
        .drop-title { display:block; width:100%; color:var(--text); font-size:1.14rem; font-weight:850; }
        .drop-note { display:block; margin-top:8px; color:var(--muted); font-size:.94rem; }
        div[data-testid="stTextInput"] input {
            min-height: 46px; border-radius: 6px; border-color: var(--line);
            background: #fafcff; font-weight: 600;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: var(--blue); box-shadow: 0 0 0 2px rgba(20,103,232,.12);
        }
        div[data-testid="stFileUploader"] {
            padding: 12px; border: 1px solid var(--line); border-radius: 8px;
            background: #fafcff;
        }
        div[data-testid="stFileUploader"] section { border: 0; background: transparent; }
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
            min-height:46px; border-radius:6px; font-weight:850; transition:transform .18s ease, box-shadow .18s ease;
        }
        .stButton button:hover, .stDownloadButton button:hover { transform:translateY(-1px); box-shadow:0 14px 30px rgba(23,105,224,.18); }
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
            .hero-grid, .image-grid, .result-grid, .awareness-hero, .emergency-section { grid-template-columns:1fr; min-height:auto; }
            .workflow-band, .health-card-grid, .footer-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
            .metric-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
            .medical-illustration { min-height:220px; }
        }
        @media (max-width: 720px) {
            .block-container { width:min(100% - 22px,1180px); padding-top:14px; }
            .topbar { align-items:stretch; flex-direction:column; margin-bottom:14px; }
            div[role="radiogroup"] { display:grid !important; grid-template-columns:repeat(2,minmax(0,1fr)); width:100%; margin:0 0 24px; }
            div[role="radiogroup"] label { min-width:0; }
            .hero-grid { gap:24px; }
            .hero-title .main { font-size:2.4rem; margin-bottom:8px; }
            .hero-title .sub { font-size:1.78rem; }
            .signal-board { display:none; }
            .status-row span { width:100%; }
            .workflow-band, .health-card-grid, .warning-grid, .footer-grid, .metric-grid, .report-tools { grid-template-columns:1fr; }
            .awareness-hero, .emergency-section, .medical-disclaimer-card { padding:16px; }
            .section-heading { align-items:stretch; flex-direction:column; }
            .infographic-card { justify-content:flex-start; }
            .medical-disclaimer-card { grid-template-columns:1fr; }
            .footer-brand { padding:22px 18px; }
            .footer-grid { padding:18px; }
            .footer-bottom { flex-direction:column; }
        }
        @media print {
            div[role="radiogroup"], .topbar, .footer, .stButton, .stDownloadButton { display:none !important; }
            .block-container { width:100%; max-width:none; padding:0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading FedDG-PneuNet model artifacts...")
def get_artifacts():
    return load_artifacts(MODEL_DIR)


def html_topbar() -> None:
    st.markdown(
        """
        <div class="topbar">
            <div class="brand">
                <span class="brand-mark">FP</span>
                <span><strong>FedDG-PneuNet</strong><small>Medical Graph AI</small></span>
            </div>
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
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return False, "Only JPG, JPEG, and PNG images are allowed."
    if uploaded_file.size <= 0 or uploaded_file.size > MAX_UPLOAD_BYTES:
        return False, "Image size must be greater than 0 and no larger than 8 MB."
    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
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
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def validate_chest_xray(uploaded_file) -> tuple[bool, str]:
    """
    Heuristic chest X-ray gate using grayscale similarity, color cues, and texture.

    Natural photos, cartoons, and colorful scenes fail before model inference runs.
    """
    rgb = _load_rgb_array(uploaded_file)
    if rgb is None:
        return False, INVALID_XRAY_MESSAGE

    rgb = _resize_for_validation(rgb)
    red = rgb[:, :, 0].astype(np.float32)
    green = rgb[:, :, 1].astype(np.float32)
    blue = rgb[:, :, 2].astype(np.float32)

    # Grayscale similarity: real X-rays keep R/G/B channels nearly aligned.
    channel_mean_diff = float(
        np.mean(np.abs(red - green) + np.abs(red - blue) + np.abs(green - blue)) / 3.0
    )
    if channel_mean_diff > 12.0:
        return False, INVALID_XRAY_MESSAGE

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_from_mean = (red + green + blue) / 3.0
    grayscale_similarity = float(np.mean(np.abs(gray.astype(np.float32) - gray_from_mean)))
    if grayscale_similarity > 8.0:
        return False, INVALID_XRAY_MESSAGE

    # Color rejection: selfies, landscapes, and cartoons carry higher saturation/colorfulness.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
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
    edge_density = float(np.count_nonzero(edges) / edges.size)
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
    return Path(path).read_bytes()


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
        st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="panel-heading">
                <div><p class="eyebrow">New analysis</p><h2>Upload X-ray</h2></div>
                <span class="secure-badge">Python</span>
            </div>
            <div class="dropzone-copy">
                <span class="drop-icon">
                    <svg viewBox="0 0 24 24" role="img"><path d="M12 3a5 5 0 0 0-5 5v1H6a4 4 0 0 0 0 8h4v-2H6a2 2 0 1 1 0-4h3V8a3 3 0 1 1 6 0v3h3a2 2 0 1 1 0 4h-4v2h4a4 4 0 0 0 0-8h-1V8a5 5 0 0 0-5-5Zm1 9h3l-4-4-4 4h3v7h2v-7Z"/></svg>
                </span>
                <span class="drop-title">Drop chest X-ray image</span>
                <span class="drop-note">JPG, JPEG, or PNG up to 8 MB</span>
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
        uploaded = st.file_uploader("Drop chest X-ray image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded is not None:
            ok, message = validate_upload(uploaded)
            if ok:
                xray_ok, xray_message = validate_chest_xray(uploaded)
                if xray_ok:
                    st.image(uploaded, caption="Selected chest X-ray preview", use_container_width=True)
                else:
                    st.error(xray_message)
            else:
                st.error(message)
        run = st.button("Run prediction", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

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
        with st.spinner("Running FedDG-PneuNet inference, graph generation, Grad-CAM, and report creation..."):
            uploaded.seek(0)
            st.session_state.result = run_prediction(uploaded, get_artifacts())
            st.session_state.patient_name = patient_name_value
            st.session_state.page = "Result"
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
            st.session_state.page = "Home"
            st.rerun()
        return

    result_class = "risk" if result["prediction"].lower() == "pneumonia" else "clear"
    confidence = min(max(float(result["confidence"]), 0.0), 100.0)
    patient_name = escape(str(st.session_state.get("patient_name", "—")))
    st.markdown(
        f"""
        <section class="result-grid">
            <article class="result-summary {result_class}">
                <p class="eyebrow">Patient · {patient_name}</p>
                <h2 class="result-title {result_class}">{patient_name}: {escape(result['prediction'])}</h2>
                <div class="confidence-meter">
                    <div class="meter-header"><span>Confidence</span><strong>{confidence:.2f}%</strong></div>
                    <div class="meter-track"><span class="meter-fill" style="width:{confidence:.2f}%"></span></div>
                </div>
                <dl class="result-metrics">
                    <div><dt>Patient name</dt><dd>{patient_name}</dd></div>
                    <div><dt>Prediction result</dt><dd>{escape(result['prediction'])}</dd></div>
                    <div><dt>Confidence score</dt><dd>{confidence:.2f}%</dd></div>
                    <div><dt>Pneumonia probability</dt><dd>{result['probability'] * 100:.2f}%</dd></div>
                    <div><dt>Decision threshold</dt><dd>{result['threshold']:.2f}</dd></div>
                    <div><dt>Graph nodes</dt><dd>{result['graph_nodes']}</dd></div>
                    <div><dt>Prediction timestamp</dt><dd>{escape(str(result['timestamp']))}</dd></div>
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
                    <div class="metric-card"><span>Normal neighbors</span><strong>{result['normal_neighbors']}</strong></div>
                    <div class="metric-card"><span>Pneumonia neighbors</span><strong>{result['pneumonia_neighbors']}</strong></div>
                    <div class="metric-card"><span>References used</span><strong>{result['neighbors']}</strong></div>
                    <div class="metric-card"><span>Preprocessing</span><strong>{escape(str(result['normalization']))}</strong></div>
                </div>
            </article>
        </section>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="image-panel"><h3>Uploaded X-ray</h3>', unsafe_allow_html=True)
        st.image(result["image_path"], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="image-panel"><h3>Grad-CAM</h3>', unsafe_allow_html=True)
        if result.get("heatmap_path"):
            st.image(result["heatmap_path"], use_container_width=True)
        else:
            st.warning("Grad-CAM was unavailable for this run.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.download_button("Download PDF Report", read_bytes(result["report_path"]), file_name=Path(result["report_path"]).name, mime="application/pdf", use_container_width=True)
    with b2:
        st.download_button("Save Prediction", read_bytes(result["saved_prediction_path"]), file_name=Path(result["saved_prediction_path"]).name, mime="application/json", use_container_width=True)
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


def main() -> None:
    css()
    html_topbar()
    pages = ["Home", "Result", "Research", "Awareness"]
    current = st.session_state.get("page", "Home")
    page = st.radio("Navigation", pages, index=pages.index(current), horizontal=True, label_visibility="collapsed")
    st.session_state.page = page
    if page == "Home":
        page_home()
    elif page == "Result":
        page_result()
    elif page == "Research":
        page_research()
    else:
        page_awareness()
    footer()


if __name__ == "__main__":
    main()
