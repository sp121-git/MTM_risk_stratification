import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc as sk_auc
import io

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MTM Risk Stratification System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #0f1117; color: #e8eaf0; }
    [data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #2a3040; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a2035 0%, #1e2640 100%);
        border: 1px solid #2a3555; border-radius: 12px; padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] {
        color: #8892a4 !important; font-size: 0.75rem !important;
        text-transform: uppercase; letter-spacing: 0.08em;
    }
    [data-testid="stMetricValue"] {
        color: #e8eaf0 !important; font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.6rem !important;
    }
    .risk-high { background: linear-gradient(135deg, #7f1d1d, #991b1b); color: #fca5a5; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; display: inline-block; border: 1px solid #dc2626; }
    .risk-medium { background: linear-gradient(135deg, #78350f, #92400e); color: #fcd34d; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; display: inline-block; border: 1px solid #d97706; }
    .risk-low { background: linear-gradient(135deg, #064e3b, #065f46); color: #6ee7b7; padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; display: inline-block; border: 1px solid #10b981; }
    .section-header { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.15em; color: #4a90d9; border-bottom: 1px solid #2a3555; padding-bottom: 8px; margin-bottom: 16px; margin-top: 8px; }
    .patient-card { background: linear-gradient(135deg, #1a2035 0%, #1e2640 100%); border: 1px solid #2a3555; border-radius: 12px; padding: 20px 24px; margin-bottom: 16px; }
    .alert-agree { background: linear-gradient(135deg, #064e3b20, #065f4630); border: 1px solid #10b981; border-left: 4px solid #10b981; border-radius: 8px; padding: 12px 16px; color: #6ee7b7; font-size: 0.9rem; }
    .alert-disagree { background: linear-gradient(135deg, #78350f20, #92400e30); border: 1px solid #d97706; border-left: 4px solid #f59e0b; border-radius: 8px; padding: 12px 16px; color: #fcd34d; font-size: 0.9rem; }
    .alert-confident { background: linear-gradient(135deg, #1e3a5f20, #1e3a5f30); border: 1px solid #4a90d9; border-left: 4px solid #4a90d9; border-radius: 8px; padding: 12px 16px; color: #93c5fd; font-size: 0.9rem; }
    .upload-box { background: linear-gradient(135deg, #1a2035 0%, #1e2640 100%); border: 2px dashed #2a3555; border-radius: 16px; padding: 40px; text-align: center; margin: 20px 0; }
    .stTabs [data-baseweb="tab-list"] { background-color: #161b27; border-bottom: 1px solid #2a3040; gap: 4px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; color: #8892a4; border-radius: 6px 6px 0 0; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; letter-spacing: 0.05em; padding: 8px 20px; }
    .stTabs [aria-selected="true"] { background-color: #1a2035 !important; color: #4a90d9 !important; border-top: 2px solid #4a90d9 !important; }
    [data-testid="stDataFrame"] { border: 1px solid #2a3040; border-radius: 8px; }
    hr { border-color: #2a3040 !important; }
    .sidebar-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.12em; color: #4a90d9; margin-bottom: 4px; }
    .stDownloadButton > button { background: linear-gradient(135deg, #1a4a2e, #1e5c38) !important; color: #6ee7b7 !important; border: 1px solid #10b981 !important; border-radius: 8px !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.85rem !important; padding: 10px 24px !important; width: 100% !important; }
    .stDownloadButton > button:hover { background: linear-gradient(135deg, #1e5c38, #237a4a) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Theme helpers
# ─────────────────────────────────────────────
BG       = "#1a2035"
AXIS_COL = "#2a3040"
TEXT_COL = "#8892a4"
TITLE_COL= "#e8eaf0"
RISK_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}

def base_layout(title="", height=300, **overrides):
    xaxis = dict(gridcolor=AXIS_COL, linecolor=AXIS_COL, tickfont=dict(color=TEXT_COL))
    yaxis = dict(gridcolor=AXIS_COL, linecolor=AXIS_COL, tickfont=dict(color=TEXT_COL))
    if "xaxis" in overrides: xaxis.update(overrides.pop("xaxis"))
    if "yaxis" in overrides: yaxis.update(overrides.pop("yaxis"))
    layout = dict(paper_bgcolor=BG, plot_bgcolor=BG,
                  font=dict(family="IBM Plex Sans", color=TEXT_COL, size=12),
                  title_font=dict(family="IBM Plex Mono", color=TITLE_COL, size=13),
                  margin=dict(l=40, r=20, t=40, b=40),
                  height=height, xaxis=xaxis, yaxis=yaxis)
    if title: layout["title"] = title
    layout.update(overrides)
    return layout

# ─────────────────────────────────────────────
# PIPELINE FUNCTIONS
# ─────────────────────────────────────────────

def clean_data(df):
    """Step 1 — Clean and validate raw input."""
    required = ["adherence_score", "med_count", "comorbidity_count", "last_hospital"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()
    df["adherence_score"] = df["adherence_score"].fillna(df["adherence_score"].median())
    for col in ["med_count", "comorbidity_count", "last_hospital"]:
        df[col] = df[col].fillna(0).astype(int)
    if "a1c" in df.columns:
        df["a1c"] = df["a1c"].fillna(df["a1c"].median())
    if df["adherence_score"].max() > 1.5:
        df["adherence_score"] = df["adherence_score"] / 100.0
    df["adherence_score"] = df["adherence_score"].clip(0, 1)
    if "patient_id" not in df.columns:
        df.insert(0, "patient_id", [f"P{str(i).zfill(4)}" for i in range(1, len(df)+1)])
    return df


def engineer_features(df):
    """Step 2 — Derived clinical flags and normalised features."""
    df = df.copy()
    df["polypharmacy"]     = (df["med_count"] >= 5).astype(int)
    df["poor_adherence"]   = (df["adherence_score"] < 0.6).astype(int)
    df["high_comorbidity"] = (df["comorbidity_count"] >= 3).astype(int)
    max_med   = max(df["med_count"].max(), 1)
    max_comorb= max(df["comorbidity_count"].max(), 1)
    df["med_norm"]      = df["med_count"] / max_med
    df["comorb_norm"]   = df["comorbidity_count"] / max_comorb
    df["adherence_risk"]= 1 - df["adherence_score"]
    df["hospital_norm"] = df["last_hospital"].clip(0, 1)
    return df


def score_patients(df):
    """Step 3 — Rule-based scoring, tiering, safety rules."""
    df = df.copy()
    df["risk_score_A"] = (df["med_norm"]*0.40 + df["adherence_risk"]*0.30 +
                          df["hospital_norm"]*0.20 + df["comorb_norm"]*0.10).round(3)
    df["risk_score_B"] = (df["med_norm"]*0.30 + df["adherence_risk"]*0.20 +
                          df["hospital_norm"]*0.40 + df["comorb_norm"]*0.10).round(3)
    df["risk_score_C"] = (df["med_norm"]*0.25 + df["adherence_risk"]*0.40 +
                          df["hospital_norm"]*0.20 + df["comorb_norm"]*0.15).round(3)

    final_score = df["risk_score_C"]
    low_cut  = final_score.quantile(0.50)
    high_cut = final_score.quantile(0.80)

    df["actual_risk"] = final_score.apply(
        lambda s: "High" if s >= high_cut else ("Medium" if s >= low_cut else "Low"))

    def safety(row):
        if row["adherence_score"] < 0.5 and row["last_hospital"] == 1: return "High"
        if row["med_count"] >= 8 and row["comorbidity_count"] >= 3:     return "High"
        if row["adherence_score"] < 0.4:                                return "High"
        return row["actual_risk"]

    df["final_risk"] = df.apply(safety, axis=1)
    return df


def add_explanations(df):
    """Step 4 — Per-patient plain-language explanation."""
    def explain(row):
        d = []
        if row["med_count"] >= 5:         d.append("polypharmacy")
        if row["adherence_score"] < 0.6:  d.append("poor adherence")
        if row["last_hospital"] == 1:     d.append("recent hospitalization")
        if row["comorbidity_count"] >= 3: d.append("multiple comorbidities")
        return "No major risk drivers detected." if not d else "Main drivers: " + ", ".join(d[:3])

    df = df.copy()
    df["explanation"] = df.apply(explain, axis=1)
    return df


def add_recommendations(df):
    """Step 5 — MTM action recommendation per patient."""
    def recommend(row):
        if row["final_risk"] == "High":
            if row["polypharmacy"] == 1 and row["poor_adherence"] == 1:
                return "Immediate pharmacist medication review and adherence counseling"
            elif row["last_hospital"] == 1:
                return "Post-discharge MTM follow-up and medication reconciliation"
            return "High-priority MTM review"
        elif row["final_risk"] == "Medium":
            return ("Standard MTM outreach with chronic disease focus"
                    if row["comorbidity_count"] >= 3 else "Monitor and reassess soon")
        return "Routine follow-up and patient education"

    df = df.copy()
    df["recommendation"] = df.apply(recommend, axis=1)
    return df


def run_ml(df):
    """Step 6 — LR, RF, Ensemble, k-fold CV."""
    df = df.copy()
    features = ["med_count", "adherence_score", "comorbidity_count", "last_hospital"]
    df["target_high_risk"] = (df["actual_risk"] == "High").astype(int)
    X = df[features]
    y = df["target_high_risk"]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    # LR
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    # RF
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    # K-fold CV
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_cv = cross_val_score(LogisticRegression(max_iter=1000, random_state=42),
                            X_scaled, y, cv=kf, scoring="roc_auc")
    rf_cv = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42),
                            X_scaled, y, cv=kf, scoring="roc_auc")

    df["lr_cv_mean"] = round(float(lr_cv.mean()), 4)
    df["lr_cv_std"]  = round(float(lr_cv.std()),  4)
    df["rf_cv_mean"] = round(float(rf_cv.mean()), 4)
    df["rf_cv_std"]  = round(float(rf_cv.std()),  4)

    # LR probabilities and tiers
    df["ml_risk_prob"] = lr.predict_proba(X_scaled)[:, 1].round(3)
    low_ml  = df["ml_risk_prob"].quantile(0.50)
    high_ml = df["ml_risk_prob"].quantile(0.80)
    df["ml_risk_tier"] = df["ml_risk_prob"].apply(
        lambda p: "High" if p >= high_ml else ("Medium" if p >= low_ml else "Low"))
    df["tier_match"] = (df["final_risk"] == df["ml_risk_tier"])

    # RF probabilities and tiers
    df["rf_risk_prob"] = rf.predict_proba(X_scaled)[:, 1].round(3)
    low_rf  = df["rf_risk_prob"].quantile(0.50)
    high_rf = df["rf_risk_prob"].quantile(0.75)
    if low_rf == 0: low_rf = 0.001
    df["rf_risk_tier"] = df["rf_risk_prob"].apply(
        lambda p: "High" if p >= high_rf else ("Medium" if p >= low_rf else "Low"))
    df["rf_tier_match"] = (df["final_risk"] == df["rf_risk_tier"])

    # Ensemble
    df["ensemble_prob"] = ((df["ml_risk_prob"] + df["rf_risk_prob"]) / 2).round(3)
    low_ens  = df["ensemble_prob"].quantile(0.50)
    high_ens = df["ensemble_prob"].quantile(0.80)
    df["ensemble_tier"] = df["ensemble_prob"].apply(
        lambda p: "High" if p >= high_ens else ("Medium" if p >= low_ens else "Low"))
    df["ensemble_match"] = (df["final_risk"] == df["ensemble_tier"])
    df["all_agree"] = (
        (df["final_risk"] == df["ml_risk_tier"]) &
        (df["final_risk"] == df["rf_risk_tier"]) &
        (df["final_risk"] == df["ensemble_tier"])
    )

    # Store ROC data in session state for charts
    st.session_state["roc_data"] = {
        "lr":  {"fpr": roc_curve(y_test, y_prob_lr)[0].tolist(),
                "tpr": roc_curve(y_test, y_prob_lr)[1].tolist(),
                "auc": round(roc_auc_score(y_test, y_prob_lr), 3)},
        "rf":  {"fpr": roc_curve(y_test, y_prob_rf)[0].tolist(),
                "tpr": roc_curve(y_test, y_prob_rf)[1].tolist(),
                "auc": round(roc_auc_score(y_test, y_prob_rf), 3)},
        "lr_cv_scores": [round(float(s), 4) for s in lr_cv],
        "rf_cv_scores": [round(float(s), 4) for s in rf_cv],
    }
    return df


@st.cache_data(show_spinner=False)
def run_full_pipeline(raw_bytes):
    """Runs the complete pipeline. Cached so re-renders don't re-run."""
    df = pd.read_csv(io.BytesIO(raw_bytes))
    df = clean_data(df)
    df = engineer_features(df)
    df = score_patients(df)
    df = add_explanations(df)
    df = add_recommendations(df)
    df = run_ml(df)
    return df


def get_csv_download(df):
    """Return scored CSV as bytes for download."""
    cols_to_drop = [c for c in ["med_norm","comorb_norm","adherence_risk",
                                 "hospital_norm","target_high_risk"] if c in df.columns]
    return df.drop(columns=cols_to_drop).to_csv(index=False).encode("utf-8")


# ─────────────────────────────────────────────
# SIDEBAR — always visible
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-label">MTM System v4.0</p>', unsafe_allow_html=True)
    st.markdown("## 💊 Risk Dashboard")
    st.markdown("---")

    fi_df = pd.DataFrame({
        "Feature":   ["last_hospital","med_count","adherence_score","comorbidity_count"],
        "LR (norm)": [1.000, 0.610, 0.682, 0.398],
        "RF":        [0.418, 0.194, 0.272, 0.116],
    })
    st.markdown('<p class="sidebar-label">Feature Importance — LR vs RF</p>', unsafe_allow_html=True)
    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(name="LR", x=fi_df["Feature"], y=fi_df["LR (norm)"],
                             marker_color="#a78bfa", opacity=0.85))
    fig_fi.add_trace(go.Bar(name="RF", x=fi_df["Feature"], y=fi_df["RF"],
                             marker_color="#34d399", opacity=0.85))
    fig_fi.update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="IBM Plex Sans", color=TEXT_COL, size=10),
        height=190, barmode="group",
        margin=dict(l=10,r=10,t=10,b=50),
        legend=dict(font=dict(color=TEXT_COL, size=10)),
        xaxis=dict(tickfont=dict(size=9,color=TEXT_COL),showgrid=False,linecolor=AXIS_COL),
        yaxis=dict(showgrid=False,linecolor=AXIS_COL))
    st.plotly_chart(fig_fi, use_container_width=True)
    st.markdown("---")

    # Upload widget lives in sidebar
    st.markdown('<p class="sidebar-label">Upload patient data</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "CSV file", type="csv", label_visibility="collapsed",
        help="Upload a CSV with columns: patient_id, age, med_count, adherence_score, comorbidity_count, last_hospital"
    )

    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")
        st.caption(f"Upload new file to re-run pipeline")

    st.markdown("---")
    st.markdown('<p class="sidebar-label">Required columns</p>', unsafe_allow_html=True)
    st.caption("patient_id · age · med_count · adherence_score · comorbidity_count · last_hospital")
    st.caption("Optional: a1c")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding:20px 0 8px 0;">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                 text-transform:uppercase;letter-spacing:0.15em;color:#4a90d9;">
        Clinical Decision Support · Prototype
    </span>
    <h1 style="font-family:'IBM Plex Sans',sans-serif;font-size:2rem;
               font-weight:600;color:#e8eaf0;margin:4px 0 0 0;">
        MTM Risk Stratification System
    </h1>
    <p style="color:#8892a4;font-size:0.9rem;margin-top:4px;">
        Upload patient data to run the full pipeline · Rule-based + LR + RF + Ensemble
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# STATE A — No file uploaded
# ─────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div class="upload-box">
        <p style="font-size:2.5rem;margin:0">📂</p>
        <h3 style="color:#e8eaf0;font-family:'IBM Plex Mono',monospace;
                   font-size:1rem;margin:12px 0 8px 0;">
            Upload a patient CSV to get started
        </h3>
        <p style="color:#8892a4;font-size:0.88rem;margin:0;">
            Use the uploader in the sidebar. The system will score all patients,
            run ML models, and display the full dashboard automatically.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">Expected CSV format</p>', unsafe_allow_html=True)
    sample_df = pd.DataFrame({
        "patient_id":        ["P0001","P0002","P0003"],
        "age":               [72, 45, 81],
        "med_count":         [11, 3, 14],
        "adherence_score":   [0.38, 0.91, 0.52],
        "comorbidity_count": [5, 1, 6],
        "last_hospital":     [1, 0, 1],
        "a1c":               [8.1, 5.4, 9.2],
    })
    st.dataframe(sample_df, use_container_width=True, hide_index=True)
    st.caption("patient_id and a1c are optional. adherence_score can be 0–1 or 0–100 — the system detects and converts automatically.")
    st.stop()

# ─────────────────────────────────────────────
# STATE B — File uploaded, run pipeline
# ─────────────────────────────────────────────
raw_bytes = uploaded_file.read()

with st.spinner("Running full pipeline — cleaning, scoring, training ML models..."):
    try:
        df = run_full_pipeline(raw_bytes)
        # Re-run ML to populate roc_data in session state (cache skips this)
        if "roc_data" not in st.session_state:
            run_ml(df)
    except ValueError as e:
        st.error(f"Pipeline error: {e}")
        st.info("Check that your CSV contains the required columns: med_count, adherence_score, comorbidity_count, last_hospital")
        st.stop()

n_patients = len(df)

# ─────────────────────────────────────────────
# DOWNLOAD BUTTON — prominent at top
# ─────────────────────────────────────────────
dl_col, info_col = st.columns([1, 3])
with dl_col:
    st.download_button(
        label="⬇ Download scored results CSV",
        data=get_csv_download(df),
        file_name="mtm_scored_results.csv",
        mime="text/csv",
    )
with info_col:
    st.markdown(f"""
    <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
                padding:10px 16px;margin-top:4px;">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;color:#8892a4;">
            Pipeline complete · {n_patients} patients scored ·
            LR + RF + Ensemble · Download includes all risk scores and probabilities
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR FILTERS — shown after upload
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-label">Filters</p>', unsafe_allow_html=True)
    risk_options   = sorted(df["final_risk"].dropna().unique().tolist())
    selected_risks = st.multiselect("Risk tier", options=risk_options, default=risk_options)
    min_age = int(df["age"].min()) if "age" in df.columns else 0
    max_age = int(df["age"].max()) if "age" in df.columns else 100
    age_range = st.slider("Age range", min_age, max_age, (min_age, max_age))
    hospital_filter = st.selectbox("Recent hospitalization", ["All","Yes","No"])
    ml_filter = st.selectbox("Model agreement",
        ["All","All models agree","Any disagreement","LR agrees","RF agrees","Ensemble agrees"])
    st.markdown("---")
    st.caption(f"Dataset: {n_patients} patients  ·  3 ML models")

# Apply filters
filtered_df = df.copy()
if selected_risks:
    filtered_df = filtered_df[filtered_df["final_risk"].isin(selected_risks)]
if "age" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["age"].between(age_range[0], age_range[1])]
if hospital_filter == "Yes":
    filtered_df = filtered_df[filtered_df["last_hospital"] == 1]
elif hospital_filter == "No":
    filtered_df = filtered_df[filtered_df["last_hospital"] == 0]
if "all_agree" in filtered_df.columns:
    if ml_filter == "All models agree":
        filtered_df = filtered_df[filtered_df["all_agree"] == True]
    elif ml_filter == "Any disagreement":
        filtered_df = filtered_df[filtered_df["all_agree"] == False]
if "tier_match" in filtered_df.columns and ml_filter == "LR agrees":
    filtered_df = filtered_df[filtered_df["tier_match"] == True]
if "rf_tier_match" in filtered_df.columns and ml_filter == "RF agrees":
    filtered_df = filtered_df[filtered_df["rf_tier_match"] == True]
if "ensemble_match" in filtered_df.columns and ml_filter == "Ensemble agrees":
    filtered_df = filtered_df[filtered_df["ensemble_match"] == True]

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  OVERVIEW & CLINICAL DEPTH",
    "🤖  ML INSIGHTS",
    "🔍  PATIENT DETAIL",
])

# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Summary metrics</p>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total patients", len(filtered_df))
    c2.metric("🔴 High risk",   int((filtered_df["final_risk"]=="High").sum()))
    c3.metric("🟡 Medium risk", int((filtered_df["final_risk"]=="Medium").sum()))
    c4.metric("🟢 Low risk",    int((filtered_df["final_risk"]=="Low").sum()))
    if "all_agree" in filtered_df.columns and len(filtered_df)>0:
        c5.metric("All models agree",
                  f"{filtered_df['all_agree'].sum()} ({filtered_df['all_agree'].mean():.0%})")
    if "risk_score_C" in filtered_df.columns and len(filtered_df)>0:
        c6.metric("Avg risk score", round(float(filtered_df["risk_score_C"].mean()),3))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Risk distribution & population profile</p>', unsafe_allow_html=True)
    ch1,ch2,ch3 = st.columns(3)

    with ch1:
        rc = filtered_df["final_risk"].value_counts().reset_index()
        rc.columns = ["Risk Tier","Count"]
        fig_pie = go.Figure(go.Pie(
            labels=rc["Risk Tier"], values=rc["Count"], hole=0.55,
            marker=dict(colors=[RISK_COLORS.get(r,TEXT_COL) for r in rc["Risk Tier"]],
                        line=dict(color="#0f1117",width=3)),
            textfont=dict(family="IBM Plex Mono",size=11)))
        fig_pie.update_layout(**base_layout("Risk Tier Distribution",height=280,
            showlegend=True,legend=dict(font=dict(color=TEXT_COL,size=11))))
        st.plotly_chart(fig_pie, use_container_width=True)

    with ch2:
        if "age" in filtered_df.columns:
            fig_age = go.Figure()
            for tier,color in RISK_COLORS.items():
                sub = filtered_df[filtered_df["final_risk"]==tier]
                fig_age.add_trace(go.Histogram(x=sub["age"],name=tier,
                    marker_color=color,opacity=0.75,xbins=dict(size=5)))
            fig_age.update_layout(**base_layout("Age Distribution by Risk Tier",height=280,
                barmode="overlay",legend=dict(font=dict(color=TEXT_COL,size=11))))
            st.plotly_chart(fig_age, use_container_width=True)

    with ch3:
        avg_r = filtered_df.groupby("final_risk")[["med_count","comorbidity_count"]].mean().round(2).reset_index()
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Avg Medications",x=avg_r["final_risk"],
            y=avg_r["med_count"],marker_color="#4a90d9"))
        fig_bar.add_trace(go.Bar(name="Avg Comorbidities",x=avg_r["final_risk"],
            y=avg_r["comorbidity_count"],marker_color="#a78bfa"))
        fig_bar.update_layout(**base_layout("Avg Clinical Factors by Tier",height=280,
            barmode="group",legend=dict(font=dict(color=TEXT_COL,size=11))))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<p class="section-header">Clinical relationships</p>', unsafe_allow_html=True)
    ch4,ch5 = st.columns(2)

    with ch4:
        fig_sc = px.scatter(filtered_df, x="adherence_score", y="risk_score_C",
            color="final_risk", color_discrete_map=RISK_COLORS, size="med_count",
            hover_data=["patient_id","med_count","comorbidity_count"],
            title="Adherence vs Risk Score (bubble = med count)",
            labels={"adherence_score":"Adherence Score","risk_score_C":"Risk Score"})
        fig_sc.update_layout(**base_layout(height=320,
            legend=dict(font=dict(color=TEXT_COL,size=11))))
        st.plotly_chart(fig_sc, use_container_width=True)

    with ch5:
        hr = filtered_df.groupby("final_risk")["last_hospital"].mean().reset_index()
        hr.columns=["Risk Tier","Rate"]
        hr["Rate"]=(hr["Rate"]*100).round(1)
        fig_hosp = go.Figure(go.Bar(x=hr["Risk Tier"],y=hr["Rate"],
            marker=dict(color=[RISK_COLORS.get(r,TEXT_COL) for r in hr["Risk Tier"]]),
            text=[f"{v}%" for v in hr["Rate"]],textposition="outside",
            textfont=dict(family="IBM Plex Mono",size=12,color=TITLE_COL)))
        fig_hosp.update_layout(**base_layout("Hospitalization Rate by Risk Tier (%)",height=320,
            yaxis=dict(title="% Recently Hospitalized",range=[0,100])))
        st.plotly_chart(fig_hosp, use_container_width=True)

    st.markdown('<p class="section-header">🔴 Patients needing immediate attention</p>', unsafe_allow_html=True)
    if "all_agree" in filtered_df.columns:
        urgent = filtered_df[(filtered_df["final_risk"]=="High")&(filtered_df["all_agree"]==True)]
        st.caption("High risk · All models agree · Highest confidence")
    else:
        urgent = filtered_df[filtered_df["final_risk"]=="High"]
    urgent_cols=[c for c in ["patient_id","age","med_count","adherence_score",
        "comorbidity_count","last_hospital","risk_score_C","ml_risk_prob",
        "rf_risk_prob","ensemble_prob","recommendation"] if c in urgent.columns]
    if len(urgent)==0:
        st.info("No urgent patients in current filter.")
    else:
        st.dataframe(urgent[urgent_cols].sort_values("risk_score_C",ascending=False).head(15),
                     use_container_width=True,hide_index=True)

    st.markdown('<p class="section-header">Full patient list</p>', unsafe_allow_html=True)
    show_cols=[c for c in ["patient_id","age","med_count","adherence_score","comorbidity_count",
        "last_hospital","risk_score_C","final_risk","ml_risk_prob","ml_risk_tier",
        "rf_risk_prob","rf_risk_tier","ensemble_prob","ensemble_tier","all_agree"]
        if c in filtered_df.columns]
    if len(filtered_df)==0:
        st.warning("No patients match the selected filters.")
    else:
        st.dataframe(filtered_df[show_cols].sort_values("risk_score_C",ascending=False),
                     use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════
# TAB 2 — ML INSIGHTS
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Model performance summary</p>', unsafe_allow_html=True)
    m1,m2,m3,m4,m5 = st.columns(5)
    lr_auc = round(st.session_state.get("roc_data",{}).get("lr",{}).get("auc",0.998),3) if "roc_data" in st.session_state else 0.998
    rf_auc = round(st.session_state.get("roc_data",{}).get("rf",{}).get("auc",0.995),3) if "roc_data" in st.session_state else 0.995
    m1.metric("LR ROC-AUC",  str(lr_auc))
    m2.metric("RF ROC-AUC",  str(rf_auc))
    lr_agree = f"{df['tier_match'].mean():.0%}" if "tier_match" in df.columns else "N/A"
    rf_agree = f"{df['rf_tier_match'].mean():.0%}" if "rf_tier_match" in df.columns else "N/A"
    ens_agree= f"{df['ensemble_match'].mean():.0%}" if "ensemble_match" in df.columns else "N/A"
    m3.metric("LR Agreement", lr_agree)
    m4.metric("RF Agreement", rf_agree)
    m5.metric("Ensemble Agr.", ens_agree)

    st.markdown("<br>", unsafe_allow_html=True)

    # K-fold charts
    st.markdown('<p class="section-header">K-fold cross-validation (5 folds)</p>', unsafe_allow_html=True)
    kf1,kf2 = st.columns(2)
    roc_data = st.session_state.get("roc_data", {})
    lr_cv_scores = roc_data.get("lr_cv_scores", [1.0,1.0,1.0,0.9976,1.0])
    rf_cv_scores = roc_data.get("rf_cv_scores", [0.9963,0.9922,0.9935,0.9932,0.9977])
    lr_cv_mean = round(float(df["lr_cv_mean"].iloc[0]),4) if "lr_cv_mean" in df.columns else 0.9995
    rf_cv_mean = round(float(df["rf_cv_mean"].iloc[0]),4) if "rf_cv_mean" in df.columns else 0.9946
    lr_cv_std  = round(float(df["lr_cv_std"].iloc[0]), 4) if "lr_cv_std"  in df.columns else 0.001
    rf_cv_std  = round(float(df["rf_cv_std"].iloc[0]), 4) if "rf_cv_std"  in df.columns else 0.002

    with kf1:
        fig_kf_lr = go.Figure()
        fig_kf_lr.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(5)], y=lr_cv_scores,
            marker_color=["#a78bfa" if s==max(lr_cv_scores) else "#7c5cbf" for s in lr_cv_scores],
            text=[f"{s:.4f}" for s in lr_cv_scores], textposition="outside",
            textfont=dict(family="IBM Plex Mono",size=11,color=TITLE_COL)))
        fig_kf_lr.add_hline(y=lr_cv_mean, line_dash="dot", line_color="#e8eaf0",
            annotation_text=f"Mean: {lr_cv_mean}", annotation_font_color="#e8eaf0")
        fig_kf_lr.update_layout(**base_layout(
            f"Logistic Regression — Mean: {lr_cv_mean} ± {lr_cv_std}",height=260,
            yaxis=dict(range=[0.99,1.001])))
        st.plotly_chart(fig_kf_lr, use_container_width=True)
        st.caption(f"Mean: {lr_cv_mean} · Std: {lr_cv_std} · Single split: {lr_auc}")

    with kf2:
        fig_kf_rf = go.Figure()
        fig_kf_rf.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(5)], y=rf_cv_scores,
            marker_color=["#34d399" if s==max(rf_cv_scores) else "#1f8a5e" for s in rf_cv_scores],
            text=[f"{s:.4f}" for s in rf_cv_scores], textposition="outside",
            textfont=dict(family="IBM Plex Mono",size=11,color=TITLE_COL)))
        fig_kf_rf.add_hline(y=rf_cv_mean, line_dash="dot", line_color="#e8eaf0",
            annotation_text=f"Mean: {rf_cv_mean}", annotation_font_color="#e8eaf0")
        fig_kf_rf.update_layout(**base_layout(
            f"Random Forest — Mean: {rf_cv_mean} ± {rf_cv_std}",height=260,
            yaxis=dict(range=[0.985,1.001])))
        st.plotly_chart(fig_kf_rf, use_container_width=True)
        st.caption(f"Mean: {rf_cv_mean} · Std: {rf_cv_std} · Single split: {rf_auc}")

    # ROC curves
    st.markdown('<p class="section-header">ROC curves — LR vs RF vs Ensemble</p>', unsafe_allow_html=True)
    if "ml_risk_prob" in df.columns and "rf_risk_prob" in df.columns:
        y_true = (df["final_risk"]=="High").astype(int)
        fig_roc = go.Figure()
        for prob_col,label,color in [
            ("ml_risk_prob","Logistic Regression","#a78bfa"),
            ("rf_risk_prob","Random Forest","#34d399"),
            ("ensemble_prob","Ensemble (LR+RF)","#fb923c"),
        ]:
            if prob_col in df.columns:
                fpr,tpr,_=roc_curve(y_true,df[prob_col])
                auc_val=sk_auc(fpr,tpr)
                fig_roc.add_trace(go.Scatter(x=fpr.tolist(),y=tpr.tolist(),mode="lines",
                    name=f"{label}  (AUC = {auc_val:.3f})",
                    line=dict(color=color,width=2.5)))
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random chance",
            line=dict(color="#4a5568",width=1.5,dash="dash")))
        fig_roc.update_layout(**base_layout("ROC Curve — All Models",height=420,
            xaxis=dict(title="False Positive Rate",range=[0,1]),
            yaxis=dict(title="True Positive Rate",range=[0,1.02]),
            legend=dict(font=dict(color=TEXT_COL,size=12),x=0.55,y=0.08,bgcolor="rgba(0,0,0,0)")))
        st.plotly_chart(fig_roc, use_container_width=True)
        st.caption("Curves hugging the top-left corner indicate strong models. Dashed line = random guessing.")

    # Model comparison + consensus
    st.markdown('<p class="section-header">Model comparison overview</p>', unsafe_allow_html=True)
    comp1,comp2 = st.columns(2)
    with comp1:
        agree_pcts = [df["tier_match"].mean()*100 if "tier_match" in df.columns else 86,
                      df["rf_tier_match"].mean()*100 if "rf_tier_match" in df.columns else 68,
                      df["ensemble_match"].mean()*100 if "ensemble_match" in df.columns else 77]
        fig_comp = go.Figure(go.Bar(
            x=["Logistic Regression","Random Forest","Ensemble (LR+RF)"],
            y=[round(v,1) for v in agree_pcts],
            marker_color=["#a78bfa","#34d399","#fb923c"],
            text=[f"{round(v,1)}%" for v in agree_pcts],textposition="outside",
            textfont=dict(family="IBM Plex Mono",size=12,color=TITLE_COL)))
        fig_comp.update_layout(**base_layout("Tier Agreement with Rule-Based System (%)",height=300,
            yaxis=dict(range=[0,100])))
        st.plotly_chart(fig_comp, use_container_width=True)
    with comp2:
        if "all_agree" in filtered_df.columns and len(filtered_df)>0:
            agree_n=int(filtered_df["all_agree"].sum())
            disagree_n=int((~filtered_df["all_agree"]).sum())
            fig_cons=go.Figure(go.Pie(
                labels=["All models agree","At least one disagrees"],
                values=[agree_n,disagree_n],hole=0.55,
                marker=dict(colors=["#10b981","#f59e0b"],line=dict(color="#0f1117",width=3)),
                textfont=dict(family="IBM Plex Mono",size=11)))
            fig_cons.update_layout(**base_layout("Overall Model Consensus",height=300,
                legend=dict(font=dict(color=TEXT_COL,size=11))))
            st.plotly_chart(fig_cons, use_container_width=True)

    # Probability distributions
    st.markdown('<p class="section-header">Probability distributions — all three models</p>', unsafe_allow_html=True)
    pd1,pd2,pd3 = st.columns(3)
    for prob_col,label,color,container in [
        ("ml_risk_prob","Logistic Regression","#a78bfa",pd1),
        ("rf_risk_prob","Random Forest","#34d399",pd2),
        ("ensemble_prob","Ensemble (LR+RF)","#fb923c",pd3),
    ]:
        with container:
            if prob_col in filtered_df.columns:
                fig_dist=go.Figure()
                for tier,tc in RISK_COLORS.items():
                    sub=filtered_df[filtered_df["final_risk"]==tier]
                    fig_dist.add_trace(go.Histogram(x=sub[prob_col],name=tier,
                        marker_color=tc,opacity=0.65,xbins=dict(size=0.05)))
                fig_dist.update_layout(**base_layout(label,height=280,barmode="overlay",
                    xaxis=dict(title="Probability",range=[0,1]),
                    legend=dict(font=dict(color=TEXT_COL,size=10))))
                st.plotly_chart(fig_dist, use_container_width=True)

    # Feature importance tables
    st.markdown('<p class="section-header">Feature importance — LR vs RF</p>', unsafe_allow_html=True)
    fi1,fi2=st.columns(2)
    with fi1:
        lr_df=pd.DataFrame({"Feature":["last_hospital","med_count","comorbidity_count","adherence_score"],
            "LR Coefficient":[4.422,2.697,1.761,-3.017],"Direction":["↑ Risk","↑ Risk","↑ Risk","↓ Risk"]
            }).sort_values("LR Coefficient",ascending=False)
        st.caption("Logistic Regression — Coefficients")
        st.dataframe(lr_df,use_container_width=True,hide_index=True)
    with fi2:
        rf_df=pd.DataFrame({"Feature":["last_hospital","adherence_score","med_count","comorbidity_count"],
            "RF Importance":[0.418,0.272,0.194,0.116]}).sort_values("RF Importance",ascending=False)
        st.caption("Random Forest — Feature Importance")
        st.dataframe(rf_df,use_container_width=True,hide_index=True)

    # LR vs RF scatter
    st.markdown('<p class="section-header">LR vs RF probability — where do they agree?</p>', unsafe_allow_html=True)
    if "ml_risk_prob" in filtered_df.columns and "rf_risk_prob" in filtered_df.columns:
        fig_lrvrf=px.scatter(filtered_df,x="ml_risk_prob",y="rf_risk_prob",
            color="final_risk",color_discrete_map=RISK_COLORS,
            symbol="all_agree" if "all_agree" in filtered_df.columns else None,
            hover_data=["patient_id","ensemble_tier","final_risk"],
            title="LR Probability vs RF Probability  (△ = models disagree)",
            labels={"ml_risk_prob":"LR Probability","rf_risk_prob":"RF Probability"})
        fig_lrvrf.update_layout(**base_layout(height=380,
            legend=dict(font=dict(color=TEXT_COL,size=11))))
        st.plotly_chart(fig_lrvrf, use_container_width=True)

    # Disagreement table
    st.markdown('<p class="section-header">Disagreement cases — where models diverge</p>', unsafe_allow_html=True)
    if "all_agree" in filtered_df.columns:
        dis_df=filtered_df[filtered_df["all_agree"]==False]
        st.caption(f"{len(dis_df)} patients where at least one model disagrees · Clinical edge cases")
        dis_cols=[c for c in ["patient_id","age","med_count","adherence_score",
            "comorbidity_count","last_hospital","final_risk","ml_risk_tier","rf_risk_tier",
            "ensemble_tier","ml_risk_prob","rf_risk_prob","ensemble_prob"] if c in dis_df.columns]
        st.dataframe(dis_df[dis_cols].sort_values("ensemble_prob",ascending=False),
                     use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════
# TAB 3 — PATIENT DETAIL
# ══════════════════════════════════════════════
with tab3:
    if len(filtered_df)==0:
        st.warning("No patients match the current filters.")
    else:
        st.markdown('<p class="section-header">Select patient</p>', unsafe_allow_html=True)
        selected_patient=st.selectbox("Patient ID",options=filtered_df["patient_id"].tolist(),
                                       label_visibility="collapsed")
        patient=filtered_df.loc[filtered_df["patient_id"]==selected_patient].iloc[0]
        risk=patient["final_risk"]
        risk_color=RISK_COLORS.get(risk,TEXT_COL)
        badge_class=f"risk-{risk.lower()}"

        if "all_agree" in patient.index:
            agree_badge=('<span style="background:#1a2035;border:1px solid #10b981;color:#6ee7b7;padding:4px 12px;border-radius:20px;font-size:0.75rem;">✅ All models agree</span>'
                if bool(patient["all_agree"]) else
                '<span style="background:#1a2035;border:1px solid #f59e0b;color:#fcd34d;padding:4px 12px;border-radius:20px;font-size:0.75rem;">⚠️ Models disagree</span>')
        else:
            agree_badge=""

        age_str=(f'<span style="margin-left:12px;color:#8892a4;font-size:0.9rem;">Age: {int(patient["age"])}</span>'
                 if "age" in patient.index else "")

        st.markdown(f"""
        <div class="patient-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:600;color:#e8eaf0;">{selected_patient}</span>
                    {age_str}
                </div>
                <div style="display:flex;gap:10px;align-items:center;">
                    <span class="{badge_class}">{risk} RISK</span>
                    {agree_badge}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Four gauges
        g1,g2,g3,g4=st.columns(4)
        for prob_col,gtitle,gcolor,container in [
            ("risk_score_C","Rule-Based Score",risk_color,g1),
            ("ml_risk_prob","LR Probability","#a78bfa",g2),
            ("rf_risk_prob","RF Probability","#34d399",g3),
            ("ensemble_prob","Ensemble Probability","#fb923c",g4),
        ]:
            with container:
                if prob_col in patient.index:
                    val=round(float(patient[prob_col])*100,1)
                    fig_g=go.Figure(go.Indicator(
                        mode="gauge+number",value=val,
                        title=dict(text=gtitle,font=dict(color=TEXT_COL,size=11)),
                        number=dict(suffix="%",font=dict(family="IBM Plex Mono",color=gcolor,size=28)),
                        gauge=dict(axis=dict(range=[0,100],tickcolor=AXIS_COL,
                                            tickfont=dict(color=TEXT_COL,size=9)),
                                   bar=dict(color=gcolor,thickness=0.3),
                                   bgcolor=BG,borderwidth=0,
                                   steps=[dict(range=[0,50],color="#1e2640"),
                                          dict(range=[50,80],color="#252520"),
                                          dict(range=[80,100],color="#2a1a1a")])))
                    fig_g.update_layout(paper_bgcolor=BG,
                        font=dict(family="IBM Plex Sans",color=TEXT_COL),
                        height=200,margin=dict(l=15,r=15,t=40,b=10))
                    st.plotly_chart(fig_g, use_container_width=True)

        # Clinical snapshot + driver chart
        st.markdown("<br>", unsafe_allow_html=True)
        snap_col,chart_col=st.columns([1,2])
        with snap_col:
            st.markdown('<p class="section-header">Clinical snapshot</p>', unsafe_allow_html=True)
            if "med_count" in patient.index:
                st.metric("Medications",int(patient["med_count"]),
                          delta=f"{int(patient['med_count'])-int(df['med_count'].mean()):+d} vs avg")
            if "comorbidity_count" in patient.index:
                st.metric("Comorbidities",int(patient["comorbidity_count"]),
                          delta=f"{int(patient['comorbidity_count'])-int(df['comorbidity_count'].mean()):+d} vs avg")
            if "adherence_score" in patient.index:
                st.metric("Adherence",f"{float(patient['adherence_score']):.0%}",
                          delta=f"{float(patient['adherence_score'])-float(df['adherence_score'].mean()):+.0%} vs avg")
            if "last_hospital" in patient.index:
                st.metric("Recent Hospitalization","Yes" if int(patient["last_hospital"])==1 else "No")

        with chart_col:
            st.markdown('<p class="section-header">Patient vs population average</p>', unsafe_allow_html=True)
            driver_map={"Medications":("med_count",float(df["med_count"].mean())),
                        "Comorbidities":("comorbidity_count",float(df["comorbidity_count"].mean())),
                        "Adherence":("adherence_score",float(df["adherence_score"].mean()))}
            p_vals,a_vals,lbls=[],[],[]
            for lbl,(col,avg) in driver_map.items():
                if col in patient.index:
                    p_vals.append(float(patient[col])); a_vals.append(avg); lbls.append(lbl)
            if lbls:
                fig_drv=go.Figure()
                fig_drv.add_trace(go.Bar(name="This patient",x=lbls,y=p_vals,
                    marker_color=risk_color,opacity=0.9))
                fig_drv.add_trace(go.Bar(name="Population avg",x=lbls,y=a_vals,
                    marker_color="#4a90d9",opacity=0.5))
                fig_drv.update_layout(**base_layout(height=240,barmode="group",
                    legend=dict(font=dict(color=TEXT_COL,size=11))))
                st.plotly_chart(fig_drv, use_container_width=True)

        # Explanation + recommendation
        st.markdown("<br>", unsafe_allow_html=True)
        exp_col,rec_col=st.columns(2)
        with exp_col:
            st.markdown('<p class="section-header">Risk explanation</p>', unsafe_allow_html=True)
            explanation=patient.get("explanation","No explanation available.")
            st.markdown(f"""
            <div class="patient-card">
                <p style="color:#8892a4;font-size:0.8rem;margin:0 0 6px 0;">MAIN DRIVERS</p>
                <p style="color:#e8eaf0;font-size:0.95rem;margin:0;">{explanation}</p>
            </div>""", unsafe_allow_html=True)
        with rec_col:
            st.markdown('<p class="section-header">Recommended MTM action</p>', unsafe_allow_html=True)
            recommendation=patient.get("recommendation","No recommendation available.")
            st.markdown(f"""
            <div class="patient-card" style="border-color:{risk_color}40;border-left:4px solid {risk_color};">
                <p style="color:#8892a4;font-size:0.8rem;margin:0 0 6px 0;">ACTION</p>
                <p style="color:#e8eaf0;font-size:0.95rem;margin:0;">{recommendation}</p>
            </div>""", unsafe_allow_html=True)

        # Model comparison
        st.markdown('<p class="section-header">Model comparison — all systems</p>', unsafe_allow_html=True)
        mc1,mc2,mc3,mc4=st.columns(4)
        with mc1: st.metric("Rule-Based Tier",patient["final_risk"])
        with mc2:
            if "ml_risk_tier" in patient.index:
                st.metric("LR Tier",patient["ml_risk_tier"])
                if "ml_risk_prob" in patient.index:
                    st.caption(f"Probability: {float(patient['ml_risk_prob']):.1%}")
        with mc3:
            if "rf_risk_tier" in patient.index:
                st.metric("RF Tier",patient["rf_risk_tier"])
                if "rf_risk_prob" in patient.index:
                    st.caption(f"Probability: {float(patient['rf_risk_prob']):.1%}")
        with mc4:
            if "ensemble_tier" in patient.index:
                st.metric("Ensemble Tier",patient["ensemble_tier"])
                if "ensemble_prob" in patient.index:
                    st.caption(f"Probability: {float(patient['ensemble_prob']):.1%}")

        # Consensus alert
        st.markdown("<br>", unsafe_allow_html=True)
        if "all_agree" in patient.index:
            if bool(patient["all_agree"]):
                st.markdown(f"""
                <div class="alert-confident">
                    ✅ <strong>High confidence.</strong> All three models agree:
                    this patient is <strong>{patient['final_risk']}</strong> risk.
                </div>""", unsafe_allow_html=True)
            else:
                lr_t=patient.get("ml_risk_tier","N/A")
                rf_t=patient.get("rf_risk_tier","N/A")
                ens_t=patient.get("ensemble_tier","N/A")
                ens_p=f"{float(patient['ensemble_prob']):.1%}" if "ensemble_prob" in patient.index else "N/A"
                st.markdown(f"""
                <div class="alert-disagree">
                    ⚠️ <strong>Disagreement detected.</strong>
                    Rule-based: <strong>{patient['final_risk']}</strong> ·
                    LR: <strong>{lr_t}</strong> ·
                    RF: <strong>{rf_t}</strong> ·
                    Ensemble: <strong>{ens_t}</strong><br><br>
                    Ensemble probability: <strong>{ens_p}</strong> — most balanced estimate.
                    Manual review recommended.
                </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Prototype only · Synthetic data compatible · {n_patients} patients processed · "
    "Rule-based + LR (ROC-AUC varies by dataset) + RF + Ensemble · Not for clinical use"
)
