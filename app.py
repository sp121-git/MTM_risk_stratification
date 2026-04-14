import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="MTM Risk Stratification System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS
# -----------------------------
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
    .risk-high {
        background: linear-gradient(135deg, #7f1d1d, #991b1b); color: #fca5a5;
        padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;
        display: inline-block; border: 1px solid #dc2626;
    }
    .risk-medium {
        background: linear-gradient(135deg, #78350f, #92400e); color: #fcd34d;
        padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;
        display: inline-block; border: 1px solid #d97706;
    }
    .risk-low {
        background: linear-gradient(135deg, #064e3b, #065f46); color: #6ee7b7;
        padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;
        display: inline-block; border: 1px solid #10b981;
    }
    .section-header {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
        text-transform: uppercase; letter-spacing: 0.15em; color: #4a90d9;
        border-bottom: 1px solid #2a3555; padding-bottom: 8px;
        margin-bottom: 16px; margin-top: 8px;
    }
    .patient-card {
        background: linear-gradient(135deg, #1a2035 0%, #1e2640 100%);
        border: 1px solid #2a3555; border-radius: 12px;
        padding: 20px 24px; margin-bottom: 16px;
    }
    .alert-agree {
        background: linear-gradient(135deg, #064e3b20, #065f4630);
        border: 1px solid #10b981; border-left: 4px solid #10b981;
        border-radius: 8px; padding: 12px 16px; color: #6ee7b7; font-size: 0.9rem;
    }
    .alert-disagree {
        background: linear-gradient(135deg, #78350f20, #92400e30);
        border: 1px solid #d97706; border-left: 4px solid #f59e0b;
        border-radius: 8px; padding: 12px 16px; color: #fcd34d; font-size: 0.9rem;
    }
    .alert-confident {
        background: linear-gradient(135deg, #1e3a5f20, #1e3a5f30);
        border: 1px solid #4a90d9; border-left: 4px solid #4a90d9;
        border-radius: 8px; padding: 12px 16px; color: #93c5fd; font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #161b27; border-bottom: 1px solid #2a3040; gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent; color: #8892a4;
        border-radius: 6px 6px 0 0; font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem; letter-spacing: 0.05em; padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a2035 !important; color: #4a90d9 !important;
        border-top: 2px solid #4a90d9 !important;
    }
    [data-testid="stDataFrame"] { border: 1px solid #2a3040; border-radius: 8px; }
    hr { border-color: #2a3040 !important; }
    .sidebar-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
        text-transform: uppercase; letter-spacing: 0.12em;
        color: #4a90d9; margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Theme constants and layout helper
# No PLOT_THEME dict — avoids all duplicate-key conflicts
# ---------------------------------------------------------------
BG        = "#1a2035"
AXIS_COL  = "#2a3040"
TEXT_COL  = "#8892a4"
TITLE_COL = "#e8eaf0"

def base_layout(title="", height=300, **overrides):
    """
    Returns a clean layout dict for fig.update_layout().
    xaxis/yaxis overrides are MERGED into defaults, not replaced,
    so there are never duplicate keyword arguments.
    """
    xaxis = dict(gridcolor=AXIS_COL, linecolor=AXIS_COL, tickfont=dict(color=TEXT_COL))
    yaxis = dict(gridcolor=AXIS_COL, linecolor=AXIS_COL, tickfont=dict(color=TEXT_COL))

    if "xaxis" in overrides:
        xaxis.update(overrides.pop("xaxis"))
    if "yaxis" in overrides:
        yaxis.update(overrides.pop("yaxis"))

    layout = dict(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="IBM Plex Sans", color=TEXT_COL, size=12),
        title_font=dict(family="IBM Plex Mono", color=TITLE_COL, size=13),
        margin=dict(l=40, r=20, t=40, b=40),
        height=height,
        xaxis=xaxis,
        yaxis=yaxis,
    )
    if title:
        layout["title"] = title
    layout.update(overrides)
    return layout


RISK_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}

# -----------------------------
# File path
# -----------------------------
DATA_PATH = Path("data/scored_data.csv")


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find file at: {path}")
    return pd.read_csv(path)


try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if "final_risk" not in df.columns:
    if "risk_score_C" in df.columns:
        df["final_risk"] = "Low"
        df.loc[df["risk_score_C"] >= 0.6, "final_risk"] = "Medium"
        df.loc[df["risk_score_C"] >= 0.8, "final_risk"] = "High"
    else:
        st.error("Neither 'final_risk' nor 'risk_score_C' found in the CSV.")
        st.stop()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown('<p class="sidebar-label">MTM System v3.0</p>', unsafe_allow_html=True)
    st.markdown("## 💊 Risk Dashboard")
    st.markdown("---")

    st.markdown('<p class="sidebar-label">Feature Importance — LR vs RF</p>', unsafe_allow_html=True)
    st.caption("LR coefficients (normalised) vs RF importance")

    fi_df = pd.DataFrame({
        "Feature":   ["last_hospital", "med_count", "adherence_score", "comorbidity_count"],
        "LR (norm)": [1.000, 0.610, 0.682, 0.398],
        "RF":        [0.418, 0.194, 0.272, 0.116],
    })
    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(name="LR", x=fi_df["Feature"], y=fi_df["LR (norm)"],
                             marker_color="#a78bfa", opacity=0.85))
    fig_fi.add_trace(go.Bar(name="RF", x=fi_df["Feature"], y=fi_df["RF"],
                             marker_color="#34d399", opacity=0.85))
    fig_fi.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="IBM Plex Sans", color=TEXT_COL, size=10),
        height=190, barmode="group",
        margin=dict(l=10, r=10, t=10, b=50),
        legend=dict(font=dict(color=TEXT_COL, size=10)),
        xaxis=dict(tickfont=dict(size=9, color=TEXT_COL), showgrid=False, linecolor=AXIS_COL),
        yaxis=dict(showgrid=False, linecolor=AXIS_COL),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="sidebar-label">Filters</p>', unsafe_allow_html=True)

    risk_options   = sorted(df["final_risk"].dropna().unique().tolist())
    selected_risks = st.multiselect("Risk tier", options=risk_options, default=risk_options)

    min_age  = int(df["age"].min()) if "age" in df.columns else 0
    max_age  = int(df["age"].max()) if "age" in df.columns else 100
    age_range = st.slider("Age range", min_age, max_age, (min_age, max_age))

    hospital_filter = st.selectbox("Recent hospitalization", ["All", "Yes", "No"])

    ml_filter = st.selectbox(
        "Model agreement",
        ["All", "All models agree", "Any disagreement",
         "LR agrees", "RF agrees", "Ensemble agrees"],
    )

    st.markdown("---")
    st.caption(f"Dataset: {len(df)} patients  ·  3 ML models")

# -----------------------------
# Apply filters
# -----------------------------
filtered_df = df.copy()

if selected_risks:
    filtered_df = filtered_df[filtered_df["final_risk"].isin(selected_risks)]
if "age" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["age"].between(age_range[0], age_range[1])]
if hospital_filter == "Yes" and "last_hospital" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["last_hospital"] == 1]
elif hospital_filter == "No" and "last_hospital" in filtered_df.columns:
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

# -----------------------------
# Page header
# -----------------------------
st.markdown("""
<div style="padding:24px 0 8px 0;">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                 text-transform:uppercase;letter-spacing:0.15em;color:#4a90d9;">
        Clinical Decision Support · Prototype
    </span>
    <h1 style="font-family:'IBM Plex Sans',sans-serif;font-size:2rem;
               font-weight:600;color:#e8eaf0;margin:4px 0 0 0;">
        MTM Risk Stratification System
    </h1>
    <p style="color:#8892a4;font-size:0.9rem;margin-top:4px;">
        Rule-based scoring &nbsp;·&nbsp; Logistic Regression &nbsp;·&nbsp;
        Random Forest &nbsp;·&nbsp; Ensemble
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "📊  OVERVIEW & CLINICAL DEPTH",
    "🤖  ML INSIGHTS",
    "🔍  PATIENT DETAIL",
])

# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW & CLINICAL DEPTH
# ══════════════════════════════════════════════
with tab1:

    st.markdown('<p class="section-header">Summary metrics</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total patients", len(filtered_df))
    c2.metric("🔴 High risk",   int((filtered_df["final_risk"] == "High").sum()))
    c3.metric("🟡 Medium risk", int((filtered_df["final_risk"] == "Medium").sum()))
    c4.metric("🟢 Low risk",    int((filtered_df["final_risk"] == "Low").sum()))
    if "all_agree" in filtered_df.columns and len(filtered_df) > 0:
        agree_pct = filtered_df["all_agree"].mean()
        c5.metric("All models agree",
                  f"{filtered_df['all_agree'].sum()} ({agree_pct:.0%})")
    if "risk_score_C" in filtered_df.columns and len(filtered_df) > 0:
        c6.metric("Avg risk score",
                  round(float(filtered_df["risk_score_C"].mean()), 3))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Risk distribution & population profile</p>',
                unsafe_allow_html=True)
    ch1, ch2, ch3 = st.columns(3)

    with ch1:
        rc = filtered_df["final_risk"].value_counts().reset_index()
        rc.columns = ["Risk Tier", "Count"]
        fig_pie = go.Figure(go.Pie(
            labels=rc["Risk Tier"], values=rc["Count"], hole=0.55,
            marker=dict(
                colors=[RISK_COLORS.get(r, TEXT_COL) for r in rc["Risk Tier"]],
                line=dict(color="#0f1117", width=3),
            ),
            textfont=dict(family="IBM Plex Mono", size=11),
        ))
        fig_pie.update_layout(**base_layout("Risk Tier Distribution", height=280,
            showlegend=True, legend=dict(font=dict(color=TEXT_COL, size=11))))
        st.plotly_chart(fig_pie, use_container_width=True)

    with ch2:
        if "age" in filtered_df.columns:
            fig_age = go.Figure()
            for tier, color in RISK_COLORS.items():
                sub = filtered_df[filtered_df["final_risk"] == tier]
                fig_age.add_trace(go.Histogram(
                    x=sub["age"], name=tier, marker_color=color,
                    opacity=0.75, xbins=dict(size=5),
                ))
            fig_age.update_layout(**base_layout("Age Distribution by Risk Tier", height=280,
                barmode="overlay", legend=dict(font=dict(color=TEXT_COL, size=11))))
            st.plotly_chart(fig_age, use_container_width=True)

    with ch3:
        avg_r = filtered_df.groupby("final_risk")[
            ["med_count", "comorbidity_count"]
        ].mean().round(2).reset_index()
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Avg Medications",
            x=avg_r["final_risk"], y=avg_r["med_count"], marker_color="#4a90d9"))
        fig_bar.add_trace(go.Bar(name="Avg Comorbidities",
            x=avg_r["final_risk"], y=avg_r["comorbidity_count"], marker_color="#a78bfa"))
        fig_bar.update_layout(**base_layout("Avg Clinical Factors by Tier", height=280,
            barmode="group", legend=dict(font=dict(color=TEXT_COL, size=11))))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<p class="section-header">Clinical relationships</p>', unsafe_allow_html=True)
    ch4, ch5 = st.columns(2)

    with ch4:
        fig_sc = px.scatter(
            filtered_df, x="adherence_score", y="risk_score_C",
            color="final_risk", color_discrete_map=RISK_COLORS,
            size="med_count",
            hover_data=["patient_id", "med_count", "comorbidity_count"],
            title="Adherence vs Risk Score (bubble = med count)",
            labels={"adherence_score": "Adherence Score", "risk_score_C": "Risk Score"},
        )
        fig_sc.update_layout(**base_layout(height=320,
            legend=dict(font=dict(color=TEXT_COL, size=11))))
        st.plotly_chart(fig_sc, use_container_width=True)

    with ch5:
        hr = filtered_df.groupby("final_risk")["last_hospital"].mean().reset_index()
        hr.columns = ["Risk Tier", "Rate"]
        hr["Rate"] = (hr["Rate"] * 100).round(1)
        fig_hosp = go.Figure(go.Bar(
            x=hr["Risk Tier"], y=hr["Rate"],
            marker=dict(color=[RISK_COLORS.get(r, TEXT_COL) for r in hr["Risk Tier"]]),
            text=[f"{v}%" for v in hr["Rate"]], textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=12, color=TITLE_COL),
        ))
        fig_hosp.update_layout(**base_layout(
            "Hospitalization Rate by Risk Tier (%)", height=320,
            yaxis=dict(title="% Recently Hospitalized", range=[0, 100])))
        st.plotly_chart(fig_hosp, use_container_width=True)

    # Urgent attention table
    st.markdown('<p class="section-header">🔴 Patients needing immediate attention</p>',
                unsafe_allow_html=True)
    if "all_agree" in filtered_df.columns:
        urgent = filtered_df[
            (filtered_df["final_risk"] == "High") &
            (filtered_df["all_agree"] == True)
        ]
        st.caption("High risk · All models agree · Highest confidence cases")
    else:
        urgent = filtered_df[filtered_df["final_risk"] == "High"]

    urgent_cols = [c for c in [
        "patient_id", "age", "med_count", "adherence_score",
        "comorbidity_count", "last_hospital", "risk_score_C",
        "ml_risk_prob", "rf_risk_prob", "ensemble_prob", "recommendation",
    ] if c in urgent.columns]

    if len(urgent) == 0:
        st.info("No urgent patients in current filter.")
    else:
        st.dataframe(
            urgent[urgent_cols].sort_values("risk_score_C", ascending=False).head(15),
            use_container_width=True, hide_index=True,
        )

    # Full patient table
    st.markdown('<p class="section-header">Full patient list</p>', unsafe_allow_html=True)
    show_cols = [c for c in [
        "patient_id", "age", "med_count", "adherence_score",
        "comorbidity_count", "last_hospital", "risk_score_C", "final_risk",
        "ml_risk_prob", "ml_risk_tier",
        "rf_risk_prob", "rf_risk_tier",
        "ensemble_prob", "ensemble_tier", "all_agree",
    ] if c in filtered_df.columns]

    if len(filtered_df) == 0:
        st.warning("No patients match the selected filters.")
    else:
        st.dataframe(
            filtered_df[show_cols].sort_values("risk_score_C", ascending=False),
            use_container_width=True, hide_index=True,
        )

# ══════════════════════════════════════════════
# TAB 2 — ML INSIGHTS
# ══════════════════════════════════════════════
with tab2:

    st.markdown('<p class="section-header">Model performance summary</p>', unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("LR ROC-AUC",         "0.998")
    m2.metric("RF ROC-AUC",         "0.995")
    m3.metric("LR Tier Agreement",  "86%")
    m4.metric("RF Tier Agreement",  "68%")
    m5.metric("Ensemble Agreement", "77%")


    # ── K-FOLD CROSS-VALIDATION RESULTS ─────────────────────────────────────
    st.markdown('<p class="section-header">K-fold cross-validation results (5 folds)</p>', unsafe_allow_html=True)
    kf1, kf2 = st.columns(2)

    with kf1:
        lr_cv_mean = round(float(df["lr_cv_mean"].iloc[0]), 4) if "lr_cv_mean" in df.columns else 0.9995
        lr_cv_std  = round(float(df["lr_cv_std"].iloc[0]),  4) if "lr_cv_std"  in df.columns else 0.0010
        fold_scores_lr = [1.0, 1.0, 1.0, 0.9976, 1.0]
        fig_kf_lr = go.Figure()
        fig_kf_lr.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(5)],
            y=fold_scores_lr,
            marker_color=["#a78bfa" if s == max(fold_scores_lr) else "#7c5cbf" for s in fold_scores_lr],
            text=[f"{s:.4f}" for s in fold_scores_lr],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11, color=TITLE_COL),
        ))
        fig_kf_lr.add_hline(y=lr_cv_mean, line_dash="dot", line_color="#e8eaf0",
            annotation_text=f"Mean: {lr_cv_mean}", annotation_font_color="#e8eaf0")
        fig_kf_lr.update_layout(**base_layout(
            f"Logistic Regression — Mean: {lr_cv_mean} \u00b1 {lr_cv_std}", height=260,
            yaxis=dict(range=[0.99, 1.001])))
        st.plotly_chart(fig_kf_lr, use_container_width=True)
        st.caption(f"Mean ROC-AUC: {lr_cv_mean} \u00b7 Std dev: {lr_cv_std} \u00b7 Single split: 0.998")

    with kf2:
        rf_cv_mean = round(float(df["rf_cv_mean"].iloc[0]), 4) if "rf_cv_mean" in df.columns else 0.9946
        rf_cv_std  = round(float(df["rf_cv_std"].iloc[0]),  4) if "rf_cv_std"  in df.columns else 0.0021
        fold_scores_rf = [0.9963, 0.9922, 0.9935, 0.9932, 0.9977]
        fig_kf_rf = go.Figure()
        fig_kf_rf.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(5)],
            y=fold_scores_rf,
            marker_color=["#34d399" if s == max(fold_scores_rf) else "#1f8a5e" for s in fold_scores_rf],
            text=[f"{s:.4f}" for s in fold_scores_rf],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11, color=TITLE_COL),
        ))
        fig_kf_rf.add_hline(y=rf_cv_mean, line_dash="dot", line_color="#e8eaf0",
            annotation_text=f"Mean: {rf_cv_mean}", annotation_font_color="#e8eaf0")
        fig_kf_rf.update_layout(**base_layout(
            f"Random Forest — Mean: {rf_cv_mean} \u00b1 {rf_cv_std}", height=260,
            yaxis=dict(range=[0.985, 1.001])))
        st.plotly_chart(fig_kf_rf, use_container_width=True)
        st.caption(f"Mean ROC-AUC: {rf_cv_mean} \u00b7 Std dev: {rf_cv_std} \u00b7 Single split: 0.995")

    # ── ROC CURVES ────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">ROC curves — LR vs RF vs Ensemble</p>', unsafe_allow_html=True)

    if "ml_risk_prob" in df.columns and "rf_risk_prob" in df.columns:
        from sklearn.metrics import roc_curve, auc as sk_auc
        y_true = (df["final_risk"] == "High").astype(int)

        fig_roc = go.Figure()
        for prob_col, label, color in [
            ("ml_risk_prob",  "Logistic Regression", "#a78bfa"),
            ("rf_risk_prob",  "Random Forest",        "#34d399"),
            ("ensemble_prob", "Ensemble (LR+RF)",     "#fb923c"),
        ]:
            if prob_col in df.columns:
                fpr, tpr, _ = roc_curve(y_true, df[prob_col])
                roc_auc_val = sk_auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(
                    x=fpr.tolist(), y=tpr.tolist(), mode="lines",
                    name=f"{label}  (AUC = {roc_auc_val:.3f})",
                    line=dict(color=color, width=2.5),
                ))

        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random chance",
            line=dict(color="#4a5568", width=1.5, dash="dash"),
        ))
        fig_roc.update_layout(**base_layout("ROC Curve — All Models", height=420,
            xaxis=dict(title="False Positive Rate", range=[0, 1]),
            yaxis=dict(title="True Positive Rate", range=[0, 1.02]),
            legend=dict(font=dict(color=TEXT_COL, size=12), x=0.55, y=0.08,
                        bgcolor="rgba(0,0,0,0)")))
        st.plotly_chart(fig_roc, use_container_width=True)
        st.caption(
            "The ROC curve shows the trade-off between catching High-risk patients (sensitivity) "
            "and false alarms (1 - specificity) across all possible thresholds. "
            "Curves hugging the top-left corner indicate strong models. The dashed line = random guessing."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Model comparison overview</p>', unsafe_allow_html=True)
    comp1, comp2 = st.columns(2)

    with comp1:
        fig_comp = go.Figure(go.Bar(
            x=["Logistic Regression", "Random Forest", "Ensemble (LR+RF)"],
            y=[86, 68, 77],
            marker_color=["#a78bfa", "#34d399", "#fb923c"],
            text=["86%", "68%", "77%"], textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=12, color=TITLE_COL),
        ))
        fig_comp.update_layout(**base_layout(
            "Tier Agreement with Rule-Based System (%)", height=300,
            yaxis=dict(range=[0, 100])))
        st.plotly_chart(fig_comp, use_container_width=True)

    with comp2:
        if "all_agree" in filtered_df.columns and len(filtered_df) > 0:
            agree_n    = int(filtered_df["all_agree"].sum())
            disagree_n = int((~filtered_df["all_agree"]).sum())
            fig_cons = go.Figure(go.Pie(
                labels=["All models agree", "At least one disagrees"],
                values=[agree_n, disagree_n], hole=0.55,
                marker=dict(
                    colors=["#10b981", "#f59e0b"],
                    line=dict(color="#0f1117", width=3),
                ),
                textfont=dict(family="IBM Plex Mono", size=11),
            ))
            fig_cons.update_layout(**base_layout("Overall Model Consensus", height=300,
                legend=dict(font=dict(color=TEXT_COL, size=11))))
            st.plotly_chart(fig_cons, use_container_width=True)

    # Probability distributions
    st.markdown('<p class="section-header">Probability distributions — all three models</p>',
                unsafe_allow_html=True)
    pd1, pd2, pd3 = st.columns(3)

    for prob_col, label, color, container in [
        ("ml_risk_prob",  "Logistic Regression", "#a78bfa", pd1),
        ("rf_risk_prob",  "Random Forest",        "#34d399", pd2),
        ("ensemble_prob", "Ensemble (LR+RF)",     "#fb923c", pd3),
    ]:
        with container:
            if prob_col in filtered_df.columns:
                fig_dist = go.Figure()
                for tier, tc in RISK_COLORS.items():
                    sub = filtered_df[filtered_df["final_risk"] == tier]
                    fig_dist.add_trace(go.Histogram(
                        x=sub[prob_col], name=tier, marker_color=tc,
                        opacity=0.65, xbins=dict(size=0.05),
                    ))
                fig_dist.update_layout(**base_layout(label, height=280,
                    barmode="overlay",
                    xaxis=dict(title="Probability", range=[0, 1]),
                    legend=dict(font=dict(color=TEXT_COL, size=10))))
                st.plotly_chart(fig_dist, use_container_width=True)

    # Feature importance
    st.markdown('<p class="section-header">Feature importance — LR coefficients vs RF importance</p>',
                unsafe_allow_html=True)
    fi1, fi2 = st.columns(2)

    with fi1:
        lr_coef_df = pd.DataFrame({
            "Feature":        ["last_hospital", "med_count", "comorbidity_count", "adherence_score"],
            "LR Coefficient": [4.422, 2.697, 1.761, -3.017],
            "Direction":      ["↑ Risk", "↑ Risk", "↑ Risk", "↓ Risk"],
        }).sort_values("LR Coefficient", ascending=False)
        st.caption("Logistic Regression — Coefficients")
        st.dataframe(lr_coef_df, use_container_width=True, hide_index=True)

    with fi2:
        rf_imp_df = pd.DataFrame({
            "Feature":       ["last_hospital", "adherence_score", "med_count", "comorbidity_count"],
            "RF Importance": [0.418, 0.272, 0.194, 0.116],
        }).sort_values("RF Importance", ascending=False)
        st.caption("Random Forest — Feature Importance")
        st.dataframe(rf_imp_df, use_container_width=True, hide_index=True)

    # LR vs RF scatter
    st.markdown('<p class="section-header">LR vs RF probability — where do they agree?</p>',
                unsafe_allow_html=True)
    if "ml_risk_prob" in filtered_df.columns and "rf_risk_prob" in filtered_df.columns:
        fig_lrvrf = px.scatter(
            filtered_df, x="ml_risk_prob", y="rf_risk_prob",
            color="final_risk", color_discrete_map=RISK_COLORS,
            symbol="all_agree" if "all_agree" in filtered_df.columns else None,
            hover_data=["patient_id", "ensemble_tier", "final_risk"],
            title="LR Probability vs RF Probability  (△ = models disagree)",
            labels={"ml_risk_prob": "LR Probability", "rf_risk_prob": "RF Probability"},
        )
        fig_lrvrf.update_layout(**base_layout(height=380,
            legend=dict(font=dict(color=TEXT_COL, size=11))))
        st.plotly_chart(fig_lrvrf, use_container_width=True)

    # Disagreement table
    st.markdown('<p class="section-header">Disagreement cases — where models diverge</p>',
                unsafe_allow_html=True)
    if "all_agree" in filtered_df.columns:
        dis_df = filtered_df[filtered_df["all_agree"] == False]
        st.caption(
            f"{len(dis_df)} patients where at least one model disagrees with "
            "rule-based · Clinical edge cases"
        )
        dis_cols = [c for c in [
            "patient_id", "age", "med_count", "adherence_score",
            "comorbidity_count", "last_hospital", "final_risk",
            "ml_risk_tier", "rf_risk_tier", "ensemble_tier",
            "ml_risk_prob", "rf_risk_prob", "ensemble_prob",
        ] if c in dis_df.columns]
        st.dataframe(
            dis_df[dis_cols].sort_values("ensemble_prob", ascending=False),
            use_container_width=True, hide_index=True,
        )

# ══════════════════════════════════════════════
# TAB 3 — PATIENT DETAIL
# ══════════════════════════════════════════════
with tab3:

    if len(filtered_df) == 0:
        st.warning("No patients match the current filters.")
    else:
        st.markdown('<p class="section-header">Select patient</p>', unsafe_allow_html=True)
        selected_patient = st.selectbox(
            "Patient ID", options=filtered_df["patient_id"].tolist(),
            label_visibility="collapsed",
        )
        patient     = filtered_df.loc[filtered_df["patient_id"] == selected_patient].iloc[0]
        risk        = patient["final_risk"]
        risk_color  = RISK_COLORS.get(risk, TEXT_COL)
        badge_class = f"risk-{risk.lower()}"

        # Agreement badge
        if "all_agree" in patient.index:
            if bool(patient["all_agree"]):
                agree_badge = (
                    '<span style="background:#1a2035;border:1px solid #10b981;'
                    'color:#6ee7b7;padding:4px 12px;border-radius:20px;'
                    'font-size:0.75rem;">✅ All models agree</span>'
                )
            else:
                agree_badge = (
                    '<span style="background:#1a2035;border:1px solid #f59e0b;'
                    'color:#fcd34d;padding:4px 12px;border-radius:20px;'
                    'font-size:0.75rem;">⚠️ Models disagree</span>'
                )
        else:
            agree_badge = ""

        age_str = (
            f'<span style="margin-left:12px;color:#8892a4;font-size:0.9rem;">'
            f'Age: {int(patient["age"])}</span>'
            if "age" in patient.index else ""
        )

        st.markdown(f"""
        <div class="patient-card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;
                                 font-weight:600;color:#e8eaf0;">{selected_patient}</span>
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
        g1, g2, g3, g4 = st.columns(4)
        gauge_configs = [
            ("risk_score_C",  "Rule-Based Score",    risk_color, g1),
            ("ml_risk_prob",  "LR Probability",      "#a78bfa",  g2),
            ("rf_risk_prob",  "RF Probability",      "#34d399",  g3),
            ("ensemble_prob", "Ensemble Probability","#fb923c",  g4),
        ]
        for prob_col, gtitle, gcolor, container in gauge_configs:
            with container:
                if prob_col in patient.index:
                    val = round(float(patient[prob_col]) * 100, 1)
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number", value=val,
                        title=dict(text=gtitle, font=dict(color=TEXT_COL, size=11)),
                        number=dict(
                            suffix="%",
                            font=dict(family="IBM Plex Mono", color=gcolor, size=28),
                        ),
                        gauge=dict(
                            axis=dict(range=[0, 100], tickcolor=AXIS_COL,
                                      tickfont=dict(color=TEXT_COL, size=9)),
                            bar=dict(color=gcolor, thickness=0.3),
                            bgcolor=BG, borderwidth=0,
                            steps=[
                                dict(range=[0,  50], color="#1e2640"),
                                dict(range=[50, 80], color="#252520"),
                                dict(range=[80,100], color="#2a1a1a"),
                            ],
                        ),
                    ))
                    fig_g.update_layout(
                        paper_bgcolor=BG,
                        font=dict(family="IBM Plex Sans", color=TEXT_COL),
                        height=200, margin=dict(l=15, r=15, t=40, b=10),
                    )
                    st.plotly_chart(fig_g, use_container_width=True)

        # Clinical snapshot + driver chart
        st.markdown("<br>", unsafe_allow_html=True)
        snap_col, chart_col = st.columns([1, 2])

        with snap_col:
            st.markdown('<p class="section-header">Clinical snapshot</p>', unsafe_allow_html=True)
            if "med_count" in patient.index:
                st.metric("Medications", int(patient["med_count"]),
                          delta=f"{int(patient['med_count']) - int(df['med_count'].mean()):+d} vs avg")
            if "comorbidity_count" in patient.index:
                st.metric("Comorbidities", int(patient["comorbidity_count"]),
                          delta=f"{int(patient['comorbidity_count']) - int(df['comorbidity_count'].mean()):+d} vs avg")
            if "adherence_score" in patient.index:
                st.metric("Adherence", f"{float(patient['adherence_score']):.0%}",
                          delta=f"{float(patient['adherence_score']) - float(df['adherence_score'].mean()):+.0%} vs avg")
            if "last_hospital" in patient.index:
                st.metric("Recent Hospitalization",
                          "Yes" if int(patient["last_hospital"]) == 1 else "No")

        with chart_col:
            st.markdown('<p class="section-header">Patient vs population average</p>',
                        unsafe_allow_html=True)
            driver_map = {
                "Medications":   ("med_count",         float(df["med_count"].mean())),
                "Comorbidities": ("comorbidity_count",  float(df["comorbidity_count"].mean())),
                "Adherence":     ("adherence_score",    float(df["adherence_score"].mean())),
            }
            p_vals, a_vals, lbls = [], [], []
            for lbl, (col, avg) in driver_map.items():
                if col in patient.index:
                    p_vals.append(float(patient[col]))
                    a_vals.append(avg)
                    lbls.append(lbl)
            if lbls:
                fig_drv = go.Figure()
                fig_drv.add_trace(go.Bar(name="This patient",
                    x=lbls, y=p_vals, marker_color=risk_color, opacity=0.9))
                fig_drv.add_trace(go.Bar(name="Population avg",
                    x=lbls, y=a_vals, marker_color="#4a90d9", opacity=0.5))
                fig_drv.update_layout(**base_layout(height=240,
                    barmode="group", legend=dict(font=dict(color=TEXT_COL, size=11))))
                st.plotly_chart(fig_drv, use_container_width=True)

        # Explanation + recommendation
        st.markdown("<br>", unsafe_allow_html=True)
        exp_col, rec_col = st.columns(2)

        with exp_col:
            st.markdown('<p class="section-header">Risk explanation</p>', unsafe_allow_html=True)
            explanation = patient.get("explanation", "No explanation available.")
            st.markdown(f"""
            <div class="patient-card">
                <p style="color:#8892a4;font-size:0.8rem;margin:0 0 6px 0;">MAIN DRIVERS</p>
                <p style="color:#e8eaf0;font-size:0.95rem;margin:0;">{explanation}</p>
            </div>
            """, unsafe_allow_html=True)

        with rec_col:
            st.markdown('<p class="section-header">Recommended MTM action</p>',
                        unsafe_allow_html=True)
            recommendation = patient.get("recommendation", "No recommendation available.")
            st.markdown(f"""
            <div class="patient-card"
                 style="border-color:{risk_color}40;border-left:4px solid {risk_color};">
                <p style="color:#8892a4;font-size:0.8rem;margin:0 0 6px 0;">ACTION</p>
                <p style="color:#e8eaf0;font-size:0.95rem;margin:0;">{recommendation}</p>
            </div>
            """, unsafe_allow_html=True)

        # Model comparison tiles
        st.markdown('<p class="section-header">Model comparison — all systems</p>',
                    unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Rule-Based Tier", patient["final_risk"])
        with mc2:
            if "ml_risk_tier" in patient.index:
                st.metric("LR Tier", patient["ml_risk_tier"])
                if "ml_risk_prob" in patient.index:
                    st.caption(f"Probability: {float(patient['ml_risk_prob']):.1%}")
        with mc3:
            if "rf_risk_tier" in patient.index:
                st.metric("RF Tier", patient["rf_risk_tier"])
                if "rf_risk_prob" in patient.index:
                    st.caption(f"Probability: {float(patient['rf_risk_prob']):.1%}")
        with mc4:
            if "ensemble_tier" in patient.index:
                st.metric("Ensemble Tier", patient["ensemble_tier"])
                if "ensemble_prob" in patient.index:
                    st.caption(f"Probability: {float(patient['ensemble_prob']):.1%}")

        # Consensus alert
        st.markdown("<br>", unsafe_allow_html=True)
        if "all_agree" in patient.index:
            if bool(patient["all_agree"]):
                st.markdown(f"""
                <div class="alert-confident">
                    ✅ <strong>High confidence classification.</strong>
                    All three models (LR, RF, Ensemble) agree with the rule-based system:
                    this patient is <strong>{patient['final_risk']}</strong> risk.
                    This classification can be acted upon with confidence.
                </div>
                """, unsafe_allow_html=True)
            else:
                lr_t  = patient.get("ml_risk_tier",  "N/A")
                rf_t  = patient.get("rf_risk_tier",  "N/A")
                ens_t = patient.get("ensemble_tier", "N/A")
                ens_p = (f"{float(patient['ensemble_prob']):.1%}"
                         if "ensemble_prob" in patient.index else "N/A")
                st.markdown(f"""
                <div class="alert-disagree">
                    ⚠️ <strong>Model disagreement detected.</strong><br>
                    Rule-based: <strong>{patient['final_risk']}</strong> &nbsp;·&nbsp;
                    LR: <strong>{lr_t}</strong> &nbsp;·&nbsp;
                    RF: <strong>{rf_t}</strong> &nbsp;·&nbsp;
                    Ensemble: <strong>{ens_t}</strong><br><br>
                    Ensemble probability: <strong>{ens_p}</strong> — most balanced estimate.
                    Manual pharmacist review is recommended for this edge case.
                </div>
                """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "Prototype only · Synthetic data · "
    "Rule-based + Logistic Regression (ROC-AUC 0.998) + "
    "Random Forest (ROC-AUC 0.995) + Ensemble · Not for clinical use"
)
