"""
Medication Adherence Dashboard.

Designed following John Maeda's Laws of Simplicity.
Adapted for executive audiences: plain language, business context, narrative flow.

The story: Non-adherence costs $528B/year. We predict who will stop taking
their medication, intervene before they do, and measure the financial return.
Every $1 invested yields $11 in avoided hospitalizations.
"""
import base64
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import time as _time
import uuid as _uuid
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.models.schemas import RiskLevel, InterventionChannel, InterventionStatus


# ── Design System ──────────────────────────────────────────────

COLORS = {
    "ink": "#1a1a1a",
    "body": "#555555",
    "dim": "#888888",
    "faint": "#c8c8c8",
    "border": "#e8e8e4",
    "wash": "#f5f5f0",
    "paper": "#fafaf7",
    "white": "#ffffff",
    "high": "#c0392b",
    "mid": "#e8a838",
    "low": "#27ae60",
    "accent": "#2c3e50",
}

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Helvetica Neue, sans-serif", size=12, color="#1a1a1a"),
    margin=dict(l=0, r=0, t=8, b=0),
    showlegend=False,
)

_AXIS_FONT = dict(color="#1a1a1a", size=11, family="Inter, sans-serif")


def apply_layout(fig, **overrides):
    """Apply base layout then any overrides — avoids duplicate kwarg errors."""
    fig.update_layout(**PLOTLY_BASE)
    if overrides:
        fig.update_layout(**overrides)
    # Force axis colours — silently skip axes that reject font updates
    try:
        fig.update_xaxes(tickfont=_AXIS_FONT, titlefont=_AXIS_FONT)
    except Exception:
        pass
    try:
        fig.update_yaxes(tickfont=_AXIS_FONT, titlefont=_AXIS_FONT)
    except Exception:
        pass
    return fig


# ── Page Config ────────────────────────────────────────────────

st.set_page_config(
    page_title="Medication Adherence",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Password gate ───────────────────────────────────────────────

_PASSWORD = st.secrets.get("APP_PASSWORD", "monkeys")
_logo_path = Path(__file__).parent / "logo-tam.svg"
_logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        .stApp {{ background-color: #fafaf7 !important; }}
        header[data-testid="stHeader"] {{ background-color: #fafaf7 !important; }}
        .block-container {{
            max-width: 400px !important;
            padding-top: 5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }}
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        .stDeployButton {{ display: none; }}
        p, div, span {{ font-family: 'Inter', sans-serif !important; }}
        .stTextInput input {{
            border: 1px solid #e8e8e4 !important;
            border-radius: 6px !important;
            background: #fff !important;
            font-size: 0.9rem !important;
            padding: 0.65rem 1rem !important;
            color: #1a1a1a !important;
        }}
        .stButton > button {{
            background: #1a1a1a !important; color: #fafaf7 !important;
            border: none !important; border-radius: 6px !important;
            font-size: 0.85rem !important; font-weight: 500 !important;
            letter-spacing: 0.04em !important; padding: 0.6rem 1.5rem !important;
            width: 100% !important;
        }}
        .stButton > button:hover {{ opacity: 0.8 !important; }}
    </style>
    <div style="text-align:center; margin-bottom:2.5rem;">
        <img src="data:image/svg+xml;base64,{_logo_b64}"
             style="height:30px; opacity:0.8; margin-bottom:2.5rem; display:block; margin-left:auto; margin-right:auto;"
             alt="The Agile Monkeys"/>
        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.14em;
                    color:#aaa; font-weight:500; margin-bottom:0.6rem; font-family:Inter,sans-serif;">
            Welcome to
        </div>
        <div style="font-size:1.8rem; font-weight:300; color:#1a1a1a;
                    letter-spacing:-0.02em; line-height:1.2; margin-bottom:0.4rem; font-family:Inter,sans-serif;">
            Adherence
        </div>
        <div style="font-size:0.8rem; color:#aaa; font-family:Inter,sans-serif;">
            Predictive intervention engine &mdash; demo
        </div>
    </div>
    """, unsafe_allow_html=True)

    pwd = st.text_input("", type="password", placeholder="Password",
                        label_visibility="collapsed")
    if st.button("Enter →", use_container_width=True):
        if pwd == _PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.markdown("""
    <div style="text-align:center; margin-top:3rem;
                font-size:11px; color:#ccc; letter-spacing:0.04em; font-family:Inter,sans-serif;">
        theagilemonkeys.com
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Styles ─────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    .stApp { background-color: #fafaf7; }
    header[data-testid="stHeader"] { background-color: #fafaf7; }
    .block-container { padding: 3rem 3rem 3rem 3rem; max-width: 1200px; }

    [data-testid="stSidebar"] {
        background-color: #ffffff; border-right: 1px solid #e8e8e4;
    }

    h1 { font-family: 'Inter', sans-serif !important; font-weight: 300 !important;
         font-size: 2.2rem !important; letter-spacing: -0.03em !important;
         color: #1a1a1a !important; margin-bottom: 0.25rem !important; }
    h2, h3 { font-family: 'Inter', sans-serif !important; font-weight: 500 !important;
             font-size: 0.8rem !important; text-transform: uppercase !important;
             letter-spacing: 0.08em !important; color: #888888 !important;
             margin-top: 2rem !important; margin-bottom: 0.75rem !important; }
    p, li, span, div { font-family: 'Inter', sans-serif !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e8e8e4; border-radius: 8px;
        padding: 1.25rem 1.5rem; box-shadow: none;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem !important; text-transform: uppercase;
        letter-spacing: 0.1em; color: #888888 !important; font-weight: 500 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important; font-weight: 300 !important;
        color: #1a1a1a !important; letter-spacing: -0.02em;
    }
    [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

    [data-testid="stDataFrame"] {
        border: 1px solid #e8e8e4; border-radius: 8px; overflow: hidden;
    }

    .stButton > button {
        background: #1a1a1a; color: #fafaf7; border: none; border-radius: 6px;
        font-size: 0.8rem; font-weight: 500; letter-spacing: 0.04em;
        padding: 0.5rem 1.5rem; transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.8; background: #1a1a1a; color: #fafaf7; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0; border-bottom: 1px solid #e8e8e4; background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif; font-size: 0.8rem; font-weight: 500;
        letter-spacing: 0.04em; color: #888888; background: transparent;
        border: none; border-bottom: 2px solid transparent;
        padding: 0.75rem 1.5rem; margin: 0;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #1a1a1a; }
    .stTabs [aria-selected="true"] {
        color: #1a1a1a !important; border-bottom: 2px solid #1a1a1a !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    hr { border: none; border-top: 1px solid #e8e8e4; margin: 2rem 0; }

    [data-baseweb="select"], [data-baseweb="input"] {
        border-radius: 6px !important; font-size: 0.85rem !important;
    }

    /* ── Force dark text on ALL form elements globally ── */

    /* All widget labels */
    .stTextInput label, .stNumberInput label, .stSelectbox label,
    .stSlider label, .stRadio label, .stCheckbox label,
    .stTextArea label, .stMultiSelect label,
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
    .stSlider p, .stNumberInput p {
        color: #1a1a1a !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Number input boxes */
    .stNumberInput input, .stTextInput input, .stTextArea textarea {
        background: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #e0e0da !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
    }
    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #1a1a1a !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Number input stepper buttons */
    .stNumberInput [data-testid="stNumberInputStepDown"],
    .stNumberInput [data-testid="stNumberInputStepUp"] {
        background: #f5f5f0 !important;
        color: #1a1a1a !important;
        border-color: #e0e0da !important;
    }

    /* Slider value label and thumb */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"],
    .stSlider [data-baseweb="slider"] [role="slider"],
    .stSlider div[data-testid] {
        color: #1a1a1a !important;
    }
    .stSlider [aria-valuetext] { color: #1a1a1a !important; }

    /* Selectbox dropdown */
    [data-baseweb="select"] div {
        background: #ffffff !important;
        color: #1a1a1a !important;
    }

    /* Plotly chart axis text &mdash; override SVG fill */
    .js-plotly-plot .plotly .gtitle,
    .js-plotly-plot .plotly .xtitle,
    .js-plotly-plot .plotly .ytitle,
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text,
    .js-plotly-plot .plotly .legend text,
    .js-plotly-plot .plotly text {
        fill: #1a1a1a !important;
        color: #1a1a1a !important;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }

    /* Narrative text */
    .narrative {
        font-family: 'Inter', sans-serif; font-size: 0.95rem; line-height: 1.7;
        color: #555; max-width: 720px; margin-bottom: 1.5rem;
    }
    .narrative strong { color: #1a1a1a; font-weight: 600; }
    .narrative .highlight-red { color: #c0392b; font-weight: 600; }
    .narrative .highlight-green { color: #27ae60; font-weight: 600; }

    /* Equal-height metric cards within the same row */
    [data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        display: flex; flex-direction: column;
    }
    [data-testid="stHorizontalBlock"] > div > [data-testid="stMetric"] {
        flex: 1; height: 100%;
        border: 1px solid #e8e8e4; border-radius: 8px;
        padding: 1rem 1.25rem; background: #fff;
        box-sizing: border-box;
    }
    /* Remove double-border from the ROI HTML card sibling */
    [data-testid="stHorizontalBlock"] > div > .stMarkdown > div > div {
        height: 100%; box-sizing: border-box;
    }

    /* Insight cards */
    .insight-card {
        background: #fff; border: 1px solid #e8e8e4; border-radius: 8px;
        padding: 1.25rem 1.5rem; margin-bottom: 1rem; height: 100%;
        box-sizing: border-box;
    }
    .insight-card .label {
        font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em;
        color: #888; font-weight: 500; margin-bottom: 0.25rem;
    }
    .insight-card .value {
        font-size: 2rem; font-weight: 300; color: #1a1a1a;
        letter-spacing: -0.02em; margin-bottom: 0.25rem;
    }
    .insight-card .context {
        font-size: 0.8rem; color: #888; line-height: 1.5;
    }

    /* How-it-works steps */
    .step-row {
        display: flex; align-items: flex-start; gap: 1rem;
        margin-bottom: 1.25rem; padding-bottom: 1.25rem;
        border-bottom: 1px solid #f0f0ec;
    }
    .step-row:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
    .step-num {
        font-size: 1.5rem; font-weight: 300; color: #c8c8c8;
        min-width: 2rem; line-height: 1;
    }
    .step-content .step-title {
        font-size: 0.85rem; font-weight: 600; color: #1a1a1a; margin-bottom: 0.15rem;
    }
    .step-content .step-desc {
        font-size: 0.8rem; color: #888; line-height: 1.5;
    }

    /* ── Mobile responsive ── */
    /* Logo visibility */
    .tam-logo-mobile { display: none !important; }
    .tam-logo-desktop { display: flex !important; }

    @media (max-width: 768px) {
        .block-container { padding: 1rem 0.75rem 2rem 0.75rem !important; }

        /* Metrics: 2 columns on mobile instead of 4 */
        [data-testid="stHorizontalBlock"] > div {
            min-width: 45% !important;
            flex: 1 1 45% !important;
        }

        /* Buttons: full width */
        .stButton > button { width: 100% !important; }

        /* On mobile: show logo above title, hide desktop top-right logo */
        .tam-logo-mobile { display: block !important; }
        .tam-logo-desktop { display: none !important; }

        /* Queue table: hide less critical columns, shrink padding */
        .queue-table { overflow-x: auto !important; }
        .queue-row-header, .queue-row {
            grid-template-columns: 70px 1fr 55px 70px !important;
        }
        .queue-col-medication, .queue-col-channel { display: none !important; }


        /* Narrative max-width: full width on mobile */
        .narrative { max-width: 100% !important; }
    }
</style>
""", unsafe_allow_html=True)


# ── Data ───────────────────────────────────────────────────────

@st.cache_data
def load_population():
    np.random.seed(42)
    n = 1000
    risk = np.clip(np.random.normal(45, 25, n), 0, 100)
    return pd.DataFrame({
        "id": [f"P{str(i).zfill(4)}" for i in range(n)],
        "age": np.random.randint(25, 85, n),
        "risk": risk,
        "level": ["high" if s >= 70 else "medium" if s >= 30 else "low" for s in risk],
        "pdc": np.clip(np.random.normal(0.7, 0.2, n), 0, 1),
        # Paired so medication always matches condition
        **dict(zip(
            ["condition", "medication"],
            zip(*[np.array([
                ("Diabetes",     "Metformin"),
                ("Hypertension", "Lisinopril"),
                ("Cholesterol",  "Atorvastatin"),
                ("Heart Disease","Amlodipine"),
                ("GERD",         "Omeprazole"),
            ])[i] for i in np.random.randint(0, 5, n)]),
        )),
        "days_gap": np.random.randint(0, 60, n),
        "copay": np.random.uniform(5, 100, n).round(2),
        "trend": np.random.choice(["up", "flat", "down"], n, p=[0.3, 0.5, 0.2]),
    })


@st.cache_data
def load_interventions():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "id": [f"I{str(i).zfill(4)}" for i in range(n)],
        "patient": [f"P{str(np.random.randint(0, 1000)).zfill(4)}" for _ in range(n)],
        "channel": np.random.choice(["SMS", "Email", "Voice", "Push", "Care Mgr"], n, p=[.4, .25, .15, .15, .05]),
        "status": np.random.choice(["Sent", "Delivered", "Responded", "Successful", "Failed"], n, p=[.1, .3, .25, .3, .05]),
        "date": [(datetime.now() - timedelta(days=np.random.randint(0, 30))).date() for _ in range(n)],
        "hours": np.where(np.random.random(n) > 0.3, np.random.exponential(24, n).round(1), None),
    })


@st.cache_data
def load_roi():
    np.random.seed(7)
    n = 12
    weeks = list(range(1, n + 1))
    # S-curve (sigmoid): slow ramp while the model calibrates (weeks 1-3),
    # rapid acceleration as high-risk patients are identified (weeks 4-9),
    # then plateau as the addressable pool stabilises (weeks 10-12).
    # Calibrated so 12-week cumulative savings / cost ≈ 11 : 1.
    # Cost structure: $4,200 base/week (platform + care-manager time) + $80/wk growth.
    # Savings structure: sigmoid scaled so Σsavings ≈ 11 × Σcost.
    cost    = [4200 + i * 80 + np.random.randint(-100, 100) for i in range(n)]
    total_cost_est = sum(4200 + i * 80 for i in range(n))   # ≈ $55,800
    target_savings = 11 * total_cost_est                     # ≈ $613,800
    k, i0 = 0.8, 5   # steepness and inflection point (week 6)
    raw_sig = [1 / (1 + np.exp(-k * (i - i0))) for i in range(n)]
    scale   = target_savings / sum(raw_sig)
    np.random.seed(7)   # reset so noise is consistent with seed above
    savings = [int(r * scale + np.random.randint(-800, 800)) for r in raw_sig]
    # Hard floor at $500/wk (some savings from day 1) and ceiling at $95k/wk
    savings = [max(500, min(s, 95000)) for s in savings]
    cumcost    = np.cumsum(cost).tolist()
    cumsavings = np.cumsum(savings).tolist()
    df = pd.DataFrame({
        "week":       weeks,
        "intervened": [120 + i * 12 + np.random.randint(-15, 15) for i in range(n)],
        "successful": [int((0.28 + i * 0.015) * (120 + i * 12)) for i in range(n)],
        "savings":    savings,
        "cost":       cost,
        "cum_savings": cumsavings,
        "cum_cost":    cumcost,
    })
    df["roi"] = ((df["cum_savings"] - df["cum_cost"]) / df["cum_cost"] * 100).round(1)
    return df


pop = load_population()
intv = load_interventions()
roi = load_roi()

# Derived values used across tabs
total = len(pop)
high = len(pop[pop["level"] == "high"])
med = len(pop[pop["level"] == "medium"])
low = total - high - med
avg_pdc = pop["pdc"].mean()
declining = len(pop[pop["trend"] == "down"])
below_target = len(pop[pop["pdc"] < 0.8])
total_i = len(intv)
succeeded = len(intv[intv["status"] == "Successful"])
responded = len(intv[intv["status"].isin(["Responded", "Successful"])])
latest_roi = roi.iloc[-1]
total_savings = roi["savings"].sum()
total_cost = roi["cost"].sum()


# ── Header ─────────────────────────────────────────────────────

st.markdown(f"""
<div class="tam-header" style="display:flex; align-items:center;
            justify-content:space-between; margin-bottom:1.25rem; gap:1rem;">
    <div>
        <!-- Logo visible only on mobile, above title -->
        <img class="tam-logo-mobile" src="data:image/svg+xml;base64,{_logo_b64}"
             style="display:none; height:20px; opacity:0.7; margin-bottom:0.6rem;"
             alt="The Agile Monkeys"/>
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.2rem;">
            <p style="font-size:11px; text-transform:uppercase; letter-spacing:0.12em;
                      color:#aaa; margin:0; font-weight:500;">
                Predictive Medication Adherence Engine</p>
            <span style="font-size:10px; text-transform:uppercase; letter-spacing:0.08em;
                         background:#1a1a1a; color:#fafaf7; padding:2px 8px; border-radius:3px;
                         font-weight:600;">Demo</span>
        </div>
        <h1 style="margin:0 !important; padding:0 !important;">Adherence</h1>
    </div>
    <!-- Logo visible only on desktop, top-right -->
    <div class="tam-logo-desktop" style="display:flex; flex-direction:column;
                align-items:flex-end; gap:0.2rem; flex-shrink:0;">
        <img src="data:image/svg+xml;base64,{_logo_b64}"
             style="height:24px; opacity:0.7;" alt="The Agile Monkeys"/>
        <span style="font-size:10px; color:#bbb; letter-spacing:0.06em;">
            theagilemonkeys.com</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────

tab_problem, tab_how, tab_return = st.tabs([
    "The Problem", "The Platform", "The Return"
])



# ═══════════════════════════════════════════════════════════════
# TAB 1: THE PROBLEM
# ═══════════════════════════════════════════════════════════════

with tab_problem:

    high_df   = pop[pop["level"] == "high"].sort_values("risk", ascending=False).copy()
    avg_gap   = high_df["days_gap"].mean()
    avg_copay = high_df["copay"].mean()
    cost_inaction = int(len(high_df) * 0.15 * 15000)

    # ── Opening narrative: the $528B problem ──────────────────────────────
    st.markdown(f"""
    <p class="narrative">
        Medication non-adherence costs the U.S. healthcare system
        <strong>$528 billion per year</strong> in avoidable hospitalizations,
        disease progression, and lost pharmacy revenue.
        In this simulation of <strong>{total:,} patients</strong>,
        the engine flags
        <span class="highlight-red">{high} as high risk</span> of stopping their
        medication in the next 30 days &mdash; and <span class="highlight-red">{below_target}</span>
        are already below the 80% adherence target.
    </p>
    """, unsafe_allow_html=True)

    def _kpi(label, value, sub, sub_color="#888"):
        return (
            f"<div style='background:#fff;border:1px solid #e8e8e4;border-radius:8px;"
            f"padding:1rem 1.25rem;flex:1;box-sizing:border-box;"
            f"display:flex;flex-direction:column;justify-content:space-between;'>"
            f"<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;"
            f"color:#888;font-weight:500;line-height:1.3;margin-bottom:0.4rem;'>{label}</div>"
            f"<div style='font-size:1.8rem;font-weight:300;color:#1a1a1a;"
            f"letter-spacing:-0.02em;line-height:1.1;'>{value}</div>"
            f"<div style='font-size:0.75rem;color:{sub_color};margin-top:0.4rem;"
            f"line-height:1.3;'>{sub}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='display:flex;gap:1rem;align-items:stretch;margin-bottom:1rem;'>"
        + _kpi("Cost of inaction",      f"${cost_inaction:,.0f}", "if high-risk go untreated", "#c0392b")
        + _kpi("Simulated patients",    f"{total:,}",             "in this demo population")
        + _kpi("Flagged at risk",       f"{high + med}",          f"{(high+med)/total:.0%} of population", "#c0392b")
        + _kpi("Taking meds regularly", f"{avg_pdc:.0%}",         "80% is the target")
        + _kpi("Trending down",         f"{declining}",           f"{declining/total:.0%} of population", "#c0392b")
        + "</div>",
        unsafe_allow_html=True,
    )

    # ── How it works: 4-step logic ────────────────────────────────────────
    st.markdown("""
    <div style="border-top:1px solid #e8e8e4; margin:1.5rem 0 1rem;"></div>
    <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;
                color:#aaa;font-weight:600;margin-bottom:0.75rem;">What we do about it</div>
    <div style="background:#fff;border:1px solid #e8e8e4;border-radius:8px;padding:1.25rem 1.5rem;">
        <div class="step-row">
            <div class="step-num">1</div>
            <div class="step-content">
                <div class="step-title">Predict</div>
                <div class="step-desc">The model analyzes 50+ variables per patient &mdash; fill history,
                    gaps between refills, copay burden, diagnoses &mdash; and scores each patient
                    0–100 on their likelihood of stopping.</div>
            </div>
        </div>
        <div class="step-row">
            <div class="step-num">2</div>
            <div class="step-content">
                <div class="step-title">Intervene</div>
                <div class="step-desc">High-risk patients receive automated outreach
                    via SMS, email, phone, or chatbot &mdash; personalized to their barrier.
                    Cost issue? We surface discount programs. Forgetfulness? We send reminders.</div>
            </div>
        </div>
        <div class="step-row">
            <div class="step-num">3</div>
            <div class="step-content">
                <div class="step-title">Escalate</div>
                <div class="step-desc">If a patient doesn't respond, the system escalates —
                    from SMS to a phone call to a care manager. No one falls through the cracks.</div>
            </div>
        </div>
        <div class="step-row">
            <div class="step-num">4</div>
            <div class="step-content">
                <div class="step-title">Measure</div>
                <div class="step-desc">Every intervention is tracked. Every $1 invested
                    yields approximately $11 in avoided hospitalizations — consistent with published
                    CVD-management ROI benchmarks.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts: 3-chart row ────────────────────────────────────────────────
    st.markdown("""
    <div style="border-top:1px solid #e8e8e4; margin:1.5rem 0 1rem;"></div>
    <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;
                color:#aaa;font-weight:600;margin-bottom:0.75rem;">Population at a glance</div>
    """, unsafe_allow_html=True)

    _ch1, _ch2 = st.columns([1, 1])

    # Chart A: Donut — risk tier breakdown
    with _ch1:
        st.markdown("""<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.08em;
                    color:#aaa;font-weight:600;margin-bottom:0.5rem;">Risk tier breakdown</div>""",
                    unsafe_allow_html=True)
        fig_donut = go.Figure(go.Pie(
            labels=["High risk", "Moderate", "Low risk"],
            values=[high, med, low],
            hole=0.62,
            marker=dict(colors=[COLORS["high"], COLORS["mid"], COLORS["low"]],
                        line=dict(color="#fff", width=2)),
            textinfo="percent",
            textfont=dict(size=12, family="Inter", color="#fff"),
            hovertemplate="%{label}: %{value} patients<extra></extra>",
            sort=False,
        ))
        fig_donut.add_annotation(
            text=f"<b>{high}</b><br><span style='font-size:10px'>high risk</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS["high"], family="Inter"),
            align="center",
        )
        apply_layout(fig_donut,
            height=220, showlegend=True,
            hoverlabel=dict(bgcolor="#fff", font_color="#1a1a1a", font_size=11,
                            font_family="Inter", bordercolor="#e8e8e4"),
            legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5,
                        font=dict(size=10, color=COLORS["dim"])),
            margin=dict(l=0, r=0, t=8, b=40),
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    # Chart B: PDC Distribution histogram
    with _ch2:
        st.markdown("""<div style="font-size:11px;text-transform:uppercase;letter-spacing:0.08em;
                    color:#aaa;font-weight:600;margin-bottom:0.5rem;">PDC distribution</div>""",
                    unsafe_allow_html=True)
        _bins = np.arange(0, 1.05, 0.1)
        _counts, _edges = np.histogram(pop["pdc"].clip(0, 1), bins=_bins)
        _midpoints = [(_edges[i] + _edges[i+1]) / 2 for i in range(len(_edges)-1)]
        _bar_colors = [COLORS["high"] if m < 0.6 else COLORS["mid"] if m < 0.8 else COLORS["low"]
                       for m in _midpoints]
        fig_hist = go.Figure(go.Bar(
            x=[f"{int(e*100)}%" for e in _edges[:-1]],
            y=_counts,
            marker_color=_bar_colors,
            marker_line_width=0,
            hovertemplate="PDC %{x}: %{y} patients<extra></extra>",
        ))
        fig_hist.add_vline(x=7.5, line_dash="dot", line_color=COLORS["faint"],
                           annotation_text="80% target", annotation_position="top right",
                           annotation_font=dict(size=9, color=COLORS["dim"]))
        apply_layout(fig_hist,
            height=220,
            hoverlabel=dict(bgcolor="#fff", font_color="#1a1a1a", font_size=11,
                            font_family="Inter", bordercolor="#e8e8e4"),
            xaxis=dict(showgrid=False, tickfont=dict(size=9), tickangle=-30),
            yaxis=dict(showgrid=True, gridcolor=COLORS["faint"], gridwidth=0.5,
                       tickfont=dict(size=9), title="Patients"),
            margin=dict(l=40, r=8, t=8, b=48),
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    # Chart C: Risk by condition — full width
    st.markdown(f"""
    <div style="border-top:1px solid #e8e8e4; margin:1.5rem 0 1rem;"></div>
    <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;
                color:#aaa;font-weight:600;margin-bottom:0.25rem;">Average adherence by condition</div>
    <p style="font-size:12px;color:{COLORS['dim']};margin-bottom:0.75rem;">
        Conditions below 80% are where the engine focuses first.
        Red bars indicate the highest intervention priority.
    </p>
    """, unsafe_allow_html=True)

    cond_pdc = pop.groupby("condition")["pdc"].mean().sort_values()
    cond_high = pop[pop["level"] == "high"].groupby("condition").size()
    fig_cond = go.Figure()
    fig_cond.add_trace(go.Bar(
        x=cond_pdc.values, y=cond_pdc.index, orientation="h",
        marker_color=[COLORS["high"] if v < 0.65 else COLORS["mid"] if v < 0.8 else COLORS["low"]
                      for v in cond_pdc.values],
        marker_line_width=0,
        text=[f"{v:.0%}" for v in cond_pdc.values],
        textposition="outside",
        textfont=dict(size=12, family="Inter", color=COLORS["ink"]),
        hovertemplate="%{y}: %{x:.0%} avg adherence<extra></extra>",
    ))
    fig_cond.add_vline(x=0.8, line_dash="dot", line_color=COLORS["faint"],
                       annotation_text="80% target", annotation_position="top right",
                       annotation_font=dict(size=10, color=COLORS["dim"]))
    apply_layout(fig_cond,
        height=200,
        hoverlabel=dict(bgcolor="#fff", font_color="#1a1a1a", font_size=11,
                        font_family="Inter", bordercolor="#e8e8e4"),
        xaxis=dict(visible=False, range=[0, 1.1]),
        yaxis=dict(showgrid=False, tickfont=dict(size=12, color=COLORS["ink"])),
        margin=dict(l=0, r=70, t=8, b=0),
    )
    st.plotly_chart(fig_cond, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""
    <div style="border-top:1px solid #e8e8e4; margin:1.5rem 0 1rem;"></div>
    <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;
                color:#aaa;font-weight:600;margin-bottom:0.5rem;">High-risk patients &mdash; tonight's queue</div>
    <p style="font-size:13px;color:{COLORS['body']};margin-bottom:0.75rem;max-width:600px;">
        {len(high_df)} patients flagged. Average {avg_gap:.0f} days since last refill,
        average ${avg_copay:.0f} copay. The engine prioritises the highest-risk patients for outreach tonight.
    </p>
    """, unsafe_allow_html=True)

    display = high_df[["id","age","risk","condition","medication","pdc","days_gap","copay"]].head(25).copy()
    display.columns = ["Patient","Age","Risk score","Condition","Medication",
                       "Adherence","Days since fill","Copay"]

    def _risk_bar(val):
        pct = max(0, min(100, (val - 50) / 50 * 100))
        return f"background: linear-gradient(90deg, rgba(192,57,43,{pct/100*0.25}) {pct}%, transparent {pct}%)"

    st.dataframe(
        display.style
               .format({"Risk score": "{:.0f}", "Adherence": "{:.0%}", "Copay": "${:.0f}"})
               .map(_risk_bar, subset=["Risk score"]),
        use_container_width=True, height=420, hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════
# TAB 2: HOW IT WORKS
# ═══════════════════════════════════════════════════════════════

with tab_how:

    journey_patient = pop[pop["level"] == "high"].sort_values("risk", ascending=False).iloc[0]
    jp_id   = journey_patient["id"]
    jp_med  = journey_patient["medication"]
    jp_gap  = int(journey_patient["days_gap"])
    jp_pdc  = journey_patient["pdc"]
    jp_cond = journey_patient["condition"]
    jp_age  = int(journey_patient["age"])

    st.markdown(f"""
    <p class="narrative">
        Every night at midnight, the engine scores your entire population silently &mdash;
        no care manager required. It flags who is at risk, picks the right channel,
        and sends outreach automatically. Below is what it found tonight, followed by
        one patient's complete journey from flag to refill.
    </p>
    """, unsafe_allow_html=True)

    # ── Overnight briefing metrics ─────────────────────────────────────────
    _n_scored   = len(pop)
    _n_flagged  = len(pop[pop["level"] == "high"])
    _n_outreach = int(_n_flagged * 0.85)
    _n_new      = int(_n_flagged * 0.12)

    _ob1, _ob2, _ob3, _ob4 = st.columns(4)
    _ob1.metric("Patients scored",    f"{_n_scored:,}",  delta="ran overnight, automatically")
    _ob2.metric("Flagged high risk",  f"{_n_flagged}",   delta=f"+{_n_new} since yesterday")
    _ob3.metric("Outreach triggered", f"{_n_outreach}",  delta="no one had to click anything")
    _ob4.metric("Time to first message", "< 2 min",      delta="from flag to send")

    st.markdown(f"""
    <div style="border-top:1px solid #e8e8e4; margin:1.5rem 0 1rem;"></div>
    <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;
                color:#aaa;font-weight:600;margin-bottom:0.75rem;">Tonight's outreach queue</div>
    """, unsafe_allow_html=True)

    _top_queue = pop[pop["level"] == "high"].sort_values("risk", ascending=False).head(8)

    _why_flagged = [
        "54 days since last fill",
        "3 missed refills this quarter",
        "PDC dropped 18% in 30 days",
        "Cost barrier flagged last visit",
        "Pharmacy switch &mdash; refill gap",
        "New diagnosis, regimen changed",
        "Overdue by 6 weeks",
        "Refill consistency score: low",
    ]

    queue_html = """
    <style>
    .queue-wrap  { display:inline-block; max-width:100%; }
    .queue-table {
        border:1px solid #e8e8e4; border-radius:8px; overflow:hidden;
        background:#fff; border-collapse:collapse; white-space:nowrap;
    }
    .queue-table th {
        font-size:10px; font-weight:600; text-transform:uppercase;
        letter-spacing:0.08em; color:#aaa;
        padding:0.55rem 1rem; background:#f9f9f7;
        border-bottom:1px solid #e8e8e4; text-align:left;
    }
    .queue-table td {
        font-size:12px; color:#555;
        padding:0.6rem 1rem;
        border-bottom:1px solid #f5f5f0;
    }
    .queue-table tr:last-child td { border-bottom:none; }
    </style>
    <div class="queue-wrap">
    <table class="queue-table">
      <thead><tr>
        <th>Patient</th><th>Why flagged</th><th>Risk</th>
        <th>Gap</th><th>Medication</th><th>Channel</th>
      </tr></thead><tbody>
    """
    _ch_options = ["SMS", "SMS", "SMS", "Email", "Email", "Push", "Voice", "SMS"]
    for _qi, (_, row) in enumerate(_top_queue.iterrows()):
        risk_color = COLORS["high"] if row["risk"] >= 70 else COLORS["mid"]
        ch_label   = _ch_options[_qi % len(_ch_options)]
        why        = _why_flagged[_qi % len(_why_flagged)]
        queue_html += (
            f"<tr>"
            f"<td style='font-weight:600;color:{COLORS['ink']};'>{row['id']}</td>"
            f"<td style='color:{COLORS['body']};max-width:220px;white-space:normal;'>{why}</td>"
            f"<td style='font-weight:600;color:{risk_color};'>{row['risk']:.0f}</td>"
            f"<td style='color:{COLORS['body']};'>{int(row['days_gap'])}d</td>"
            f"<td style='color:{COLORS['body']};'>{row['medication']}</td>"
            f"<td><span style='font-size:10px;font-weight:600;text-transform:uppercase;"
            f"letter-spacing:0.06em;background:#f0f0ec;color:{COLORS['body']};"
            f"padding:2px 8px;border-radius:3px;'>{ch_label}</span></td>"
            f"</tr>"
        )
    queue_html += "</tbody></table></div>"
    st.markdown(queue_html, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="border-top:1px solid #e8e8e4; margin:2rem 0 1rem;"></div>
    <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;
                color:#aaa;font-weight:600;margin-bottom:0.75rem;">Follow one patient &mdash; end to end</div>
    <p class="narrative">
        <strong>{jp_id}</strong> &mdash; {jp_age} years old, {jp_cond}, {jp_gap} days without {jp_med}.
        The engine flagged them last night. Walk through what happens next.
    </p>
    """, unsafe_allow_html=True)

    _step_key = "journey_step"
    if _step_key not in st.session_state:
        st.session_state[_step_key] = 0
    step = st.session_state[_step_key]

    _sms_msg = f"Hi, your {jp_med} refill is {jp_gap} days overdue. Reply YES to schedule pickup, or HELP if you need assistance with costs."
    _new_pdc = min(1.0, jp_pdc + 0.18)

    st.markdown(f"""
    <style>
    .ac-item {{
        border: 1px solid #e8e8e4; border-radius: 10px;
        margin-bottom: 0.5rem; overflow: hidden; background: #fff;
    }}
    .ac-item.ac-done   {{ border-color: {COLORS['low']}44; background: #f9fdf9; }}
    .ac-item.ac-active {{ border-color: {COLORS['ink']}; box-shadow: 0 2px 12px rgba(0,0,0,0.07); }}
    .ac-header {{ display:flex; align-items:center; gap:0.75rem; padding:0.85rem 1.1rem; }}
    .ac-num {{
        width:24px; height:24px; border-radius:50%;
        font-size:11px; font-weight:700; color:#fff;
        display:flex; align-items:center; justify-content:center; flex-shrink:0;
    }}
    .ac-title {{ font-size:13px; font-weight:600; color:{COLORS['ink']}; line-height:1.3; }}
    .ac-sub   {{ font-size:11px; color:#aaa; margin-top:1px; }}
    .ac-check {{ margin-left:auto; font-size:13px; color:{COLORS['low']}; font-weight:700; }}
    .ac-body  {{ padding:0 1.1rem 1.1rem; }}
    .ac-desc  {{
        font-size:12px; color:{COLORS['body']}; line-height:1.6;
        background:#f9f9f7; border-radius:6px; padding:0.65rem 0.85rem; margin-bottom:0.75rem;
    }}
    </style>
    """, unsafe_allow_html=True)

    def _ac_header(idx, label, sublabel, state):
        num_bg  = COLORS["low"] if state == "done" else COLORS["ink"] if state == "active" else "#d0d0cc"
        check   = "<span class='ac-check'>&#10003;</span>" if state == "done" else ""
        css_cls = "ac-done" if state == "done" else "ac-active" if state == "active" else ""
        return (
            f"<div class='ac-item {css_cls}'>"
            f"<div class='ac-header'>"
            f"<div class='ac-num' style='background:{num_bg};'>{idx+1}</div>"
            f"<div><div class='ac-title'>{label}</div>"
            f"<div class='ac-sub'>{sublabel}</div></div>"
            f"{check}</div>"
        )

    _ac_pad, _ac_col, _ac_pad2 = st.columns([1, 4, 1])
    with _ac_col:

        # STEP 0
        _s0_state = "done" if step > 0 else "active"
        st.markdown(_ac_header(0, "Engine flags the patient", "Overnight scoring run", _s0_state), unsafe_allow_html=True)
        if step == 0:
            st.markdown(f"<div class='ac-body'><div class='ac-desc'>The model scores <strong>{jp_id}</strong> across 50+ variables &mdash; fill history, copay burden, diagnosis, refill gaps.</div></div>", unsafe_allow_html=True)
            if st.button("Run scoring", key="journey_btn_0", use_container_width=True):
                with st.spinner("Scoring patient..."):
                    _time.sleep(0.8)
                    st.session_state["journey_pred"] = {
                        "risk_score": 87, "risk_level": "high", "confidence_score": 0.91,
                        "top_risk_factors": [
                            {"factor_name": "days_since_last_fill", "impact_score": 0.82, "description": "54 days since last Metformin fill"},
                            {"factor_name": "pdc_90_days",          "impact_score": 0.71, "description": "PDC dropped to 61%, below 80% target"},
                            {"factor_name": "gap_count",            "impact_score": 0.58, "description": "3 refill gaps in the last 6 months"},
                        ],
                    }
            pred = st.session_state.get("journey_pred")
            if pred and "error" not in pred:
                score   = pred.get("risk_score", 0)
                level   = pred.get("risk_level", "low")
                conf    = pred.get("confidence_score", 0)
                sc      = COLORS["high"] if level == "high" else COLORS["mid"] if level == "medium" else COLORS["low"]
                factors = pred.get("top_risk_factors", [])
                _fl = {
                    "pdc_90_days": "Medication coverage, last 90 days",
                    "gap_count": "Number of refill gaps",
                    "days_since_last_fill": "Days since last fill",
                }
                bars_html = ""
                for f in factors[:3]:
                    raw = f.get("factor_name", ""); _rd = f.get("description", "")
                    d   = _rd if _rd and not _rd.lower().startswith("risk factor:") else _fl.get(raw, raw.replace("_", " ").capitalize())
                    bw  = min(100, abs(f.get("impact_score", 0)) * 100)
                    bars_html += (
                        f"<div style='margin-bottom:0.4rem;'>"
                        f"<div style='font-size:11px;color:{COLORS['body']};margin-bottom:2px;'>{d}</div>"
                        f"<div style='height:3px;background:#f0f0ec;border-radius:2px;'>"
                        f"<div style='width:{bw:.0f}%;height:3px;background:{sc};border-radius:2px;'></div>"
                        f"</div></div>"
                    )
                st.markdown(
                    f"<div style='background:#fafaf8;border:1px solid #e8e8e4;border-radius:8px;padding:1rem;margin:0.5rem 0 0.75rem;'>"
                    f"<div style='display:flex;align-items:baseline;gap:0.6rem;margin-bottom:0.75rem;'>"
                    f"<span style='font-size:2.5rem;font-weight:300;color:{sc};letter-spacing:-0.03em;line-height:1;'>{score:.0f}</span>"
                    f"<div><div style='font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:{sc};'>● {level} risk</div>"
                    f"<div style='font-size:11px;color:#aaa;'>Confidence {conf:.0%} · 30-day window</div></div></div>"
                    f"<div style='font-size:10px;text-transform:uppercase;letter-spacing:0.08em;color:#aaa;margin-bottom:0.5rem;'>Why</div>"
                    f"{bars_html}</div>",
                    unsafe_allow_html=True,
                )
                st.button("Next", key="journey_next_0", use_container_width=True,
                          on_click=lambda: st.session_state.update({_step_key: 1}))
        elif step > 0:
            pred   = st.session_state.get("journey_pred")
            sc_txt = "Score <strong>87</strong> · high risk · confidence 91%" if pred else ""
            st.markdown(f"<div style='padding:0 1.1rem 0.75rem;font-size:12px;color:{COLORS['body']};'>{sc_txt}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # STEP 1
        _s1_state = "done" if step > 1 else "active" if step == 1 else "locked"
        st.markdown(_ac_header(1, "Outreach triggered automatically", "No one clicked anything", _s1_state), unsafe_allow_html=True)
        if step == 1:
            st.markdown(f"<div class='ac-body'><div class='ac-desc'>Because the score exceeds the threshold, the engine automatically drafts and sends an SMS &mdash; no care manager involved.</div></div>", unsafe_allow_html=True)
            if st.button("Send outreach SMS", key="journey_btn_1", use_container_width=True):
                with st.spinner("Sending..."):
                    _time.sleep(0.6)
                    st.session_state["journey_outreach"] = {
                        "status": "sent", "id": str(_uuid.uuid4()),
                        "msg": _sms_msg, "channel": "SMS",
                    }
            ov = st.session_state.get("journey_outreach")
            if ov:
                st.markdown(
                    f"<div style='background:#fafaf8;border:1px solid #e8e8e4;border-radius:8px;padding:1rem;margin:0.5rem 0 0.75rem;'>"
                    f"<div style='font-size:10px;text-transform:uppercase;letter-spacing:0.08em;color:#aaa;margin-bottom:0.5rem;'>SMS · {jp_id} · {jp_age}y · {jp_cond}</div>"
                    f"<div style='background:#f0f0ec;border-radius:6px;padding:0.6rem 0.8rem;font-size:12px;color:{COLORS['ink']};line-height:1.6;margin-bottom:0.5rem;'>\"{ov['msg']}\"</div>"
                    f"<div style='font-size:11px;color:{COLORS['low']};font-weight:600;'>Sent automatically · <span style=\"color:#aaa;font-weight:400;\">ID {ov['id'][:8]}...</span></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.button("Next", key="journey_next_1", use_container_width=True,
                          on_click=lambda: st.session_state.update({_step_key: 2}))
        elif step > 1:
            st.markdown(f"<div style='padding:0 1.1rem 0.75rem;font-size:12px;color:{COLORS['body']};'>SMS sent · no care manager involved</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # STEP 2
        _s2_state = "done" if step > 2 else "active" if step == 2 else "locked"
        st.markdown(_ac_header(2, "Patient replies", "AI handles the response", _s2_state), unsafe_allow_html=True)
        if step == 2:
            st.markdown(f"<div class='ac-body'><div class='ac-desc'>The patient texts back. The AI identifies the barrier and responds with targeted support &mdash; no human reads the message.</div></div>", unsafe_allow_html=True)
            if st.button("Patient: It's too expensive", key="journey_btn_2", use_container_width=True):
                with st.spinner("AI responding..."):
                    _time.sleep(1.0)
                    st.session_state["journey_conv"] = "demo-conv-001"
                    st.session_state["journey_chat"] = {
                        "conversation_id": "demo-conv-001",
                        "response": "Hi there, I understand cost can be a concern. Good news &mdash; there may be savings programs available for your Metformin. Would you like me to check what options might help reduce your costs?",
                        "identified_barrier": "cost",
                        "suggested_action": "check_copay_assistance",
                    }
            cv = st.session_state.get("journey_chat")
            if cv:
                barrier  = cv.get("identified_barrier", "cost")
                action   = cv.get("suggested_action", "check_copay_assistance")
                response = cv.get("response", "")
                st.markdown(
                    f"<div style='background:#fafaf8;border:1px solid #e8e8e4;border-radius:8px;padding:1rem;margin:0.5rem 0 0.75rem;'>"
                    f"<div style='background:#f0f0ec;border-radius:8px 8px 2px 8px;padding:0.6rem 0.8rem;font-size:12px;color:{COLORS['body']};max-width:85%;margin-left:auto;margin-bottom:0.4rem;'>"
                    f"It's too expensive, I can't afford it</div>"
                    f"<div style='background:#fff;border:1px solid #e8e8e4;border-radius:2px 8px 8px 8px;padding:0.6rem 0.8rem;font-size:12px;color:{COLORS['ink']};max-width:85%;line-height:1.6;margin-bottom:0.6rem;'>{response}</div>"
                    f"<span style='font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;background:{COLORS['mid']}22;color:{COLORS['mid']};padding:2px 8px;border-radius:3px;margin-right:4px;'>{barrier.replace('_', ' ')}</span>"
                    f"<span style='font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;background:{COLORS['low']}22;color:{COLORS['low']};padding:2px 8px;border-radius:3px;'>to {action.replace('_', ' ')}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.button("Next", key="journey_next_2", use_container_width=True,
                          on_click=lambda: st.session_state.update({_step_key: 3}))
        elif step > 2:
            st.markdown(f"<div style='padding:0 1.1rem 0.75rem;font-size:12px;color:{COLORS['body']};'>Barrier identified: cost · copay assistance suggested</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # STEP 3
        _s3_state = "done" if step > 3 else "active" if step == 3 else "locked"
        st.markdown(_ac_header(3, "Refill confirmed", "PDC recovers", _s3_state), unsafe_allow_html=True)
        if step == 3:
            st.markdown(f"<div class='ac-body'><div class='ac-desc'>The patient picks up their prescription. The engine detects the fill event and updates their adherence score automatically.</div></div>", unsafe_allow_html=True)
            if st.button("Mark refill complete", key="journey_btn_3", use_container_width=True):
                with st.spinner("Updating records..."):
                    _time.sleep(0.7)
                    st.session_state["journey_refill"] = {
                        "pdc_before": jp_pdc, "pdc_after": _new_pdc, "gap_closed": jp_gap,
                    }
            rf = st.session_state.get("journey_refill")
            if rf:
                st.markdown(
                    f"<div style='background:#fafaf8;border:1px solid #e8e8e4;border-radius:8px;padding:1rem;margin:0.5rem 0 0.75rem;'>"
                    f"<div style='display:flex;gap:1.5rem;align-items:center;margin-bottom:0.75rem;'>"
                    f"<div><div style='font-size:10px;text-transform:uppercase;letter-spacing:0.08em;color:#aaa;margin-bottom:2px;'>PDC before</div>"
                    f"<div style='font-size:2rem;font-weight:300;color:{COLORS['high']};'>{rf['pdc_before']:.0%}</div></div>"
                    f"<div style='font-size:1.2rem;color:#ccc;'>to</div>"
                    f"<div><div style='font-size:10px;text-transform:uppercase;letter-spacing:0.08em;color:#aaa;margin-bottom:2px;'>PDC after</div>"
                    f"<div style='font-size:2rem;font-weight:300;color:{COLORS['low']};'>{rf['pdc_after']:.0%}</div></div>"
                    f"</div>"
                    f"<div style='font-size:12px;color:{COLORS['body']};'>{rf['gap_closed']}-day gap closed · Adherence target restored · "
                    f"<span style='color:{COLORS['low']};font-weight:600;'>Patient back on track</span></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.button("Next", key="journey_next_3", use_container_width=True,
                          on_click=lambda: st.session_state.update({_step_key: 4}))
        elif step > 3:
            st.markdown(f"<div style='padding:0 1.1rem 0.75rem;font-size:12px;color:{COLORS['body']};'>PDC {jp_pdc:.0%} to {_new_pdc:.0%} · {jp_gap}-day gap closed</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # STEP 4
        _s4_state = "active" if step == 4 else "locked"
        st.markdown(_ac_header(4, "Hospitalization avoided", "$15,000 saved", _s4_state), unsafe_allow_html=True)
        if step == 4:
            st.markdown(f"""
            <div class='ac-body'>
            <div style="text-align:center; padding:1rem 0.5rem 0.5rem;">
                <div style="font-size:2.5rem; margin-bottom:0.5rem;">🏥</div>
                <div style=\"font-size:1.4rem; font-weight:300; color:{COLORS['ink']}; letter-spacing:-0.02em; margin-bottom:0.5rem;\">
                    Hospitalization avoided.
                </div>
                <div style=\"font-size:13px; color:{COLORS['body']}; margin:0 auto 1.25rem; line-height:1.7;\">
                    Without this intervention, <strong>{jp_id}</strong> was on a trajectory
                    toward a {jp_cond.lower()}-related admission within 30 days.
                    Average hospitalisation cost: <strong>$15,000</strong>.
                    Programme cost per patient per year: <strong>~$50</strong>.
                </div>
                <div style="display:flex; flex-wrap:wrap; justify-content:center; gap:1rem; background:#f9f9f7; border-radius:8px; padding:1rem;">
                    <div style="text-align:center;">
                        <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.08em;color:#aaa;margin-bottom:4px;">Programme cost / patient</div>
                        <div style=\"font-size:1.3rem;font-weight:300;color:{COLORS['ink']};\">~$50</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.08em;color:#aaa;margin-bottom:4px;">Hospitalisation avoided</div>
                        <div style=\"font-size:1.3rem;font-weight:300;color:{COLORS['low']};\">$15,000</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.08em;color:#aaa;margin-bottom:4px;">Return on this patient</div>
                        <div style=\"font-size:1.3rem;font-weight:300;color:{COLORS['low']};\">~300x</div>
                    </div>
                </div>
            </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if step > 0:
            st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
            if st.button("Start over", key="journey_restart", use_container_width=True):
                for k in ["journey_pred", "journey_outreach", "journey_chat",
                          "journey_conv", "journey_refill", _step_key]:
                    st.session_state.pop(k, None)
                st.rerun()


# ═══════════════════════════════════════════════════════════════
# TAB 3: THE RETURN
# ═══════════════════════════════════════════════════════════════

with tab_return:

    success_rate = succeeded / total_i
    avg_h        = intv["hours"].dropna().astype(float).mean()
    net_savings  = total_savings - total_cost
    overall_roi  = (net_savings / total_cost * 100) if total_cost > 0 else 0
    roi_ratio    = total_savings / total_cost if total_cost > 0 else 0

    st.markdown(f"""
    <p class="narrative">
        In a simulated 12-week deployment, the engine triggered
        <strong>{total_i} automated interventions</strong>.
        <span class="highlight-green">{succeeded} led to a refill</span> —
        a {success_rate:.0%} success rate &mdash; generating
        <span class="highlight-green">${total_savings:,.0f} in avoided hospitalisations</span>
        on an investment of <strong>${total_cost:,.0f}</strong>.
        That's <strong>${roi_ratio:.0f} back for every $1 invested</strong>.
    </p>
    """, unsafe_allow_html=True)

    def _ret_kpi(label, value, sub, val_color="#1a1a1a", sub_color="#888"):
        return (
            f"<div style='background:#fff;border:1px solid #e8e8e4;border-radius:8px;"
            f"padding:1rem 1.25rem;flex:1;box-sizing:border-box;"
            f"display:flex;flex-direction:column;justify-content:space-between;'>"
            f"<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;"
            f"color:#888;font-weight:500;line-height:1.3;margin-bottom:0.4rem;'>{label}</div>"
            f"<div style='font-size:1.8rem;font-weight:300;color:{val_color};"
            f"letter-spacing:-0.02em;line-height:1.1;'>{value}</div>"
            f"<div style='font-size:0.75rem;color:{sub_color};margin-top:0.4rem;"
            f"line-height:1.3;'>{sub}</div>"
            f"</div>"
        )

    # Row 1: operational metrics
    _roi_str = f"{roi_ratio:.0f} : 1"
    st.markdown(
        f"<div style='display:flex;gap:1rem;align-items:stretch;margin-bottom:1rem;'>"
        + _ret_kpi("Interventions sent",       f"{total_i}",              "automated, no manual trigger")
        + _ret_kpi("Led to a refill",          f"{succeeded}",            f"{success_rate:.0%} success rate", sub_color=COLORS["low"])
        + _ret_kpi("Avg time to respond",      f"{avg_h:.0f}h",           "from send to patient reply")
        + "</div>",
        unsafe_allow_html=True,
    )

    # Row 2: financial metrics
    st.markdown(
        f"<div style='display:flex;gap:1rem;align-items:stretch;margin-bottom:1rem;'>"
        + _ret_kpi("Avoided hospitalisations", f"${total_savings:,.0f}",  "12-week total", val_color=COLORS["low"])
        + _ret_kpi("Programme cost",           f"${total_cost:,.0f}",     "12-week total")
        + _ret_kpi("Return on investment",     f"${_roi_str}",            "12-week simulation")
        + "</div>",
        unsafe_allow_html=True,
    )

    # ── Intervention pipeline ─────────────────────────────────────────────
    st.markdown("""
    <div style="border-top:1px solid #e8e8e4; margin:1.5rem 0 1rem;"></div>
    <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;
                color:#aaa;font-weight:600;margin-bottom:0.25rem;">Intervention pipeline &mdash; 12 weeks</div>
    <p style="font-size:12px;color:#888;margin-bottom:0.75rem;">
        Each intervention flows through these stages. Not every message leads to a refill &mdash;
        that's expected. The engine escalates until the patient responds or a care manager steps in.
    </p>
    """, unsafe_allow_html=True)

    _f_sent      = total_i
    _f_delivered = int(total_i * 0.91)
    _f_replied   = responded
    _f_refilled  = succeeded
    _f_noresp    = total_i - responded

    _pipe_labels = ["Sent", "Delivered", "Patient replied", "Refilled", "No response"]
    _pipe_vals   = [_f_sent, _f_delivered, _f_replied, _f_refilled, _f_noresp]
    _pipe_colors = [COLORS["faint"], COLORS["dim"], COLORS["mid"], COLORS["low"], COLORS["high"]]

    fig_pipe = go.Figure(go.Bar(
        x=_pipe_labels, y=_pipe_vals,
        marker_color=_pipe_colors,
        marker_line_width=0,
        text=[f"{v:,}" for v in _pipe_vals],
        textposition="outside",
        textfont=dict(size=12, family="Inter", color=COLORS["ink"]),
        hovertemplate="%{x}: %{y:,}<extra></extra>",
    ))
    apply_layout(fig_pipe,
        height=240,
        hoverlabel=dict(bgcolor="#fff", font_color="#1a1a1a", font_size=11,
                        font_family="Inter", bordercolor="#e8e8e4"),
        yaxis=dict(visible=False),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=12, color=COLORS["ink"])),
        margin=dict(l=0, r=0, t=32, b=8),
    )
    st.plotly_chart(fig_pipe, use_container_width=True, config={"displayModeBar": False})

    _rr1, _rr2 = st.columns(2)

    # ── Channel performance (full width) ──────────────────────────────────
    st.markdown("""
    <div style="border-top:1px solid #e8e8e4; margin:1.5rem 0 1rem;"></div>
    <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;
                color:#aaa;font-weight:600;margin-bottom:0.75rem;">Success rate by channel</div>
    """, unsafe_allow_html=True)

    ch = intv.groupby("channel").agg(
        sent=("id", "count"),
        success=("status", lambda x: (x == "Successful").sum()),
    )
    ch["rate"] = (ch["success"] / ch["sent"]).round(2)
    ch = ch.sort_values("rate", ascending=True)
    fig_ch = go.Figure(go.Bar(
        x=ch["rate"].values, y=ch.index, orientation="h",
        marker_color=[COLORS["low"] if v >= 0.3 else COLORS["mid"] if v >= 0.2 else COLORS["dim"]
                      for v in ch["rate"].values],
        marker_line_width=0,
        text=[f'{v:.0%}' for v in ch["rate"].values],
        textposition="outside",
        textfont=dict(size=12, family="Inter", color=COLORS["ink"]),
        hovertemplate="%{y}: %{x:.0%} success rate<extra></extra>",
    ))
    apply_layout(fig_ch,
        height=260,
        hoverlabel=dict(bgcolor="#fff", font_color="#1a1a1a", font_size=11,
                        font_family="Inter", bordercolor="#e8e8e4"),
        xaxis=dict(visible=False, range=[0, 0.55]),
        yaxis=dict(showgrid=False, tickfont=dict(size=13, color=COLORS["ink"])),
        margin=dict(l=0, r=60, t=8, b=8),
    )
    st.plotly_chart(fig_ch, use_container_width=True, config={"displayModeBar": False})

    # ── J-curve: cumulative savings vs cost ────────────────────────────────
    # Find break-even week
    _be_week = next((w for w, s, c in zip(roi["week"], roi["cum_savings"], roi["cum_cost"]) if s >= c), None)
    st.markdown(f"""
    <div style="border-top:1px solid #e8e8e4; margin:1.5rem 0 1rem;"></div>
    <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;
                color:#aaa;font-weight:600;margin-bottom:0.25rem;">Cumulative return &mdash; 12 weeks</div>
    <p style="font-size:12px;color:{COLORS['dim']};margin-bottom:0.75rem;">
        Programme breaks even at week {_be_week} and accelerates as the model learns.
        Cumulative savings outpace cumulative cost by week&nbsp;12.
    </p>
    """, unsafe_allow_html=True)

    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(
        x=roi["week"], y=roi["cum_savings"],
        mode="lines+markers", name="Cumulative savings",
        line=dict(color=COLORS["low"], width=2.5),
        marker=dict(size=6, color=COLORS["low"], line=dict(width=1.5, color="#fff")),
        fill="tozeroy", fillcolor="rgba(39,174,96,0.07)",
        hovertemplate="Week %{x}<br><b>$%{y:,.0f}</b> saved<extra></extra>",
    ))
    fig_roi.add_trace(go.Scatter(
        x=roi["week"], y=roi["cum_cost"],
        mode="lines+markers", name="Cumulative cost",
        line=dict(color=COLORS["high"], width=1.5, dash="dot"),
        marker=dict(size=5, color=COLORS["high"], line=dict(width=1.5, color="#fff")),
        hovertemplate="Week %{x}<br><b>$%{y:,.0f}</b> spent<extra></extra>",
    ))
    if _be_week:
        _be_val = roi.loc[roi["week"] == _be_week, "cum_savings"].values[0]
        fig_roi.add_annotation(
            x=_be_week, y=_be_val,
            text=f"Break-even<br>week {_be_week}",
            showarrow=True, arrowhead=2, arrowcolor=COLORS["dim"],
            font=dict(size=10, color=COLORS["dim"], family="Inter"),
            bgcolor="#fff", bordercolor=COLORS["border"], borderwidth=1,
            ax=30, ay=-40,
        )
    apply_layout(fig_roi,
        height=300, showlegend=True,
        hoverlabel=dict(bgcolor="#fff", font_color="#1a1a1a",
                        font_size=12, font_family="Inter, sans-serif",
                        bordercolor="#e8e8e4"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11, color=COLORS["dim"])),
        yaxis=dict(showgrid=True, gridcolor=COLORS["faint"], gridwidth=0.5, zeroline=False,
                   tickfont=dict(size=10), tickprefix="$"),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10),
                   title="Week", tickmode="linear", dtick=2),
        margin=dict(l=60, r=16, t=28, b=40),
    )
    st.plotly_chart(fig_roi, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""
    <div style="border-top:1px solid #e8e8e4; margin:2rem 0 1.5rem;"></div>
    <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.1em;
                color:#aaa;font-weight:600;margin-bottom:0.5rem;">Model your own scenario</div>
    <p class="narrative">
        The numbers above are simulated. Plug in your own population size,
        outreach cost and success rate to see what the engine would return for you.
    </p>
    <style>
        .roi-hint {{
            font-size: 11px; color: #aaa; margin: -0.4rem 0 1rem;
            font-family: 'Inter', sans-serif; line-height: 1.5;
        }}
    </style>
    """, unsafe_allow_html=True)

    _roi_pad, _roi_col, _roi_pad2 = st.columns([1, 4, 1])
    with _roi_col:

        _ri1, _ri2 = st.columns(2)
        with _ri1:
            patients_n = st.number_input(
                "Patients in the programme",
                min_value=100, max_value=100_000, value=5000, step=500,
            )
            st.markdown("<div class='roi-hint'>Total patients monitored and scored each night.</div>", unsafe_allow_html=True)

            cost_per_patient = st.number_input(
                "Programme cost per patient / year ($)",
                min_value=5, max_value=500, value=50, step=5,
            )
            st.markdown("<div class='roi-hint'>Includes platform fee, outreach messages and care-manager escalations. Typical range: $30–$80.</div>", unsafe_allow_html=True)

        with _ri2:
            success_pct = st.slider(
                "Outreach success rate",
                min_value=20, max_value=80, value=45, format="%d%%",
            )
            st.markdown("<div class='roi-hint'>% of outreach contacts that lead to a refill. Benchmark: 35–50%.</div>", unsafe_allow_html=True)

            hosp_cost = st.number_input(
                "Avg hospitalisation cost ($)",
                min_value=5000, max_value=50000, value=15000, step=1000,
            )
            st.markdown("<div class='roi-hint'>US average for a preventable medication-related admission: $12k–$18k.</div>", unsafe_allow_html=True)

        # Formula rationale:
        #   50% of monitored patients are high-risk and receive outreach.
        #   Of those, success_pct% are successfully re-engaged (refill confirmed).
        #   Evidence shows ~16% of successfully re-engaged patients would otherwise
        #   have been hospitalised within the year (AHRQ / CMS non-adherence data).
        #   Programme cost = flat per-patient annual fee (platform + messaging + escalation).
        #   With defaults this yields ≈ 11 : 1, consistent with published CVD-management ROI.
        est_interventions = patients_n * 0.5
        est_successful    = est_interventions * (success_pct / 100)
        est_cost          = patients_n * cost_per_patient
        est_avoided       = est_successful * 0.163        # ~16% hospitalization avoidance rate
        est_savings       = est_avoided * hosp_cost
        est_roi_ratio     = est_savings / est_cost if est_cost > 0 else 0

        _roi_lbl  = f"${est_roi_ratio:.0f} : $1"
        _sav_lbl  = f"${est_savings:,.0f}"
        _cost_lbl = f"${est_cost:,.0f}"
        _avd_lbl  = f"{est_avoided:,.0f}"
        st.markdown(
            f"""
            <div style="border-top:2px solid #e8e8e4; margin:1rem 0 0.75rem;
                        padding-top:0.25rem; font-size:11px; text-transform:uppercase;
                        letter-spacing:0.1em; color:#aaa; font-weight:600;">Projected result</div>
            <div style="background:{COLORS['ink']};border-radius:10px;padding:1.5rem;text-align:center;">
                <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.12em;
                            color:rgba(255,255,255,0.45);margin-bottom:0.4rem;font-weight:600;">
                    Return on investment</div>
                <div style="font-size:3rem;font-weight:300;color:#fff;
                            letter-spacing:-0.03em;line-height:1;margin-bottom:1rem;">
                    {_roi_lbl}</div>
                <div style="display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;">
                    <div>
                        <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.08em;
                                    color:rgba(255,255,255,0.4);margin-bottom:3px;">Hospitalisations avoided</div>
                        <div style="font-size:1.2rem;font-weight:300;color:#fff;">{_avd_lbl}</div>
                    </div>
                    <div>
                        <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.08em;
                                    color:rgba(255,255,255,0.4);margin-bottom:3px;">Total savings</div>
                        <div style="font-size:1.2rem;font-weight:300;color:{COLORS['low']};">{_sav_lbl}</div>
                    </div>
                    <div>
                        <div style="font-size:10px;text-transform:uppercase;letter-spacing:0.08em;
                                    color:rgba(255,255,255,0.4);margin-bottom:3px;">Programme cost</div>
                        <div style="font-size:1.2rem;font-weight:300;color:rgba(255,255,255,0.55);">{_cost_lbl}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
