"""
Medication Adherence Dashboard.

Designed following John Maeda's Laws of Simplicity.
Adapted for executive audiences: plain language, business context, narrative flow.

The story: Non-adherence costs $528B/year. We predict who will stop taking
their medication, intervene before they do, and measure the financial return.
Every $1 invested yields $4 in avoided hospitalizations.
"""
import base64
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
    font=dict(family="Inter, Helvetica Neue, sans-serif", size=12, color=COLORS["ink"]),
    margin=dict(l=0, r=0, t=8, b=0),
    showlegend=False,
)


def apply_layout(fig, **overrides):
    """Apply base layout then any overrides — avoids duplicate kwarg errors."""
    fig.update_layout(**PLOTLY_BASE)
    if overrides:
        fig.update_layout(**overrides)
    return fig


# ── Page Config ────────────────────────────────────────────────

st.set_page_config(
    page_title="Medication Adherence",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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

    /* Insight cards */
    .insight-card {
        background: #fff; border: 1px solid #e8e8e4; border-radius: 8px;
        padding: 1.25rem 1.5rem; margin-bottom: 1rem;
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
        "medication": np.random.choice(
            ["Metformin", "Lisinopril", "Atorvastatin", "Amlodipine", "Omeprazole"], n,
        ),
        "condition": np.random.choice(
            ["Diabetes", "Hypertension", "Cholesterol", "Heart Disease", "GERD"], n,
        ),
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
    dates = pd.date_range(end=date.today(), periods=12, freq="W")
    df = pd.DataFrame({
        "week": dates,
        "intervened": [150 + i * 10 + np.random.randint(-20, 20) for i, _ in enumerate(dates)],
        "successful": [80 + i * 5 + np.random.randint(-10, 10) for i, _ in enumerate(dates)],
        "savings": [25000 + i * 2000 + np.random.randint(-5000, 5000) for i, _ in enumerate(dates)],
        "cost": [3000 + i * 100 for i, _ in enumerate(dates)],
    })
    df["roi"] = ((df["savings"] - df["cost"]) / df["cost"] * 100).round(1)
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


# ── Logo ───────────────────────────────────────────────────────

_logo_path = Path(__file__).parent / "logo-tam.svg"
_logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()

# ── Header ─────────────────────────────────────────────────────

st.markdown(f"""
<div style="display: flex; align-items: flex-start; justify-content: space-between;
            margin-bottom: 2rem;">
    <div>
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
            <p style="font-size: 12px; text-transform: uppercase; letter-spacing: 0.12em;
                      color: #888; margin: 0; font-weight: 500;">
                Predictive Medication Adherence Engine</p>
            <span style="font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em;
                         background: #1a1a1a; color: #fafaf7; padding: 3px 10px; border-radius: 3px;
                         font-weight: 600;">Demo</span>
        </div>
        <h1 style="margin: 0 !important; padding: 0 !important;">Adherence</h1>
        <p style="font-size: 14px; color: #888; margin-top: 0.25rem; max-width: 600px; line-height: 1.6;">
            Simulated data showing how the engine identifies at-risk patients,
            triggers interventions, and measures financial return.</p>
    </div>
    <div style="display: flex; flex-direction: column; align-items: flex-end;
                gap: 0.25rem; padding-top: 0.25rem;">
        <img src="data:image/svg+xml;base64,{_logo_b64}"
             style="height: 28px; opacity: 0.75;" alt="The Agile Monkeys" />
        <span style="font-size: 10px; color: #aaa; letter-spacing: 0.06em;">
            theagilemonkeys.com</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────

tab_story, tab_platform, tab_patients, tab_actions, tab_returns = st.tabs([
    "The Problem", "The Platform", "Who Needs Help", "What We're Doing", "Is It Working"
])


# ═══════════════════════════════════════════════════════════════
# TAB 1: THE PROBLEM
# Executive summary — the narrative arc in 10 seconds.
# ═══════════════════════════════════════════════════════════════

with tab_story:

    st.markdown(f"""
    <p class="narrative">
        Medication non-adherence costs the U.S. healthcare system
        <strong>$528 billion per year</strong> in avoidable hospitalizations,
        disease progression, and lost pharmacy revenue.
        In this simulation of <strong>{total:,} patients</strong>,
        the engine flags
        <span class="highlight-red">{high} as high risk</span> of
        stopping their medication in the next 30 days
        and <span class="highlight-red">{below_target}</span> already
        below the 80% adherence target.
    </p>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Simulated patients", f"{total:,}")
    c2.metric("Flagged at risk", f"{high + med}", delta=f"{(high+med)/total:.0%} of population")
    c3.metric("Taking meds regularly", f"{avg_pdc:.0%}", delta="80% is the target")
    c4.metric("Getting worse", f"{declining}", delta=f"{declining/total:.0%} trending down")

    st.markdown("### How it works")

    st.markdown(f"""
    <div style="background: #fff; border: 1px solid #e8e8e4; border-radius: 8px;
                padding: 1.5rem; max-width: 720px;">
        <div class="step-row">
            <div class="step-num">1</div>
            <div class="step-content">
                <div class="step-title">Predict</div>
                <div class="step-desc">The model analyzes 50+ variables per patient &mdash; fill history,
                    gaps between refills, copay burden, diagnoses &mdash; and scores each patient
                    from 0 to 100 on their likelihood of stopping.</div>
            </div>
        </div>
        <div class="step-row">
            <div class="step-num">2</div>
            <div class="step-content">
                <div class="step-title">Intervene</div>
                <div class="step-desc">High-risk patients receive automated outreach
                    via SMS, email, phone, or chatbot &mdash; personalized to their barrier.
                    If cost is the issue, we surface discount programs. If it's forgetfulness,
                    we send reminders.</div>
            </div>
        </div>
        <div class="step-row">
            <div class="step-num">3</div>
            <div class="step-content">
                <div class="step-title">Escalate</div>
                <div class="step-desc">If a patient doesn't respond, the system escalates &mdash;
                    from SMS to a phone call to a care manager, even notifying a family member.
                    No one falls through the cracks.</div>
            </div>
        </div>
        <div class="step-row">
            <div class="step-num">4</div>
            <div class="step-content">
                <div class="step-title">Measure</div>
                <div class="step-desc">Every intervention is tracked. Every $1 invested
                    yields approximately $4 in avoided hospitalizations and retained pharmacy
                    revenue. Results are measured weekly.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Population risk at a glance")

    fig = go.Figure()
    fig.add_trace(go.Bar(y=[""], x=[low], name=f"Low risk  ({low})", orientation="h",
                         marker_color=COLORS["low"], text=f"{low}", textposition="inside",
                         textfont=dict(color="white", size=13, family="Inter")))
    fig.add_trace(go.Bar(y=[""], x=[med], name=f"Moderate  ({med})", orientation="h",
                         marker_color=COLORS["mid"], text=f"{med}", textposition="inside",
                         textfont=dict(color="white", size=13, family="Inter")))
    fig.add_trace(go.Bar(y=[""], x=[high], name=f"High risk  ({high})", orientation="h",
                         marker_color=COLORS["high"], text=f"{high}", textposition="inside",
                         textfont=dict(color="white", size=13, family="Inter")))
    apply_layout(fig,
        barmode="stack", height=64, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.4, xanchor="left", x=0,
                    font=dict(size=11, color=COLORS["dim"]), tracegroupgap=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=28, b=0),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════
# TAB 2: THE PLATFORM
# Executive view: what happened overnight + one patient's journey.
# ═══════════════════════════════════════════════════════════════

with tab_platform:

    import requests as _requests
    import time as _time, uuid as _uuid

    API = "http://localhost:8000/api/v1"

    # ── Fixed demo patient for the journey walkthrough ────────────
    # Pick the highest-risk patient so the demo is always compelling
    journey_patient = pop[pop["level"] == "high"].sort_values("risk", ascending=False).iloc[0]
    jp_id  = journey_patient["id"]
    jp_med = journey_patient["medication"]
    jp_gap = int(journey_patient["days_gap"])
    jp_pdc = journey_patient["pdc"]
    jp_cond = journey_patient["condition"]
    jp_age  = int(journey_patient["age"])
    jp_copay = journey_patient["copay"]

    # ── SECTION 1: OVERNIGHT BRIEFING ────────────────────────────

    st.markdown("""
    <p class="narrative">
        Every night at midnight, the engine runs silently across the full patient
        population &mdash; no care manager required. Here is what it found last night.
    </p>
    """, unsafe_allow_html=True)

    # Overnight stats — derived from simulated population
    n_scored   = len(pop)
    n_flagged  = len(pop[pop["level"] == "high"])
    n_outreach = int(n_flagged * 0.85)   # engine auto-sends to 85% of high-risk
    n_new_flags = int(n_flagged * 0.12)  # 12% newly flagged since yesterday

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients scored", f"{n_scored:,}", delta="ran overnight, automatically")
    c2.metric("Flagged high risk", f"{n_flagged}", delta=f"+{n_new_flags} since yesterday")
    c3.metric("Outreach triggered", f"{n_outreach}", delta="no one had to click anything")
    c4.metric("Avg time to act", "< 2 min", delta="from flag to message sent")

    st.markdown("""
    <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.1em;
                color:#888; margin:2rem 0 0.75rem;">
        Today's queue — highest risk first
    </div>
    """, unsafe_allow_html=True)

    # Build a compact patient queue (top 8 high-risk patients)
    queue_df = pop[pop["level"] == "high"].sort_values("risk", ascending=False).head(8).copy()
    _channel_defaults = ["SMS", "SMS", "Email", "Voice", "SMS", "Email", "SMS", "Push"]
    _reasons = [
        "54 days since last fill",
        "3 missed refills this quarter",
        "PDC dropped 18% in 30 days",
        "Cost barrier flagged last visit",
        "Pharmacy switch — refill gap",
        "New diagnosis, regimen changed",
        "Overdue by 6 weeks",
        "Refill consistency score: low",
    ]

    queue_html = """
    <div style="background:#fff; border:1px solid #e8e8e4; border-radius:8px;
                overflow:hidden; margin-bottom:2rem;">
      <div style="display:grid; grid-template-columns:80px 1fr 80px 90px 110px 90px;
                  gap:0; padding:0.5rem 1.25rem;
                  border-bottom:1px solid #f0f0ec;
                  font-size:10px; text-transform:uppercase; letter-spacing:0.08em; color:#aaa;">
        <span>Patient</span><span>Why flagged</span><span>Risk</span>
        <span>Days overdue</span><span>Medication</span><span>Channel</span>
      </div>
    """
    for i, (_, row) in enumerate(queue_df.iterrows()):
        score_color = COLORS["high"] if row["risk"] > 70 else COLORS["mid"]
        bg = "#fafaf7" if i % 2 == 0 else "#fff"
        reason = _reasons[i % len(_reasons)]
        channel = _channel_defaults[i % len(_channel_defaults)]
        queue_html += f"""
      <div style="display:grid; grid-template-columns:80px 1fr 80px 90px 110px 90px;
                  gap:0; padding:0.65rem 1.25rem; background:{bg};
                  border-bottom:1px solid #f5f5f0; align-items:center;">
        <span style="font-size:12px; font-weight:600; color:{COLORS['ink']};">{row['id']}</span>
        <span style="font-size:12px; color:{COLORS['body']};">{reason}</span>
        <span style="font-size:12px; font-weight:600; color:{score_color};">{row['risk']:.0f}</span>
        <span style="font-size:12px; color:{COLORS['body']};">{int(row['days_gap'])}d</span>
        <span style="font-size:12px; color:{COLORS['body']};">{row['medication']}</span>
        <span style="font-size:10px; font-weight:600; text-transform:uppercase;
                     letter-spacing:0.06em; background:#f0f0ec; color:{COLORS['body']};
                     padding:2px 8px; border-radius:3px;">{channel}</span>
      </div>"""
    queue_html += "</div>"
    st.markdown(queue_html, unsafe_allow_html=True)

    # ── SECTION 2: ONE PATIENT'S JOURNEY ─────────────────────────

    st.markdown(f"""
    <div style="border-top:1px solid #e8e8e4; margin:0.5rem 0 1.5rem;"></div>
    <p class="narrative">
        Now follow one patient through the full loop &mdash; from the moment the engine
        flags them, to the message they receive, to the refill that prevents a hospitalization.
        This is what production looks like for <strong>{jp_id}</strong>,
        {jp_age}-year-old with {jp_cond}, {jp_gap} days without {jp_med}.
    </p>
    """, unsafe_allow_html=True)

    # Journey state — which steps are complete
    _step_key = "journey_step"
    if _step_key not in st.session_state:
        st.session_state[_step_key] = 0

    step = st.session_state[_step_key]

    # Step definitions
    _steps = [
        {
            "label": "Engine flags the patient",
            "icon": "🔍",
            "sublabel": "Overnight scoring run",
            "action_label": "Run scoring",
            "result_key": "journey_pred",
        },
        {
            "label": "Outreach is triggered automatically",
            "icon": "📲",
            "sublabel": "No one clicked anything",
            "action_label": "Send outreach",
            "result_key": "journey_outreach",
        },
        {
            "label": "Patient replies",
            "icon": "💬",
            "sublabel": "AI handles the response",
            "action_label": "Patient: \"It's too expensive\"",
            "result_key": "journey_chat",
        },
        {
            "label": "Refill confirmed",
            "icon": "✅",
            "sublabel": "PDC recovers",
            "action_label": "Mark refill complete",
            "result_key": "journey_refill",
        },
        {
            "label": "Hospitalization avoided",
            "icon": "🏥",
            "sublabel": f"${15000:,} saved",
            "action_label": None,
            "result_key": None,
        },
    ]

    # ── Timeline — one column per step, no cross-column CSS ──────
    _tcols = st.columns(len(_steps))
    for i, (tc, s) in enumerate(zip(_tcols, _steps)):
        if i < step:
            dot_bg = COLORS["low"]
            lc     = COLORS["low"]
        elif i == step:
            dot_bg = COLORS["ink"]
            lc     = COLORS["ink"]
        else:
            dot_bg = "#e0e0da"
            lc     = "#aaa"

        # Step number badge + label — no absolute positioning, no emoji in HTML
        tc.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='width:22px;height:22px;border-radius:50%;"
            f"background:{dot_bg};color:#fff;font-size:11px;font-weight:700;"
            f"display:flex;align-items:center;justify-content:center;"
            f"margin:0 auto 6px;line-height:1;'>{i+1}</div>"
            f"<div style='font-size:10px;font-weight:600;color:{lc};"
            f"line-height:1.3;'>{s['label']}</div>"
            f"<div style='font-size:10px;color:#aaa;margin-top:2px;'>{s['sublabel']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)

    # ── Active step panel ─────────────────────────────────────────
    if step < len(_steps):
        current = _steps[step]

        # ── STEP 0: Engine scores patient ────────────────────────
        if step == 0:
            col_btn, col_result = st.columns([1, 2])
            with col_btn:
                st.markdown(f"""
                <div style="background:#fff; border:1px solid #e8e8e4; border-radius:8px;
                            padding:1.25rem; height:100%;">
                    <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em;
                                color:#888; margin-bottom:0.5rem;">What happens</div>
                    <div style="font-size:13px; color:{COLORS['body']}; line-height:1.6;
                                margin-bottom:1rem;">
                        The model scores <strong>{jp_id}</strong> across 50+ variables —
                        fill history, copay burden, diagnosis, refill gaps.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_result:
                if st.button("▶  Run scoring", key="journey_btn_0", use_container_width=True):
                    with st.spinner("Scoring patient…"):
                        import time as _time
                        _time.sleep(0.8)
                        # Simulated response — no API call in demo mode
                        st.session_state["journey_pred"] = {
                            "risk_score": 87,
                            "risk_level": "high",
                            "confidence_score": 0.91,
                            "top_risk_factors": [
                                {"factor_name": "days_since_last_fill", "impact_score": 0.82, "description": "54 days since last Metformin fill"},
                                {"factor_name": "pdc_90_days",          "impact_score": 0.71, "description": "PDC dropped to 61% — below 80% target"},
                                {"factor_name": "gap_count",            "impact_score": 0.58, "description": "3 refill gaps in the last 6 months"},
                            ],
                        }

                pred = st.session_state.get("journey_pred")
                if pred and "error" not in pred:
                    score = pred.get("risk_score", 0)
                    level = pred.get("risk_level", "low")
                    conf  = pred.get("confidence_score", 0)
                    sc    = COLORS["high"] if level == "high" else COLORS["mid"] if level == "medium" else COLORS["low"]
                    factors = pred.get("top_risk_factors", [])
                    _fl = {
                        "pdc_90_days": "Medication coverage — last 90 days",
                        "pdc_180_days": "Medication coverage — last 6 months",
                        "pdc_365_days": "Medication coverage — last year",
                        "gap_count": "Number of refill gaps",
                        "max_gap_days": "Longest gap without medication",
                        "days_since_last_fill": "Days since last fill",
                        "is_overdue": "Prescription overdue",
                        "high_cost_medication_flag": "High-cost medication",
                        "has_depression_diagnosis": "Depression diagnosis",
                        "average_gap_days": "Avg days between refills",
                        "refill_consistency_score": "Refill consistency",
                    }

                    # Score header — separate markdown call
                    st.markdown(
                        f"<div style='background:#fff;border:1px solid #e8e8e4;border-radius:8px;padding:1.25rem;'>"
                        f"<div style='display:flex;align-items:baseline;gap:0.75rem;margin-bottom:0.75rem;'>"
                        f"<span style='font-size:2.8rem;font-weight:300;color:{sc};"
                        f"letter-spacing:-0.03em;line-height:1;'>{score:.0f}</span>"
                        f"<div><div style='font-size:12px;font-weight:700;text-transform:uppercase;"
                        f"letter-spacing:0.08em;color:{sc};'>&#9679; {level} risk</div>"
                        f"<div style='font-size:11px;color:#aaa;'>Confidence {conf:.0%} &middot; 30-day window</div>"
                        f"</div></div>"
                        f"<div style='font-size:10px;text-transform:uppercase;letter-spacing:0.08em;"
                        f"color:#aaa;margin-bottom:0.5rem;'>Why</div></div>",
                        unsafe_allow_html=True,
                    )
                    # Factor bars — one st.markdown per factor
                    for f in factors[:3]:
                        raw = f.get("factor_name", "")
                        _rd = f.get("description", "")
                        d = _rd if _rd and not _rd.lower().startswith("risk factor:") else _fl.get(raw, raw.replace("_", " ").capitalize())
                        bw = min(100, abs(f.get("impact_score", 0)) * 100)
                        st.markdown(
                            f"<div style='margin-bottom:0.4rem;padding:0 1.25rem;'>"
                            f"<div style='font-size:11px;color:{COLORS['body']};margin-bottom:2px;'>{d}</div>"
                            f"<div style='height:3px;background:#f0f0ec;border-radius:2px;'>"
                            f"<div style='width:{bw:.0f}%;height:3px;background:{sc};border-radius:2px;'></div>"
                            f"</div></div>",
                            unsafe_allow_html=True,
                        )

                    st.button("Next →", key="journey_next_0", on_click=lambda: st.session_state.update({_step_key: 1}))

        # ── STEP 1: Outreach triggered ────────────────────────────
        elif step == 1:
            col_btn, col_result = st.columns([1, 2])
            _sms_msg = f"Hi, your {jp_med} refill is {jp_gap} days overdue. Reply YES to schedule pickup, or HELP if you need assistance with costs."
            with col_btn:
                st.markdown(f"""
                <div style="background:#fff; border:1px solid #e8e8e4; border-radius:8px;
                            padding:1.25rem;">
                    <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em;
                                color:#888; margin-bottom:0.5rem;">What happens</div>
                    <div style="font-size:13px; color:{COLORS['body']}; line-height:1.6;">
                        Because the score exceeds the threshold, the engine
                        automatically drafts and sends an SMS — no care manager involved.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_result:
                if st.button("▶  Send outreach", key="journey_btn_1", use_container_width=True):
                    with st.spinner("Sending…"):
                        _time.sleep(0.6)
                        st.session_state["journey_outreach"] = {
                            "status": "sent",
                            "id": str(_uuid.uuid4()),
                            "msg": _sms_msg,
                            "channel": "SMS",
                        }

                ov = st.session_state.get("journey_outreach")
                if ov:
                    st.markdown(f"""
                    <div style="background:#fff; border:1px solid #e8e8e4; border-radius:8px; padding:1.25rem;">
                        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em;
                                    color:#888; margin-bottom:0.4rem;">SMS to {jp_id} · {jp_age}y · {jp_cond}</div>
                        <div style="background:#f5f5f0; border-radius:6px; padding:0.75rem 1rem;
                                    font-size:13px; color:{COLORS['ink']}; line-height:1.6;
                                    margin-bottom:0.75rem;">
                            "{ov['msg']}"
                        </div>
                        <div style="font-size:11px; color:{COLORS['low']}; font-weight:600;">
                            &#10003; Sent automatically &nbsp;·&nbsp;
                            <span style="color:#aaa; font-weight:400;">ID: {ov['id'][:8]}…</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.button("Next →", key="journey_next_1", on_click=lambda: st.session_state.update({_step_key: 2}))

        # ── STEP 2: Patient replies ───────────────────────────────
        elif step == 2:
            col_btn, col_result = st.columns([1, 2])
            with col_btn:
                st.markdown(f"""
                <div style="background:#fff; border:1px solid #e8e8e4; border-radius:8px;
                            padding:1.25rem;">
                    <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em;
                                color:#888; margin-bottom:0.5rem;">What happens</div>
                    <div style="font-size:13px; color:{COLORS['body']}; line-height:1.6;">
                        The patient texts back. The AI identifies the barrier
                        and responds with targeted support — no human reads the message.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_result:
                if st.button("▶  Patient: \"It's too expensive\"", key="journey_btn_2", use_container_width=True):
                    with st.spinner("AI responding…"):
                        import time as _time
                        _time.sleep(1.0)
                        # Simulated response — no API call in demo mode
                        st.session_state["journey_conv"] = "demo-conv-001"
                        st.session_state["journey_chat"] = {
                            "conversation_id": "demo-conv-001",
                            "response": "Hi there, I understand cost can be a concern. Good news — there may be savings programs available for your Metformin. Would you like me to check what options might help reduce your costs?",
                            "identified_barrier": "cost",
                            "suggested_action": "check_copay_assistance",
                        }

                cv = st.session_state.get("journey_chat")
                if cv:
                    barrier = cv.get("identified_barrier", "cost")
                    action  = cv.get("suggested_action", "check_copay_assistance")
                    response = cv.get("response", "")
                    st.markdown(f"""
                    <div style="background:#fff; border:1px solid #e8e8e4; border-radius:8px; padding:1.25rem;">
                        <div style="background:#f5f5f0; border-radius:8px 8px 2px 8px;
                                    padding:0.65rem 0.9rem; font-size:13px; color:{COLORS['body']};
                                    max-width:85%; margin-left:auto; margin-bottom:0.5rem;">
                            "It's too expensive, I can't afford it"
                        </div>
                        <div style="background:#fff; border:1px solid #e8e8e4;
                                    border-radius:8px 2px 8px 8px;
                                    padding:0.65rem 0.9rem; font-size:13px; color:{COLORS['ink']};
                                    max-width:85%; margin-bottom:0.5rem; line-height:1.6;">
                            {response}
                        </div>
                        <div style="margin-top:0.5rem;">
                            <span style="font-size:10px; font-weight:600; text-transform:uppercase;
                                letter-spacing:0.06em; background:{COLORS['mid']}22; color:{COLORS['mid']};
                                padding:2px 8px; border-radius:3px; margin-right:4px;">
                                {barrier.replace('_',' ')}</span>
                            <span style="font-size:10px; font-weight:600; text-transform:uppercase;
                                letter-spacing:0.06em; background:{COLORS['low']}22; color:{COLORS['low']};
                                padding:2px 8px; border-radius:3px;">
                                &#8594; {action.replace('_',' ')}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.button("Next →", key="journey_next_2", on_click=lambda: st.session_state.update({_step_key: 3}))

        # ── STEP 3: Refill confirmed ──────────────────────────────
        elif step == 3:
            col_btn, col_result = st.columns([1, 2])
            new_pdc = min(1.0, jp_pdc + 0.18)
            with col_btn:
                st.markdown(f"""
                <div style="background:#fff; border:1px solid #e8e8e4; border-radius:8px;
                            padding:1.25rem;">
                    <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em;
                                color:#888; margin-bottom:0.5rem;">What happens</div>
                    <div style="font-size:13px; color:{COLORS['body']}; line-height:1.6;">
                        The patient picks up their prescription. The engine
                        detects the fill event and updates their adherence score automatically.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_result:
                if st.button("▶  Mark refill complete", key="journey_btn_3", use_container_width=True):
                    with st.spinner("Updating records…"):
                        _time.sleep(0.7)
                        st.session_state["journey_refill"] = {
                            "pdc_before": jp_pdc,
                            "pdc_after": new_pdc,
                            "gap_closed": jp_gap,
                        }

                rf = st.session_state.get("journey_refill")
                if rf:
                    st.markdown(f"""
                    <div style="background:#fff; border:1px solid #e8e8e4; border-radius:8px; padding:1.25rem;">
                        <div style="display:flex; gap:2rem; align-items:center; margin-bottom:1rem;">
                            <div>
                                <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em;
                                            color:#aaa; margin-bottom:2px;">PDC before</div>
                                <div style="font-size:2rem; font-weight:300; color:{COLORS['high']};">
                                    {rf['pdc_before']:.0%}</div>
                            </div>
                            <div style="font-size:1.5rem; color:#aaa;">→</div>
                            <div>
                                <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em;
                                            color:#aaa; margin-bottom:2px;">PDC after</div>
                                <div style="font-size:2rem; font-weight:300; color:{COLORS['low']};">
                                    {rf['pdc_after']:.0%}</div>
                            </div>
                        </div>
                        <div style="font-size:13px; color:{COLORS['body']}; line-height:1.6;">
                            {rf['gap_closed']}-day gap closed &nbsp;·&nbsp;
                            Adherence target restored &nbsp;·&nbsp;
                            <span style="color:{COLORS['low']}; font-weight:600;">
                            Patient back on track</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.button("Next →", key="journey_next_3", on_click=lambda: st.session_state.update({_step_key: 4}))

        # ── STEP 4: Hospitalization avoided ──────────────────────
        elif step == 4:
            st.markdown(f"""
            <div style="background:#fff; border:1px solid #e8e8e4; border-radius:8px;
                        padding:2rem 2.5rem; text-align:center;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">🏥</div>
                <div style="font-size:1.6rem; font-weight:300; color:{COLORS['ink']};
                            letter-spacing:-0.02em; margin-bottom:0.5rem;">
                    Hospitalization avoided.
                </div>
                <div style="font-size:14px; color:{COLORS['body']}; max-width:480px;
                            margin:0 auto 1.5rem; line-height:1.7;">
                    Without this intervention, <strong>{jp_id}</strong> was on a trajectory
                    toward a {jp_cond.lower()}-related admission within 30 days —
                    an average cost of <strong>$15,000</strong>.
                    Total engine cost for this patient: <strong>$0.05</strong> (one SMS).
                </div>
                <div style="display:inline-flex; gap:3rem; background:#f9f9f7;
                            border-radius:8px; padding:1rem 2rem;">
                    <div>
                        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em;
                                    color:#aaa; margin-bottom:4px;">Cost of outreach</div>
                        <div style="font-size:1.4rem; font-weight:300; color:{COLORS['ink']};">$0.05</div>
                    </div>
                    <div>
                        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em;
                                    color:#aaa; margin-bottom:4px;">Hospitalization avoided</div>
                        <div style="font-size:1.4rem; font-weight:300; color:{COLORS['low']};">$15,000</div>
                    </div>
                    <div>
                        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.08em;
                                    color:#aaa; margin-bottom:4px;">Return on this one patient</div>
                        <div style="font-size:1.4rem; font-weight:300; color:{COLORS['low']};">300,000×</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            if st.button("↩  Start over", key="journey_restart"):
                for k in ["journey_pred", "journey_outreach", "journey_chat",
                          "journey_conv", "journey_refill", _step_key]:
                    st.session_state.pop(k, None)
                st.rerun()


# ═══════════════════════════════════════════════════════════════
# TAB 3: WHO NEEDS HELP
# The patients who need attention, in language a VP understands.
# ═══════════════════════════════════════════════════════════════

with tab_patients:

    high_df = pop[pop["level"] == "high"].sort_values("risk", ascending=False).copy()
    avg_gap = high_df["days_gap"].mean()
    avg_copay = high_df["copay"].mean()

    st.markdown(f"""
    <p class="narrative">
        In this scenario, the model would flag
        <span class="highlight-red">{len(high_df)} patients</span> as likely
        to stop their medication within 30 days. Their simulated profile:
        an average of <strong>{avg_gap:.0f} days</strong> since last refill
        and a <strong>${avg_copay:.0f} copay</strong> per fill &mdash;
        the kind of pattern that precedes hospitalization.
    </p>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Flagged high risk", len(high_df))
    c2.metric("Avg days without refill", f"{avg_gap:.0f}")
    c3.metric("Avg copay burden", f"${avg_copay:.0f}")

    st.markdown("### Where adherence would be lowest")

    cond_pdc = pop.groupby("condition")["pdc"].mean().sort_values()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cond_pdc.values, y=cond_pdc.index, orientation="h",
        marker_color=[COLORS["high"] if v < 0.65 else COLORS["mid"] if v < 0.8 else COLORS["low"] for v in cond_pdc.values],
        text=[f"{v:.0%}" for v in cond_pdc.values], textposition="outside",
        textfont=dict(size=12, family="Inter", color=COLORS["ink"]),
    ))
    fig.add_vline(x=0.8, line_dash="dot", line_color=COLORS["faint"],
                  annotation_text="80% target",
                  annotation_font=dict(size=10, color=COLORS["dim"]))
    apply_layout(fig,
        height=220, xaxis=dict(visible=False, range=[0, 1.05]),
        yaxis=dict(showgrid=False), margin=dict(l=0, r=60, t=8, b=0),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("### Simulated high-risk patient list")

    display = high_df[["id", "age", "risk", "condition", "medication", "pdc", "days_gap", "copay"]].head(25).copy()
    display.columns = ["Patient", "Age", "Risk score", "Condition", "Medication",
                        "Adherence", "Days since fill", "Copay"]

    def _risk_bar(val):
        pct = max(0, min(100, (val - 50) / 50 * 100))
        return f"background: linear-gradient(90deg, rgba(192,57,43,{pct/100*0.25}) {pct}%, transparent {pct}%)"

    st.dataframe(
        display.style
              .format({"Risk score": "{:.0f}", "Adherence": "{:.0%}", "Copay": "${:.0f}"})
              .map(_risk_bar, subset=["Risk score"]),
        use_container_width=True,
        height=480,
        hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════
# TAB 4: WHAT WE'RE DOING
# Interventions — the actions the system takes, and what's working.
# ═══════════════════════════════════════════════════════════════

with tab_actions:

    success_rate = succeeded / total_i
    avg_h = intv["hours"].dropna().astype(float).mean()

    st.markdown(f"""
    <p class="narrative">
        In this demo, the engine would have triggered
        <strong>{total_i} automated interventions</strong> over 30 days.
        Based on simulated outcomes,
        <span class="highlight-green">{succeeded} would lead to a refill</span>
        &mdash; a <strong>{success_rate:.0%} success rate</strong>,
        with an average response time of <strong>{avg_h:.0f} hours</strong>.
    </p>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Interventions sent", f"{total_i}")
    c2.metric("Patients responded", f"{responded}", delta=f"{responded/total_i:.0%}")
    c3.metric("Led to a refill", f"{succeeded}", delta=f"{success_rate:.0%} success")
    c4.metric("Avg time to respond", f"{avg_h:.0f}h")

    st.markdown("### Which channels would work best")

    st.markdown("""
    <p style="font-size: 0.85rem; color: #888; margin-bottom: 0.5rem; max-width: 600px;">
        Success rate = the patient refilled their medication after outreach.
        Based on simulated channel distribution.
    </p>
    """, unsafe_allow_html=True)

    ch = intv.groupby("channel").agg(
        sent=("id", "count"),
        success=("status", lambda x: (x == "Successful").sum()),
    )
    ch["rate"] = (ch["success"] / ch["sent"]).round(2)
    ch = ch.sort_values("rate", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ch["rate"].values, y=ch.index, orientation="h",
        marker_color=[COLORS["low"] if v >= 0.3 else COLORS["mid"] if v >= 0.2 else COLORS["dim"] for v in ch["rate"].values],
        text=[f'{v:.0%}  ({ch.loc[ch.index[i], "sent"]} sent)' for i, v in enumerate(ch["rate"].values)],
        textposition="outside",
        textfont=dict(size=11, family="Inter", color=COLORS["ink"]),
    ))
    apply_layout(fig,
        height=200, xaxis=dict(visible=False, range=[0, 0.55]),
        yaxis=dict(showgrid=False), margin=dict(l=0, r=120, t=8, b=0),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("### Intervention pipeline")

    st.markdown("""
    <p style="font-size: 0.85rem; color: #888; margin-bottom: 0.5rem; max-width: 600px;">
        Each intervention flows through these stages. The engine moves patients
        from "Sent" all the way to "Refilled." Simulated distribution below.
    </p>
    """, unsafe_allow_html=True)

    status_order = ["Sent", "Delivered", "Responded", "Successful", "Failed"]
    status_labels = ["Sent", "Delivered", "Patient replied", "Refilled", "No response"]
    st_counts = intv["status"].value_counts().reindex(status_order).fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=status_labels, y=st_counts.values,
        marker_color=[COLORS["faint"], COLORS["dim"], COLORS["mid"], COLORS["low"], COLORS["high"]],
        text=[f"{int(v)}" for v in st_counts.values], textposition="outside",
        textfont=dict(size=12, family="Inter", color=COLORS["ink"]),
    ))
    apply_layout(fig,
        height=240, yaxis=dict(visible=False),
        xaxis=dict(showgrid=False, zeroline=False),
        margin=dict(l=0, r=0, t=8, b=0),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════
# TAB 5: IS IT WORKING
# Financial return — the bottom line.
# ═══════════════════════════════════════════════════════════════

with tab_returns:

    net_savings = total_savings - total_cost
    overall_roi = (net_savings / total_cost * 100) if total_cost > 0 else 0
    roi_ratio = total_savings / total_cost if total_cost > 0 else 0

    st.markdown(f"""
    <p class="narrative">
        In a simulated 12-week deployment, the program would generate
        <span class="highlight-green">${total_savings:,.0f} in avoided
        hospitalizations</span> on an investment of
        <strong>${total_cost:,.0f}</strong> &mdash; a projected return of
        <strong>${roi_ratio:.1f} for every $1 invested</strong>.
    </p>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background: #fff; border: 1px solid #e8e8e4; border-radius: 10px;
                padding: 2rem; text-align: center; max-width: 480px; margin-bottom: 1.5rem;">
        <p style="font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.12em;
                  color: #888; margin: 0 0 0.5rem 0; font-weight: 500;">Projected return on investment</p>
        <p style="font-size: 3.5rem; font-weight: 300; color: #1a1a1a; margin: 0;
                  letter-spacing: -0.03em; line-height: 1;">${roi_ratio:.0f} : $1</p>
        <p style="font-size: 0.85rem; color: #888; margin: 0.75rem 0 0 0;">
            ${total_savings:,.0f} saved &middot; ${total_cost:,.0f} invested &middot;
            {overall_roi:,.0f}% return</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Program cost", f"${latest_roi['cost']:,.0f}/wk")
    c2.metric("Projected savings", f"${latest_roi['savings']:,.0f}/wk")
    c3.metric("Weekly ROI", f"{latest_roi['roi']:.0f}%")

    st.markdown("### Projected savings vs. cost")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=roi["week"], y=roi["savings"], mode="lines+markers", name="Avoided hospitalizations",
        line=dict(color=COLORS["low"], width=2.5), marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(39,174,96,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=roi["week"], y=roi["cost"], mode="lines", name="Program cost",
        line=dict(color=COLORS["high"], width=1.5, dash="dot"),
    ))
    apply_layout(fig,
        height=300, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11, color=COLORS["dim"])),
        yaxis=dict(showgrid=True, gridcolor=COLORS["faint"], gridwidth=0.5, zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        margin=dict(l=0, r=0, t=28, b=0),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("### Model your own scenario")

    st.markdown("""
    <p style="font-size: 0.85rem; color: #888; margin-bottom: 1rem; max-width: 600px;">
        Adjust the inputs below to see how the program scales.
        The industry average hospitalization cost is $15,000.
    </p>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        patients_n = st.number_input("Patients in the program", 100, 100_000, 5000, step=500)
        success_pct = st.slider("Expected success rate", 20, 80, 45, format="%d%%",
                                help="Percentage of interventions that lead to a refill")
    with c2:
        cost_per = st.number_input("Cost per outreach ($)", 0.10, 50.0, 0.50, format="%.2f",
                                   help="SMS ~$0.05, Email ~$0.02, Voice ~$2.50, Care Mgr ~$25")
        hosp_cost = st.number_input("Avg hospitalization cost ($)", 5000, 50000, 15000, step=1000)

    est_interventions = patients_n * 0.5
    est_successful = est_interventions * (success_pct / 100)
    est_cost = est_interventions * cost_per
    est_avoided = est_successful * 0.15
    est_savings = est_avoided * hosp_cost
    est_roi_ratio = est_savings / est_cost if est_cost > 0 else 0

    st.markdown(f"""
    <div style="background: #fff; border: 1px solid #e8e8e4; border-radius: 8px;
                padding: 1.5rem; margin-top: 0.5rem; text-align: center;">
        <p style="font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em;
                  color: #888; margin: 0 0 0.5rem 0; font-weight: 500;">Projected return</p>
        <p style="font-size: 2.8rem; font-weight: 300; color: #1a1a1a; margin: 0;
                  letter-spacing: -0.03em;">${est_roi_ratio:.0f} : $1</p>
        <p style="font-size: 0.85rem; color: #888; margin: 0.75rem 0 0 0; line-height: 1.6;">
            {est_avoided:,.0f} hospitalizations avoided &middot;
            ${est_savings:,.0f} saved &middot;
            ${est_cost:,.0f} invested</p>
    </div>
    """, unsafe_allow_html=True)
