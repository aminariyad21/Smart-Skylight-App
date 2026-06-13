# =============================================================================
#  SMART SKYLIGHT  -  dashboard  (self-contained Streamlit app, dark theme)
#  CCAS 2.6 - Intelligent Systems Design  ·  Amina Riyad
#
#  A generative shading facade + a machine-learning surrogate that predicts
#  Solar Heat Gain live - in Grasshopper, and right here.
#
#  Predictions run from the portable forest (shading_rf_portable.json) in pure
#  Python - identical to the scikit-learn .pkl, so it deploys with no version pain.
#
#  RUN:  pip install -r requirements.txt   then   streamlit run app.py
#  Files needed (same folder): shading_units_dataset.csv, shading_units_raw.csv,
#       shading_rf_portable.json, facade_render.jpg, charts/, .streamlit/config.toml
# =============================================================================
import os, json, struct
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except Exception:
    pass

st.set_page_config(page_title="Smart Skylight", page_icon="\U0001F7E7", layout="wide")

# ----------------------------------------------------------------- dark CSS
CORAL, TEAL, BLUE = "#E8643C", "#5FA89F", "#5B8DC9"
st.markdown("""
<style>
.block-container {padding-top:1.4rem; max-width:1200px;}
h1,h2,h3,h4 {color:#FFFFFF !important; font-weight:700; letter-spacing:-.01em;}
h2 {font-size:30px; margin:.2em 0 .1em;}
h3 {font-size:21px;}
.tagline {color:#9BA1AC; font-size:16px; margin:-4px 0 2px;}
.idea {color:#C8CCD4; font-size:16px; line-height:1.62;}
.idea b {color:#FFFFFF;} .idea i {color:#C8CCD4;}
.metric {padding:4px 0 16px;}
.metric .ml {color:#8B919C; font-size:14px; margin-bottom:1px;}
.metric .mv {color:#FFFFFF; font-size:40px; font-weight:700; line-height:1.04;}
.metric .mv small {font-size:18px; color:#9BA1AC; font-weight:600;}
.flow {margin:6px 0 2px; line-height:2.7;}
.pill {display:inline-block; background:#171B24; border:1px solid #2A2F3A; border-radius:20px;
  padding:7px 15px; margin:3px 1px; color:#D8DCE2; font-size:14px;}
.arr {color:#E8643C; margin:0 4px; font-weight:700;}
table.vt {width:100%; border-collapse:collapse; margin-top:8px; font-size:14px;}
table.vt th {text-align:left; color:#8B919C; font-weight:600; padding:8px 10px;
  border-bottom:1px solid #2A2F3A; font-size:13px;}
table.vt td {color:#D8DCE2; padding:8px 10px; border-bottom:1px solid #1C212B;}
table.vt td:first-child {color:#EDEFF2; font-family:ui-monospace,monospace;}
.kicker {font-size:12px; letter-spacing:.13em; text-transform:uppercase; color:#5FA89F; margin:16px 0 3px;}
.verdict {border-radius:12px; padding:18px 22px; margin:6px 0 10px; border:1px solid;}
.note {background:#12161E; border:1px solid #232936; border-radius:10px; padding:14px 18px; color:#C8CCD4; font-size:15px; line-height:1.55;}
.note b {color:#FFD9A8;}
.stTabs [data-baseweb="tab-list"] {gap:24px; border-bottom:1px solid #21262F; flex-wrap:wrap;}
.stTabs [data-baseweb="tab"] {font-size:15px; color:#9BA1AC; padding:8px 2px; background:transparent;}
.stTabs [aria-selected="true"] {color:#E8643C !important;}
.stTabs [data-baseweb="tab-highlight"] {background:#E8643C;}
[data-testid="stMetric"] {background:#12161E; border:1px solid #232936; border-radius:8px; padding:10px 14px;}
[data-testid="stMetricValue"] {color:#FFFFFF;}
.stSlider label {color:#9BA1AC !important;}
img {border-radius:8px;}
</style>
""", unsafe_allow_html=True)

pio.templates["skydark"] = go.layout.Template(layout=go.Layout(
    font=dict(family="sans-serif", size=13, color="#C8CCD4"),
    title=dict(font=dict(size=17, color="#FFFFFF")),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[CORAL, TEAL, BLUE, "#E0A33C", "#9BA1AC"],
    xaxis=dict(gridcolor="#21262F", zerolinecolor="#2A2F3A"),
    yaxis=dict(gridcolor="#21262F", zerolinecolor="#2A2F3A"),
    margin=dict(l=10, r=10, t=48, b=10), legend=dict(bgcolor="rgba(0,0,0,0)")))
pio.templates.default = "plotly_dark+skydark"

# ----------------------------------------------------------------- constants
NUM = ["U_Divisions", "V_Divisions", "Depth_Factor",
       "Attractor_Strength", "Aperture_Target", "Max_Rotation_deg"]
FEATURES = NUM + ["Orientation"]
FEATURE_COLS = FEATURES
TARGET   = "Solar_Heat_Gain_kWh_m2"
ORIENTS  = ["N", "E", "S", "W"]
EXPOSURE = {"S": 1294, "W": 1090, "E": 1090, "N": 552}   # real Cairo annual irradiance (pvlib clear-sky)
NICE = {"U_Divisions": "Panels across (U)", "V_Divisions": "Panels up (V)",
        "Depth_Factor": "Fin depth", "Attractor_Strength": "Attractor pull",
        "Aperture_Target": "Aperture (openness)", "Max_Rotation_deg": "Panel rotation",
        "Orientation": "Orientation"}
# upgraded model metrics (Random Forest deployed)
METRICS = {"r2": 0.952, "cv": 0.933, "rmse": 37.7}
COMPARE = pd.DataFrame({
    "Model": ["Gradient Boosting", "Random Forest  \u2605", "Decision Tree", "Linear Regression"],
    "R\u00b2 test": [0.957, 0.952, 0.897, 0.870],
    "R\u00b2 5-fold CV": [0.943, 0.933, 0.849, 0.881],
    "RMSE (kWh/m\u00b2)": [35.95, 37.71, 55.64, 62.40]})

def kicker(t): st.markdown(f'<div class="kicker">{t}</div>', unsafe_allow_html=True)
def metric_card(label, value):
    st.markdown(f'<div class="metric"><div class="ml">{label}</div><div class="mv">{value}</div></div>',
                unsafe_allow_html=True)
def vtable(headers, rows):
    h = "".join(f"<th>{x}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    st.markdown(f'<table class="vt"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>',
                unsafe_allow_html=True)

# ----------------------------------------------------------------- data + model
@st.cache_data(show_spinner=False)
def load_csv(name): return pd.read_csv(name) if os.path.exists(name) else None
@st.cache_resource(show_spinner=False)
def load_forest():
    with open("shading_rf_portable.json") as f: return json.load(f)
def f32(v): return struct.unpack("f", struct.pack("f", v))[0]

def predict_df(forest, frame):
    cats, order = forest["orientation_categories"], forest["numeric_order"]
    trees, nt = forest["trees"], forest["n_trees"]; out = []
    for _, r in frame.iterrows():
        oh = [1.0 if c == str(r["Orientation"]).upper() else 0.0 for c in cats]
        x  = [f32(v) for v in (oh + [float(r[n]) for n in order])]; tot = 0.0
        for tr in trees:
            cl, cr, fe, th, va = tr["cl"], tr["cr"], tr["f"], tr["t"], tr["v"]; n = 0
            while cl[n] != cr[n]: n = cl[n] if x[fe[n]] <= th[n] else cr[n]
            tot += va[n]
        out.append(tot / nt)
    return np.array(out)
def predict_one(forest, d): return float(predict_df(forest, pd.DataFrame([d]))[0])

def img(name, caption=None):
    p = os.path.join("charts", name)
    if os.path.exists(p): st.image(p, width='stretch', caption=caption)
    else: st.info(f"chart not found: {name}")

def gh_export_block(d, shg, key):
    row = {k: d[k] for k in FEATURE_COLS}; row["Predicted_SHG_kWh_m2"] = round(float(shg), 1)
    kicker("Output \u2014 send this design to Grasshopper")
    cc = st.columns(2)
    cc[0].download_button("\u2b07  Parameters as CSV", pd.DataFrame([row]).to_csv(index=False).encode(),
        file_name="skylight_design.csv", mime="text/csv", key=f"csv_{key}", width='stretch')
    cc[1].download_button("\u2b07  Parameters as JSON", json.dumps(row, indent=2).encode(),
        file_name="skylight_design.json", mime="application/json", key=f"json_{key}", width='stretch')
    st.caption("Wire these six values + orientation into the **GH predictive node** to rebuild this exact "
               "screen in Rhino \u2014 the node returns the same predicted heat gain.")

def facade_figure(d):
    nx, ny = 26, 16
    xs, ys = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny)); xs, ys = xs.ravel(), ys.ravel()
    ax_, ay_ = 0.74, 0.58
    falloff = 2.0 * np.exp(-3.0 * np.sqrt((xs-ax_)**2 + (ys-ay_)**2))
    tilt = np.clip(d["Max_Rotation_deg"] * (0.5 + 5.0*d["Attractor_Strength"]*falloff), 0, 90)
    open_ = np.clip((0.2 + 0.8*d["Aperture_Target"]/100.0) * (tilt/90.0), 0, 1)
    sym = np.where(((xs*(nx-1)).round() + (ys*(ny-1)).round()) % 2 == 0, "triangle-up", "triangle-down")
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers",
        marker=dict(symbol=sym, angle=tilt, size=11+7*d["Depth_Factor"], color=open_, colorscale="YlOrRd",
                    cmin=0, cmax=1, colorbar=dict(title="openness"), line=dict(width=.4, color="#3A3F4A")),
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[ax_], y=[ay_], mode="markers+text", marker=dict(symbol="star", size=15, color=CORAL),
        text=["attractor"], textposition="bottom center", textfont=dict(color="#C8CCD4", size=12), hoverinfo="skip"))
    fig.update_xaxes(visible=False, range=[-.04, 1.04])
    fig.update_yaxes(visible=False, range=[-.06, 1.06], scaleanchor="x")
    fig.update_layout(height=470, showlegend=False, title="The screen, live \u2014 panels rotate & open as you design")
    return fig

df  = load_csv("shading_units_dataset.csv")
raw = load_csv("shading_units_raw.csv")
forest = load_forest()
if df is not None:
    SHG_LO, SHG_HI = float(df[TARGET].quantile(.05)), float(df[TARGET].quantile(.95))

# ----------------------------------------------------------------- header
st.markdown('<div class="tagline">A generative shading facade + a machine-learning surrogate that predicts '
            'Solar Heat Gain live \u2014 in Grasshopper, and right here.</div>', unsafe_allow_html=True)

T = st.tabs(["\U0001F3E0 Project", "\U0001F4D3 Data journey", "\U0001F4CA Analysis",
             "\u2699\uFE0F Model", "\U0001F3A8 Design studio", "\U0001F997 Grasshopper & files"])

# ============================================================ 0  PROJECT
with T[0]:
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("## The idea")
        st.markdown('<div class="idea">An <b>attractor-driven triangular shading facade</b> is built in '
            'Grasshopper. Six sliders generate the geometry \u2014 and the question every designer has is: '
            '<i>how much solar energy will this variant let through?</i> Normally that needs a slow separate '
            'analysis. This project trains a <b>surrogate model</b> on sampled designs so the answer arrives '
            '<b>instantly, while the sliders move.</b></div>', unsafe_allow_html=True)
        st.markdown('<div class="idea" style="margin-top:14px;"><b>What we predict:</b> Solar Heat Gain '
            '(kWh/m\u00b2) \u2014 the number an architect minimises to fight cooling loads in a hot city like '
            'Cairo.</div>', unsafe_allow_html=True)
    with right:
        a, b = st.columns(2)
        with a: metric_card("Designs in dataset", "500")
        with b: metric_card("Test accuracy", "R\u00b2 \u2248 0.95")
        c, d2 = st.columns(2)
        with c: metric_card("Features", "6 + orientation")
        with d2: metric_card("Deployed model", "Random Forest")

    st.markdown("## The workflow (end to end)")
    steps = ["Grasshopper model", "CSV dataset", "Data cleaning", "Feature engineering", "Train / test split",
             "Train model", "Evaluate", "Export .pkl", "GH predictive node", "Dashboard & website"]
    html = '<div class="flow">'
    for i, s in enumerate(steps):
        html += f'<span class="pill">{s}</span>'
        if i < len(steps) - 1: html += '<span class="arr">\u2192</span>'
    st.markdown(html + "</div>", unsafe_allow_html=True)

    st.markdown("## The variables")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Features (X)** \u2014 the six Grasshopper sliders + context")
        vtable(["feature", "range", "meaning"], [
            ["U_Divisions", "1 \u2013 100", "panels horizontally"],
            ["V_Divisions", "1 \u2013 100", "panels vertically"],
            ["Depth_Factor", "0.1 \u2013 1.0", "fin projection depth"],
            ["Attractor_Strength", "0 \u2013 0.1", "local rotation pull"],
            ["Aperture_Target", "0 \u2013 100", "opening size"],
            ["Max_Rotation_deg", "0 \u2013 100", "panel tilt"],
            ["Orientation", "N/E/S/W", "facade direction"]])
    with v2:
        st.markdown("**Targets (Y)** \u2014 derived per design; \u2b50 is what the model predicts")
        vtable(["target", "unit", "meaning"], [
            ["Total_Shading_Area_m2", "m\u00b2", "facade + slanted fins"],
            ["Openness_Ratio", "0\u20131", "how open the screen is"],
            ["Avg_Panel_Tilt_deg", "\u00b0", "average panel tilt"],
            ["Projection_Depth_m", "m", "fin depth"],
            ["Material_Volume_m3", "m\u00b3", "material used"],
            ["\u2b50 Solar_Heat_Gain_kWh_m2", "kWh/m\u00b2", "what the model predicts"]])

# ============================================================ 1  DATA JOURNEY
with T[1]:
    st.markdown("## From a messy export to a model-ready table")
    st.markdown('<div class="idea">The pipeline turns 500 sampled designs into a clean dataset in three '
                'classic steps \u2014 cleansing, transformation, integration.</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a: metric_card("Raw rows", "508")
    with b: metric_card("Removed / fixed", "5 dup \u00b7 13 NA \u00b7 3 bad")
    with c: metric_card("Clean rows", "500")

    kicker("Step 1 \u2014 cleansing")
    st.markdown('<div class="note">The raw export carries realistic mess: <b>5 duplicate rows</b>, '
        '<b>13 missing cells</b>, a stray <b>text value</b> in a number column, and <b>3 impossible outliers</b> '
        '(rotation 999\u00b0, U = \u22125, depth 7.5). Cleansing coerces numbers, drops duplicates, '
        'median-imputes the blanks, and removes out-of-range rows \u2192 <b>508 \u2192 500</b>.</div>',
        unsafe_allow_html=True)
    if raw is not None:
        with st.expander("Peek at the raw file (the dirty rows are in here)"):
            st.dataframe(raw.head(12), width='stretch', height=300)

    kicker("Step 2 \u2014 transformation")
    st.markdown('<div class="note">A model only reads numbers, so the one text column, <b>Orientation</b>, '
        'is <b>one-hot encoded</b> into four 0/1 flags (Orient_N/E/S/W). "S" becomes [0,0,1,0]. The six '
        'numeric sliders pass through unchanged \u2014 a Random Forest needs no scaling.</div>',
        unsafe_allow_html=True)

    kicker("Step 3 \u2014 integration (the target)")
    st.markdown('<div class="note">The heat-gain target is built from <b>real Cairo solar exposure</b> '
        '(annual irradiance on each vertical facade, computed with pvlib: S\u22481294, E/W\u22481090, '
        'N\u2248552 kWh/m\u00b2) \u00d7 a <b>geometry-based transmittance</b> (open-area fraction from aperture '
        'and rotation, reduced by fin shading). <b>SHG = E_beam\u00b7\u03c4_beam + E_diffuse\u00b7\u03c4_diffuse</b>. '
        'Honest note: the exposure is real; the shading is an engineering estimate, not a ray-traced '
        'simulation.</div>', unsafe_allow_html=True)
    if df is not None:
        st.markdown("**Heat gain by orientation** (mean of the clean dataset)")
        m = df.groupby("Orientation")[TARGET].mean().reindex(["S", "W", "E", "N"]).reset_index()
        fig = px.bar(m, x="Orientation", y=TARGET, color="Orientation",
                     color_discrete_sequence=[CORAL, "#E0A33C", BLUE, TEAL])
        fig.update_layout(height=320, showlegend=False, yaxis_title="mean SHG (kWh/m\u00b2)")
        st.plotly_chart(fig, width='stretch')

# ============================================================ 2  ANALYSIS
with T[2]:
    st.markdown("## What the data and the model reveal")
    c1, c2 = st.columns(2)
    with c1:
        kicker("Correlation matrix")
        img("correlation_matrix.png")
        st.caption("Linear relationships. Openness (0.89) and aperture (0.81) drive heat gain; depth works "
                   "slightly against it (\u22120.15).")
    with c2:
        kicker("Feature importance")
        img("feature_importance.png")
        st.caption("What the forest leans on: aperture dominates (0.66), then rotation and orientation.")

    kicker("Design map \u2014 the model swept across the design space")
    img("design_map.png")
    st.caption("For each orientation, predicted heat gain across rotation \u00d7 aperture. South & west run "
               "hot as the screen opens; north stays cool.")

    if df is not None:
        kicker("Distribution of heat gain by orientation (interactive)")
        fig = px.histogram(df, x=TARGET, color="Orientation", nbins=40, barmode="overlay", opacity=.7,
                           color_discrete_map={"S": CORAL, "W": "#E0A33C", "E": BLUE, "N": TEAL})
        fig.update_layout(height=360, xaxis_title="Solar Heat Gain (kWh/m\u00b2)")
        st.plotly_chart(fig, width='stretch')

# ============================================================ 3  MODEL
with T[3]:
    st.markdown("## Supervised regression, evaluated honestly")
    a, b, c = st.columns(3)
    with a: metric_card("R\u00b2 (test, unseen 20%)", f'{METRICS["r2"]:.3f}')
    with b: metric_card("R\u00b2 (5-fold CV)", f'{METRICS["cv"]:.3f}')
    with c: metric_card("RMSE", f'{METRICS["rmse"]:.1f} <small>kWh/m\u00b2</small>')

    c1, c2 = st.columns(2)
    with c1:
        kicker("Four models compared")
        img("model_comparison.png")
    with c2:
        kicker("Predicted vs actual (held-out test)")
        img("pred_vs_actual.png")

    st.dataframe(COMPARE, hide_index=True, width='stretch')

    kicker("Learning curve")
    img("learning_curve.png")
    st.markdown('<div class="note"><b>Why Random Forest?</b> Gradient Boosting edged it (0.957 vs 0.952) but '
        'within noise; the forest is more robust on a small, noisy dataset, its test and CV scores stay close '
        '(no overfitting), and it gives clean feature importances. A Random Forest is 300 decision trees, each '
        'trained on a different resample of the data, with their predictions averaged.</div>',
        unsafe_allow_html=True)

# ============================================================ 4  DESIGN STUDIO
with T[4]:
    st.markdown("## Design a screen \u2014 get the heat gain instantly")
    cL, cR = st.columns([1, 1.15])
    with cL:
        o   = st.selectbox("Orientation", ORIENTS, index=2,
                           format_func=lambda x: f"{x}  \u00b7  {EXPOSURE[x]} kWh/m\u00b2 exposure")
        u   = st.slider("Panels across (U)", 1, 100, 50)
        v   = st.slider("Panels up (V)", 1, 100, 50)
        dep = st.slider("Fin depth", 0.10, 1.00, 0.55)
        att = st.slider("Attractor pull", 0.00, 0.10, 0.05)
        ap  = st.slider("Aperture (openness)", 0, 100, 55)
        rot = st.slider("Panel rotation", 0, 100, 45)
        d = {"U_Divisions": u, "V_Divisions": v, "Depth_Factor": dep, "Attractor_Strength": att,
             "Aperture_Target": ap, "Max_Rotation_deg": rot, "Orientation": o}
    with cR:
        shg = predict_one(forest, d)
        frac = 0 if SHG_HI == SHG_LO else np.clip((shg - SHG_LO)/(SHG_HI - SHG_LO), 0, 1)
        col = CORAL if frac > 0.66 else ("#E0A33C" if frac > 0.33 else TEAL)
        verdict = "runs hot" if frac > 0.66 else ("moderate" if frac > 0.33 else "stays cool")
        st.markdown(f'<div class="verdict" style="border-color:{col}; background:{col}1A;">'
            f'<div style="font-size:12px; letter-spacing:.1em; color:{col}; text-transform:uppercase;">'
            f'Predicted solar heat gain \u00b7 {verdict}</div>'
            f'<div style="font-size:46px; font-weight:700; color:#FFFFFF; line-height:1.05;">{shg:,.0f} '
            f'<span style="font-size:18px; color:#9BA1AC;">kWh/m\u00b2/yr</span></div></div>',
            unsafe_allow_html=True)
        st.plotly_chart(facade_figure(d), width='stretch')
    gh_export_block(d, shg, "studio")

    st.divider()
    st.markdown("## Or let the model find the best design")
    oc1, oc2 = st.columns([1, 1.25])
    with oc1:
        o2 = st.selectbox("Orientation ", ORIENTS, index=2, key="opt_o",
                          format_func=lambda x: f"{x}  \u00b7  {EXPOSURE[x]} kWh/m\u00b2")
        mode = st.radio("Goal", ["Coolest possible screen", "Most open screen under a heat budget"], key="opt_m")
        budget = None
        if mode.startswith("Most open"):
            lo, hi = int(df[TARGET].min()), int(df[TARGET].max())
            budget = st.slider("Heat-gain budget (kWh/m\u00b2/yr)", lo, hi, int((lo+hi)/2), key="opt_b")
        go_ = st.button("\U0001F50D  Find best design", type="primary", width='stretch', key="opt_go")
    with oc2:
        if go_:
            with st.spinner("Scoring 3,000 candidate screens\u2026"):
                n = 3000; rng = np.random.default_rng(0)
                cand = pd.DataFrame({"Orientation": o2,
                    "U_Divisions": rng.integers(1,101,n), "V_Divisions": rng.integers(1,101,n),
                    "Depth_Factor": rng.uniform(.1,1,n).round(2), "Attractor_Strength": rng.uniform(0,.1,n).round(3),
                    "Aperture_Target": rng.integers(0,101,n), "Max_Rotation_deg": rng.integers(0,101,n)})
                cand["SHG"] = predict_df(forest, cand)
                if budget is None:
                    best = cand.loc[cand["SHG"].idxmin()]
                else:
                    ok = cand[cand["SHG"] <= budget]
                    best = (ok.loc[ok["Aperture_Target"].idxmax()] if len(ok) else cand.loc[cand["SHG"].idxmin()])
                    if not len(ok): st.warning(f"Nothing stays under {budget} facing {o2}; showing coolest.")
            ik = ("U_Divisions","V_Divisions","Aperture_Target","Max_Rotation_deg")
            d2 = {k:(int(best[k]) if k in ik else round(float(best[k]),3) if k in ("Depth_Factor","Attractor_Strength")
                  else o2) for k in FEATURE_COLS}
            shg2 = float(best["SHG"])
            st.markdown(f'<div class="verdict" style="border-color:{TEAL}; background:{TEAL}1A;">'
                f'<div style="font-size:12px; letter-spacing:.1em; color:{TEAL}; text-transform:uppercase;">'
                f'Recommended screen \u00b7 predicted heat gain</div>'
                f'<div style="font-size:40px; font-weight:700; color:#FFFFFF;">{shg2:,.0f} '
                f'<span style="font-size:17px; color:#9BA1AC;">kWh/m\u00b2/yr</span></div></div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({"Parameter": [NICE[k] for k in FEATURE_COLS],
                                       "Value": [d2[k] for k in FEATURE_COLS]}), hide_index=True, width='stretch')
            gh_export_block(d2, shg2, "opt")
        else:
            st.info("Pick a goal and press **Find best design** \u2014 the model searches and returns a screen "
                    "you can send to Grasshopper.")

# ============================================================ 5  GRASSHOPPER & FILES
with T[5]:
    st.markdown("## The predictive node, and everything to download")
    st.markdown('<div class="note">The <b>predictive node</b> is a Python component inside Grasshopper that '
        'loads this trained model and outputs predicted heat gain from the same six sliders \u2014 so geometry '
        'and its cooling consequence are generated together, live. The portable build reads the JSON below and '
        'needs <b>no scikit-learn</b>, so it runs even where Rhino can\u2019t install packages.</div>',
        unsafe_allow_html=True)

    kicker("Downloads")
    files = [("shading_shg_rf_model.pkl", "Trained Random Forest (scikit-learn)"),
             ("shading_rf_portable.json", "Same model, dependency-free (for the GH node)"),
             ("shading_units_dataset.csv", "Clean 500-design dataset"),
             ("shading_units_raw.csv", "Raw 508-row dataset (pre-cleaning)"),
             ("gh_node_predict_portable.py", "Grasshopper predictive-node script")]
    for fn, desc in files:
        if os.path.exists(fn):
            with open(fn, "rb") as fh:
                st.download_button(f"\u2b07  {fn}  \u2014  {desc}", fh.read(), file_name=fn,
                                   key=f"dl_{fn}", width='stretch')

    st.caption("Drop the script into a Python 3 component in Grasshopper, wire the six sliders + orientation "
               "in, point it at the .pkl (or the JSON), and it returns the predicted Solar Heat Gain live.")
