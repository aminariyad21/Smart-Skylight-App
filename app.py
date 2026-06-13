# =============================================================================
#  SMART SKYLIGHT  -  interactive design studio  (self-contained Streamlit app)
#  CCAS 2.6 - Intelligent Systems Design
#
#  An adaptive triangular shading screen for a hot climate. Move the sliders,
#  watch the screen react, and read the predicted Solar Heat Gain instantly.
#
#  ZERO ML DEPENDENCY: predictions run from the portable forest
#  (shading_rf_portable.json) in pure Python - byte-for-byte identical to the
#  scikit-learn .pkl, so this deploys on Streamlit Cloud with NO version pain.
#
#  RUN LOCALLY:   pip install -r requirements.txt
#                 streamlit run app.py
#  FILES NEEDED (same folder): shading_units_dataset.csv, shading_units_raw.csv,
#                 shading_rf_portable.json, facade_render.jpg
# =============================================================================
import os, json, struct
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Anchor relative file reads (CSV / JSON / image) to THIS file's folder, so the
# app works no matter where the host launches it from (e.g. Streamlit Cloud runs
# from the repo root, which may differ from the app's directory).
try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
except Exception:
    pass

st.set_page_config(page_title="Smart Skylight", page_icon="\U0001F7E7", layout="wide")

# ----------------------------------------------------------------- palette/CSS
INK, MUTED   = "#17130F", "#8C8576"
CARD, LINE   = "#FBF8F1", "#C9C0AC"
AMBER, TEAL, NAVY = "#C75B39", "#1F6F6B", "#1F3A5F"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,700;1,9..144,500&family=Spline+Sans:wght@400;500;600&family=Space+Mono:wght@400;700&display=swap');
html, body, [data-testid="stAppViewContainer"] {background-color:#F5F1E8; color:#17130F;}
[data-testid="stHeader"] {background:rgba(245,241,232,.85);}
.block-container {padding-top:1.1rem; max-width:1180px;}
p, li, label, [data-testid="stMarkdownContainer"] p {font-family:'Spline Sans',sans-serif;}
code, pre {font-family:'Space Mono',monospace;}
h1,h2,h3,h4 {font-family:'Fraunces',Georgia,serif !important; color:#17130F !important; letter-spacing:-.01em;}
.sk-title {font-family:'Fraunces',serif; font-weight:700; font-size:46px; line-height:1.04; margin:0 0 4px;}
.sk-title em {font-style:italic; font-weight:500; color:#C75B39;}
.sk-mark {color:#C75B39; font-size:24px; line-height:1;}
.sk-sub {color:#5F5E5A; font-size:17px; max-width:820px;}
.sk-chip {display:inline-block; font-family:'Space Mono',monospace; font-size:11.5px; letter-spacing:.04em;
  color:#A8500F; border:1px solid #C75B39; border-radius:20px; padding:3px 12px; margin:10px 8px 0 0; background:#FBF8F1;}
.sk-rule {height:2px; background:#17130F; margin:16px 0 3px;}
.sk-rule2 {height:1px; background:#C9C0AC; margin:0 0 4px;}
.sk-kicker {font-family:'Space Mono',monospace; font-size:11.5px; letter-spacing:.16em;
  text-transform:uppercase; color:#1F6F6B; margin:10px 0 2px;}
.sk-verdict {border-radius:10px; padding:16px 20px; margin:2px 0 8px; border:1px solid;}
.sk-step {background:#FBF8F1; border:1px solid #C9C0AC; border-left:4px solid #C75B39;
  border-radius:6px; padding:10px 14px; margin:8px 0;}
.sk-eq {background:#1F3A5F; color:#F5F1E8; border-radius:8px; padding:14px 18px; font-family:'Space Mono',monospace;
  font-size:13px; line-height:1.7; overflow-x:auto;}
.stTabs [data-baseweb="tab-list"] {gap:16px; border-bottom:1px solid #DDD6C5; flex-wrap:wrap;}
.stTabs [data-baseweb="tab"] {font-family:'Space Mono',monospace; text-transform:uppercase;
  letter-spacing:.05em; font-size:12px; color:#5F5E5A; padding:10px 2px; background:transparent;}
.stTabs [aria-selected="true"] {color:#A8500F !important;}
.stTabs [data-baseweb="tab-highlight"] {background-color:#C75B39;}
[data-testid="stMetric"] {background:#FBF8F1; border:1px solid #C9C0AC; border-radius:6px; padding:12px 16px;}
[data-testid="stMetricLabel"] p {font-family:'Space Mono',monospace !important; font-size:11px !important;
  letter-spacing:.07em; text-transform:uppercase; color:#8C8576;}
[data-testid="stMetricValue"] {font-family:'Fraunces',serif; color:#17130F;}
[data-testid="stExpander"] {background:#FBF8F1; border:1px solid #C9C0AC; border-radius:6px;}
.stSlider label {font-family:'Space Mono',monospace !important; font-size:11.5px !important;
  letter-spacing:.04em; text-transform:uppercase; color:#5F5E5A;}
</style>
""", unsafe_allow_html=True)

pio.templates["sky"] = go.layout.Template(layout=go.Layout(
    font=dict(family="Spline Sans, sans-serif", size=13, color=INK),
    title=dict(font=dict(family="Fraunces, serif", size=18, color=INK)),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[AMBER, TEAL, NAVY, "#BA7517", "#5F5E5A"],
    xaxis=dict(gridcolor="#E7E1D2", zerolinecolor=LINE),
    yaxis=dict(gridcolor="#E7E1D2", zerolinecolor=LINE),
    margin=dict(l=10, r=10, t=48, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)")))
pio.templates.default = "plotly_white+sky"

# ----------------------------------------------------------------- constants
NUM = ["U_Divisions", "V_Divisions", "Depth_Factor",
       "Attractor_Strength", "Aperture_Target", "Max_Rotation_deg"]
FEATURES = NUM + ["Orientation"]
TARGET   = "Solar_Heat_Gain_kWh_m2"
ORIENTS  = ["N", "E", "S", "W"]
ORIENT_COLORS = {"S": AMBER, "W": "#BA7517", "E": NAVY, "N": TEAL}
EXPOSURE = {"S": 900, "W": 820, "E": 720, "N": 450}
NICE = {"U_Divisions": "Panels across (U)", "V_Divisions": "Panels up (V)",
        "Depth_Factor": "Fin depth", "Attractor_Strength": "Attractor pull",
        "Aperture_Target": "Aperture (openness)", "Max_Rotation_deg": "Panel rotation"}

def kicker(t): st.markdown(f'<div class="sk-kicker">{t}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------- data + model
@st.cache_data(show_spinner=False)
def load_csv(name):
    return pd.read_csv(name) if os.path.exists(name) else None

@st.cache_resource(show_spinner=False)
def load_forest():
    with open("shading_rf_portable.json") as f:
        return json.load(f)

def f32(v): return struct.unpack("f", struct.pack("f", v))[0]

def predict_df(forest, frame):
    cats, order = forest["orientation_categories"], forest["numeric_order"]
    trees, nt = forest["trees"], forest["n_trees"]
    out = []
    for _, r in frame.iterrows():
        oh = [1.0 if c == str(r["Orientation"]).upper() else 0.0 for c in cats]
        x  = [f32(v) for v in (oh + [float(r[n]) for n in order])]
        tot = 0.0
        for tr in trees:
            cl, cr, fe, th, va = tr["cl"], tr["cr"], tr["f"], tr["t"], tr["v"]
            n = 0
            while cl[n] != cr[n]:
                n = cl[n] if x[fe[n]] <= th[n] else cr[n]
            tot += va[n]
        out.append(tot / nt)
    return np.array(out)

def predict_one(forest, d):
    return float(predict_df(forest, pd.DataFrame([d]))[0])

# ----------------------------------------------------------------- live facade
def facade_figure(d):
    nx, ny = 26, 16
    xs, ys = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    xs, ys = xs.ravel(), ys.ravel()
    ax_, ay_ = 0.74, 0.58
    dist = np.sqrt((xs - ax_)**2 + (ys - ay_)**2)
    falloff = 2.0 * np.exp(-3.0 * dist)
    tilt = np.clip(d["Max_Rotation_deg"] * (0.5 + 5.0 * d["Attractor_Strength"] * falloff), 0, 90)
    open_ = np.clip((0.2 + 0.8 * d["Aperture_Target"]/100.0) * (tilt/90.0), 0, 1)
    sym = np.where(((xs*(nx-1)).round() + (ys*(ny-1)).round()) % 2 == 0, "triangle-up", "triangle-down")
    size = 11 + 7*d["Depth_Factor"]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers",
        marker=dict(symbol=sym, angle=tilt, size=size, color=open_, colorscale="YlOrRd",
                    cmin=0, cmax=1, colorbar=dict(title="openness"), line=dict(width=.4, color="#4A3A2C")),
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[ax_], y=[ay_], mode="markers+text",
        marker=dict(symbol="star", size=14, color=NAVY), text=["attractor"],
        textposition="bottom center", textfont=dict(color=INK, size=12), hoverinfo="skip"))
    fig.update_xaxes(visible=False, range=[-.04, 1.04])
    fig.update_yaxes(visible=False, range=[-.06, 1.06], scaleanchor="x")
    fig.update_layout(height=480, showlegend=False,
                      title="The screen, live \u2014 panels rotate & open as you design")
    return fig

df  = load_csv("shading_units_dataset.csv")
raw = load_csv("shading_units_raw.csv")
forest = load_forest()

# precompute SHG bounds for the verdict scale
if df is not None:
    SHG_LO, SHG_HI = float(df[TARGET].quantile(.05)), float(df[TARGET].quantile(.95))

# ----------------------------------------------------------------- header
st.markdown("""
<div>
  <div class="sk-mark">&#9650;</div>
  <div class="sk-title">Smart <em>Skylight</em></div>
  <div class="sk-sub">An adaptive triangular shading screen for a hot climate &mdash; design it with
  six sliders and feel the solar-heat consequence instantly. A trained Random Forest powers the numbers,
  quietly, in milliseconds.</div>
  <div>
    <span class="sk-chip">Cairo &middot; hot, high-irradiance</span>
    <span class="sk-chip">attractor-driven screen</span>
    <span class="sk-chip">R&sup2; = 0.937 on unseen designs</span>
    <span class="sk-chip">zero-install prediction</span>
  </div>
  <div class="sk-rule"></div><div class="sk-rule2"></div>
</div>
""", unsafe_allow_html=True)

if df is None:
    st.error("shading_units_dataset.csv not found next to app.py.")
    st.stop()

T = st.tabs(["The idea", "Facade studio", "Data processing", "Explore the data",
             "Design map", "The model"])

# ============================================================ 1  THE IDEA
with T[0]:
    kicker("The problem")
    st.subheader("In a hot city, the sun is the bill")
    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.markdown(
            "Cooling dominates a building's energy use in a high-irradiance climate, and the **facade "
            "decides how much sun reaches the glass**. A fixed screen is wrong twice \u2014 too closed and "
            "you lose daylight and views, too open and the building cooks.\n\n"
            "**Smart Skylight** is a screen of triangular panels that doesn't have to choose: an "
            "**attractor curve** drives each panel's rotation and opening, so the screen closes hard "
            "where exposure is brutal and relaxes where it isn't. The question this project answers \u2014 "
            "*how much solar heat does any given screen let in?* \u2014 normally needs a slow simulation. "
            "Here a surrogate model answers it the instant a slider moves.")
        st.markdown('<div class="sk-step"><b>What we predict.</b> Solar Heat Gain '
                    '(kWh/m\u00b2/yr) \u2014 the quantity an architect minimises to cut cooling load.</div>',
                    unsafe_allow_html=True)
    with c2:
        if os.path.exists("facade_render.jpg"):
            st.image("facade_render.jpg", caption="The triangular shading screen (Rhino / Grasshopper)",
                     use_container_width=True)
    kicker("The stakes \u2014 orientation first")
    ex = pd.DataFrame({"Orientation": list(EXPOSURE), "Annual exposure (kWh/m\u00b2)": list(EXPOSURE.values())})
    st.plotly_chart(px.bar(ex, x="Orientation", y="Annual exposure (kWh/m\u00b2)", color="Orientation",
                           color_discrete_map=ORIENT_COLORS,
                           title="A south facade receives roughly twice a north one \u2014 so one screen setting can't serve every wall"),
                    use_container_width=True)

# ============================================================ 2  FACADE STUDIO
with T[1]:
    kicker("Design it \u2014 the model answers live")
    left, right = st.columns([1, 1.25])
    with left:
        st.markdown("**Move the six sliders.** The screen and the predicted heat update instantly.")
        d = {}
        d["Orientation"] = st.selectbox("Facade orientation", ORIENTS, index=2,
                                        format_func=lambda o: f"{o}  \u00b7  {EXPOSURE[o]} kWh/m\u00b2 exposure")
        d["U_Divisions"]       = st.slider(NICE["U_Divisions"], 1, 100, 50)
        d["V_Divisions"]       = st.slider(NICE["V_Divisions"], 1, 100, 50)
        d["Depth_Factor"]      = st.slider(NICE["Depth_Factor"], 0.10, 1.00, 0.55, 0.01)
        d["Attractor_Strength"]= st.slider(NICE["Attractor_Strength"], 0.00, 0.10, 0.05, 0.005)
        d["Aperture_Target"]   = st.slider(NICE["Aperture_Target"]+" (%)", 0, 100, 60)
        d["Max_Rotation_deg"]  = st.slider(NICE["Max_Rotation_deg"]+" (deg)", 0, 100, 50)
    with right:
        shg = predict_one(forest, d)
        frac = np.clip((shg - SHG_LO) / max(SHG_HI - SHG_LO, 1e-6), 0, 1)
        if   frac < .33: col, label, msg = TEAL,  "COOL SCREEN",     "Low solar gain \u2014 this screen keeps the cooling load down."
        elif frac < .66: col, label, msg = "#BA7517", "MODERATE",    "Middle of the range \u2014 a workable daylight/heat balance."
        else:            col, label, msg = AMBER, "HOT SCREEN",      "High solar gain \u2014 close the aperture or cut rotation to cool it."
        st.markdown(f'<div class="sk-verdict" style="border-color:{col}; background:{col}18;">'
                    f'<div style="font-family:Space Mono,monospace; font-size:11px; letter-spacing:.1em; color:{col};">PREDICTED SOLAR HEAT GAIN \u00b7 {label}</div>'
                    f'<div style="font-family:Fraunces,serif; font-size:44px; color:{INK}; line-height:1.1;">{shg:,.0f} '
                    f'<span style="font-size:18px; color:{MUTED};">kWh/m\u00b2/yr</span></div>'
                    f'<div style="color:#5F5E5A; font-size:14px;">{msg}</div></div>', unsafe_allow_html=True)
        # where this design sits among the 500
        fig = px.histogram(df, x=TARGET, nbins=40, title="Where this design sits among the 500 sampled screens")
        fig.update_traces(marker_color=LINE)
        fig.add_vline(x=shg, line_color=col, line_width=3,
                      annotation_text="your design", annotation_position="top")
        fig.update_layout(height=240, showlegend=False, xaxis_title="Solar Heat Gain (kWh/m\u00b2)")
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(facade_figure(d), use_container_width=True)
    st.caption("Schematic preview: panels near the attractor (star) rotate and open more. The real screen "
               "carries 2\u00b7U\u00b7V triangular panels; the prediction uses the full trained forest.")

# ============================================================ 3  DATA PROCESSING
with T[2]:
    kicker("Phase 2 \u2014 auditable, reproducible")
    st.subheader("From a messy raw export to a clean, model-ready table")
    st.markdown("This page runs the cleaning **live on the raw file** so every step is visible and the "
                "before/after counts are auditable \u2014 nothing is hidden inside a script.")

    # ---- show the raw problems
    st.markdown("##### 1 \u00b7 The raw export and its problems")
    rawN = len(raw)
    coerced = {c: pd.to_numeric(raw[c], errors="coerce") for c in NUM}
    n_dups   = int(raw.duplicated().sum())
    n_missing= int(sum(v.isna().sum() for v in coerced.values()))
    oor = 0
    bounds = {"U_Divisions":(1,100),"V_Divisions":(1,100),"Depth_Factor":(.1,1),
              "Attractor_Strength":(0,.1),"Aperture_Target":(0,100),"Max_Rotation_deg":(0,100)}
    for c,(lo,hi) in bounds.items():
        v = coerced[c]; oor += int(((v<lo)|(v>hi)).sum())
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Raw rows", f"{rawN}")
    c2.metric("Duplicate rows", f"{n_dups}")
    c3.metric("Missing / bad cells", f"{n_missing}")
    c4.metric("Out-of-range cells", f"{oor}")
    st.dataframe(raw.head(8), use_container_width=True, height=240)
    st.markdown('<div class="sk-step">The raw CSV reads as <b>mixed text</b> (a stray <code>n/a</code> '
                'in a numeric column), carries <b>exact duplicate rows</b>, scattered <b>missing values</b>, '
                'and a few <b>physically impossible outliers</b> (e.g. rotation = 999\u00b0, depth = 7.5, '
                'divisions = \u22125). All four must be fixed before training.</div>', unsafe_allow_html=True)

    # ---- run the four cleansing operations live
    st.markdown("##### 2 \u00b7 Four cleansing operations \u2014 with before / after counts")
    def cleanse(rw):
        d = rw.copy(); rep = {"rows_in": len(d)}
        for c in NUM: d[c] = pd.to_numeric(d[c], errors="coerce")   # (a) coerce -> text becomes NaN
        b = len(d); d = d.drop_duplicates().reset_index(drop=True)  # (b) drop duplicates
        rep["duplicates_removed"] = b - len(d)
        rep["missing_cells_imputed"] = int(d[NUM].isna().sum().sum())
        for c in NUM: d[c] = d[c].fillna(d[c].median())             # (c) median impute
        b = len(d); m = pd.Series(True, index=d.index)              # (d) drop out-of-range
        for c,(lo,hi) in bounds.items(): m &= d[c].between(lo,hi)
        d = d[m].reset_index(drop=True); rep["out_of_range_removed"] = b - len(d)
        for c in ["U_Divisions","V_Divisions","Aperture_Target","Max_Rotation_deg"]:
            d[c] = d[c].round().astype(int)                          # (e) restore int dtypes
        rep["rows_out"] = len(d); return d, rep
    clean, rep = cleanse(raw)
    steps = [("a","Coerce to numeric","stray text \u2192 NaN so it can be repaired"),
             ("b","Drop exact duplicates", f'{rep["duplicates_removed"]} identical rows removed'),
             ("c","Median-impute missing", f'{rep["missing_cells_imputed"]} isolated gaps filled with each column median'),
             ("d","Remove out-of-range", f'{rep["out_of_range_removed"]} rows outside each slider\u2019s real Min..Max dropped'),
             ("e","Restore integer dtypes","division & angle columns back to whole numbers")]
    for k,t,why in steps:
        st.markdown(f'<div class="sk-step"><b>{k})&nbsp; {t}</b> &mdash; {why}</div>', unsafe_allow_html=True)
    st.success(f"Result: **{rep['rows_in']} raw rows \u2192 {rep['rows_out']} clean rows**, "
               f"every column numeric and within range.")

    # ---- transformation (one-hot)
    st.markdown("##### 3 \u00b7 Data transformation \u2014 qualitative \u2192 quantitative")
    st.markdown("The only text feature, **Orientation**, is one-hot encoded so the model can read it as numbers.")
    demo = pd.DataFrame({"Orientation": ["S","N","E","W","S"]})
    oh = pd.get_dummies(demo["Orientation"]).reindex(columns=["E","N","S","W"], fill_value=0).astype(int)
    oh.columns = [f"Orient_{c}" for c in oh.columns]
    st.dataframe(pd.concat([demo, oh], axis=1), use_container_width=True, height=220)

    # ---- integration (derive targets)
    st.markdown("##### 4 \u00b7 Data integration \u2014 deriving the performance targets")
    st.markdown("New target columns are computed from the features with documented, physically-motivated "
                "equations (faithful to the Grasshopper geometry). The Random Forest learns the relationship "
                "between the design parameters and **Solar Heat Gain**.")
    st.markdown('<div class="sk-eq">'
                'Panel_Count&nbsp; = 2 &middot; U &middot; V<br>'
                'Avg_Tilt&deg;&nbsp;&nbsp;&nbsp; = clip( MaxRot &middot; (0.5 + 5&middot;Attractor), 0, 90 )<br>'
                'Openness&nbsp;&nbsp;&nbsp; = clip( (0.2 + 0.8&middot;Aperture/100) &middot; Tilt/90, 0, 1 )<br>'
                'Shading_Area = A_facade + 2&middot;Depth&middot;&radic;(N&middot;A_facade)<br>'
                'Solar_Heat_Gain = Exposure(orient) &middot; Openness &middot; (1 \u2212 0.55&middot;g_block) &middot; density &nbsp;(+4% noise)'
                '</div>', unsafe_allow_html=True)
    st.caption("Honest note: targets come from a parametric reconstruction + illustrative equations, not "
               "measured simulation. Swap this step for a Grasshopper-exported CSV to make them measured.")
    st.download_button("\u2b07 Download the clean dataset (CSV)",
                       data=df.to_csv(index=False).encode(), file_name="shading_units_dataset.csv",
                       mime="text/csv")

# ============================================================ 4  EXPLORE
with T[3]:
    kicker("Phase 3 \u2014 exploratory data analysis")
    st.subheader("What the 500 designs look like")
    DERIV = ["Total_Shading_Area_m2","Openness_Ratio","Avg_Panel_Tilt_deg","Projection_Depth_m","Material_Volume_m3",TARGET]
    sub = st.radio("View", ["Correlation matrix","Feature distributions","Solar gain vs drivers"], horizontal=True)
    if sub == "Correlation matrix":
        cc = df[NUM+DERIV].corr()
        fig = px.imshow(cc, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                        text_auto=".2f", title="Correlation matrix \u2014 features + targets")
        fig.update_layout(height=620); fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("Solar Heat Gain tracks **openness, rotation and aperture** most strongly \u2014 the levers "
                    "the model later confirms are the dominant drivers.")
    elif sub == "Feature distributions":
        cols = st.columns(3)
        for i, c in enumerate(NUM):
            with cols[i % 3]:
                st.plotly_chart(px.histogram(df, x=c, nbins=30, title=NICE[c]).update_layout(height=260, showlegend=False),
                                use_container_width=True)
    else:
        drv = st.selectbox("Driver", ["Max_Rotation_deg","Aperture_Target","Depth_Factor","Attractor_Strength"])
        st.plotly_chart(px.scatter(df, x=drv, y=TARGET, color="Orientation", color_discrete_map=ORIENT_COLORS,
                                    opacity=.7, title=f"{NICE.get(drv,drv)} vs Solar Heat Gain").update_layout(height=520),
                        use_container_width=True)

# ============================================================ 5  DESIGN MAP
with T[4]:
    kicker("The surrogate as a design tool")
    st.subheader("Every pixel is one design, evaluated by the model")
    st.markdown("Hundreds of screens, predicted in about a second. Each map fixes U=V=50, depth 0.55, "
                "attractor 0.05 and sweeps **rotation \u00d7 aperture** for one orientation.")
    o = st.select_slider("Orientation", ORIENTS, value="S")
    n = 50
    rot = np.linspace(0,100,n); ap = np.linspace(0,100,n)
    R,A = np.meshgrid(rot,ap)
    grid = pd.DataFrame({"U_Divisions":50,"V_Divisions":50,"Depth_Factor":.55,"Attractor_Strength":.05,
                         "Aperture_Target":A.ravel(),"Max_Rotation_deg":R.ravel(),"Orientation":o})
    Z = predict_df(forest, grid).reshape(n,n)
    fig = go.Figure(go.Heatmap(z=Z, x=rot, y=ap, colorscale="YlOrRd",
                               colorbar=dict(title="SHG<br>kWh/m\u00b2")))
    fig.update_layout(height=520, title=f"Predicted Solar Heat Gain \u2014 facing {o}",
                      xaxis_title="Max rotation (deg)", yaxis_title="Aperture target (%)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Cool sits where rotation and aperture are low; the screen heats up toward the top-right.** "
                "Point at the colour you can afford and read off the sliders that get you there.")

# ============================================================ 6  THE MODEL
with T[5]:
    kicker("Phase 4 \u2014 supervised regression, evaluated honestly")
    st.subheader("Trained, validated and tested")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("R\u00b2 (test, unseen 20%)", "0.937")
    c2.metric("R\u00b2 (5-fold CV)", "0.921")
    c3.metric("RMSE", "20.4 kWh/m\u00b2")
    c4.metric("Forest size", "300 trees")
    st.markdown("**Approach.** 500 sampled designs, an 80/20 train/test split, four candidate regressors "
                "compared, 5-fold cross-validation as the overfitting check. The **Random Forest** is deployed "
                "\u2014 within the noise band of Gradient Boosting but more interpretable and robust on a small, "
                "noisy dataset.")
    comp = pd.DataFrame({
        "Model": ["Gradient Boosting","Random Forest","Decision Tree","Linear Reg."],
        "R\u00b2 test": [0.965, 0.937, 0.846, 0.798],
        "R\u00b2 5-fold CV": [0.937, 0.921, 0.849, 0.836],
        "RMSE": [15.2, 20.4, 31.9, 36.5]})
    cc1, cc2 = st.columns([1.1,1])
    with cc1:
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.caption("\u2605 Random Forest deployed.")
    with cc2:
        bar = px.bar(comp, x="R\u00b2 test", y="Model", orientation="h", title="Test R\u00b2 by model",
                     color="Model", color_discrete_sequence=[AMBER, TEAL, NAVY, "#BA7517"])
        bar.update_layout(height=300, showlegend=False, xaxis_range=[0,1])
        st.plotly_chart(bar, use_container_width=True)

    st.markdown("##### Why your professor can open the model now")
    st.markdown(
        "A scikit-learn `.pkl` only re-opens reliably on the **exact library version** that wrote it \u2014 "
        "the original file was saved on scikit-learn 1.6.1 and failed to unpickle elsewhere. Two fixes ship "
        "together:")
    st.markdown(
        "- **`shading_shg_rf_model.pkl`** \u2014 retrained as a *plain* Random Forest (no ColumnTransformer), "
        "which is far more stable across scikit-learn versions.\n"
        "- **`shading_rf_portable.json`** \u2014 the same 300-tree forest exported to pure JSON. "
        "`load_model.py` tries the `.pkl` first and **auto-falls-back** to this, so prediction needs **no "
        "scikit-learn at all** \u2014 it is exactly what powers this dashboard. Verified identical to the "
        "`.pkl` (max\u2009|diff|\u2009=\u20090.0).")
    st.caption("This app itself is the running proof: it predicts entirely from the portable JSON.")

st.markdown('<div class="sk-rule2" style="margin-top:24px;"></div>', unsafe_allow_html=True)
st.caption("Smart Skylight \u00b7 CCAS 2.6 Intelligent Systems Design \u00b7 predictions from the portable Random Forest (zero-install).")
