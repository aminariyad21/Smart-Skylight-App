# =============================================================================
#  SMART SKYLIGHT  ·  dashboard   (self-contained Streamlit app)
#  CCAS 2.6 — Intelligent Systems Design
#
#  A generative shading facade + a machine-learning surrogate that predicts
#  Solar Heat Gain live — in Grasshopper, and right here.
#
#  RUN:  pip install -r requirements.txt   then   streamlit run app.py
#  Needs (same folder): shading_units_dataset.csv, shading_units_raw.csv,
#        shading_rf_portable.json, dashboard_meta.json, .streamlit/config.toml
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

CORAL = "#EC5242"; SKY = "#7FB2F0"; INK = "#E6E6E6"
# --------------------------------------------------------------- styling
st.markdown("""
<style>
.block-container {padding-top:1.3rem; max-width:1180px;}
h1,h2,h3,h4 {font-family: Georgia,'Source Serif Pro','Times New Roman',serif !important;
  color:#FFFFFF !important; font-weight:700; letter-spacing:-.005em;}
h2 {font-size:31px; margin:.15em 0 .15em;}
h3 {font-size:22px;}
.tag {color:#9AA0AB; font-size:16px; margin:-2px 0 2px;}
.idea {color:#C9CDD4; font-size:16px; line-height:1.62;}
.idea b {color:#FFFFFF;} .idea i {color:#C9CDD4;}
.mc {padding:2px 0 14px;}
.mc .l {color:#8A909B; font-size:14px; margin-bottom:1px;}
.mc .v {color:#FFFFFF; font-family:Georgia,serif; font-size:38px; font-weight:700; line-height:1.05;
  overflow-wrap:normal; word-break:keep-all;}
.mc .v.sm {font-size:22px; line-height:1.2;}
.mc .v small {font-size:17px; color:#9AA0AB;}
.flow {margin:6px 0 2px; line-height:2.7;}
.pill {display:inline-block; background:#1A1D26; border:1px solid #2C313C; border-radius:18px;
  padding:6px 14px; margin:3px 1px; color:#D6DAE0; font-size:14px;}
.arr {color:#EC5242; margin:0 4px; font-weight:700;}
table.vt {width:100%; border-collapse:collapse; margin-top:6px; font-size:14px;}
table.vt th {text-align:left; color:#8A909B; font-weight:600; padding:9px 11px;
  border-bottom:1px solid #2C313C; font-size:13px;}
table.vt td {color:#D6DAE0; padding:9px 11px; border-bottom:1px solid #1E222B; vertical-align:top;}
table.vt td.k {color:#EEF0F3; font-family:ui-monospace,monospace;}
.note {background:#10243A; border:1px solid #1E4368; border-radius:9px; padding:14px 18px;
  color:#9CC4EA; font-size:15px; line-height:1.55; margin-top:14px;}
.mono {background:#161A22; border:1px solid #262B35; border-radius:8px; padding:13px 17px;
  color:#C9CDD4; font-family:ui-monospace,monospace; font-size:14px;}
.mono b {color:#E0A33C;}
.verdict .l {color:#8A909B; font-size:15px;}
.verdict .v {font-family:Georgia,serif; font-size:50px; font-weight:700; color:#FFFFFF; line-height:1.05;}
.verdict .v small {font-size:20px; color:#9AA0AB;}
.sub {color:#9AA0AB; font-size:14px;}
.stTabs [data-baseweb="tab-list"] {gap:22px; border-bottom:1px solid #232730; flex-wrap:wrap;}
.stTabs [data-baseweb="tab"] {font-size:15px; color:#9AA0AB; padding:8px 1px; background:transparent;}
.stTabs [aria-selected="true"] {color:#EC5242 !important;}
.stTabs [data-baseweb="tab-highlight"] {background:#EC5242;}
.codepill {color:#86C293; font-family:ui-monospace,monospace; background:#161A22;
  padding:1px 6px; border-radius:5px; font-size:.92em;}
</style>
""", unsafe_allow_html=True)

pio.templates["sky"] = go.layout.Template(layout=go.Layout(
    font=dict(family="sans-serif", size=13, color="#C9CDD4"),
    title=dict(font=dict(size=15, color="#FFFFFF")),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[CORAL, SKY, "#5FA89F", "#E0A33C", "#9AA0AB"],
    xaxis=dict(gridcolor="#232730", zerolinecolor="#2C313C"),
    yaxis=dict(gridcolor="#232730", zerolinecolor="#2C313C"),
    margin=dict(l=10, r=10, t=44, b=10), legend=dict(bgcolor="rgba(0,0,0,0)")))
pio.templates.default = "plotly_dark+sky"

# --------------------------------------------------------------- constants / data
NUM = ["U_Divisions","V_Divisions","Depth_Factor","Attractor_Strength","Aperture_Target","Max_Rotation_deg"]
TARGET = "Solar_Heat_Gain_kWh_m2"
ORI = ["N","E","S","W"]; OCOL = {"S":CORAL,"W":"#E0A33C","E":SKY,"N":"#5FA89F"}
NICE = {"U_Divisions":"U_Divisions","V_Divisions":"V_Divisions","Depth_Factor":"Depth_Factor",
        "Attractor_Strength":"Attractor_Strength","Aperture_Target":"Aperture_Target",
        "Max_Rotation_deg":"Max_Rotation_deg"}
CORR_COLS = NUM + ["Panel_Count","Avg_Panel_Tilt_deg","Projection_Depth_m","Openness_Ratio",
                   "Total_Shading_Area_m2","Material_Volume_m3", TARGET]

@st.cache_data(show_spinner=False)
def load_csv(n): return pd.read_csv(n) if os.path.exists(n) else None
@st.cache_resource(show_spinner=False)
def load_forest():
    with open("shading_rf_portable.json") as f: return json.load(f)
@st.cache_data(show_spinner=False)
def load_meta():
    with open("dashboard_meta.json") as f: return json.load(f)
def f32(v): return struct.unpack("f", struct.pack("f", float(v)))[0]

def predict_df(forest, frame):
    cats, order, trees, nt = (forest["orientation_categories"], forest["numeric_order"],
                              forest["trees"], forest["n_trees"]); out=[]
    for _, r in frame.iterrows():
        oh=[1.0 if c==str(r["Orientation"]).upper() else 0.0 for c in cats]
        x=[f32(v) for v in (oh+[float(r[n]) for n in order])]; tot=0.0
        for tr in trees:
            cl,cr,fe,th,va=tr["cl"],tr["cr"],tr["f"],tr["t"],tr["v"]; n=0
            while cl[n]!=cr[n]: n=cl[n] if x[fe[n]]<=th[n] else cr[n]
            tot+=va[n]
        out.append(tot/nt)
    return np.array(out)
def predict_one(forest,d): return float(predict_df(forest, pd.DataFrame([d]))[0])

def mcard(label, value, big=True):
    cls = "v" if big else "v sm"
    st.markdown(f'<div class="mc"><div class="l">{label}</div><div class="{cls}">{value}</div></div>',
                unsafe_allow_html=True)
def vtable(headers, rows, kcol=0):
    h="".join(f"<th>{x}</th>" for x in headers)
    body=""
    for r in rows:
        cells="".join(f'<td class="{"k" if i==kcol else ""}">{c}</td>' for i,c in enumerate(r))
        body+=f"<tr>{cells}</tr>"
    st.markdown(f'<table class="vt"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>',
                unsafe_allow_html=True)

df  = load_csv("shading_units_dataset.csv")
raw = load_csv("shading_units_raw.csv")
forest = load_forest(); M = load_meta()
SMAX = float(df[TARGET].max())

# --------------------------------------------------------------- header
st.markdown(f'<h1 style="display:inline;"><span style="background:{CORAL};color:{CORAL};'
            f'border-radius:5px;">▮▮</span> Smart Skylight</h1>', unsafe_allow_html=True)
st.markdown('<div class="tag">A generative shading facade + a machine-learning surrogate that predicts '
            'Solar Heat Gain live — in Grasshopper, and right here.</div>', unsafe_allow_html=True)

T = st.tabs(["\U0001F3E0 Project", "\u270F\uFE0F Data journey", "\U0001F4CA Analysis",
             "\u2699\uFE0F Model", "\U0001F3A8 Design studio", "\U0001F997 Grasshopper & files"])

# ============================================================ PROJECT
with T[0]:
    L, R = st.columns([1.32, 1])
    with L:
        st.markdown("## The idea")
        st.markdown('<div class="idea">An <b>attractor-driven triangular shading facade</b> is built in '
            'Grasshopper. Six sliders generate the geometry — and the question every designer has is: '
            '<i>how much solar energy will this variant let through?</i> Normally that needs a slow separate '
            'analysis. This project trains a <b>surrogate model</b> on sampled designs so the answer arrives '
            '<b>instantly, while the sliders move</b>.</div>', unsafe_allow_html=True)
        st.markdown('<div class="idea" style="margin-top:13px;"><b>What we predict:</b> Solar Heat Gain '
            '(kWh/m²) — the number an architect minimises to fight cooling loads in a hot city like Cairo.</div>',
            unsafe_allow_html=True)
    with R:
        a,b = st.columns(2)
        with a: mcard("Designs in dataset", "500")
        with b: mcard("Test accuracy", "R² ≈ 0.94")
        c,d = st.columns(2)
        with c: mcard("Features", "6 + orientation", big=False)
        with d: mcard("Deployed model", "Random Forest", big=False)

    st.markdown("## The workflow (end to end)")
    steps=["Grasshopper model","CSV dataset","Data cleaning","Feature engineering","Train / test split",
           "Train model","Evaluate","Export .pkl","GH predictive node","Dashboard & website"]
    html='<div class="flow">'
    for i,s in enumerate(steps):
        html+=f'<span class="pill">{s}</span>'+('<span class="arr">→</span>' if i<len(steps)-1 else '')
    st.markdown(html+"</div>", unsafe_allow_html=True)

    st.markdown("## The variables")
    v1,v2 = st.columns(2)
    with v1:
        st.markdown("**Features (X)** — the six Grasshopper sliders + context")
        vtable(["feature","range","meaning"], [
            ["U_Divisions","1 – 100","panels horizontally"],
            ["V_Divisions","1 – 100","panels vertically"],
            ["Depth_Factor","0.1 – 1.0","fin projection depth"],
            ["Attractor_Strength","0.0 – 0.1","attractor influence"],
            ["Aperture_Target","0 – 100","target openness (%)"],
            ["Max_Rotation_deg","0 – 100","max panel rotation (deg)"],
            ["Orientation","N / E / S / W","facade direction (context)"]])
    with v2:
        st.markdown("**Targets (Y)** — derived per design; ⭐ is what the model predicts")
        vtable(["target","unit","meaning"], [
            ["Total_Shading_Area_m2","m²","facade + slanted fins"],
            ["Openness_Ratio","0–1","how open the screen is"],
            ["Avg_Panel_Tilt_deg","deg","attractor-amplified rotation"],
            ["Projection_Depth_m","m","fin depth"],
            ["Material_Volume_m3","m³","panel material cost"],
            ["⭐ Solar_Heat_Gain_kWh_m2","kWh/m²","the model's target"]])

    st.markdown('<div class="note"><b>Honest note:</b> targets come from a documented physics-style '
        'reconstruction (+4% noise), not a measured simulation — the GH data recorder exists to swap in '
        'measured values later with the identical pipeline.</div>', unsafe_allow_html=True)

# ============================================================ DATA JOURNEY
with T[1]:
    st.markdown("## Phase 2 — from messy export to model-ready table")
    a,b,c,d = st.columns(4)
    with a: mcard("Raw rows","508")
    with b: mcard("Duplicates found","5")
    with c: mcard("Missing / garbled cells","13")
    with d: mcard("Clean rows","500")

    with st.expander("See the raw (pre-cleaning) data"):
        if raw is not None: st.dataframe(raw.head(10), use_container_width=True, height=390)

    st.markdown('<div class="idea" style="margin-top:14px;"><b>Cleansing — four auditable operations:</b> '
        '(1) coerce text like <span class="codepill">"n/a"</span> to <span class="codepill">NaN</span> · '
        '(2) drop exact duplicate rows · (3) median-impute missing numeric cells · '
        '(4) drop rows outside each slider\'s real Min/Max.</div>', unsafe_allow_html=True)
    st.markdown('<div class="idea" style="margin-top:12px;"><b>Transformation (qualitative → quantitative):</b> '
        '<span class="codepill">Orientation</span> is one-hot encoded — and the encoder lives <i>inside</i> the '
        'saved pipeline, so any future input is transformed exactly like the training data.</div>',
        unsafe_allow_html=True)
    oh = pd.get_dummies(df["Orientation"].head(5), prefix="Orientation").astype(int)
    oh = oh.reindex(columns=["Orientation_E","Orientation_N","Orientation_S","Orientation_W"], fill_value=0)
    st.dataframe(oh, use_container_width=True)

    st.markdown('<div class="idea" style="margin-top:10px;"><b>Integration — deriving the targets</b> from the '
                'features, e.g.:</div>', unsafe_allow_html=True)
    st.latex(r"\Omega \;=\; \mathrm{clip}\!\left[\left(0.2 + 0.8\,\tfrac{Aperture}{100}\right)\cdot"
             r"\tfrac{\theta}{90},\; 0,\; 1\right]")
    st.latex(r"SHG \;=\; E(\text{orientation})\cdot \Omega \cdot \left(1 - 0.55\,g_{block}\right)\cdot"
             r"\rho \quad (+\,4\%\ \text{noise})")

# ============================================================ ANALYSIS
with T[2]:
    st.markdown("## Exploratory analysis")
    fc1, fc2 = st.columns([1,1.1])
    with fc1:
        sel = st.multiselect("Orientation", ORI, default=ORI)
    with fc2:
        rng = st.slider("Solar Heat Gain range (kWh/m²)", 0.0, round(SMAX,2),
                        (0.0, round(SMAX,2)))
    d = df[df["Orientation"].isin(sel) & df[TARGET].between(*rng)]
    st.markdown(f'<div class="sub">{len(d)} designs match the filters.</div>', unsafe_allow_html=True)

    cc = [c for c in CORR_COLS if c in d.columns]
    corr = d[cc].corr().round(2)
    fig = go.Figure(go.Heatmap(z=corr.values, x=cc, y=cc, colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text:.2f}", textfont=dict(size=9),
        colorbar=dict(title="")))
    fig.update_layout(height=560, margin=dict(l=10,r=10,t=10,b=10),
                      yaxis=dict(autorange="reversed"))
    fig.update_xaxes(tickangle=-40)
    st.plotly_chart(fig, use_container_width=True)

    dc1, dc2 = st.columns(2)
    with dc1: distf = st.selectbox("Distribution of", NUM, index=2)
    with dc2: vsf = st.selectbox("Solar Heat Gain vs", NUM, index=5)
    g1, g2 = st.columns(2)
    with g1:
        h = px.histogram(d, x=distf, nbins=26, color_discrete_sequence=[SKY])
        h.update_layout(height=420, bargap=.03, yaxis_title="count"); st.plotly_chart(h, use_container_width=True)
    with g2:
        s = px.scatter(d, x=vsf, y=TARGET, color="Orientation", opacity=.75,
                       color_discrete_map=OCOL, category_orders={"Orientation":["S","W","E","N"]})
        s.update_traces(marker=dict(size=6)); s.update_layout(height=420)
        st.plotly_chart(s, use_container_width=True)

# ============================================================ MODEL
with T[3]:
    st.markdown("## Supervised regression, evaluated honestly")
    st.markdown('<div class="idea"><b>Protocol:</b> 80% train / 20% held-out test, with 5-fold '
        'cross-validation for validation. Four models compared (full workflow in the notebook):</div>',
        unsafe_allow_html=True)
    cmp = M["compare"]; star = {"Random Forest":"⭐ "}
    rows=[]
    for name in ["Gradient Boosting","Random Forest","Decision Tree","Linear Regression"]:
        s=cmp[name]; label=("⭐ "+name+" (deployed)") if name=="Random Forest" else name
        rows.append([label, f'{s["mse"]:.1f}', f'{s["rmse"]:.1f}', f'{s["mae"]:.1f}',
                     f'{s["r2"]:.3f}', f'{s["cv"]:.3f}'])
    vtable(["Model","MSE","RMSE","MAE","R² test","R² CV"], rows, kcol=-1)
    st.markdown('<div class="idea" style="margin-top:12px;"><b>Random Forest is deployed:</b> within noise of '
        'Gradient Boosting, but more interpretable (feature importances) and more robust for retraining on '
        'measured data later.</div>', unsafe_allow_html=True)

    st.markdown("## Inside the deployed .pkl (read live from the file)")
    a,b,c = st.columns(3)
    with a: mcard("Trees", f'{M["n_trees"]}')
    with b: mcard("Decision nodes", f'{M["n_nodes"]:,}')
    with c: mcard("Avg tree depth", f'{M["avg_depth"]}')

    imp = pd.Series(M["importances"]).sort_values()
    fi = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=SKY))
    fi.update_layout(height=430, xaxis_title="importance", yaxis_title="feature",
                     margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fi, use_container_width=True)
    st.markdown(f'<div class="mono">root rule of tree #1:&nbsp; if <b>{M["root_feat"]}</b> &lt;= '
                f'{M["root_thr"]} -&gt; go left, else go right</div>', unsafe_allow_html=True)

# ============================================================ DESIGN STUDIO
with T[4]:
    st.markdown("## Move the sliders — exactly like moving them in Grasshopper")
    sL, sR = st.columns([1.05, 1])
    with sL:
        c1,c2 = st.columns(2)
        with c1:
            u = st.slider("U_Divisions",1,100,24); dep = st.slider("Depth_Factor",0.10,1.00,0.17)
            ap = st.slider("Aperture_Target",0,100,17)
        with c2:
            v = st.slider("V_Divisions",1,100,18); att = st.slider("Attractor_Strength",0.00,0.10,0.02)
            rot = st.slider("Max_Rotation_deg",0,100,47)
        o = st.radio("Orientation", ORI, index=3, horizontal=True)
        dd = {"U_Divisions":u,"V_Divisions":v,"Depth_Factor":dep,"Attractor_Strength":att,
              "Aperture_Target":ap,"Max_Rotation_deg":rot,"Orientation":o}
    with sR:
        shg = predict_one(forest, dd)
        pct = int((df[TARGET] < shg).mean()*100)
        st.markdown('<div class="verdict"><div class="l">Predicted Solar Heat Gain</div>'
                    f'<div class="v">{shg:,.1f} <small>kWh/m²</small></div></div>', unsafe_allow_html=True)
        st.progress(min(max(shg/SMAX,0),1.0))
        st.markdown(f'<div class="sub">Hotter than {pct}% of the 500 sampled designs '
                    f'(dataset range 0–{SMAX:.0f}).</div>', unsafe_allow_html=True)

    st.markdown("## The design map — your design is the dot")
    st.markdown(f'<div class="idea"><b>Predicted SHG across rotation × aperture — facing {o}</b> '
                '(other sliders as set above)</div>', unsafe_allow_html=True)
    rax = np.linspace(0,100,21); aax = np.linspace(0,100,21)
    grid = pd.DataFrame([{"U_Divisions":u,"V_Divisions":v,"Depth_Factor":dep,"Attractor_Strength":att,
                          "Aperture_Target":a,"Max_Rotation_deg":r,"Orientation":o}
                         for a in aax for r in rax])
    grid["z"] = predict_df(forest, grid)
    Z = grid.pivot_table(index="Aperture_Target", columns="Max_Rotation_deg", values="z").values
    cm = go.Figure(go.Contour(z=Z, x=rax, y=aax, colorscale="YlOrRd",
                              colorbar=dict(title="kWh/m²"), contours=dict(showlines=True)))
    cm.add_trace(go.Scatter(x=[rot], y=[ap], mode="markers+text", marker=dict(symbol="x", size=15,
                 color="#1B3A6B", line=dict(width=2,color="#1B3A6B")),
                 text=["your design"], textposition="top center", textfont=dict(color=SKY, size=11)))
    cm.update_layout(height=560, showlegend=False, xaxis_title="Max_Rotation_deg",
                     yaxis_title="Aperture_Target", margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(cm, use_container_width=True)

# ============================================================ GRASSHOPPER & FILES
with T[5]:
    st.markdown("## Deployment — the model goes back to where design happens")
    st.markdown('<div class="idea">A Grasshopper <b>Python 3 Script component</b> wraps the trained model. '
        'Its inputs are the <b>same six sliders</b> that drive the geometry plus the orientation; its output '
        'is the predicted Solar Heat Gain — so geometry and predicted performance regenerate <b>together, in '
        'real time</b>. It ships in two interchangeable forms, verified to give identical predictions:</div>',
        unsafe_allow_html=True)
    st.markdown('<div class="idea" style="margin-top:8px;">• <span class="codepill">gh_node_predict_rhino8.py</span> '
        '— loads the <span class="codepill">.pkl</span> (scikit-learn 1.6.1, matching Rhino 8\'s Python 3.9)'
        '<br>• <span class="codepill">gh_node_predict_portable.py</span> — loads '
        '<span class="codepill">shading_rf_portable.json</span>, <b>zero dependencies</b>, for machines where '
        'pip is blocked</div>', unsafe_allow_html=True)
    st.markdown('<div class="idea" style="margin-top:8px;">A third component, '
        '<span class="codepill">gh_data_recorder.py</span>, <b>closes the loop</b>: it writes the live slider '
        'values (and a measured value) back to CSV, so the same pipeline can retrain on real measured data.</div>',
        unsafe_allow_html=True)

    with st.expander("Predictive node — wiring (inputs / output)"):
        vtable(["input","type hint","wire to"], [
            ["model_path","str","panel with full path to the .pkl"],
            ["U V Depth Attractor Aperture Rotation","float","the six geometry sliders"],
            ["Orientation","str","value list: N / E / S / W"],
            ["→ output: SHG","•","a panel (predicted kWh/m²)"]])

    st.markdown("## Project files")
    vtable(["file","what"], [
        ["shading_units_dataset.csv / _raw.csv","clean dataset + messy raw"],
        ["smart_skylight_analysis.ipynb / _colab.ipynb","executed notebook + Colab version"],
        ["shading_shg_rf_model_rhino8.pkl / portable.json","the model, two formats"],
        ["inspect_model.py / load_model.py","open the .pkl, print what's inside"],
        ["gh_node_predict_*.py / gh_data_recorder.py","Grasshopper nodes"],
        ["smart_skylight_report.pdf / RUN_GUIDE.pdf","full report + how-to-run"],
        ["Shading_units_grad.gh","the Grasshopper definition"]])
    st.markdown('<div class="sub" style="margin-top:8px;">Everything is downloadable on the project '
                'website.</div>', unsafe_allow_html=True)
