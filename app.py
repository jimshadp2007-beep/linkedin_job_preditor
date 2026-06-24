import sys, os, json, pathlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib
 
_app_dir = pathlib.Path(__file__).resolve().parent
for _src in [_app_dir / "src", _app_dir.parent / "src"]:
    if (_src / "preprocessing.py").exists():
        sys.path.insert(0, str(_src)); break
import preprocessing as prep
 
# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="LinkedIn Job Intelligence", page_icon="💼",
                   layout="wide", initial_sidebar_state="collapsed")
 
# ── glassmorphism CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
 
/* ── global reset & background ── */
*, *::before, *::after { box-sizing: border-box; }
 
.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #0d1b4b 30%, #0a3060 60%, #0e4a8a 100%);
    min-height: 100vh;
    font-family: 'Inter', sans-serif;
}
 
/* animated background orbs */
.stApp::before {
    content: '';
    position: fixed;
    top: -20%;  left: -10%;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(0,119,181,0.35) 0%, transparent 70%);
    border-radius: 50%;
    animation: float1 8s ease-in-out infinite;
    z-index: 0; pointer-events: none;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: -10%; right: -10%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(10,102,194,0.3) 0%, transparent 70%);
    border-radius: 50%;
    animation: float2 10s ease-in-out infinite;
    z-index: 0; pointer-events: none;
}
@keyframes float1 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(40px,30px)} }
@keyframes float2 { 0%,100%{transform:translate(0,0)} 50%{transform:translate(-30px,-40px)} }
 
/* ── hide streamlit chrome ── */
#MainMenu, footer, header, .stDeployButton { display: none !important; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1400px !important; }
 
/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(20px);
    border-radius: 16px;
    padding: 6px;
    border: 1px solid rgba(255,255,255,0.12);
    gap: 4px;
    margin-bottom: 1.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px !important;
    color: rgba(255,255,255,0.6) !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.25s ease !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0077B5, #0a66c2) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(0,119,181,0.4) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }
 
/* ── glass card ── */
.glass-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.13);
    padding: 24px;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}
 
/* ── KPI metric cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin-bottom: 1.5rem; }
.kpi-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(24px);
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.13);
    padding: 22px 20px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
.kpi-icon  { font-size: 2rem; margin-bottom: 6px; }
.kpi-value { font-size: 1.75rem; font-weight: 700; color: #fff; line-height: 1; }
.kpi-label { font-size: 0.78rem; color: rgba(255,255,255,0.55); margin-top: 6px; letter-spacing: 0.04em; text-transform: uppercase; }
 
/* ── cluster pill cards ── */
.cluster-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: 14px; margin-bottom: 1.5rem; }
.cluster-pill {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(20px);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.13);
    padding: 18px 14px;
    text-align: center;
    cursor: pointer;
    transition: all 0.25s;
}
.cluster-pill:hover { background: rgba(0,119,181,0.25); border-color: #0077B5; transform: translateY(-2px); }
.cluster-pill .badge {
    display: inline-block;
    background: linear-gradient(135deg,#0077B5,#0a66c2);
    color: white; font-size: 0.65rem; font-weight: 700;
    border-radius: 30px; padding: 3px 10px; margin-bottom: 8px; letter-spacing: 0.05em;
}
.cluster-pill .sal  { font-size: 1.1rem; font-weight: 700; color: #5de0a0; margin: 6px 0 2px; }
.cluster-pill .cnt  { font-size: 0.72rem; color: rgba(255,255,255,0.5); }
.cluster-pill .name { font-size: 0.78rem; color: rgba(255,255,255,0.8); font-weight: 500; margin-top: 4px; }
 
/* ── section headers ── */
.section-header {
    font-size: 1.1rem; font-weight: 600; color: rgba(255,255,255,0.9);
    margin: 1.2rem 0 0.8rem; letter-spacing: 0.01em;
    display: flex; align-items: center; gap: 8px;
}
.section-header::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(0,119,181,0.5), transparent);
}
 
/* ── predict result card ── */
.result-card {
    background: linear-gradient(135deg, rgba(0,119,181,0.25), rgba(10,102,194,0.15));
    backdrop-filter: blur(24px);
    border-radius: 20px;
    border: 1px solid rgba(0,119,181,0.4);
    padding: 28px;
    box-shadow: 0 8px 32px rgba(0,119,181,0.2);
}
.result-cluster { font-size: 2.2rem; font-weight: 800; color: #fff; margin: 0; }
.result-label   { font-size: 1rem; color: rgba(255,255,255,0.7); margin-top: 4px; }
.result-badge   {
    display: inline-block; background: linear-gradient(135deg,#0077B5,#00a0dc);
    color: #fff; border-radius: 30px; padding: 4px 16px; font-size: 0.8rem;
    font-weight: 600; margin-top: 10px;
}
 
/* ── insight row ── */
.insight-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; margin: 1rem 0; }
.insight-box {
    background: rgba(255,255,255,0.06);
    border-radius: 14px; border: 1px solid rgba(255,255,255,0.1);
    padding: 16px; text-align: center;
}
.insight-box .val { font-size: 1.4rem; font-weight: 700; color: #5de0a0; }
.insight-box .lbl { font-size: 0.72rem; color: rgba(255,255,255,0.5); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
 
/* ── similar jobs table ── */
.similar-job {
    background: rgba(255,255,255,0.05);
    border-radius: 12px; border: 1px solid rgba(255,255,255,0.09);
    padding: 14px 18px; margin-bottom: 8px;
    display: flex; justify-content: space-between; align-items: center;
}
.sj-title   { font-size: 0.9rem; font-weight: 600; color: rgba(255,255,255,0.92); }
.sj-company { font-size: 0.75rem; color: rgba(255,255,255,0.45); margin-top: 2px; }
.sj-salary  { font-size: 0.95rem; font-weight: 700; color: #5de0a0; white-space: nowrap; }
 
/* ── form inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stMultiSelect > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: white !important;
}
.stCheckbox > label { color: rgba(255,255,255,0.85) !important; }
label, .stSelectbox label, .stNumberInput label, .stMultiSelect label {
    color: rgba(255,255,255,0.7) !important; font-size: 0.82rem !important; font-weight: 500 !important;
}
 
/* ── submit button ── */
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #0077B5, #0a66c2) !important;
    color: white !important; border: none !important; border-radius: 14px !important;
    padding: 14px 40px !important; font-size: 1rem !important; font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: 0 4px 20px rgba(0,119,181,0.45) !important;
    transition: all 0.25s !important; width: 100% !important;
}
.stFormSubmitButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(0,119,181,0.6) !important;
}
 
/* ── plotly charts ── */
.js-plotly-plot { border-radius: 16px; overflow: hidden; }
 
/* ── scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 3px; }
 
/* ── salary insights bar chart labels ── */
.exp-bar { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.exp-bar-label { font-size: 0.75rem; color: rgba(255,255,255,0.6); width: 130px; flex-shrink: 0; }
.exp-bar-fill  { height: 28px; border-radius: 8px; display: flex; align-items: center; padding-left: 10px; }
.exp-bar-val   { font-size: 0.8rem; font-weight: 600; color: white; }
</style>
""", unsafe_allow_html=True)
 
# ── artifact discovery ────────────────────────────────────────────────────────
def _find_root():
    cwd = pathlib.Path.cwd()
    for c in [_app_dir/"models", _app_dir/"notebooks"/"output"/"models",
              _app_dir/"output"/"models", cwd/"models",
              cwd/"notebooks"/"output"/"models", cwd/"output"/"models"]:
        if (c/"preprocessor.pkl").exists(): return c
    return None
 
def _find_csv(mr):
    for c in [mr.parent.parent/"data"/"processed", mr.parent/"data", mr.parent]:
        if (c/"clustered_jobs.csv").exists(): return c
    return None
 
@st.cache_resource
def load_models(d):
    p = pathlib.Path(d)
    pre  = joblib.load(p/"preprocessor.pkl")
    km   = joblib.load(p/"kmeans_model.pkl")
    pca  = joblib.load(p/"pca.pkl")
    with open(p/"cluster_profiles.json") as f:
        prof = pd.DataFrame(json.load(f))
    return pre, km, pca, prof
 
@st.cache_data
def load_jobs(d):
    return pd.read_csv(pathlib.Path(d)/"clustered_jobs.csv")
 
mr = _find_root()
if not mr:
    st.error("Model artifacts not found. Run `notebooks/job_clustering_pipeline.ipynb` first.")
    st.stop()
cr = _find_csv(mr)
if not cr:
    st.error("clustered_jobs.csv not found. Re-run the notebook.")
    st.stop()
 
preprocessor, kmeans, pca, profiles = load_models(str(mr))
jobs_df = load_jobs(str(cr))
 
# ── helpers ───────────────────────────────────────────────────────────────────
CLUSTER_COLORS = ["#0077B5","#00a0dc","#5de0a0","#f5a623","#e05d5d"]
CLUSTER_NAMES  = {
    0: "Senior Tech & Healthcare",
    1: "Entry-Level Support",
    2: "Entry-Level Operations",
    3: "Senior Tech & Sales",
    4: "Senior IT & Engineering",
}
EXP_ORDER = ["Internship","Entry level","Associate","Mid-Senior level","Director","Executive"]
def hex_to_rgba(hex_color, alpha=1.0):
    """Convert #RRGGBB hex to rgba() string safe for all plotly properties."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"
 
 
# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:18px;padding:28px 0 18px;border-bottom:1px solid rgba(255,255,255,0.1);margin-bottom:24px">
  <svg xmlns="http://www.w3.org/2000/svg" width="52" height="52" viewBox="0 0 72 72">
    <rect width="72" height="72" rx="12" fill="#0A66C2"/>
    <path d="M15.6 29h7.2v23.4h-7.2V29zm3.6-11.5a4.17 4.17 0 110 8.34A4.17 4.17 0 0119.2 17.5zM30 29h6.9v3.2h.1C38.1 30 41 27.8 45.3 27.8c7.3 0 8.7 4.8 8.7 11.1v13.5h-7.2V40.3c0-2.9-.1-6.6-4-6.6s-4.7 3.2-4.7 6.4v12.3H30V29z" fill="white"/>
  </svg>
  <div>
    <div style="font-size:1.8rem;font-weight:800;color:#fff;line-height:1">Job Market Intelligence</div>
    <div style="font-size:0.88rem;color:rgba(255,255,255,0.5);margin-top:4px">AI-powered clustering of 35,509 LinkedIn job postings</div>
  </div>
</div>
""", unsafe_allow_html=True)
 
# ── tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3 = st.tabs(["📊 Market Pulse", "🧠 Cluster Intelligence", "🎯 Job Classifier"])
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MARKET PULSE
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    # ── KPI row ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-icon">💼</div>
        <div class="kpi-value">35,509</div>
        <div class="kpi-label">Total Job Postings</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon">💰</div>
        <div class="kpi-value">$95.8K</div>
        <div class="kpi-label">Avg Annual Salary</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon">🏢</div>
        <div class="kpi-value">10,210</div>
        <div class="kpi-label">Unique Companies</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon">🌐</div>
        <div class="kpi-value">13.4%</div>
        <div class="kpi-label">Remote Postings</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
 
    col1, col2 = st.columns([3, 2])
 
    with col1:
        st.markdown('<div class="section-header">📈 Salary Distribution Across Market</div>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        for c in sorted(jobs_df["cluster"].unique()):
            sub = jobs_df[jobs_df["cluster"]==c]
            fig_hist.add_trace(go.Histogram(
                x=sub["normalized_salary"], name=CLUSTER_NAMES.get(c, f"Cluster {c}"),
                opacity=0.75, nbinsx=50,
                marker_color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
            ))
        fig_hist.update_layout(
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
            margin=dict(l=0,r=0,t=10,b=0), height=260,
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.3, font_size=11),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Annual Salary (USD)",
                       tickprefix="$", tickformat=","),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="# Postings"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
 
    with col2:
        st.markdown('<div class="section-header">🎓 Salary by Experience</div>', unsafe_allow_html=True)
        exp_salary = jobs_df.groupby("formatted_experience_level")["normalized_salary"].median()
        max_sal = 191270
        bars_html = ""
        colors = ["#e05d5d","#f5a623","#f5d623","#5de0a0","#0077B5","#00a0dc"]
        for i, lvl in enumerate(EXP_ORDER):
            if lvl not in exp_salary.index: continue
            sal = exp_salary[lvl]
            pct = sal / max_sal * 100
            col = colors[i % len(colors)]
            bars_html += f"""
            <div class="exp-bar">
              <div class="exp-bar-label">{lvl}</div>
              <div style="flex:1">
                <div class="exp-bar-fill" style="background:linear-gradient(90deg,{col}aa,{col});width:{pct:.0f}%">
                  <span class="exp-bar-val">${sal/1000:.0f}K</span>
                </div>
              </div>
            </div>"""
        st.markdown(f'<div class="glass-card" style="padding:20px">{bars_html}</div>', unsafe_allow_html=True)
 
    col3, col4 = st.columns(2)
 
    with col3:
        st.markdown('<div class="section-header">🏆 Top Hiring Companies</div>', unsafe_allow_html=True)
        top_co = jobs_df["company_name"].value_counts().head(8)
        fig_co = go.Figure(go.Bar(
            x=top_co.values, y=top_co.index, orientation="h",
            marker=dict(color=list(range(len(top_co))),
                        colorscale=[[0,"#004182"],[1,"#00a0dc"]],
                        showscale=False),
        ))
        fig_co.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
            margin=dict(l=0,r=0,t=10,b=0), height=240,
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_co, use_container_width=True)
 
    with col4:
        st.markdown('<div class="section-header">📍 Top Hiring Locations</div>', unsafe_allow_html=True)
        top_loc = jobs_df["location"].value_counts().head(7)
        fig_loc = go.Figure(go.Pie(
            labels=top_loc.index, values=top_loc.values,
            hole=0.55,
            marker=dict(colors=["#0077B5","#00a0dc","#5de0a0","#f5a623","#e05d5d","#a78bfa","#fb923c"]),
            textfont=dict(size=11, color="white"),
        ))
        fig_loc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
            margin=dict(l=0,r=0,t=10,b=0), height=240,
            legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10, orientation="v"),
            showlegend=True,
        )
        st.plotly_chart(fig_loc, use_container_width=True)
 
    st.markdown('<div class="section-header">🔥 Most In-Demand Job Titles</div>', unsafe_allow_html=True)
    top_titles = jobs_df["title"].value_counts().head(12)
    fig_t = go.Figure(go.Treemap(
        labels=top_titles.index.tolist(),
        parents=[""]*len(top_titles),
        values=top_titles.values.tolist(),
        marker=dict(colorscale=[[0,"#004182"],[0.5,"#0077B5"],[1,"#00c4ff"]],
                    cmid=top_titles.mean(),
                    line=dict(color="rgba(0,0,0,0.3)", width=2)),
        textfont=dict(color="white", size=13),
    ))
    fig_t.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=0,b=0), height=220,
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig_t, use_container_width=True)
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLUSTER INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    # cluster pill selector
    st.markdown('<div class="section-header">🗂 Select a Market Segment to Explore</div>', unsafe_allow_html=True)
 
    pills_html = '<div class="cluster-grid">'
    for _, row in profiles.sort_values("median_salary", ascending=False).iterrows():
        c = int(row["cluster"])
        name = CLUSTER_NAMES.get(c, row["label"])
        pills_html += f"""
        <div class="cluster-pill">
          <div class="badge">CLUSTER {c}</div>
          <div class="name">{name}</div>
          <div class="sal">${row['median_salary']/1000:.0f}K</div>
          <div class="cnt">{row['n_postings']:,} postings</div>
        </div>"""
    pills_html += "</div>"
    st.markdown(pills_html, unsafe_allow_html=True)
 
    selected_c = st.selectbox("Deep-dive into cluster:", options=sorted(profiles["cluster"].unique()),
                               format_func=lambda c: f"Cluster {c} — {CLUSTER_NAMES.get(c,'')}")
 
    prof = profiles[profiles["cluster"]==selected_c].iloc[0]
    c_jobs = jobs_df[jobs_df["cluster"]==selected_c]
    col_color = CLUSTER_COLORS[selected_c % len(CLUSTER_COLORS)]
 
    # 3 insight boxes
    st.markdown(f"""
    <div class="insight-row">
      <div class="insight-box">
        <div class="val">${prof['median_salary']/1000:.0f}K</div>
        <div class="lbl">Median Salary</div>
      </div>
      <div class="insight-box">
        <div class="val">{prof['remote_share']:.0%}</div>
        <div class="lbl">Remote Share</div>
      </div>
      <div class="insight-box">
        <div class="val">{prof['avg_num_skills']:.1f}</div>
        <div class="lbl">Avg Skills Tagged</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
 
    col_a, col_b = st.columns(2)
 
    with col_a:
        st.markdown('<div class="section-header">💼 Experience Mix</div>', unsafe_allow_html=True)
        exp_counts = c_jobs["formatted_experience_level"].value_counts()
        fig_exp = go.Figure(go.Bar(
            x=exp_counts.index, y=exp_counts.values,
            marker_color=col_color, opacity=0.85,
        ))
        fig_exp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
            margin=dict(l=0,r=0,t=10,b=0), height=220,
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        )
        st.plotly_chart(fig_exp, use_container_width=True)
 
        st.markdown('<div class="section-header">📊 Salary Spread</div>', unsafe_allow_html=True)
        fig_box = go.Figure(go.Box(
            y=c_jobs["normalized_salary"], name=f"Cluster {selected_c}",
            marker_color=col_color, line_color=col_color,
            boxmean=True, fillcolor=hex_to_rgba(col_color, 0.2),
        ))
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
            margin=dict(l=0,r=0,t=10,b=0), height=200,
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickprefix="$", tickformat=","),
        )
        st.plotly_chart(fig_box, use_container_width=True)
 
    with col_b:
        st.markdown('<div class="section-header">🗺 Cluster on Market Map (PCA)</div>', unsafe_allow_html=True)
        bg = jobs_df[jobs_df["cluster"]!=selected_c].sample(min(4000,len(jobs_df)), random_state=42)
        fig_map = go.Figure()
        fig_map.add_trace(go.Scatter(
            x=bg["pca_1"], y=bg["pca_2"], mode="markers",
            marker=dict(color="rgba(255,255,255,0.08)", size=4),
            name="Other clusters", showlegend=True,
        ))
        fig_map.add_trace(go.Scatter(
            x=c_jobs["pca_1"], y=c_jobs["pca_2"], mode="markers",
            marker=dict(color=col_color, size=6, opacity=0.7,
                        line=dict(color="white", width=0.3)),
            name=f"Cluster {selected_c}", showlegend=True,
        ))
        fig_map.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
            margin=dict(l=0,r=0,t=10,b=0), height=280,
            legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showticklabels=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showticklabels=False),
        )
        st.plotly_chart(fig_map, use_container_width=True)
 
        st.markdown('<div class="section-header">🏢 Top Companies in This Cluster</div>', unsafe_allow_html=True)
        top_co_c = c_jobs["company_name"].value_counts().head(6)
        co_html = ""
        for co, cnt in top_co_c.items():
            co_html += f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.06)">
              <span style="color:rgba(255,255,255,0.85);font-size:0.83rem">{co}</span>
              <span style="background:{col_color}33;color:{col_color};
                           border-radius:20px;padding:2px 10px;font-size:0.75rem;font-weight:600">
                {cnt} jobs</span>
            </div>"""
        st.markdown(f'<div class="glass-card" style="padding:14px 20px">{co_html}</div>',
                    unsafe_allow_html=True)
 
    st.markdown('<div class="section-header">🔑 Top Job Titles in This Cluster</div>', unsafe_allow_html=True)
    top_t = c_jobs["title"].value_counts().head(10)
    fig_tt = go.Figure(go.Bar(
        x=top_t.values, y=top_t.index, orientation="h",
        marker=dict(color=top_t.values,
                    colorscale=[[0, hex_to_rgba(col_color, 0.4)],[1, col_color]],
                    showscale=False),
        text=top_t.values, textposition="outside",
        textfont=dict(color="rgba(255,255,255,0.7)", size=11),
    ))
    fig_tt.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
        margin=dict(l=0,r=0,t=10,b=0), height=300,
        yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    )
    st.plotly_chart(fig_tt, use_container_width=True)
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — JOB CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    st.markdown("""
    <div class="glass-card" style="padding:20px 24px;margin-bottom:1.5rem">
      <div style="font-size:1rem;font-weight:600;color:rgba(255,255,255,0.9)">🎯 Classify a Job Posting</div>
      <div style="font-size:0.82rem;color:rgba(255,255,255,0.45);margin-top:4px">
        Enter the details of any job posting and our AI will instantly segment it into one of 5 market clusters.
      </div>
    </div>
    """, unsafe_allow_html=True)
 
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            salary = st.number_input("💰 Annual Salary (USD)", min_value=15_000,
                                      max_value=500_000, value=90_000, step=5_000)
            experience = st.selectbox("🎓 Experience Level",
                ["Not Specified","Internship","Entry level","Associate",
                 "Mid-Senior level","Director","Executive"], index=4)
        with c2:
            work_type = st.selectbox("🏢 Work Type", prep.WORK_TYPES, index=0)
            remote = st.checkbox("🌐 Remote Position", value=False)
        with c3:
            skill_name_to_abr = {v: k for k, v in prep.SKILL_MAP.items()}
            selected_skills = st.multiselect("🛠 Required Skill Categories",
                options=sorted(skill_name_to_abr.keys()),
                default=["Information Technology","Engineering"])
        submitted = st.form_submit_button("🔍  Classify This Job Posting")
 
    if submitted:
        abrs = [skill_name_to_abr[s] for s in selected_skills]
        new_row  = prep.build_single_job_row(salary, experience, work_type, remote, abrs)
        feat_row = prep.engineer_features(new_row)
        X_new    = preprocessor.transform(feat_row[prep.get_feature_columns()])
        pred_c   = int(kmeans.predict(X_new)[0])
        coords   = pca.transform(X_new)[0]
        prof     = profiles[profiles["cluster"]==pred_c].iloc[0]
        col_c    = CLUSTER_COLORS[pred_c % len(CLUSTER_COLORS)]
        c_jobs_p = jobs_df[jobs_df["cluster"]==pred_c]
 
        # result card
        st.markdown(f"""
        <div class="result-card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px">
            <div>
              <div style="font-size:0.75rem;color:rgba(255,255,255,0.5);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px">Predicted Segment</div>
              <div class="result-cluster">Cluster {pred_c}</div>
              <div class="result-label">{CLUSTER_NAMES.get(pred_c, prof['label'])}</div>
              <div class="result-badge">{'🌐 Remote-friendly' if prof['remote_share']>0.15 else '🏢 On-site dominant'}</div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;min-width:320px">
              <div style="text-align:center;background:rgba(0,0,0,0.2);border-radius:14px;padding:14px">
                <div style="font-size:1.3rem;font-weight:700;color:#5de0a0">${prof['median_salary']/1000:.0f}K</div>
                <div style="font-size:0.7rem;color:rgba(255,255,255,0.45);margin-top:3px;text-transform:uppercase">Cluster Median</div>
              </div>
              <div style="text-align:center;background:rgba(0,0,0,0.2);border-radius:14px;padding:14px">
                <div style="font-size:1.3rem;font-weight:700;color:#00a0dc">{prof['n_postings']:,}</div>
                <div style="font-size:0.7rem;color:rgba(255,255,255,0.45);margin-top:3px;text-transform:uppercase">Postings</div>
              </div>
              <div style="text-align:center;background:rgba(0,0,0,0.2);border-radius:14px;padding:14px">
                <div style="font-size:1.3rem;font-weight:700;color:#f5a623">{prof['remote_share']:.0%}</div>
                <div style="font-size:0.7rem;color:rgba(255,255,255,0.45);margin-top:3px;text-transform:uppercase">Remote</div>
              </div>
            </div>
          </div>
          <div style="margin-top:16px;padding-top:14px;border-top:1px solid rgba(255,255,255,0.1);
                      font-size:0.83rem;color:rgba(255,255,255,0.55)">
            🛠 Top skills in this cluster: <span style="color:rgba(255,255,255,0.85)">{prof['top_3_skills']}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown("")
        col_m, col_n = st.columns([2, 3])
 
        with col_m:
            st.markdown('<div class="section-header">📍 Position on Market Map</div>', unsafe_allow_html=True)
            all_c = sorted(jobs_df["cluster"].unique())
            fig_p = go.Figure()
            for cc in all_c:
                sub_c = jobs_df[jobs_df["cluster"]==cc].sample(min(1200,len(jobs_df[jobs_df["cluster"]==cc])),random_state=42)
                fig_p.add_trace(go.Scatter(
                    x=sub_c["pca_1"], y=sub_c["pca_2"], mode="markers",
                    marker=dict(color=CLUSTER_COLORS[cc % len(CLUSTER_COLORS)],
                                size=4 if cc==pred_c else 3,
                                opacity=0.7 if cc==pred_c else 0.25),
                    name=f"C{cc}", showlegend=False,
                ))
            fig_p.add_trace(go.Scatter(
                x=[coords[0]], y=[coords[1]], mode="markers",
                marker=dict(symbol="star", size=20, color="white",
                            line=dict(color=col_c, width=2)),
                name="Your Job ⭐", showlegend=True,
            ))
            fig_p.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
                margin=dict(l=0,r=0,t=10,b=0), height=300,
                legend=dict(bgcolor="rgba(0,0,0,0)",x=0,y=1),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showticklabels=False),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showticklabels=False),
            )
            st.plotly_chart(fig_p, use_container_width=True)
 
        with col_n:
            st.markdown('<div class="section-header">🔗 Most Similar Postings in This Cluster</div>', unsafe_allow_html=True)
            similar = c_jobs_p.copy()
            similar["_dist"] = (similar["normalized_salary"] - salary).abs()
            top10 = similar.sort_values("_dist").head(8)
            jobs_html = ""
            for _, row in top10.iterrows():
                sal_str = f"${row['normalized_salary']/1000:.0f}K"
                remote_badge = "🌐 Remote" if row["remote_allowed"] else "🏢 On-site"
                jobs_html += f"""
                <div class="similar-job">
                  <div>
                    <div class="sj-title">{str(row['title'])[:45]}</div>
                    <div class="sj-company">{str(row.get('company_name',''))[:40]} · {remote_badge}</div>
                  </div>
                  <div class="sj-salary">{sal_str}</div>
                </div>"""
            st.markdown(f'<div class="glass-card" style="padding:14px 18px">{jobs_html}</div>',
                        unsafe_allow_html=True)
