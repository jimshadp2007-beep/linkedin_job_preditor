"""
LinkedIn Job Market Segmentation — Streamlit App
--------------------------------------------------
Two tabs:
  1. Explore   — interactive dashboard over the pre-clustered dataset
  2. Predict   — classify a brand-new job posting into one of the 5 clusters

Run with:  streamlit run app.py
"""

import sys
import os
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import preprocessing as prep

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="LinkedIn Job Market Segmentation",
    page_icon="📊",
    layout="wide",
)

PALETTE = px.colors.qualitative.Set2

# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------

@st.cache_resource
def load_models():
    preprocessor = joblib.load("models/preprocessor.pkl")
    kmeans = joblib.load("models/kmeans_model.pkl")
    pca = joblib.load("models/pca.pkl")
    with open("models/cluster_profiles.json") as f:
        profiles = pd.DataFrame(json.load(f))
    return preprocessor, kmeans, pca, profiles


@st.cache_data
def load_clustered_data():
    return pd.read_csv("data/processed/clustered_jobs.csv")


def cluster_color_map(profiles):
    return {int(row.cluster): PALETTE[i % len(PALETTE)] for i, row in profiles.iterrows()}


def cluster_label(profiles, cluster_id):
    row = profiles[profiles["cluster"] == cluster_id]
    if len(row) == 0:
        return f"Cluster {cluster_id}"
    return row.iloc[0]["label"]


# --------------------------------------------------------------------------
# Load everything once
# --------------------------------------------------------------------------
try:
    preprocessor, kmeans, pca, profiles = load_models()
    jobs_df = load_clustered_data()
    color_map = cluster_color_map(profiles)
except FileNotFoundError as e:
    st.error(
        "Couldn't find the trained model artifacts. Run "
        "`notebooks/job_clustering_pipeline.ipynb` first to generate "
        "`models/*.pkl` and `data/processed/clustered_jobs.csv`.\n\n"
        f"Details: {e}"
    )
    st.stop()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("📊 LinkedIn Job Market Segmentation")
st.caption(
    "Job postings clustered with K-Means on salary, skills, experience "
    "level, and remote-work availability."
)

tab_explore, tab_predict = st.tabs(["🔎 Explore Clusters", "🧭 Classify a New Posting"])

# ==========================================================================
# TAB 1 — EXPLORE
# ==========================================================================
with tab_explore:
    st.sidebar.header("Filters")

    cluster_options = sorted(jobs_df["cluster"].unique())
    cluster_labels_map = {c: f"Cluster {c} — {cluster_label(profiles, c)}" for c in cluster_options}
    selected_clusters = st.sidebar.multiselect(
        "Clusters",
        options=cluster_options,
        default=cluster_options,
        format_func=lambda c: cluster_labels_map[c],
    )

    exp_options = sorted(jobs_df["formatted_experience_level"].unique())
    selected_exp = st.sidebar.multiselect("Experience level", exp_options, default=exp_options)

    remote_filter = st.sidebar.radio("Remote", ["All", "Remote only", "On-site / unspecified"], index=0)

    salary_min, salary_max = int(jobs_df["normalized_salary"].min()), int(jobs_df["normalized_salary"].max())
    salary_range = st.sidebar.slider(
        "Salary range (USD/yr)", salary_min, salary_max, (salary_min, salary_max), step=5000
    )

    filtered = jobs_df[
        jobs_df["cluster"].isin(selected_clusters)
        & jobs_df["formatted_experience_level"].isin(selected_exp)
        & jobs_df["normalized_salary"].between(*salary_range)
    ]
    if remote_filter == "Remote only":
        filtered = filtered[filtered["remote_allowed"] == 1]
    elif remote_filter == "On-site / unspecified":
        filtered = filtered[filtered["remote_allowed"] == 0]

    # ---- KPI row ----
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Postings (filtered)", f"{len(filtered):,}")
    k2.metric("Median salary", f"${filtered['normalized_salary'].median():,.0f}" if len(filtered) else "—")
    k3.metric("Remote share", f"{filtered['remote_allowed'].mean():.0%}" if len(filtered) else "—")
    k4.metric("Avg. skills tagged", f"{filtered['num_skills'].mean():.1f}" if len(filtered) else "—")

    st.divider()

    if len(filtered) == 0:
        st.warning("No postings match the current filters. Try widening them in the sidebar.")
    else:
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("Cluster map (PCA projection)")
            plot_df = filtered.copy()
            plot_df["Cluster"] = plot_df["cluster"].map(lambda c: cluster_labels_map[c])
            fig = px.scatter(
                plot_df.sample(min(8000, len(plot_df)), random_state=42),
                x="pca_1", y="pca_2", color="Cluster",
                color_discrete_map={cluster_labels_map[c]: color_map[c] for c in cluster_options},
                hover_data=["title", "company_name", "normalized_salary"],
                opacity=0.55,
                height=480,
            )
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.35))
            st.plotly_chart(fig, width='stretch')

        with col_right:
            st.subheader("Salary by cluster")
            box_df = jobs_df[jobs_df["cluster"].isin(selected_clusters)].copy()
            box_df["Cluster"] = box_df["cluster"].map(lambda c: cluster_labels_map[c])
            fig2 = px.box(
                box_df, x="Cluster", y="normalized_salary",
                color="Cluster",
                color_discrete_map={cluster_labels_map[c]: color_map[c] for c in cluster_options},
                points=False, height=480,
            )
            fig2.update_layout(showlegend=False, xaxis_title="", yaxis_title="Annual salary (USD)")
            fig2.update_xaxes(tickfont=dict(size=9))
            st.plotly_chart(fig2, width='stretch')

        st.subheader("Cluster profiles")
        prof_display = profiles.copy()
        prof_display["median_salary"] = prof_display["median_salary"].map(lambda v: f"${v:,.0f}")
        prof_display["remote_share"] = prof_display["remote_share"].map(lambda v: f"{v:.0%}")
        prof_display["avg_num_skills"] = prof_display["avg_num_skills"].round(1)
        st.dataframe(
            prof_display[[
                "cluster", "label", "n_postings", "median_salary",
                "top_experience", "top_work_type", "remote_share",
                "avg_num_skills", "top_3_skills",
            ]].rename(columns={
                "cluster": "Cluster", "label": "Label", "n_postings": "# Postings",
                "median_salary": "Median Salary", "top_experience": "Top Experience",
                "top_work_type": "Top Work Type", "remote_share": "Remote Share",
                "avg_num_skills": "Avg. Skills", "top_3_skills": "Top Skills",
            }),
            width='stretch', hide_index=True,
        )

        st.subheader("Sample postings")
        sample_cols = ["title", "company_name", "location", "normalized_salary",
                        "formatted_experience_level", "work_type", "remote_allowed", "cluster"]
        st.dataframe(
            filtered[sample_cols].sort_values("normalized_salary", ascending=False).head(200),
            width='stretch', hide_index=True,
        )

# ==========================================================================
# TAB 2 — PREDICT
# ==========================================================================
with tab_predict:
    st.subheader("Classify a new job posting")
    st.caption(
        "Enter the details of a job posting (real or hypothetical) and see "
        "which market segment it falls into."
    )

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            salary = st.number_input(
                "Annual salary (USD)", min_value=15_000, max_value=500_000,
                value=90_000, step=1_000,
            )
            experience = st.selectbox(
                "Experience level",
                ["Not Specified", "Internship", "Entry level", "Associate",
                 "Mid-Senior level", "Director", "Executive"],
                index=4,
            )
            work_type = st.selectbox("Work type", prep.WORK_TYPES, index=0)
        with c2:
            remote = st.checkbox("Remote allowed", value=False)
            skill_name_to_abr = {v: k for k, v in prep.SKILL_MAP.items()}
            selected_skill_names = st.multiselect(
                "Required skill categories",
                options=sorted(skill_name_to_abr.keys()),
                default=["Information Technology"],
            )
        submitted = st.form_submit_button("Classify posting", type="primary")

    if submitted:
        selected_abrs = [skill_name_to_abr[s] for s in selected_skill_names]
        new_row = prep.build_single_job_row(
            normalized_salary=salary,
            experience_level=experience,
            work_type=work_type,
            remote_allowed=remote,
            selected_skill_abrs=selected_abrs,
        )
        feat_row = prep.engineer_features(new_row)
        X_new = preprocessor.transform(feat_row[prep.get_feature_columns()])
        predicted_cluster = int(kmeans.predict(X_new)[0])
        new_coords = pca.transform(X_new)[0]

        prof_row = profiles[profiles["cluster"] == predicted_cluster].iloc[0]

        st.success(f"**Predicted segment: Cluster {predicted_cluster} — {prof_row['label']}**")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cluster median salary", f"${prof_row['median_salary']:,.0f}")
        m2.metric("Cluster size", f"{prof_row['n_postings']:,} postings")
        m3.metric("Remote share in cluster", f"{prof_row['remote_share']:.0%}")
        m4.metric("Most common experience", prof_row["top_experience"])

        st.write(f"**Dominant skills in this cluster:** {prof_row['top_3_skills']}")

        st.subheader("Where this posting lands on the cluster map")
        base = jobs_df.sample(min(6000, len(jobs_df)), random_state=42).copy()
        base["Cluster"] = base["cluster"].map(lambda c: f"Cluster {c}")
        all_clusters = sorted(jobs_df["cluster"].unique())
        fig3 = px.scatter(
            base, x="pca_1", y="pca_2", color="Cluster",
            color_discrete_map={f"Cluster {c}": color_map[c] for c in all_clusters},
            opacity=0.35, height=480,
        )
        fig3.add_scatter(
            x=[new_coords[0]], y=[new_coords[1]], mode="markers",
            marker=dict(size=18, color="black", symbol="star"),
            name="Your posting",
        )
        st.plotly_chart(fig3, width='stretch')

        st.subheader("Similar existing postings in this cluster")
        similar = jobs_df[jobs_df["cluster"] == predicted_cluster].copy()
        similar["dist"] = (similar["normalized_salary"] - salary).abs()
        similar = similar.sort_values("dist").head(10)
        st.dataframe(
            similar[["title", "company_name", "location", "normalized_salary",
                     "formatted_experience_level", "work_type", "remote_allowed"]],
            width='stretch', hide_index=True,
        )
