# LinkedIn Job Market Segmentation — Clustering Project

Segments LinkedIn job postings into data-driven market clusters using
**salary, required skills, experience level, and remote-work availability**,
via **K-Means clustering**. Includes a full Jupyter notebook pipeline and a
deployed **Streamlit app** for interactive exploration and live classification
of new postings.

## Project structure

```
project/
├── data/
│   ├── raw/                       # put the original Kaggle CSVs here
│   │   ├── postings.csv           # ⚠️ NOT included (493MB) — copy your own in
│   │   ├── job_skills.csv         # included
│   │   └── skills.csv             # included
│   └── processed/
│       └── clustered_jobs.csv     # output of the notebook — used by the app
├── models/
│   ├── preprocessor.pkl           # fitted ColumnTransformer (scaler + encoder)
│   ├── kmeans_model.pkl           # fitted K-Means model (k=5)
│   ├── pca.pkl                    # fitted PCA (for the 2D cluster map)
│   └── cluster_profiles.json      # human-readable cluster summaries
├── notebooks/
│   └── job_clustering_pipeline.ipynb   # the full ML pipeline, already executed
├── src/
│   └── preprocessing.py           # shared feature-engineering logic
├── app.py                         # Streamlit app (Explore + Predict tabs)
├── requirements.txt
└── README.md
```

`src/preprocessing.py` is the key design choice here: every feature-engineering
step (salary log-transform, experience ordinal encoding, skill multi-hot
encoding, etc.) is written **once** and imported by both the notebook and
`app.py`. That means a job posting you type into the Streamlit app is
transformed in *exactly* the same way as the training data — no train/serve
skew.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The repo ships with `job_skills.csv` and `skills.csv` already in
`data/raw/`. **You'll need to copy your own `postings.csv`** (from the
LinkedIn job postings dataset you already have) into `data/raw/` — it was
left out of this delivery because it's ~500MB.

## 1. Run the notebook (trains the model)

```bash
jupyter notebook notebooks/job_clustering_pipeline.ipynb
```

Run all cells top to bottom (the notebook is already pre-executed with
outputs/plots visible, so you can also just read through it). It will:

1. Load & merge `postings.csv` + `job_skills.csv`
2. Run EDA (missingness, salary distribution, skill frequency)
3. Clean/filter to USD postings with a plausible salary ($15k–$500k/yr)
4. Engineer features (log-salary, ordinal experience rank, work-type
   one-hot, remote flag, 35 multi-hot skill categories, skill count)
5. Fit a `ColumnTransformer` preprocessing pipeline
6. Select k via the **elbow method** + **silhouette score** (k=2..10 tested)
7. Train the final K-Means model (**k=5**)
8. Profile and visualize the clusters (PCA map, salary boxplots, skill
   heatmap, remote-share bars)
9. Save all artifacts to `models/` and `data/processed/`

## 2. Run the Streamlit app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with two tabs:

- **🔎 Explore Clusters** — filter by cluster, experience level, remote
  status, and salary range; see KPIs, an interactive PCA cluster map,
  salary-by-cluster boxplots, the full cluster-profile table, and a
  sortable table of sample postings.
- **🧭 Classify a New Posting** — enter a salary, experience level, work
  type, remote flag, and required skills for a hypothetical (or real) job
  posting, and the app runs it through the exact same trained pipeline to
  predict which of the 5 market segments it belongs to — showing the
  cluster's profile, where the new posting lands on the cluster map, and
  the most similar existing postings in that cluster.

## The 5 clusters (from this run)

| Cluster | Label | Median Salary | Dominant Experience | Top Skills |
|---|---|---|---|---|
| 4 | High-Pay · Mid-Senior · IT | $117,500 | Mid-Senior level | IT, Sales, Engineering |
| 0 | High-Pay · Mid-Senior · IT/Healthcare | $115,000 | Mid-Senior level | IT, Health Care, Other |
| 3 | High-Pay · Unspecified Level · IT | $113,650 | Not Specified | IT, Sales, Engineering |
| 1 | Entry-Pay · Entry Level · Support | $50,000 | Entry level | Other, Health Care, Admin |
| 2 | Entry-Pay · Entry Level · Manufacturing/Sales | $46,800 | Entry level | Management, Manufacturing, Sales |

(Exact numbers will shift slightly if you re-run the notebook on an updated
dataset — K-Means uses random initialization, though `random_state=42` is
fixed for reproducibility on the same data.)

## Why K-Means with k=5?

The notebook tests k=2 through k=10, tracking both **inertia** (elbow
method) and **silhouette score**. The elbow visibly flattens after k≈5,
and k=5 sits at a local peak in the silhouette curve — a better balance of
cluster separation than k=4 or k=6 — while still producing segments that
are large and distinct enough to be practically useful (entry-level vs.
senior, technical vs. operational, etc.).

## Notes & assumptions

- Only USD-denominated postings with salary in **$15,000–$500,000/yr** are
  used for clustering (filters out currency-conversion noise and obvious
  data-entry errors) — about 35.5k of the original 124k postings.
- Missing `formatted_experience_level` and `remote_allowed` are kept as
  their own explicit categories rather than dropped or imputed, since
  "the recruiter didn't specify this" is itself a meaningful signal.
- Skill categories come from the 35 LinkedIn skill taxonomy buckets in
  `skills.csv` (e.g. IT, Sales, Engineering), not free-text skill keywords.
