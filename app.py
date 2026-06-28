"""
app.py - interactive Streamlit dashboard for the Teiko technical.

Run locally:
    streamlit run app.py
(`make dashboard` runs the same command.)

Features:
  * Orientation header   - cohort size and composition at a glance.
  * Part 2 frequencies   - per-sample table with single-sample inspect + CSV export.
  * Part 3 response view - boxplot + significance tests. Defaults to the required
                           cohort (melanoma + miraclib + PBMC) but the filters are
                           adjustable so an analyst can explore other cuts.
  * Part 4 baseline      - subset breakdowns and the headline B-cell average.

If cell_count.db is missing (e.g. a fresh deploy), it is built automatically from
cell-count.csv via load_data.py, so the app is self-contained.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import mannwhitneyu

import analysis
import load_data

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "cell_count.db"

st.set_page_config(page_title="Teiko Immune Profiling", layout="wide")


def ensure_database() -> None:
    """Build the SQLite database from the CSV if it does not already exist."""
    if not DB_PATH.exists():
        load_data.main()


ensure_database()


# --------------------------------------------------------------------------- #
# Data loaders (cached so interactions don't re-query)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading data...")
def load_summary() -> pd.DataFrame:
    conn = analysis.get_connection()
    try:
        return pd.read_sql_query(analysis.SUMMARY_SQL, conn)
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def load_overview():
    """Top-line scalars plus the per-dimension breakdowns shown when a metric
    card's 'Breakdown' is opened."""
    conn = analysis.get_connection()
    try:
        ov = {
            "scalars": pd.read_sql_query("""
                SELECT
                    (SELECT COUNT(*) FROM samples)  AS n_samples,
                    (SELECT COUNT(*) FROM subjects) AS n_subjects,
                    (SELECT COUNT(*) FROM projects) AS n_projects
            """, conn).iloc[0],
            "samples_by_type": pd.read_sql_query(
                "SELECT sample_type, COUNT(*) AS samples FROM samples "
                "GROUP BY sample_type ORDER BY sample_type;", conn),
            "samples_by_timepoint": pd.read_sql_query(
                "SELECT time_from_treatment_start AS timepoint, COUNT(*) AS samples "
                "FROM samples GROUP BY time_from_treatment_start "
                "ORDER BY time_from_treatment_start;", conn),
            "subjects_by_condition": pd.read_sql_query(
                "SELECT condition, COUNT(*) AS subjects FROM subjects "
                "GROUP BY condition ORDER BY condition;", conn),
            "subjects_by_treatment": pd.read_sql_query(
                "SELECT treatment, COUNT(*) AS subjects FROM subjects "
                "GROUP BY treatment ORDER BY treatment;", conn),
            "subjects_by_sex": pd.read_sql_query(
                "SELECT sex, COUNT(*) AS subjects FROM subjects "
                "GROUP BY sex ORDER BY sex;", conn),
            "by_project": pd.read_sql_query("""
                SELECT p.project_id AS project,
                       COUNT(DISTINCT sub.subject_id) AS subjects,
                       COUNT(s.sample_id)             AS samples
                FROM projects p
                JOIN subjects sub ON sub.project_id = p.project_id
                JOIN samples  s   ON s.subject_id   = sub.subject_id
                GROUP BY p.project_id ORDER BY p.project_id;
            """, conn),
        }
    finally:
        conn.close()
    return ov


@st.cache_data(show_spinner=False)
def load_baseline():
    conn = analysis.get_connection()
    try:
        per_project, response_split, sex_split = analysis.baseline_breakdowns(conn)
        avg_b, n = analysis.baseline_male_responder_bcell_avg(conn)
    finally:
        conn.close()
    return per_project, response_split, sex_split, avg_b, n


@st.cache_data(show_spinner=False)
def compute_cohort(condition, treatment, sample_type, timepoints):
    """Per-(sample, population) frequencies for an arbitrary filter combination.
    The default call (melanoma, miraclib, PBMC, all timepoints) reproduces the
    cohort required by Part 3."""
    placeholders = ",".join("?" for _ in timepoints)
    sql = f"""
        WITH totals AS (
            SELECT sample_id, SUM(count) AS total_count
            FROM cell_counts GROUP BY sample_id
        )
        SELECT s.sample_id                       AS sample,
               sub.response                      AS response,
               cc.population                     AS population,
               100.0 * cc.count / t.total_count  AS percentage
        FROM cell_counts cc
        JOIN totals   t   ON cc.sample_id = t.sample_id
        JOIN samples  s   ON cc.sample_id = s.sample_id
        JOIN subjects sub ON s.subject_id = sub.subject_id
        WHERE sub.condition = ?
          AND sub.treatment = ?
          AND s.sample_type = ?
          AND s.time_from_treatment_start IN ({placeholders})
          AND sub.response IN ('yes', 'no');
    """
    conn = analysis.get_connection()
    try:
        return pd.read_sql_query(sql, conn, params=[condition, treatment, sample_type, *timepoints])
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def compute_timecourse(condition, treatment, sample_type):
    """Mean population frequency at each timepoint, split by response, for the
    given cohort filters. Always spans all timepoints (it is a time series)."""
    sql = """
        WITH totals AS (
            SELECT sample_id, SUM(count) AS total_count
            FROM cell_counts GROUP BY sample_id
        )
        SELECT s.time_from_treatment_start          AS timepoint,
               sub.response                          AS response,
               cc.population                         AS population,
               AVG(100.0 * cc.count / t.total_count) AS mean_percentage
        FROM cell_counts cc
        JOIN totals   t   ON cc.sample_id = t.sample_id
        JOIN samples  s   ON cc.sample_id = s.sample_id
        JOIN subjects sub ON s.subject_id = sub.subject_id
        WHERE sub.condition = ?
          AND sub.treatment = ?
          AND s.sample_type = ?
          AND sub.response IN ('yes', 'no')
        GROUP BY s.time_from_treatment_start, sub.response, cc.population
        ORDER BY cc.population, sub.response, s.time_from_treatment_start;
    """
    conn = analysis.get_connection()
    try:
        return pd.read_sql_query(sql, conn, params=[condition, treatment, sample_type])
    finally:
        conn.close()


def compute_stats(cohort: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U per population with Benjamini-Hochberg correction. Assumes
    the cohort contains both responder and non-responder samples."""
    rows = []
    for pop in analysis.POPULATIONS:
        resp = cohort[(cohort.population == pop) & (cohort.response == "yes")]["percentage"]
        nonr = cohort[(cohort.population == pop) & (cohort.response == "no")]["percentage"]
        u, p = mannwhitneyu(resp, nonr, alternative="two-sided")
        rows.append({
            "population": pop,
            "n_responder": len(resp), "n_nonresponder": len(nonr),
            "median_responder_pct": round(resp.median(), 2),
            "median_nonresponder_pct": round(nonr.median(), 2),
            "u_statistic": round(u, 1), "p_value": p,
        })
    stats = pd.DataFrame(rows)
    stats["p_value_bh"] = analysis.benjamini_hochberg(stats["p_value"].values)
    stats["significant"] = stats["p_value_bh"] < 0.05
    stats["p_value"] = stats["p_value"].round(4)
    stats["p_value_bh"] = stats["p_value_bh"].round(4)
    return stats


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
summary = load_summary()
ov = load_overview()
sc = ov["scalars"]

st.title("Immune Cell Population Dashboard")
st.caption("Loblaw Bio clinical trial - cell-count analysis (Parts 2-4). "
           "Click any metric card to drill into its breakdown.")

# st.metric is not clickable, so each KPI card below is a full-width st.button
# restyled to look like a card. st.button is used ONLY here, so this CSS does not
# affect the download buttons or filter controls elsewhere in the app.
st.markdown("""
<style>
div[data-testid="stButton"] > button {
    width: 100%;
    padding: 0.85rem 1.1rem;
    border-radius: 0.6rem;
    border: 1px solid rgba(140,160,190,0.25);
    background: rgba(130,150,190,0.06);
    text-align: left;
    line-height: 1.25;
    transition: border-color .15s ease, background .15s ease;
}
div[data-testid="stButton"] > button:hover {
    border-color: #4a90d9;
    background: rgba(74,144,217,0.10);
}
div[data-testid="stButton"] > button p { margin: 0; color: inherit; }
div[data-testid="stButton"] > button strong { font-size: 1.7rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

if "open_card" not in st.session_state:
    st.session_state.open_card = None


def kpi_card(col, key, label, value):
    """Full-width clickable card; clicking toggles its breakdown panel."""
    with col:
        if st.button(f"{label}\n\n**{value}**", key=f"kpi_{key}", use_container_width=True):
            st.session_state.open_card = None if st.session_state.open_card == key else key


k1, k2, k3 = st.columns(3)
kpi_card(k1, "samples", "SAMPLES", f"{int(sc.n_samples):,}")
kpi_card(k2, "subjects", "SUBJECTS", f"{int(sc.n_subjects):,}")
kpi_card(k3, "projects", "PROJECTS", f"{int(sc.n_projects)}")

# Breakdown panel for whichever card is currently open.
open_card = st.session_state.open_card
if open_card == "samples":
    st.markdown("**Samples** - by type and timepoint")
    x, y = st.columns(2)
    x.dataframe(ov["samples_by_type"], hide_index=True, use_container_width=True)
    y.dataframe(ov["samples_by_timepoint"], hide_index=True, use_container_width=True)
elif open_card == "subjects":
    st.markdown("**Subjects** - by condition, treatment, and sex")
    x, y, z = st.columns(3)
    x.dataframe(ov["subjects_by_condition"], hide_index=True, use_container_width=True)
    y.dataframe(ov["subjects_by_treatment"], hide_index=True, use_container_width=True)
    z.dataframe(ov["subjects_by_sex"], hide_index=True, use_container_width=True)
elif open_card == "projects":
    st.markdown("**Per project** - subjects and samples")
    st.dataframe(ov["by_project"], hide_index=True, use_container_width=True)

tab2, tab3, tab4 = st.tabs(
    ["Frequencies (Part 2)", "Response analysis (Part 3)", "Baseline subset (Part 4)"]
)

# --- Part 2 -------------------------------------------------------------- #
with tab2:
    st.subheader("Relative frequency of each population, per sample")
    samples = sorted(summary["sample"].unique())
    pick = st.selectbox("Inspect a single sample (or view all)", ["(all samples)"] + samples)
    view = summary if pick == "(all samples)" else summary[summary["sample"] == pick]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download full summary table (CSV)", to_csv_bytes(summary),
                       "summary_table.csv", "text/csv")
    st.caption(f"{summary['sample'].nunique():,} samples x 5 populations = {len(summary):,} rows")

# --- Part 3 -------------------------------------------------------------- #
with tab3:
    st.subheader("Responders vs non-responders")
    st.caption("Default is the required cohort: melanoma + miraclib + PBMC, all timepoints. "
               "Adjust the filters to explore other indications, drugs, sample types, or timepoints.")

    f1, f2, f3, f4 = st.columns(4)
    condition = f1.selectbox("Indication", ["melanoma", "carcinoma", "healthy"], index=0)
    treatment = f2.selectbox("Treatment", ["miraclib", "phauximab", "none"], index=0)
    sample_type = f3.selectbox("Sample type", ["PBMC", "WB"], index=0)
    timepoints = f4.multiselect("Timepoints", [0, 7, 14], default=[0, 7, 14])
    pops = st.multiselect("Populations to show", analysis.POPULATIONS, default=analysis.POPULATIONS)

    if not timepoints:
        st.warning("Select at least one timepoint.")
    else:
        cohort = compute_cohort(condition, treatment, sample_type, tuple(timepoints))
        if cohort.empty or cohort["response"].nunique() < 2:
            st.warning("This filter combination has no responder vs non-responder split to compare "
                       "(untreated / healthy subjects have no response label). Try different filters.")
        else:
            n_pop = len(analysis.POPULATIONS)
            n_resp = int((cohort["response"] == "yes").sum() // n_pop)
            n_nonr = int((cohort["response"] == "no").sum() // n_pop)
            st.caption(f"Cohort: {n_resp} responder samples vs {n_nonr} non-responder samples")

            show_points = st.checkbox("Show individual sample points on the boxplot", value=False)
            plot_df = cohort[cohort["population"].isin(pops)] if pops else cohort
            fig = px.box(
                plot_df, x="population", y="percentage", color="response",
                points=("all" if show_points else False),
                category_orders={"population": analysis.POPULATIONS, "response": ["yes", "no"]},
                labels={"percentage": "Relative frequency (%)",
                        "population": "Immune cell population", "response": "Responder"},
            )
            st.plotly_chart(fig, use_container_width=True)

            stats = compute_stats(cohort)
            st.markdown("**Significance tests** - Mann-Whitney U, Benjamini-Hochberg corrected:")
            st.dataframe(stats, use_container_width=True, hide_index=True)
            st.download_button("Download stats (CSV)", to_csv_bytes(stats),
                               "response_stats.csv", "text/csv")
            sig = stats.loc[stats["significant"], "population"].tolist()
            if sig:
                st.success(f"Significant after correction: {', '.join(sig)}")
            else:
                st.info("No population is significant after correcting for multiple comparisons.")

            # --- time-course: how each population shifts across the trial ---- #
            st.markdown("---")
            st.subheader("Time-course: mean frequency over treatment timepoints")
            st.caption("Mean relative frequency at each timepoint (0 to 7 to 14 days), responders "
                       "vs non-responders. Spans all timepoints regardless of the filter above.")
            tc = compute_timecourse(condition, treatment, sample_type)
            tc = tc[tc["population"].isin(pops)] if pops else tc
            line = px.line(
                tc, x="timepoint", y="mean_percentage", color="response",
                facet_col="population", facet_col_wrap=3, markers=True,
                category_orders={"population": analysis.POPULATIONS, "response": ["yes", "no"]},
                labels={"timepoint": "Days from treatment start",
                        "mean_percentage": "Mean frequency (%)", "response": "Responder"},
            )
            line.update_xaxes(tickvals=[0, 7, 14])
            line.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(line, use_container_width=True)

# --- Part 4 -------------------------------------------------------------- #
with tab4:
    per_project, response_split, sex_split, avg_b, n = load_baseline()
    st.subheader("Baseline cohort: melanoma + miraclib + PBMC, time = 0")

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg B cells - melanoma male responders (t=0)", f"{avg_b:,.2f}", help=f"n = {n} samples")
    c2.metric("Subjects in cohort", f"{int(response_split['n_subjects'].sum()):,}")
    c3.metric("Samples in cohort", f"{int(per_project['n_samples'].sum()):,}")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("**Samples per project**")
        st.dataframe(per_project, hide_index=True, use_container_width=True)
    with d2:
        st.markdown("**Subjects by response**")
        st.dataframe(response_split, hide_index=True, use_container_width=True)
    with d3:
        st.markdown("**Subjects by sex**")
        st.dataframe(sex_split, hide_index=True, use_container_width=True)
