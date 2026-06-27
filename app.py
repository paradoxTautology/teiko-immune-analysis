"""
app.py - interactive Streamlit dashboard for the Teiko technical.

Run locally:
    streamlit run app.py
(`make dashboard` runs the same command.)

The dashboard reads cell_count.db. If that database is missing (e.g. on a fresh
deploy), it is built automatically from cell-count.csv via load_data.py, so the
app is self-contained.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import analysis
import load_data

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "cell_count.db"

st.set_page_config(page_title="Teiko Immune Profiling", layout="wide")


def ensure_database() -> None:
    """Build the SQLite database from the CSV if it does not already exist."""
    if not DB_PATH.exists():
        load_data.main()


@st.cache_data(show_spinner="Running pipeline...")
def load_all():
    ensure_database()
    conn = analysis.get_connection()
    try:
        summary = pd.read_sql_query(analysis.SUMMARY_SQL, conn)
        cohort = analysis.responder_cohort(conn)
        stats = analysis.compare_responders(conn)
        per_project, response_split, sex_split = analysis.baseline_breakdowns(conn)
        avg_b, n = analysis.baseline_male_responder_bcell_avg(conn)
    finally:
        conn.close()
    return summary, cohort, stats, per_project, response_split, sex_split, avg_b, n


(summary, cohort, stats, per_project, response_split,
 sex_split, avg_b, n) = load_all()

st.title("Immune Cell Population Dashboard")
st.caption("Loblaw Bio clinical trial - cell-count analysis (Parts 2-4)")

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
    st.caption(f"{summary['sample'].nunique():,} samples x 5 populations "
               f"= {len(summary):,} rows")

# --- Part 3 -------------------------------------------------------------- #
with tab3:
    st.subheader("Responders vs non-responders (melanoma + miraclib, PBMC only)")
    pops = st.multiselect("Populations to show", analysis.POPULATIONS,
                          default=analysis.POPULATIONS)
    plot_df = cohort[cohort["population"].isin(pops)]
    fig = px.box(
        plot_df, x="population", y="percentage", color="response",
        category_orders={"population": analysis.POPULATIONS, "response": ["yes", "no"]},
        labels={"percentage": "Relative frequency (%)",
                "population": "Immune cell population", "response": "Responder"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Significance tests** - Mann-Whitney U, Benjamini-Hochberg corrected:")
    st.dataframe(stats, use_container_width=True, hide_index=True)
    if not stats["significant"].any():
        st.info("No population shows a statistically significant difference after "
                "correcting for testing five populations at once. cd4_t_cell is the "
                "closest (nominal p = 0.013, not significant after correction).")

# --- Part 4 -------------------------------------------------------------- #
with tab4:
    st.subheader("Baseline cohort: melanoma + miraclib + PBMC, time = 0")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg B cells - melanoma male responders (t=0)",
              f"{avg_b:,.2f}", help=f"n = {n} samples")
    c2.metric("Subjects in cohort", f"{int(response_split['n_subjects'].sum()):,}")
    c3.metric("Samples in cohort", f"{int(per_project['n_samples'].sum()):,}")

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown("**Samples per project**")
        st.dataframe(per_project, hide_index=True, use_container_width=True)
    with c5:
        st.markdown("**Subjects by response**")
        st.dataframe(response_split, hide_index=True, use_container_width=True)
    with c6:
        st.markdown("**Subjects by sex**")
        st.dataframe(sex_split, hide_index=True, use_container_width=True)
