"""
analysis.py
-----------
Analytical pipeline for the Teiko technical. Reads cell_count.db (built by
load_data.py) and produces the outputs for Parts 2-4.

    Part 2  build_summary_table()  -> outputs/summary_table.csv
    Part 3  compare_responders()   -> outputs/part3_boxplot.png, outputs/part3_stats.csv
    Part 4  (added next)

Run:
    python analysis.py
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use("Agg")  # headless backend so plots save without a display
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "cell_count.db"
OUTPUT_DIR = ROOT / "outputs"

POPULATIONS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


# --------------------------------------------------------------------------- #
# Part 2: relative frequency of each population per sample
# --------------------------------------------------------------------------- #
SUMMARY_SQL = """
    SELECT
        cc.sample_id                                    AS sample,
        totals.total_count                              AS total_count,
        cc.population                                   AS population,
        cc.count                                        AS count,
        ROUND(100.0 * cc.count / totals.total_count, 2) AS percentage
    FROM cell_counts AS cc
    JOIN (
        SELECT sample_id, SUM(count) AS total_count
        FROM cell_counts
        GROUP BY sample_id
    ) AS totals ON cc.sample_id = totals.sample_id
    ORDER BY cc.sample_id, cc.population;
"""


def build_summary_table(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(SUMMARY_SQL, conn)
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / "summary_table.csv", index=False)
    return df


# --------------------------------------------------------------------------- #
# Part 3: responders vs non-responders (melanoma + miraclib, PBMC only)
# --------------------------------------------------------------------------- #
# Full-precision frequencies (no rounding) so the statistical test isn't
# affected by display rounding.
COHORT_SQL = """
    WITH totals AS (
        SELECT sample_id, SUM(count) AS total_count
        FROM cell_counts
        GROUP BY sample_id
    )
    SELECT
        s.sample_id                       AS sample,
        sub.response                      AS response,
        cc.population                     AS population,
        100.0 * cc.count / t.total_count  AS percentage
    FROM cell_counts AS cc
    JOIN totals   AS t   ON cc.sample_id = t.sample_id
    JOIN samples  AS s   ON cc.sample_id = s.sample_id
    JOIN subjects AS sub ON s.subject_id = sub.subject_id
    WHERE sub.condition = 'melanoma'
      AND sub.treatment = 'miraclib'
      AND s.sample_type = 'PBMC'
      AND sub.response IN ('yes', 'no');
"""


def responder_cohort(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(COHORT_SQL, conn)


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (controls false discovery rate
    across the 5 population tests)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = p.argsort()
    adj = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]  # enforce monotonicity
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def compare_responders(conn: sqlite3.Connection) -> pd.DataFrame:
    """Boxplot each population by response and test for significant differences."""
    df = responder_cohort(conn)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- boxplot: one population group per x tick, split responder vs not ---
    plt.figure(figsize=(11, 6))
    sns.boxplot(
        data=df, x="population", y="percentage", hue="response",
        order=POPULATIONS, hue_order=["yes", "no"],
    )
    plt.title("Melanoma + miraclib (PBMC): population frequency by treatment response")
    plt.xlabel("Immune cell population")
    plt.ylabel("Relative frequency (%)")
    plt.legend(title="Responder")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "part3_boxplot.png", dpi=150)
    plt.close()

    # --- Mann-Whitney U per population (non-parametric: no normality assumed) ---
    rows = []
    for pop in POPULATIONS:
        resp = df[(df.population == pop) & (df.response == "yes")]["percentage"]
        nonr = df[(df.population == pop) & (df.response == "no")]["percentage"]
        u, p = mannwhitneyu(resp, nonr, alternative="two-sided")
        rows.append({
            "population": pop,
            "n_responder": len(resp),
            "n_nonresponder": len(nonr),
            "median_responder_pct": round(resp.median(), 2),
            "median_nonresponder_pct": round(nonr.median(), 2),
            "u_statistic": round(u, 1),
            "p_value": p,
        })

    stats = pd.DataFrame(rows)
    stats["p_value_bh"] = benjamini_hochberg(stats["p_value"].values)
    stats["significant"] = stats["p_value_bh"] < 0.05
    stats["p_value"] = stats["p_value"].round(4)
    stats["p_value_bh"] = stats["p_value_bh"].round(4)
    stats.to_csv(OUTPUT_DIR / "part3_stats.csv", index=False)
    return stats


# --------------------------------------------------------------------------- #
# Part 4: baseline subset analysis (melanoma + miraclib + PBMC, time = 0)
# --------------------------------------------------------------------------- #
# The cohort Part 4 is scoped to. Defined once and reused by every query so the
# subset is guaranteed identical across the breakdowns and the average.
BASELINE_WHERE = """
        sub.condition = 'melanoma'
    AND sub.treatment = 'miraclib'
    AND s.sample_type = 'PBMC'
    AND s.time_from_treatment_start = 0
"""


def baseline_breakdowns(conn: sqlite3.Connection):
    """Within the baseline cohort: samples per project, and subject-level
    responder and sex splits."""
    per_project = pd.read_sql_query(f"""
        SELECT sub.project_id AS project, COUNT(*) AS n_samples
        FROM samples s JOIN subjects sub ON s.subject_id = sub.subject_id
        WHERE {BASELINE_WHERE}
        GROUP BY sub.project_id ORDER BY sub.project_id;
    """, conn)

    response_split = pd.read_sql_query(f"""
        SELECT sub.response AS response, COUNT(DISTINCT sub.subject_id) AS n_subjects
        FROM samples s JOIN subjects sub ON s.subject_id = sub.subject_id
        WHERE {BASELINE_WHERE}
        GROUP BY sub.response ORDER BY sub.response;
    """, conn)

    sex_split = pd.read_sql_query(f"""
        SELECT sub.sex AS sex, COUNT(DISTINCT sub.subject_id) AS n_subjects
        FROM samples s JOIN subjects sub ON s.subject_id = sub.subject_id
        WHERE {BASELINE_WHERE}
        GROUP BY sub.sex ORDER BY sub.sex;
    """, conn)
    return per_project, response_split, sex_split


def baseline_male_responder_bcell_avg(conn: sqlite3.Connection):
    """Form question: average B-cell COUNT for melanoma male responders at
    baseline (within the PBMC + miraclib cohort). Returns (average, n)."""
    val = pd.read_sql_query(f"""
        SELECT ROUND(AVG(cc.count), 2) AS avg_b_cells, COUNT(*) AS n
        FROM samples s
        JOIN subjects sub ON s.subject_id = sub.subject_id
        JOIN cell_counts cc ON cc.sample_id = s.sample_id AND cc.population = 'b_cell'
        WHERE {BASELINE_WHERE}
          AND sub.sex = 'M'
          AND sub.response = 'yes';
    """, conn)
    return float(val["avg_b_cells"].iloc[0]), int(val["n"].iloc[0])


def part4_subset_analysis(conn: sqlite3.Connection):
    per_project, response_split, sex_split = baseline_breakdowns(conn)
    avg_b, n = baseline_male_responder_bcell_avg(conn)

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "part4_summary.txt", "w") as f:
        f.write("Baseline cohort: melanoma + miraclib + PBMC, time_from_treatment_start = 0\n\n")
        f.write("Samples per project:\n" + per_project.to_string(index=False) + "\n\n")
        f.write("Subjects by response:\n" + response_split.to_string(index=False) + "\n\n")
        f.write("Subjects by sex:\n" + sex_split.to_string(index=False) + "\n\n")
        f.write(f"Average B cells (melanoma male responders, time=0): {avg_b:.2f}  (n={n})\n")
    return per_project, response_split, sex_split, avg_b, n


def main() -> None:
    conn = get_connection()
    try:
        summary = build_summary_table(conn)
        print(f"Part 2 - summary table: {summary.shape[0]:,} rows")

        stats = compare_responders(conn)
        n_resp = stats["n_responder"].iloc[0]
        n_nonr = stats["n_nonresponder"].iloc[0]
        print(f"\nPart 3 - responders vs non-responders "
              f"({n_resp} responder samples, {n_nonr} non-responder samples):")
        print(stats.to_string(index=False))
        sig = stats.loc[stats["significant"], "population"].tolist()
        print("\nSignificant (BH-adjusted p < 0.05):", sig if sig else "none")

        per_project, response_split, sex_split, avg_b, n = part4_subset_analysis(conn)
        print("\nPart 4 - baseline cohort (melanoma + miraclib + PBMC, time=0):")
        print("  samples per project: " +
              ", ".join(f"{r.project}={r.n_samples}" for r in per_project.itertuples()))
        print("  subjects by response: " +
              ", ".join(f"{r.response}={r.n_subjects}" for r in response_split.itertuples()))
        print("  subjects by sex: " +
              ", ".join(f"{r.sex}={r.n_subjects}" for r in sex_split.itertuples()))
        print(f"  AVG B cells (melanoma male responders, time=0): {avg_b:.2f}  (n={n})")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
