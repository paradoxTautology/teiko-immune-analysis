# Teiko Technical: Immune Cell Population Analysis

A Python pipeline and interactive dashboard for a clinical trial of the drug **miraclib** (Loblaw Bio). It loads raw immune cell counts into a normalized SQLite database, computes per-sample population frequencies, tests for differences between treatment responders and non-responders, and serves the results in a Streamlit dashboard.

**Live dashboard:** https://teiko-immune-analysis-bbryfilsageonz6kfuxpja.streamlit.app

---

## Running the project

Graded in GitHub Codespaces via three `make` targets:

```bash
make setup       # install dependencies from requirements.txt
make pipeline    # build the database, load the data, generate all outputs
make dashboard   # launch the interactive dashboard
```

`make pipeline` runs `load_data.py` then `analysis.py`: it creates `cell_count.db`, loads every row (Part 1), and writes all Part 2-4 tables and plots to `outputs/`. The pipeline is idempotent, so it can be re-run without manual cleanup.

`load_data.py` also runs on its own:

```bash
python load_data.py      # creates cell_count.db in the repository root
```

---

## Repository structure

| File | Purpose |
| --- | --- |
| `load_data.py` | **Part 1.** Defines the schema and loads `cell-count.csv` into `cell_count.db`. |
| `analysis.py` | **Parts 2-4.** Summary table, responder statistics and boxplot, baseline subset analysis. |
| `app.py` | Streamlit dashboard. |
| `Makefile` | `setup`, `pipeline`, and `dashboard` targets. |
| `requirements.txt` | Python dependencies. |
| `cell-count.csv` | Raw input data. |
| `outputs/` | Generated tables and plots. |

Data loading (`load_data.py`) is kept separate from analysis (`analysis.py`) so each has a single responsibility. The dashboard (`app.py`) reads the same database and reuses the analysis functions, so every number shown in the UI comes from the same code path as the pipeline outputs: one source of truth.

---

## Database schema

Four normalized tables:

```
projects (project_id)
    |
    | 1-to-many
    v
subjects (subject_id, project_id*, condition, age, sex, treatment, response)
    |
    | 1-to-many
    v
samples (sample_id, subject_id*, sample_type, time_from_treatment_start)
    |
    | 1-to-many
    v
cell_counts (sample_id*, population, count)     # long format
```

`*` = foreign key.

### Design rationale

- **Normalized to the subject.** Every subject-level attribute (condition, treatment, response, demographics) is constant across that subject's samples, so it is stored once per subject instead of being repeated on each sample row. This eliminates redundancy and update anomalies.
- **Long-format `cell_counts`.** Each (sample, population) pair is a row rather than a column. Adding a new cell population is just new rows, never a schema change, and computing each population's relative frequency is a single `GROUP BY` over the per-sample total.
- **Integrity and speed.** Foreign keys enforce the project to subject to sample hierarchy. Indexes sit on every column used to join or filter: `subjects(project_id)`, `samples(subject_id)`, `samples(sample_type, time_from_treatment_start)`, and `cell_counts(population)`.
- **`response` is nullable** because healthy and untreated subjects have no response label.

### Scaling to hundreds of projects and thousands of samples

- New projects, subjects, and samples are plain inserts; no structural change is required.
- The long-format counts table absorbs new assays and new cell populations without any migration, and supports new analytics as additional queries rather than new columns.
- The indexes keep filtered joins fast as the tables grow.
- The schema ports directly to a server database such as PostgreSQL. Frequently requested summaries (for example the Part 2 frequency table) can become materialized views, and the analytical queries are standard SQL that a warehouse can optimize.

---

## Analysis summary

**Part 2 - Data overview.** Relative frequency of each population per sample, with columns `sample`, `total_count`, `population`, `count`, `percentage` (`outputs/summary_table.csv`).

**Part 3 - Responders vs non-responders** (melanoma, miraclib, PBMC only). Each population's relative frequency is compared between responders and non-responders with the **Mann-Whitney U test**: a non-parametric test that makes no normality assumption and is robust to the skew and outliers visible in frequency data. P-values are corrected across the five populations with the **Benjamini-Hochberg** procedure to control the false discovery rate. Result: no population shows a statistically significant difference after correction; `cd4_t_cell` is the closest (nominal p = 0.013, not significant once corrected). Boxplot and full statistics are in `outputs/`.

**Part 4 - Baseline subset** (melanoma, miraclib, PBMC, time from treatment start = 0). Breakdown of samples per project, responder versus non-responder counts, and sex distribution (`outputs/part4_summary.txt`). All breakdowns share one cohort filter so the subset is identical across every figure. Average B-cell count for melanoma male responders at baseline: **10401.28**.

---

## Dashboard

Built with Streamlit. Three tabs:

- **Frequencies (Part 2)** - the per-sample frequency table, with single-sample inspection and CSV export.
- **Response analysis (Part 3)** - boxplot and significance tests, plus a time-course view of each population's mean frequency across timepoints (0, 7, 14) for responders versus non-responders. The default view is the required melanoma + miraclib + PBMC cohort; filters let an analyst explore other indications, drugs, sample types, and timepoints.
- **Baseline subset (Part 4)** - the baseline breakdowns and the headline B-cell average.

The top metric cards (samples, subjects, projects) are clickable and expand into their own breakdowns.
