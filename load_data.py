"""
load_data.py
------------
Initializes a SQLite database from cell-count.csv and loads every row into a
normalized relational schema (projects -> subjects -> samples -> cell_counts).

Usage:
    python load_data.py

Produces:
    cell_count.db   (SQLite database, written to the repository root)

The script is idempotent: re-running it drops and rebuilds the tables, so
`make pipeline` can be run repeatedly without manual cleanup.
"""

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "cell-count.csv"
DB_PATH = ROOT / "cell_count.db"

# The five immune-cell populations stored as columns in the CSV. Kept in one
# place so adding/removing a population is a one-line change.
POPULATIONS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the normalized schema, dropping any existing tables first."""
    conn.executescript(
        """
        DROP TABLE IF EXISTS cell_counts;
        DROP TABLE IF EXISTS samples;
        DROP TABLE IF EXISTS subjects;
        DROP TABLE IF EXISTS projects;

        -- One row per project.
        CREATE TABLE projects (
            project_id TEXT PRIMARY KEY
        );

        -- One row per subject. All subject-level attributes (condition,
        -- treatment, response, demographics) live here exactly once.
        CREATE TABLE subjects (
            subject_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            condition  TEXT,
            age        INTEGER,
            sex        TEXT,
            treatment  TEXT,
            response   TEXT,            -- NULL for healthy / untreated subjects
            FOREIGN KEY (project_id) REFERENCES projects (project_id)
        );

        -- One row per biological sample (a subject at one timepoint).
        CREATE TABLE samples (
            sample_id                 TEXT PRIMARY KEY,
            subject_id                TEXT NOT NULL,
            sample_type               TEXT,            -- e.g. PBMC, WB
            time_from_treatment_start INTEGER,
            FOREIGN KEY (subject_id) REFERENCES subjects (subject_id)
        );

        -- Long format: one row per (sample, population). Adding a new cell
        -- population means new rows, never a schema change.
        CREATE TABLE cell_counts (
            sample_id  TEXT NOT NULL,
            population TEXT NOT NULL,
            count      INTEGER NOT NULL,
            PRIMARY KEY (sample_id, population),
            FOREIGN KEY (sample_id) REFERENCES samples (sample_id)
        );

        -- Indexes that keep filters/joins fast as the data grows.
        CREATE INDEX idx_subjects_project   ON subjects (project_id);
        CREATE INDEX idx_samples_subject     ON samples (subject_id);
        CREATE INDEX idx_samples_type_time   ON samples (sample_type, time_from_treatment_start);
        CREATE INDEX idx_cell_counts_pop     ON cell_counts (population);
        """
    )
    conn.commit()


def load_data(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """Populate all four tables from the raw CSV dataframe."""
    # projects: distinct project ids
    projects = (
        df[["project"]]
        .drop_duplicates()
        .rename(columns={"project": "project_id"})
    )
    projects.to_sql("projects", conn, if_exists="append", index=False)

    # subjects: one row per subject (attributes are constant within a subject)
    subjects = (
        df[["subject", "project", "condition", "age", "sex", "treatment", "response"]]
        .drop_duplicates(subset=["subject"])
        .rename(columns={"subject": "subject_id", "project": "project_id"})
    )
    subjects.to_sql("subjects", conn, if_exists="append", index=False)

    # samples: one row per sample
    samples = (
        df[["sample", "subject", "sample_type", "time_from_treatment_start"]]
        .drop_duplicates(subset=["sample"])
        .rename(columns={"sample": "sample_id", "subject": "subject_id"})
    )
    samples.to_sql("samples", conn, if_exists="append", index=False)

    # cell_counts: melt the five population columns from wide to long
    cell_counts = (
        df.melt(
            id_vars=["sample"],
            value_vars=POPULATIONS,
            var_name="population",
            value_name="count",
        )
        .rename(columns={"sample": "sample_id"})
    )
    cell_counts.to_sql("cell_counts", conn, if_exists="append", index=False)

    conn.commit()


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Expected {CSV_PATH.name} in the repository root.")

    df = pd.read_csv(CSV_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        create_schema(conn)
        load_data(conn, df)

        print("Loaded cell_count.db:")
        for table in ["projects", "subjects", "samples", "cell_counts"]:
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:<12} {n:>7,} rows")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
