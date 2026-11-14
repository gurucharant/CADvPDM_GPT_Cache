"""
CADvPDM GPT - Local Iceberg Data Loader

Loads analytical datasets from the local Apache Iceberg warehouse
and converts them to Pandas DataFrames.

Semantic memory remains in PostgreSQL + pgvector.
"""

from pathlib import Path

import pandas as pd

from pyiceberg.catalog import load_catalog


# ==============================================================
# Paths
# ==============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

WAREHOUSE = (
    PROJECT_ROOT
    / "iceberg_warehouse"
).resolve()

CATALOG_DB = (
    WAREHOUSE
    / "catalog.db"
)


# ==============================================================
# Catalog
# ==============================================================

def get_catalog():

    return load_catalog(
        "cadvpdm_local",

        type="sql",

        uri=(
            f"sqlite:///"
            f"{CATALOG_DB.as_posix()}"
        ),

        warehouse=str(
            WAREHOUSE
        ),
    )


# ==============================================================
# Generic table reader
# ==============================================================

def load_table_to_pandas(
    table_name: str,
) -> pd.DataFrame:
    """
    Load one Iceberg table into a Pandas DataFrame.

    Example:
        load_table_to_pandas("support_tickets")
    """

    catalog = get_catalog()

    identifier = (
        f"cadvpdm.{table_name}"
    )

    table = catalog.load_table(
        identifier
    )

    arrow_table = (
        table
        .scan()
        .to_arrow()
    )

    return arrow_table.to_pandas()


# ==============================================================
# Individual logical datasets
# ==============================================================

def load_support():
    return load_table_to_pandas(
        "support_tickets"
    )


def load_jira():
    return load_table_to_pandas(
        "jira_issues"
    )


def load_logs():
    return load_table_to_pandas(
        "execution_logs"
    )


def load_confluence():
    return load_table_to_pandas(
        "confluence_index"
    )


def load_git():
    return load_table_to_pandas(
        "git_activity"
    )


def load_teams():
    return load_table_to_pandas(
        "teams_transcripts"
    )


def load_engineers():
    return load_table_to_pandas(
        "engineers"
    )


# ==============================================================
# Load all analytical datasets
# ==============================================================

def load_all():

    return {
        "support": load_support(),
        "jira": load_jira(),
        "logs": load_logs(),
        "confluence": load_confluence(),
        "git": load_git(),
        "teams": load_teams(),
        "engineers": load_engineers(),
    }


# ==============================================================
# Standalone verification
# ==============================================================

if __name__ == "__main__":

    print("=" * 90)
    print(
        "CADvPDM GPT - ICEBERG DATA LOADER TEST"
    )
    print("=" * 90)

    datasets = load_all()

    for name, df in datasets.items():

        print(
            f"{name:<12} "
            f"rows={len(df):>7,} "
            f"columns={len(df.columns):>3}"
        )

    print(
        "\nIceberg DataFrame loading "
        "completed successfully."
    )