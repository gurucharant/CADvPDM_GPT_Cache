"""
Create a local Apache Iceberg analytical-memory warehouse
from the CADvPDM GPT POC CSV datasets.

POC architecture:

CSV
  ↓
Pandas
  ↓
PyArrow
  ↓
Apache Iceberg
  ↓
SQLite catalog
  ↓
Local Windows filesystem

Semantic embeddings remain in PostgreSQL + pgvector.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa

from pyiceberg.catalog import load_catalog


# ==============================================================
# Project paths
# ==============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

WAREHOUSE = (
    PROJECT_ROOT
    / "iceberg_warehouse"
).resolve()

WAREHOUSE.mkdir(
    parents=True,
    exist_ok=True,
)

CATALOG_DB = (
    WAREHOUSE
    / "catalog.db"
)


# ==============================================================
# Iceberg catalog
#
# IMPORTANT FOR WINDOWS:
# Use str(WAREHOUSE), NOT WAREHOUSE.as_uri()
# ==============================================================

catalog = load_catalog(
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
# Namespace
# ==============================================================

catalog.create_namespace_if_not_exists(
    "cadvpdm"
)


# ==============================================================
# Source datasets
# ==============================================================

SOURCES = {

    "support_tickets":
        "data/support/support_tickets.csv",

    "jira_issues":
        "data/jira/jira_issues.csv",

    "execution_logs":
        "data/logs/logs_index.csv",

    "confluence_index":
        "data/confluence/confluence_index.csv",

    "git_activity":
        "data/git/git_activity.csv",

    "teams_transcripts":
        "data/teams/teams_transcripts.csv",

    "engineers":
        "data/directory/engineers.csv",
}


# ==============================================================
# Data cleaning
# ==============================================================

def clean_for_arrow(df):
    """
    Convert object columns to strings so PyArrow
    receives consistent data types.
    """

    df = df.copy()

    for column in df.columns:

        if df[column].dtype == "object":

            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
            )

    return df


# ==============================================================
# Build
# ==============================================================

print("=" * 90)
print("CADvPDM GPT - LOCAL APACHE ICEBERG BUILD")
print("=" * 90)

print(
    f"\nProject root : {PROJECT_ROOT}"
)

print(
    f"Warehouse    : {WAREHOUSE}"
)

print(
    f"Catalog DB   : {CATALOG_DB}"
)

print()


loaded_tables = []
skipped_tables = []


for table_name, relative_path in SOURCES.items():

    csv_path = (
        PROJECT_ROOT
        / relative_path
    )

    identifier = (
        f"cadvpdm.{table_name}"
    )

    print("-" * 90)

    print(
        f"Source : {csv_path}"
    )

    print(
        f"Target : {identifier}"
    )

    # ----------------------------------------------------------
    # Check source file
    # ----------------------------------------------------------

    if not csv_path.exists():

        print(
            "WARNING: Source file does not exist."
        )

        print(
            "Skipping this table."
        )

        skipped_tables.append(
            table_name
        )

        continue

    # ----------------------------------------------------------
    # Read CSV
    # ----------------------------------------------------------

    df = pd.read_csv(
        csv_path
    )

    print(
        f"Rows read: {len(df):,}"
    )

    print(
        f"Columns  : {len(df.columns)}"
    )

    # ----------------------------------------------------------
    # Clean data
    # ----------------------------------------------------------

    df = clean_for_arrow(
        df
    )

    # ----------------------------------------------------------
    # Convert to Arrow
    # ----------------------------------------------------------

    arrow_table = (
        pa.Table.from_pandas(
            df,
            preserve_index=False,
        )
    )

    # ----------------------------------------------------------
    # Create Iceberg table
    # ----------------------------------------------------------

    iceberg_table = (
        catalog.create_table_if_not_exists(
            identifier,
            schema=arrow_table.schema,
        )
    )

    # ----------------------------------------------------------
    # Overwrite table content
    #
    # Good for reproducible POC builds.
    # ----------------------------------------------------------

    iceberg_table.overwrite(
        arrow_table
    )

    print(
        "SUCCESS"
    )

    print(
        f"Rows written: {len(df):,}"
    )

    print(
        "Metadata:"
    )

    print(
        iceberg_table.metadata_location
    )

    loaded_tables.append(
        table_name
    )


# ==============================================================
# Summary
# ==============================================================

print(
    "\n" + "=" * 90
)

print(
    "ICEBERG BUILD COMPLETE"
)

print(
    "=" * 90
)

print(
    "\nWarehouse:"
)

print(
    WAREHOUSE
)

print(
    "\nLoaded tables:"
)

for table_name in loaded_tables:

    print(
        f"  - cadvpdm.{table_name}"
    )


if skipped_tables:

    print(
        "\nSkipped tables:"
    )

    for table_name in skipped_tables:

        print(
            f"  - {table_name}"
        )


print(
    "\nArchitecture:"
)

print(
    "Analytical memory : Apache Iceberg"
)

print(
    "Semantic memory   : PostgreSQL + pgvector"
)