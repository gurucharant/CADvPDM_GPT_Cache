"""
Read and verify the local CADvPDM GPT Apache Iceberg warehouse.
"""

from pathlib import Path

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
# Load Iceberg catalog
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


print("=" * 90)
print("CADvPDM GPT - LOCAL ICEBERG READ TEST")
print("=" * 90)


# ==============================================================
# List tables
# ==============================================================

tables = catalog.list_tables(
    "cadvpdm"
)

print(
    "\nAvailable Iceberg tables:"
)

for table in tables:

    print(
        f"  - {'.'.join(table)}"
    )


# ==============================================================
# Support table
# ==============================================================

print(
    "\n" + "-" * 90
)

print(
    "Testing cadvpdm.support_tickets"
)

support_table = catalog.load_table(
    "cadvpdm.support_tickets"
)

support_arrow = (
    support_table
    .scan(limit=5)
    .to_arrow()
)

print(
    support_arrow
)


# ==============================================================
# Jira table
# ==============================================================

print(
    "\n" + "-" * 90
)

print(
    "Testing cadvpdm.jira_issues"
)

jira_table = catalog.load_table(
    "cadvpdm.jira_issues"
)

jira_arrow = (
    jira_table
    .scan(limit=5)
    .to_arrow()
)

print(
    jira_arrow
)


# ==============================================================
# Execution logs
# ==============================================================

print(
    "\n" + "-" * 90
)

print(
    "Testing cadvpdm.execution_logs"
)

logs_table = catalog.load_table(
    "cadvpdm.execution_logs"
)

logs_arrow = (
    logs_table
    .scan(limit=5)
    .to_arrow()
)

print(
    logs_arrow
)


# ==============================================================
# Confluence
# ==============================================================

print(
    "\n" + "-" * 90
)

print(
    "Testing cadvpdm.confluence_index"
)

confluence_table = catalog.load_table(
    "cadvpdm.confluence_index"
)

confluence_arrow = (
    confluence_table
    .scan(limit=5)
    .to_arrow()
)

print(
    confluence_arrow
)


# ==============================================================
# Git
# ==============================================================

print(
    "\n" + "-" * 90
)

print(
    "Testing cadvpdm.git_activity"
)

git_table = catalog.load_table(
    "cadvpdm.git_activity"
)

git_arrow = (
    git_table
    .scan(limit=5)
    .to_arrow()
)

print(
    git_arrow
)


# ==============================================================
# Teams
# ==============================================================

print(
    "\n" + "-" * 90
)

print(
    "Testing cadvpdm.teams_transcripts"
)

teams_table = catalog.load_table(
    "cadvpdm.teams_transcripts"
)

teams_arrow = (
    teams_table
    .scan(limit=5)
    .to_arrow()
)

print(
    teams_arrow
)


# ==============================================================
# Engineers
# ==============================================================

print(
    "\n" + "-" * 90
)

print(
    "Testing cadvpdm.engineers"
)

engineers_table = catalog.load_table(
    "cadvpdm.engineers"
)

engineers_arrow = (
    engineers_table
    .scan(limit=5)
    .to_arrow()
)

print(
    engineers_arrow
)


# ==============================================================
# Metadata
# ==============================================================

print(
    "\n" + "-" * 90
)

print(
    "Support metadata location:"
)

print(
    support_table.metadata_location
)

print(
    "\nJira metadata location:"
)

print(
    jira_table.metadata_location
)


# ==============================================================
# Snapshot information
# ==============================================================

print(
    "\n" + "-" * 90
)

print(
    "Support current snapshot:"
)

print(
    support_table.current_snapshot()
)

print(
    "\nJira current snapshot:"
)

print(
    jira_table.current_snapshot()
)


print(
    "\n" + "=" * 90
)

print(
    "ICEBERG READ TEST COMPLETED SUCCESSFULLY"
)

print(
    "=" * 90
)