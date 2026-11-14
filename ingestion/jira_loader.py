from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

JIRA_FILE = (
    PROJECT_ROOT
    / "data"
    / "jira"
    / "jira_issues.csv"
)


def load_jira_documents():

    print("Reading Jira data from:")
    print(JIRA_FILE)

    df = pd.read_csv(JIRA_FILE)

    documents = []

    for _, row in df.iterrows():

        content = f"""
Jira Issue: {row.get('issue_id', '')}

Issue Type:
{row.get('issue_type', '')}

Summary:
{row.get('summary', '')}

Description:
{row.get('description', '')}

Application:
{row.get('application', '')}

Component:
{row.get('component', '')}

Site:
{row.get('site', '')}

Country:
{row.get('country', '')}

Error Code:
{row.get('error_code', '')}

Root Cause:
{row.get('root_cause', '')}

Resolution:
{row.get('resolution', '')}

Comments / Investigation:
{row.get('comments_summary', '')}
"""

        metadata = {
            "project": str(row.get("project", "")),
            "priority": str(row.get("priority", "")),
            "status": str(row.get("status", "")),
            "assignee": str(row.get("assignee", "")),
            "created_ts": str(row.get("created_ts", "")),
            "resolved_ts": str(row.get("resolved_ts", "")),
            "related_kb_id": str(row.get("related_kb_id", "")),
            "resolution_minutes": str(
                row.get("resolution_minutes", "")
            )
        }

        document = {
            "source_type": "JIRA",
            "source_id": str(row.get("issue_id", "")),
            "chunk_index": 0,
            "title": str(row.get("summary", "")),
            "content": content.strip(),
            "site": str(row.get("site", "")),
            "application": str(row.get("application", "")),
            "error_code": str(row.get("error_code", "")),
            "metadata": metadata
        }

        documents.append(document)

    print(
        f"Loaded {len(documents):,} Jira documents."
    )

    return documents


if __name__ == "__main__":

    docs = load_jira_documents()

    print("\nSample Jira document:\n")
    print(docs[0])