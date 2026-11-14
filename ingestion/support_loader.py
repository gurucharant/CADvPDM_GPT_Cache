from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUPPORT_FILE = (
    PROJECT_ROOT
    / "data"
    / "support"
    / "support_tickets.csv"
)


def load_support_documents():

    print("Reading Support data from:")
    print(SUPPORT_FILE)

    df = pd.read_csv(SUPPORT_FILE)

    documents = []

    for _, row in df.iterrows():

        content = f"""
Support Ticket: {row.get('ticket_id', '')}

Problem:
{row.get('problem_description', '')}

Short Description:
{row.get('short_description', '')}

Symptoms:
{row.get('symptoms', '')}

Application:
{row.get('application', '')}

Module:
{row.get('module', '')}

Site:
{row.get('site', '')}

Country:
{row.get('country', '')}

Error Code:
{row.get('error_code', '')}

Diagnostic Notes:
{row.get('diagnostic_notes', '')}

Root Cause:
{row.get('root_cause', '')}

Resolution:
{row.get('resolution_summary', '')}

Workaround:
{row.get('workaround', '')}

Resolved By:
{row.get('assigned_engineer', '')}

Assignment Group:
{row.get('assignment_group', '')}
"""

        metadata = {
            "priority": str(row.get("priority", "")),
            "severity": str(row.get("severity", "")),
            "impact": str(row.get("impact", "")),
            "urgency": str(row.get("urgency", "")),
            "channel": str(row.get("channel", "")),
            "requester_department": str(
                row.get("requester_department", "")
            ),
            "opened_ts": str(row.get("opened_ts", "")),
            "resolved_ts": str(row.get("resolved_ts", "")),
            "resolution_minutes": str(
                row.get("resolution_minutes", "")
            ),
            "assigned_engineer": str(
                row.get("assigned_engineer", "")
            ),
            "assignment_group": str(
                row.get("assignment_group", "")
            ),
            "sla_met": str(row.get("sla_met", "")),
            "related_jira_id": str(
                row.get("related_jira_id", "")
            ),
            "related_kb_id": str(
                row.get("related_kb_id", "")
            ),
            "related_log_file": str(
                row.get("related_log_file", "")
            )
        }

        document = {
            "source_type": "SUPPORT",
            "source_id": str(row.get("ticket_id", "")),
            "chunk_index": 0,
            "title": str(
                row.get("short_description", "")
            ),
            "content": content.strip(),
            "site": str(row.get("site", "")),
            "application": str(
                row.get("application", "")
            ),
            "error_code": str(
                row.get("error_code", "")
            ),
            "metadata": metadata
        }

        documents.append(document)

    print(
        f"Loaded {len(documents):,} Support documents."
    )

    return documents


if __name__ == "__main__":

    docs = load_support_documents()

    print("\nSample Support document:\n")
    print(docs[0])