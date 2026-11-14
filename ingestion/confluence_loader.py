from pathlib import Path

import pandas as pd

from ingestion.chunker import chunk_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFLUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "confluence"
)

INDEX_FILE = (
    CONFLUENCE_DIR
    / "confluence_index.csv"
)


def clean_value(value):
    """
    Convert pandas NaN / None values
    into clean strings.
    """

    if pd.isna(value):
        return ""

    return str(value)


def load_confluence_documents(
        max_pages=None
):

    print("Reading Confluence index from:")
    print(INDEX_FILE)

    df = pd.read_csv(INDEX_FILE)

    if max_pages is not None:
        df = df.head(max_pages)

    documents = []

    pages_processed = 0

    for _, row in df.iterrows():

        page_id = clean_value(
            row.get("page_id")
        )

        title = clean_value(
            row.get("title")
        )

        file_name = clean_value(
            row.get("file_name")
        )

        file_path = (
            CONFLUENCE_DIR
            / file_name
        )

        if not file_path.exists():

            print(
                f"WARNING: File not found: "
                f"{file_path}"
            )

            continue

        text = file_path.read_text(
            encoding="utf-8"
        )

        chunks = chunk_text(
            text,
            chunk_size=160,
            overlap=30
        )

        for chunk_index, chunk in enumerate(
                chunks
        ):

            # Add title/context to each chunk
            content = f"""
Confluence Knowledge Article

Page ID:
{page_id}

Title:
{title}

Application:
{clean_value(row.get('application'))}

Module:
{clean_value(row.get('module'))}

Error Code:
{clean_value(row.get('error_code'))}

Content:
{chunk}
""".strip()

            metadata = {
                "page_type": clean_value(
                    row.get("page_type")
                ),
                "owner": clean_value(
                    row.get("owner")
                ),
                "country": clean_value(
                    row.get("country")
                ),
                "labels": clean_value(
                    row.get("labels")
                ),
                "updated_ts": clean_value(
                    row.get("updated_ts")
                ),
                "file_name": file_name
            }

            document = {

                "source_type":
                    "CONFLUENCE",

                "source_id":
                    page_id,

                "chunk_index":
                    chunk_index,

                "title":
                    title,

                "content":
                    content,

                "site":
                    clean_value(
                        row.get("site_scope")
                    ),

                "application":
                    clean_value(
                        row.get("application")
                    ),

                "error_code":
                    clean_value(
                        row.get("error_code")
                    ),

                "metadata":
                    metadata
            }

            documents.append(
                document
            )

        pages_processed += 1

    print(
        f"\nProcessed "
        f"{pages_processed:,} "
        f"Confluence pages."
    )

    print(
        f"Created "
        f"{len(documents):,} "
        f"Confluence chunks."
    )

    return documents


if __name__ == "__main__":

    # Only 5 pages for initial test
    docs = load_confluence_documents(
        max_pages=5
    )

    print(
        "\nSample Confluence document:\n"
    )

    print(
        docs[0]
    )