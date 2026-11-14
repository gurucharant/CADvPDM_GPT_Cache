def build_context(
        search_results,
        max_chars_per_result=1200
):
    """
    Converts retrieved database records
    into structured context for the LLM.
    """

    context_blocks = []

    for index, row in enumerate(
            search_results,
            start=1
    ):

        (
            source_type,
            source_id,
            chunk_index,
            title,
            site,
            application,
            error_code,
            content,
            metadata,
            similarity
        ) = row

        block = f"""
SOURCE #{index}

Source Type: {source_type}
Source ID: {source_id}
Title: {title}
Site: {site}
Application: {application}
Error Code: {error_code}
Similarity: {float(similarity):.4f}

Evidence:
{content[:max_chars_per_result]}
""".strip()

        context_blocks.append(block)

    separator = "\n\n" + "=" * 80 + "\n\n"

    return separator.join(context_blocks)