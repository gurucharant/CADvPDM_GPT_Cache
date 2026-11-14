def _source_ids(search_results, max_sources=12):
    """
    Extract unique source IDs from CrossSourceSearch tuples.

    Expected row structure:
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
    """

    evidence = []
    seen = set()

    for row in search_results or []:

        if len(row) < 2:
            continue

        source_type = str(row[0])
        source_id = str(row[1])

        key = (
            source_type,
            source_id,
        )

        if key in seen:
            continue

        seen.add(key)

        evidence.append(
            f"- **{source_type}: {source_id}**"
        )

        if len(evidence) >= max_sources:
            break

    return "\n".join(evidence)


def compose_mixed_answer(
    analytics_result,
    semantic_summary,
    search_results,
):
    """
    Build a concise MIXED-mode narrative answer.

    Important design rule:

    - Python / Streamlit owns:
        counts
        KPIs
        charts
        rankings
        resolver tables
        resolution-frequency tables

    - LLM / narrative owns:
        historical interpretation
        recurring root causes
        recurring resolutions
        documented guidance

    This avoids duplicating analytical tables in the chat response.
    """

    sections = analytics_result.get(
        "sections",
        {}
    )

    output = []

    # ==========================================================
    # Executive summary
    # ==========================================================

    total = sections.get(
        "total_incidents"
    )

    top_sites = sections.get(
        "top_sites"
    )

    top_resolvers = sections.get(
        "top_resolvers"
    )

    common_resolutions = sections.get(
        "common_resolutions"
    )

    summary_lines = []

    if total is not None:
        summary_lines.append(
            f"**{int(total):,} matching incidents** "
            "were identified in the structured Jira + Support "
            "incident history."
        )

    # ----------------------------------------------------------
    # Mention top site in one sentence only.
    # Do not reproduce the whole table.
    # ----------------------------------------------------------

    if (
        top_sites is not None
        and hasattr(top_sites, "empty")
        and not top_sites.empty
    ):
        site = top_sites.iloc[0]["site"]
        count = int(
            top_sites.iloc[0]["count"]
        )

        summary_lines.append(
            f"The most affected site was "
            f"**{site}** with **{count:,} incidents**."
        )

    # ----------------------------------------------------------
    # Mention top resolver in one sentence only.
    # ----------------------------------------------------------

    if (
        top_resolvers is not None
        and hasattr(top_resolvers, "empty")
        and not top_resolvers.empty
    ):
        resolver = (
            top_resolvers.iloc[0][
                "resolver"
            ]
        )

        count = int(
            top_resolvers.iloc[0][
                "count"
            ]
        )

        summary_lines.append(
            f"The leading resolver was "
            f"**{resolver}** with "
            f"**{count:,} resolved cases**."
        )

    # ----------------------------------------------------------
    # Mention leading resolution percentage only.
    # Detailed table remains in Streamlit.
    # ----------------------------------------------------------

    if (
        common_resolutions is not None
        and hasattr(
            common_resolutions,
            "empty",
        )
        and not common_resolutions.empty
    ):

        first = common_resolutions.iloc[0]

        if (
            "percentage"
            in common_resolutions.columns
        ):
            percentage = float(
                first["percentage"]
            )

            summary_lines.append(
                f"The leading normalized resolution pattern "
                f"accounts for **{percentage:.1f}%** "
                f"of matching resolved incidents."
            )

    if summary_lines:

        output.append(
            "## Executive Summary\n\n"
            + "\n\n".join(
                summary_lines
            )
        )

    # ==========================================================
    # Historical semantic intelligence
    # ==========================================================

    if semantic_summary:

        output.append(
            "## Historical Knowledge & Patterns\n\n"
            + semantic_summary.strip()
        )


    return "\n\n".join(output)