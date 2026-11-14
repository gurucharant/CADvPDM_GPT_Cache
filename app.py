import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rag.rag_pipeline import CADvPDMRAG


# ==============================================================
# Page configuration
# ==============================================================

st.set_page_config(
    page_title="CADvPDM GPT",
    page_icon="🧠",
    layout="wide",
)


# ==============================================================
# Styling
# ==============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================
# Load engine once
# ==============================================================

@st.cache_resource
def load_engine():
    return CADvPDMRAG()


rag = load_engine()


# ==============================================================
# Source conversion
# ==============================================================

def convert_sources(rows):
    out = []

    for row in rows or []:

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
            similarity,
        ) = row

        out.append({
            "source_type": source_type,
            "source_id": source_id,
            "chunk_index": chunk_index,
            "title": title,
            "site": site,
            "application": application,
            "error_code": error_code,
            "content": content,
            "metadata": metadata,
            "similarity": float(similarity),
        })

    return out


# ==============================================================
# Supporting evidence
# ==============================================================

def display_sources(sources):

    if not sources:
        return

    with st.expander(
        f"🔎 Supporting organizational evidence "
        f"({len(sources)})"
    ):

        for s in sources:

            st.markdown(
                f"**{s['source_type']} — "
                f"{s['source_id']}**  \n"
                f"{s['title']}  \n"
                f"Application: {s['application']} | "
                f"Site: {s['site']} | "
                f"Error: {s['error_code']} | "
                f"Similarity: "
                f"{s['similarity']:.3f}"
            )

            with st.expander(
                "View retrieved content"
            ):
                st.text(
                    s["content"][:1800]
                )

            st.divider()


# ==============================================================
# Generic chart
# ==============================================================

def display_chart(df, spec, chart_key):

    if (
        df is None
        or spec is None
        or df.empty
    ):
        return

    typ = spec.get("type")

    if typ == "line":

        fig = px.line(
            df,
            x=spec["x"],
            y=spec["y"],
            markers=True,
            title=spec.get("title"),
        )

    elif typ == "pareto":

        fig = make_subplots(
            specs=[
                [
                    {
                        "secondary_y": True
                    }
                ]
            ]
        )

        fig.add_trace(
            go.Bar(
                x=df[spec["x"]],
                y=df[spec["y"]],
                name="Count",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=df[spec["x"]],
                y=df[
                    spec["secondary"]
                ],
                name="Cumulative %",
                mode="lines+markers",
            ),
            secondary_y=True,
        )

        fig.update_yaxes(
            title_text="Count",
            secondary_y=False,
        )

        fig.update_yaxes(
            title_text="Cumulative %",
            range=[0, 105],
            secondary_y=True,
        )

        fig.update_layout(
            title=spec.get("title")
        )

    else:

        fig = px.bar(
            df,
            x=spec["x"],
            y=spec["y"],
            title=spec.get("title"),
        )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=chart_key,
    )


# ==============================================================
# Convert section DataFrames for session state
# ==============================================================

def serialize_sections(sections):

    serialized = {}

    for key, value in (
        sections or {}
    ).items():

        if isinstance(
            value,
            pd.DataFrame,
        ):
            serialized[key] = {
                "type": "dataframe",
                "data": value.to_dict(
                    "records"
                ),
            }

        else:
            serialized[key] = {
                "type": "value",
                "data": value,
            }

    return serialized


def deserialize_sections(sections):

    restored = {}

    for key, value in (
        sections or {}
    ).items():

        if (
            isinstance(value, dict)
            and value.get("type")
            == "dataframe"
        ):

            restored[key] = (
                pd.DataFrame(
                    value.get(
                        "data",
                        [],
                    )
                )
            )

        elif isinstance(
            value,
            dict,
        ):

            restored[key] = (
                value.get("data")
            )

        else:
            restored[key] = value

    return restored


# ==============================================================
# MIXED intelligence dashboard
# ==============================================================

def display_mixed_dashboard(
    sections,
    key_prefix,
):

    if not sections:
        return

    # ----------------------------------------------------------
    # KPI row
    # ----------------------------------------------------------

    total = sections.get(
        "total_incidents"
    )

    top_sites = sections.get(
        "top_sites"
    )

    top_resolvers = sections.get(
        "top_resolvers"
    )

    common_resolutions = (
        sections.get(
            "common_resolutions"
        )
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "Matching incidents",
            f"{int(total):,}"
            if total is not None
            else "N/A",
        )

    with col2:

        if (
            isinstance(
                top_sites,
                pd.DataFrame,
            )
            and not top_sites.empty
        ):

            st.metric(
                "Top affected site",
                str(
                    top_sites.iloc[0][
                        "site"
                    ]
                ),
                f"{int(top_sites.iloc[0]['count']):,} incidents",
            )

        else:

            st.metric(
                "Top affected site",
                "N/A",
            )

    with col3:

        if (
            isinstance(
                top_resolvers,
                pd.DataFrame,
            )
            and not top_resolvers.empty
        ):

            st.metric(
                "Top resolver",
                str(
                    top_resolvers.iloc[
                        0
                    ]["resolver"]
                ),
                f"{int(top_resolvers.iloc[0]['count']):,} cases",
            )

        else:

            st.metric(
                "Top resolver",
                "N/A",
            )

    with col4:

        if (
            isinstance(
                common_resolutions,
                pd.DataFrame,
            )
            and not common_resolutions.empty
            and "percentage"
            in common_resolutions.columns
        ):

            percentage = float(
                common_resolutions.iloc[
                    0
                ]["percentage"]
            )

            st.metric(
                "Leading resolution",
                f"{percentage:.1f}%",
                "of resolved incidents",
            )

        else:

            st.metric(
                "Leading resolution",
                "N/A",
            )

    st.divider()

    # ----------------------------------------------------------
    # Sites + resolution patterns
    # ----------------------------------------------------------

    left, right = st.columns(
        [1, 1]
    )

    with left:

        if (
            isinstance(
                top_sites,
                pd.DataFrame,
            )
            and not top_sites.empty
        ):

            st.markdown(
                "### 🌍 Top Affected Sites"
            )

            fig = px.bar(
                top_sites,
                x="site",
                y="count",
                title=(
                    "Incidents by site"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"{key_prefix}_top_sites",
            )

    with right:

        if (
            isinstance(
                common_resolutions,
                pd.DataFrame,
            )
            and not common_resolutions.empty
        ):

            st.markdown(
                "### 🛠 Resolution Patterns"
            )

            resolution_chart = (
                common_resolutions.copy()
            )

            # Short display labels
            resolution_chart[
                "resolution_short"
            ] = (
                resolution_chart[
                    "resolution_pattern"
                ]
                .astype(str)
                .str.slice(0, 55)
            )

            fig = px.bar(
                resolution_chart,
                x="count",
                y="resolution_short",
                orientation="h",
                title=(
                    "Historical resolution "
                    "frequency"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"{key_prefix}_resolution_patterns",
            )

            display_cols = [
                c
                for c in [
                    "resolution_pattern",
                    "count",
                    "percentage",
                ]
                if c
                in common_resolutions.columns
            ]

            st.dataframe(
                common_resolutions[
                    display_cols
                ],
                use_container_width=True,
                hide_index=True,
            )

    # ----------------------------------------------------------
    # Resolver table
    # ----------------------------------------------------------

    resolver_contacts = (
        sections.get(
            "resolver_contacts"
        )
    )

    if (
        isinstance(
            resolver_contacts,
            pd.DataFrame,
        )
        and not resolver_contacts.empty
    ):

        st.markdown(
            "### 👥 Main Resolvers / Contacts"
        )

        st.dataframe(
            resolver_contacts,
            use_container_width=True,
            hide_index=True,
        )

    # ----------------------------------------------------------
    # Trend if present
    # ----------------------------------------------------------

    trend = sections.get(
        "trend"
    )

    if (
        isinstance(
            trend,
            pd.DataFrame,
        )
        and not trend.empty
    ):

        st.markdown(
            "### 📈 Incident Trend"
        )

        fig = px.line(
            trend,
            x="period",
            y="issue_count",
            markers=True,
            title=(
                "Incident volume over time"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"{key_prefix}_trend",
        )

    # ----------------------------------------------------------
    # SLA if present
    # ----------------------------------------------------------

    sla = sections.get(
        "sla"
    )

    if (
        isinstance(
            sla,
            pd.DataFrame,
        )
        and not sla.empty
    ):

        st.markdown(
            "### ⏱ Resolution Performance"
        )

        st.dataframe(
            sla,
            use_container_width=True,
            hide_index=True,
        )


# ==============================================================
# Sidebar
# ==============================================================

with st.sidebar:

    st.title(
        "🧠 CADvPDM GPT"
    )

    st.caption(
        "Hybrid Organizational Intelligence"
    )

    st.markdown(
        """
🎫 Support  
🛠 Jira  
📘 Confluence  
📋 Logs  
💻 Git  
👥 Teams
"""
    )

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ==============================================================
# Page title
# ==============================================================

st.markdown(
    """
    <div class="main-title">
        🧠 CADvPDM GPT
    </div>

    <div class="subtitle">
        Organizational intelligence across incidents,
        engineering history, analytics and knowledge.
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================
# Chat history
# ==============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# if not st.session_state.messages:
#
#     st.info(
#         "Try: "
#         "‘Show issue counts over time for ENOVIA’, "
#         "‘Which error codes contribute to 80% of incidents?’, "
#         "‘Who resolved AUTH-401 incidents?’, "
#         "or "
#         "‘How many AUTH-401 incidents occurred, "
#         "which sites had the most, who resolved them, "
#         "and what were the most common historical resolutions?’"
#     )


# ==============================================================
# Render previous messages
# ==============================================================

for message_index, m in enumerate(st.session_state.messages):

    with st.chat_message(
        m["role"]
    ):

        st.markdown(
            m["content"]
        )

        if m.get("mode"):

            st.caption(
                f"Query path: "
                f"{m['mode']}"
            )

        sections = deserialize_sections(
            m.get(
                "sections",
                {},
            )
        )

        if (
            m.get("mode")
            == "MIXED"
            and sections
        ):

            display_mixed_dashboard(
                sections,
                key_prefix=f"history_{message_index}_mixed",
            )

        else:

            if (
                m.get("data")
                is not None
            ):

                st.dataframe(
                    pd.DataFrame(
                        m["data"]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            if (
                m.get(
                    "chart_data"
                )
                is not None
            ):

                display_chart(
                    pd.DataFrame(
                        m[
                            "chart_data"
                        ]
                    ),
                    m.get("chart"),
                    chart_key=f"history_{message_index}_chart",
                )

        display_sources(
            m.get(
                "sources",
                [],
            )
        )


# ==============================================================
# User input
# ==============================================================

question = st.chat_input(
    "Ask CADvPDM GPT anything about "
    "organizational engineering history "
    "or knowledge..."
)


# ==============================================================
# Process question
# ==============================================================

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Routing the question and querying "
            "the right organizational memory..."
        ):

            try:

                result = rag.ask(
                    question,
                    results_per_source=3,
                )

                answer = result[
                    "answer"
                ]

                sources = (
                    convert_sources(
                        result.get(
                            "results",
                            [],
                        )
                    )
                )

                mode = result.get(
                    "mode",
                    "",
                )

                sections = result.get(
                    "sections",
                    {},
                )

                # ------------------------------------------------
                # Main answer
                # ------------------------------------------------

                st.markdown(
                    answer
                )

                cache_hit = result.get(
                    "cache_hit",
                    False,
                )

                if cache_hit:
                    cache_similarity = result.get(
                        "cache_similarity",
                        0.0,
                    )
                    st.caption(
                        f"Query path: {mode} · "
                        f"Semantic cache hit "
                        f"({cache_similarity:.1%}) ⚡"
                    )
                else:
                    st.caption(
                        f"Query path: {mode}"
                    )

                # ------------------------------------------------
                # MIXED dashboard
                # ------------------------------------------------

                if (
                    mode == "MIXED"
                    and sections
                ):

                    display_mixed_dashboard(
                        sections,
                        key_prefix=f"current_{len(st.session_state.messages)}_mixed",
                    )

                # ------------------------------------------------
                # Non-MIXED legacy display
                # ------------------------------------------------

                else:

                    df = result.get(
                        "data"
                    )

                    chart = result.get(
                        "chart"
                    )

                    if (
                        df is not None
                        and not df.empty
                    ):

                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True,
                        )

                        display_chart(
                            df,
                            chart,
                            chart_key=f"current_{len(st.session_state.messages)}_chart",
                        )

                # ------------------------------------------------
                # Evidence
                # ------------------------------------------------

                display_sources(
                    sources
                )

                # ------------------------------------------------
                # Save chat state
                # ------------------------------------------------

                df = result.get(
                    "data"
                )

                chart = result.get(
                    "chart"
                )

                st.session_state.messages.append({

                    "role":
                        "assistant",

                    "content":
                        answer,

                    "mode":
                        mode,

                    "sources":
                        sources,

                    "sections":
                        serialize_sections(
                            sections
                        ),

                    "data": (
                        df.to_dict(
                            "records"
                        )
                        if df is not None
                        else None
                    ),

                    "chart_data": (
                        df.to_dict(
                            "records"
                        )
                        if (
                            df is not None
                            and chart
                        )
                        else None
                    ),

                    "chart":
                        chart,
                })

            except Exception as e:

                st.error(
                    "CADvPDM GPT encountered "
                    "an error."
                )

                st.exception(
                    e
                )