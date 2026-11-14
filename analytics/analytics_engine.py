import re
import pandas as pd

from iceberg.iceberg_loader import load_all


class AnalyticsEngine:
    """
    POC analytical memory engine over structured source data.

    Production:
        Point the same logical datasets at Iceberg tables through
        Spark / Trino / Athena.

    POC:
        Pandas keeps the setup simple while demonstrating the
        final organizational-intelligence behaviour.
    """

    def __init__(self):

        print("=" * 90)
        print("Initializing AnalyticsEngine from Apache Iceberg")
        print("=" * 90)

        # ==========================================================
        # Load all structured analytical datasets from Iceberg
        # ==========================================================

        datasets = load_all()

        self.support = datasets["support"]
        self.jira = datasets["jira"]
        self.logs = datasets["logs"]
        self.engineers = datasets["engineers"]
        self.git = datasets["git"]
        self.teams = datasets["teams"]

        self.confluence = datasets["confluence"]

        # ==========================================================
        # Convert string timestamp columns from Iceberg to datetime
        # ==========================================================

        date_columns = [
            (
                self.support,
                [
                    "opened_ts",
                    "resolved_ts",
                ],
            ),
            (
                self.jira,
                [
                    "created_ts",
                    "resolved_ts",
                ],
            ),
            (
                self.logs,
                [
                    "start_ts",
                    "end_ts",
                ],
            ),
            (
                self.git,
                [
                    "timestamp",
                ],
            ),
            (
                self.teams,
                [
                    "meeting_ts",
                ],
            ),
        ]

        for df, cols in date_columns:

            for col in cols:

                if col in df.columns:

                    df[col] = pd.to_datetime(
                        df[col],
                        errors="coerce",
                    )

        # ==========================================================
        # Startup confirmation
        # ==========================================================

        print(
            f"Support     : {len(self.support):,} rows"
        )

        print(
            f"Jira        : {len(self.jira):,} rows"
        )

        print(
            f"Logs        : {len(self.logs):,} rows"
        )

        print(
            f"Confluence  : {len(self.confluence):,} rows"
        )

        print(
            f"Git         : {len(self.git):,} rows"
        )

        print(
            f"Teams       : {len(self.teams):,} rows"
        )

        print(
            f"Engineers   : {len(self.engineers):,} rows"
        )

        print(
            "\nAnalytics source: Apache Iceberg"
        )

    # ==============================================================
    # Filtering
    # ==============================================================

    def _filters(self, question, df):
        out = df.copy()
        ql = question.lower()

        filter_columns = [
            "application",
            "site",
            "country",
            "error_code",
            "priority",
            "severity",
            "assignment_group",
            "assigned_engineer",
            "assignee",
            "problem_category",
            "problem_subcategory",
            "status",
        ]

        for col in filter_columns:
            if col not in out.columns:
                continue

            values = [
                str(v)
                for v in out[col].dropna().unique()
                if str(v).strip()
            ]

            matches = [
                v
                for v in values
                if v.lower() in ql
            ]

            if matches:
                out = out[
                    out[col].astype(str).isin(matches)
                ]

        id_patterns = [
            ("ticket_id", r"SUP-\d+"),
            (
                "issue_id",
                r"(?:CADOPS|PDM|PLMOPS)-\d+"
            ),
            ("execution_id", r"EXEC-\d+"),
        ]

        for col, pattern in id_patterns:
            if col not in out.columns:
                continue

            match = re.search(
                pattern,
                question,
                re.IGNORECASE
            )

            if match:
                out = out[
                    out[col]
                    .astype(str)
                    .str.upper()
                    == match.group(0).upper()
                ]

        return out

    # ==============================================================
    # Unified incident model
    # ==============================================================

    def _unified_incidents(self):
        support = self.support.copy()

        support_incidents = pd.DataFrame({
            "record_type": "SUPPORT",
            "id": support.ticket_id,
            "event_ts": support.opened_ts,
            "resolved_ts": support.resolved_ts,
            "application": support.application,
            "site": support.site,
            "country": support.country,
            "error_code": support.error_code,
            "issue": support.short_description,
            "resolver": support.assigned_engineer,
            "resolution": support.resolution_summary,
            "resolution_minutes": support.resolution_minutes,
            "problem_category": support.problem_category,
            "priority": support.priority,
            "severity": support.severity,
            "assignment_group": support.assignment_group,
        })

        jira = self.jira.copy()

        jira_incidents = pd.DataFrame({
            "record_type": "JIRA",
            "id": jira.issue_id,
            "event_ts": jira.created_ts,
            "resolved_ts": jira.resolved_ts,
            "application": jira.application,
            "site": jira.site,
            "country": jira.country,
            "error_code": jira.error_code,
            "issue": jira.summary,
            "resolver": jira.assignee,
            "resolution": jira.resolution,
            "resolution_minutes": jira.resolution_minutes,
            "problem_category": jira.component,
            "priority": jira.priority,
            "severity": "",
            "assignment_group": jira.reporter_team,
        })

        return pd.concat(
            [
                support_incidents,
                jira_incidents
            ],
            ignore_index=True,
        )

    # ==============================================================
    # Contact enrichment
    # ==============================================================

    def _contact_enrich(self, df):
        if "resolver" not in df.columns:
            return df

        return df.merge(
            self.engineers,
            how="left",
            left_on="resolver",
            right_on="engineer_name",
        )

    # ==============================================================
    # Resolution normalization
    # ==============================================================

    @staticmethod
    def _normalize_resolution_text(text):

        if pd.isna(text):
            return ""

        text = str(text).strip()

        if not text:
            return ""

        text = re.sub(
            r"\s*Related engineering record\s+"
            r"[A-Z0-9_-]+\s+was reviewed\.?",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if text and text[-1] not in ".!?":
            text += "."

        return text

    # ==============================================================
    # Top counts
    # ==============================================================

    @staticmethod
    def _top_counts(df, column, n=10):
        if column not in df.columns or df.empty:
            return pd.DataFrame()

        return (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .rename_axis(column)
            .reset_index(name="count")
            .head(n)
        )

    # ==============================================================
    # Format table for summary
    # ==============================================================

    @staticmethod
    def _table_text(df, rows=10):
        if df is None or df.empty:
            return "No matching data."

        return df.head(rows).to_string(
            index=False
        )

    # ==============================================================
    # Main analytics entry point
    # ==============================================================

    def answer(self, question: str):
        q = question.lower()

        incidents = self._filters(
            question,
            self._unified_incidents()
        )

        result = {
            "answer": "",
            "data": None,
            "chart": None,
            "sources": [],
            "query_mode": "ANALYTICS",
            "sections": {},
        }

        total = len(incidents)

        wants_count = any(
            x in q
            for x in [
                "how many",
                "count",
                "counts",
                "number of",
                "total",
                "frequency",
            ]
        )

        wants_site = any(
            x in q
            for x in [
                "which site",
                "which sites",
                "top site",
                "top sites",
                "most by site",
                "by site",
                "site had the most",
                "sites had the most",
            ]
        )

        wants_resolver = any(
            x in q
            for x in [
                "who solved",
                "who resolved",
                "resolver",
                "resolved by",
                "solver",
                "engineer",
            ]
        )

        wants_contact = any(
            x in q
            for x in [
                "contact",
                "email",
                "phone",
                "telephone",
            ]
        )

        wants_resolution = any(
            x in q
            for x in [
                "resolution",
                "resolutions",
                "resolved",
                "fixed",
                "fix",
                "remediation",
            ]
        )

        wants_trend = any(
            x in q
            for x in [
                "over time",
                "trend",
                "line graph",
                "line chart",
                "per month",
                "monthly",
                "per week",
                "weekly",
                "per year",
                "yearly",
            ]
        )

        wants_pareto = (
            "80%" in q
            or "80 percent" in q
            or "pareto" in q
        )

        wants_sla = any(
            x in q
            for x in [
                "sla",
                "resolution time",
                "average",
                "mean",
                "median",
                "p90",
            ]
        )

        result["sections"]["total_incidents"] = total

        top_sites = self._top_counts(
            incidents,
            "site",
            n=15,
        )

        if wants_site:
            result["sections"]["top_sites"] = top_sites

        top_resolvers = self._top_counts(
            incidents,
            "resolver",
            n=15,
        )

        if wants_resolver or wants_contact:

            result["sections"][
                "top_resolvers"
            ] = top_resolvers

            if not top_resolvers.empty:

                resolver_details = (
                    top_resolvers
                    .merge(
                        self.engineers,
                        how="left",
                        left_on="resolver",
                        right_on="engineer_name",
                    )
                )

                keep_cols = [
                    c
                    for c in [
                        "resolver",
                        "count",
                        "team",
                        "email",
                        "phone",
                    ]
                    if c in resolver_details.columns
                ]

                result["sections"][
                    "resolver_contacts"
                ] = resolver_details[
                    keep_cols
                ]

        if wants_resolution:

            resolution_data = (
                incidents[
                    incidents["resolution"].notna()
                ][["resolution"]]
                .copy()
            )

            resolution_data[
                "normalized_resolution"
            ] = resolution_data[
                "resolution"
            ].apply(
                self._normalize_resolution_text
            )

            resolution_data = (
                resolution_data[
                    resolution_data[
                        "normalized_resolution"
                    ] != ""
                ]
            )

            resolution_counts = (
                resolution_data[
                    "normalized_resolution"
                ]
                .value_counts()
                .rename_axis(
                    "resolution_pattern"
                )
                .reset_index(
                    name="count"
                )
            )

            total_resolved = (
                resolution_counts[
                    "count"
                ].sum()
            )

            if total_resolved > 0:

                resolution_counts[
                    "percentage"
                ] = (
                    100
                    * resolution_counts["count"]
                    / total_resolved
                ).round(1)

            else:

                resolution_counts[
                    "percentage"
                ] = 0.0

            resolution_counts = (
                resolution_counts.head(10)
            )

            result["sections"][
                "common_resolutions"
            ] = resolution_counts

        if wants_pareto:

            dimension = "error_code"

            candidates = [
                (
                    "problem_category",
                    [
                        "category",
                        "issue type",
                    ],
                ),
                (
                    "application",
                    [
                        "application",
                    ],
                ),
                (
                    "site",
                    [
                        "site",
                    ],
                ),
                (
                    "resolver",
                    [
                        "solver",
                        "engineer",
                        "resolver",
                    ],
                ),
                (
                    "error_code",
                    [
                        "error",
                        "error code",
                        "incident",
                    ],
                ),
            ]

            for candidate, terms in candidates:

                if any(
                    term in q
                    for term in terms
                ):

                    dimension = candidate
                    break

            counts = (
                incidents[dimension]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .rename_axis(
                    dimension
                )
                .reset_index(
                    name="count"
                )
            )

            counts[
                "cumulative_pct"
            ] = (
                100
                * counts["count"].cumsum()
                / max(
                    counts["count"].sum(),
                    1
                )
            )

            counts[
                "within_80_pct"
            ] = (
                counts[
                    "cumulative_pct"
                ] <= 80
            )

            if (
                not counts.empty
                and not counts[
                    "within_80_pct"
                ].any()
            ):

                counts.loc[
                    0,
                    "within_80_pct"
                ] = True

            result["sections"][
                "pareto"
            ] = counts

            result["data"] = counts

            result["chart"] = {
                "type": "pareto",
                "x": dimension,
                "y": "count",
                "secondary":
                    "cumulative_pct",
                "title": (
                    f"Pareto of incidents by "
                    f"{dimension.replace('_', ' ')}"
                ),
            }

        if wants_trend:

            freq = "M"
            label = "month"

            if "week" in q:
                freq = "W"
                label = "week"

            elif "year" in q:
                freq = "Y"
                label = "year"

            trend_source = (
                incidents
                .dropna(
                    subset=[
                        "event_ts"
                    ]
                )
                .copy()
            )

            trend_source[
                "period"
            ] = (
                trend_source[
                    "event_ts"
                ]
                .dt.to_period(
                    freq
                )
                .dt.to_timestamp()
            )

            trend = (
                trend_source
                .groupby(
                    "period"
                )
                .size()
                .reset_index(
                    name="issue_count"
                )
                .sort_values(
                    "period"
                )
            )

            result["sections"][
                "trend"
            ] = trend

            if result["data"] is None:

                result["data"] = trend

                result["chart"] = {
                    "type": "line",
                    "x": "period",
                    "y": "issue_count",
                    "title":
                        f"Issue counts by {label}",
                }

        if wants_sla:

            d = incidents.dropna(
                subset=[
                    "resolution_minutes"
                ]
            )

            if not d.empty:

                stats = pd.DataFrame([
                    {
                        "incident_count":
                            len(d),

                        "average_resolution_hours":
                            round(
                                d[
                                    "resolution_minutes"
                                ].mean()
                                / 60,
                                2,
                            ),

                        "median_resolution_hours":
                            round(
                                d[
                                    "resolution_minutes"
                                ].median()
                                / 60,
                                2,
                            ),

                        "p90_resolution_hours":
                            round(
                                d[
                                    "resolution_minutes"
                                ].quantile(
                                    0.9
                                )
                                / 60,
                                2,
                            ),
                    }
                ])

                result["sections"][
                    "sla"
                ] = stats

                if result["data"] is None:
                    result["data"] = stats

        wants_details = any(
            x in q
            for x in [
                "when did",
                "when was",
                "incident details",
                "show incidents",
                "list incidents",
                "recent incidents",
            ]
        )

        if wants_details:

            detail = self._contact_enrich(
                incidents
                .sort_values(
                    "event_ts",
                    ascending=False,
                )
                .head(25)
            )

            detail_cols = [
                c
                for c in [
                    "record_type",
                    "id",
                    "event_ts",
                    "resolved_ts",
                    "application",
                    "site",
                    "error_code",
                    "resolver",
                    "team",
                    "email",
                    "phone",
                    "resolution",
                ]
                if c in detail.columns
            ]

            detail = detail[
                detail_cols
            ]

            result["sections"][
                "incident_details"
            ] = detail

            if result["data"] is None:
                result["data"] = detail

        if result["data"] is None:

            if (
                wants_site
                and not top_sites.empty
            ):

                result["data"] = (
                    top_sites
                )

                result["chart"] = {
                    "type": "bar",
                    "x": "site",
                    "y": "count",
                    "title":
                        "Incident count by site",
                }

            elif (
                wants_resolver
                and not top_resolvers.empty
            ):

                result["data"] = (
                    result[
                        "sections"
                    ].get(
                        "resolver_contacts",
                        top_resolvers,
                    )
                )

                result["chart"] = {
                    "type": "bar",
                    "x": "resolver",
                    "y": "count",
                    "title":
                        "Incident count by resolver",
                }

        summary_parts = [
            "STRUCTURED ANALYTICS RESULTS",
            f"Total matching incidents: "
            f"{total:,}",
        ]

        if (
            wants_site
            and not top_sites.empty
        ):

            summary_parts.append(
                "\nTop affected sites:\n"
                + self._table_text(
                    top_sites,
                    rows=10,
                )
            )

        if (
            wants_resolver
            or wants_contact
        ):

            resolver_contacts = (
                result[
                    "sections"
                ].get(
                    "resolver_contacts"
                )
            )

            if (
                resolver_contacts
                is not None
                and not resolver_contacts.empty
            ):

                summary_parts.append(
                    "\nTop resolvers / contacts:\n"
                    + self._table_text(
                        resolver_contacts,
                        rows=10,
                    )
                )

        if wants_resolution:

            common_resolutions = (
                result[
                    "sections"
                ].get(
                    "common_resolutions"
                )
            )

            if (
                common_resolutions
                is not None
                and not common_resolutions.empty
            ):

                summary_parts.append(
                    "\nMost common normalized "
                    "resolution patterns:\n"
                    + self._table_text(
                        common_resolutions,
                        rows=10,
                    )
                )

        if wants_trend:

            trend = (
                result[
                    "sections"
                ].get(
                    "trend"
                )
            )

            if (
                trend is not None
                and not trend.empty
            ):

                summary_parts.append(
                    "\nTrend data:\n"
                    + self._table_text(
                        trend,
                        rows=24,
                    )
                )

        if wants_sla:

            sla = (
                result[
                    "sections"
                ].get(
                    "sla"
                )
            )

            if (
                sla is not None
                and not sla.empty
            ):

                summary_parts.append(
                    "\nResolution-time statistics:\n"
                    + self._table_text(
                        sla
                    )
                )

        result["answer"] = (
            "\n".join(
                summary_parts
            )
        )

        return result