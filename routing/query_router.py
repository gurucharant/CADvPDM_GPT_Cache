import re
from dataclasses import dataclass

@dataclass
class QueryPlan:
    mode: str
    needs_chart: bool = False
    chart_type: str | None = None
    prefer_sources: list[str] | None = None

class QueryRouter:
    """Lightweight deterministic router. Keeps the POC reliable without needing an LLM just to classify intent."""

    ANALYTIC_TERMS = (
        "how many", "count", "counts", "trend", "over time", "per month", "per week",
        "per year", "80%", "pareto", "top ", "most common", "frequency", "average",
        "mean", "median", "sla", "resolution time", "distribution", "percentage", "percent",
        "chart", "graph", "visualization", "visualisation", "line graph", "bar chart"
    )
    KNOWLEDGE_TERMS = (
        "method", "methodology", "procedure", "process", "runbook", "guide", "explain",
        "documentation", "confluence", "how does", "what is the procedure", "steps"
    )
    INCIDENT_TERMS = (
        "incident", "ticket", "issue", "when did", "who solved", "who resolved", "contact",
        "happened", "occurred", "history", "experienced before"
    )

    def route(self, question: str) -> QueryPlan:
        q = question.lower().strip()
        has_analytics = any(t in q for t in self.ANALYTIC_TERMS)
        has_knowledge = any(t in q for t in self.KNOWLEDGE_TERMS)
        has_incident = any(t in q for t in self.INCIDENT_TERMS)
        needs_chart = any(t in q for t in ("chart", "graph", "visual", "trend", "over time", "80%", "pareto"))
        chart_type = None
        if "80%" in q or "pareto" in q:
            chart_type = "pareto"
        elif "line" in q or "trend" in q or "over time" in q:
            chart_type = "line"
        elif needs_chart:
            chart_type = "bar"

        if has_analytics and has_knowledge:
            return QueryPlan("MIXED", needs_chart, chart_type, ["CONFLUENCE", "SUPPORT", "JIRA", "LOG"])
        if has_analytics:
            return QueryPlan("ANALYTICS", needs_chart, chart_type, None)
        if has_knowledge and not has_incident:
            return QueryPlan("KNOWLEDGE", False, None, ["CONFLUENCE"])
        if has_incident and ("who solved" in q or "contact" in q or "when did" in q or "how many" in q):
            return QueryPlan("ANALYTICS", needs_chart, chart_type, None)
        return QueryPlan("SEMANTIC", False, None, ["SUPPORT", "JIRA", "CONFLUENCE", "LOG", "TEAMS", "GIT"])
