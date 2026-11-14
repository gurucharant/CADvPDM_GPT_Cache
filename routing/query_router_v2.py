import re
from dataclasses import dataclass, field


# ======================================================================
# QUERY PLAN
# ======================================================================

@dataclass
class QueryPlan:

    route: str

    # ------------------------------------------------------------------
    # Analytics requirements
    # ------------------------------------------------------------------

    needs_count: bool = False
    needs_trend: bool = False
    needs_visualization: bool = False

    needs_people: bool = False
    needs_contact: bool = False

    needs_pareto: bool = False
    needs_grouping: bool = False

    # ------------------------------------------------------------------
    # Organizational-memory requirements
    # ------------------------------------------------------------------

    needs_semantic_history: bool = False
    needs_resolution: bool = False
    needs_root_cause: bool = False
    needs_knowledge: bool = False

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    reasons: list = field(
        default_factory=list
    )


# ======================================================================
# HELPERS
# ======================================================================

def _contains_any(
    text,
    terms,
):

    return any(
        term in text
        for term in terms
    )


def _matches_any(
    text,
    patterns,
):

    """
    Regex-based intent detection.

    Used where natural-language wording can vary substantially.

    Example:

        how were these issues typically resolved
        how was AUTH-401 usually fixed
        what fixes worked previously
    """

    return any(
        re.search(
            pattern,
            text,
            re.IGNORECASE,
        )
        is not None

        for pattern in patterns
    )


# ======================================================================
# MAIN ROUTER
# ======================================================================

def route_query(
    question: str
) -> QueryPlan:

    q = question.lower().strip()

    reasons = []

    # ==================================================================
    # ANALYTICS INTENT
    # ==================================================================

    count_terms = [
        "how many",
        "count",
        "counts",
        "number of",
        "total",
        "frequency",
    ]

    trend_terms = [
        "over time",
        "trend",
        "trends",
        "monthly",
        "per month",
        "weekly",
        "per week",
        "yearly",
        "per year",
        "line chart",
        "line graph",
    ]

    people_terms = [
        "who solved",
        "who resolved",
        "resolver",
        "resolvers",
        "resolved by",
        "solver",
        "solvers",
        "engineer",
        "engineers",
    ]

    contact_terms = [
        "contact",
        "contact details",
        "email",
        "phone",
        "telephone",
    ]

    grouping_terms = [
        "which site",
        "which sites",
        "top site",
        "top sites",
        "most affected site",
        "most affected sites",
        "most by site",
        "by site",
        "sites had the most",
        "site had the most",
        "top error",
        "top errors",
        "most common error",
        "most common errors",
        "ranking",
        "rank",
        "breakdown",
        "distribution",
    ]

    visualization_terms = [
        "chart",
        "graph",
        "visualize",
        "visualise",
        "visualisation",
        "visualization",
        "plot",
    ]

    pareto_terms = [
        "pareto",
        "80%",
        "80 percent",
    ]

    # ------------------------------------------------------------------
    # Detect analytics requirements
    # ------------------------------------------------------------------

    needs_count = _contains_any(
        q,
        count_terms
    )

    needs_trend = _contains_any(
        q,
        trend_terms
    )

    needs_people = _contains_any(
        q,
        people_terms
    )

    needs_contact = _contains_any(
        q,
        contact_terms
    )

    needs_grouping = _contains_any(
        q,
        grouping_terms
    )

    needs_visualization = _contains_any(
        q,
        visualization_terms
    )

    needs_pareto = _contains_any(
        q,
        pareto_terms
    )

    analytics_intent = any([
        needs_count,
        needs_trend,
        needs_people,
        needs_contact,
        needs_grouping,
        needs_visualization,
        needs_pareto,
    ])

    # ------------------------------------------------------------------
    # Analytics routing reasons
    # ------------------------------------------------------------------

    if needs_count:

        reasons.append(
            "count/statistical intent detected"
        )

    if needs_trend:

        reasons.append(
            "trend/time-series intent detected"
        )

    if needs_people:

        reasons.append(
            "resolver/people intent detected"
        )

    if needs_contact:

        reasons.append(
            "contact-information intent detected"
        )

    if needs_grouping:

        reasons.append(
            "grouping/ranking intent detected"
        )

    if needs_visualization:

        reasons.append(
            "visualization intent detected"
        )

    if needs_pareto:

        reasons.append(
            "Pareto intent detected"
        )

    # ==================================================================
    # HISTORICAL / ORGANIZATIONAL MEMORY INTENT
    # ==================================================================

    history_terms = [
        "have we seen",
        "have we experienced",
        "seen before",
        "experienced before",
        "previous incident",
        "previous incidents",
        "previous issue",
        "previous issues",
        "previously",
        "historical",
        "historically",
        "history of",
        "past incident",
        "past incidents",
        "past issue",
        "past issues",
        "similar incident",
        "similar incidents",
        "similar issue",
        "similar issues",
        "recurring incident",
        "recurring incidents",
        "recurring issue",
        "recurring issues",
        "what happened before",
        "in the past",
        "earlier incidents",
        "earlier cases",
    ]

    needs_semantic_history = (
        _contains_any(
            q,
            history_terms
        )
    )

    if needs_semantic_history:

        reasons.append(
            "historical organizational-memory intent detected"
        )

    # ==================================================================
    # RESOLUTION INTENT
    # ==================================================================

    resolution_terms = [

        # --------------------------------------------------------------
        # Historical resolution wording
        # --------------------------------------------------------------

        "resolved previously",
        "resolved before",

        "fixed previously",
        "fixed before",

        "previous resolution",
        "previous resolutions",

        "historical resolution",
        "historical resolutions",

        "past resolution",
        "past resolutions",

        # --------------------------------------------------------------
        # Direct questions
        # --------------------------------------------------------------

        "how did we fix",
        "how did we resolve",

        "how was it fixed",
        "how was it resolved",

        "how was this fixed",
        "how was this resolved",

        "how were they fixed",
        "how were they resolved",

        "how were these fixed",
        "how were these resolved",

        "how was the issue fixed",
        "how was the issue resolved",

        "how were the issues fixed",
        "how were the issues resolved",

        # --------------------------------------------------------------
        # Pattern-oriented resolution questions
        # --------------------------------------------------------------

        "typically resolved",
        "typically fixed",

        "usually resolved",
        "usually fixed",

        "commonly resolved",
        "commonly fixed",

        "normally resolved",
        "normally fixed",

        "most common resolution",
        "most common resolutions",

        "common resolution",
        "common resolutions",

        "resolution pattern",
        "resolution patterns",

        "fix pattern",
        "fix patterns",

        "typical resolution",
        "typical fix",

        # --------------------------------------------------------------
        # General remediation wording
        # --------------------------------------------------------------

        "remediation",
        "remediated",

        "workaround",
        "workarounds",
    ]

    # ------------------------------------------------------------------
    # Flexible resolution patterns
    #
    # These catch wording not covered by exact phrase matching.
    # ------------------------------------------------------------------

    resolution_patterns = [

        # how ... resolved
        r"\bhow\b.{0,80}\bresolved\b",

        # how ... fixed
        r"\bhow\b.{0,80}\bfixed\b",

        # how ... remediated
        r"\bhow\b.{0,80}\bremediated\b",

        # typically/usually/commonly ... resolved
        r"\b(?:typically|usually|commonly|normally)\b"
        r".{0,50}\bresolved\b",

        # typically/usually/commonly ... fixed
        r"\b(?:typically|usually|commonly|normally)\b"
        r".{0,50}\bfixed\b",

        # resolved ... typically/usually
        r"\bresolved\b"
        r".{0,50}\b(?:typically|usually|commonly|normally)\b",

        # fixed ... typically/usually
        r"\bfixed\b"
        r".{0,50}\b(?:typically|usually|commonly|normally)\b",
    ]

    needs_resolution = any([

        _contains_any(
            q,
            resolution_terms
        ),

        _matches_any(
            q,
            resolution_patterns
        ),
    ])

    if needs_resolution:

        reasons.append(
            "historical resolution intent detected"
        )

    # ==================================================================
    # ROOT-CAUSE INTENT
    # ==================================================================

    root_cause_terms = [

        "root cause",
        "root causes",

        "probable cause",
        "probable causes",

        "underlying cause",
        "underlying causes",

        "historical cause",
        "historical causes",

        "cause of the incident",
        "cause of this incident",

        "cause of the issue",
        "cause of this issue",

        "why did it fail",
        "why did this fail",

        "why did they fail",
        "why were they failing",

        "what caused it",
        "what caused this",

        "what caused the issue",
        "what caused these issues",

        "typically caused",
        "usually caused",
        "commonly caused",

        "cause pattern",
        "cause patterns",
    ]

    root_cause_patterns = [

        r"\bwhat\b.{0,60}\bcaused\b",

        r"\bwhy\b.{0,60}\b(?:fail|failed|failing)\b",

        r"\b(?:typically|usually|commonly|normally)\b"
        r".{0,50}\bcaused\b",
    ]

    needs_root_cause = any([

        _contains_any(
            q,
            root_cause_terms
        ),

        _matches_any(
            q,
            root_cause_patterns
        ),
    ])

    if needs_root_cause:

        reasons.append(
            "root-cause intent detected"
        )

    # ==================================================================
    # INTERNAL KNOWLEDGE INTENT
    # ==================================================================

    knowledge_terms = [
        "documented methodology",
        "documented procedure",
        "documented process",

        "troubleshooting guide",
        "troubleshooting procedure",

        "runbook",

        "confluence",

        "knowledge base",
        "kb article",

        "according to our documentation",
        "our documentation",

        "documented guidance",

        "procedure to investigate",
        "procedure for investigating",

        "methodology for investigating",

        "documented troubleshooting",
        "documented workaround",
    ]

    needs_knowledge = (
        _contains_any(
            q,
            knowledge_terms
        )
    )

    if needs_knowledge:

        reasons.append(
            "internal documented-knowledge intent detected"
        )

    # ==================================================================
    # ORGANIZATIONAL CONTEXT
    # ==================================================================

    organizational_terms = [
        "our incident",
        "our incidents",

        "our issue",
        "our issues",

        "our environment",

        "our system",
        "our systems",

        "our organization",
        "our organisation",

        "our company",

        "our support",

        "our jira",
        "our confluence",

        "our logs",

        "our ticket",
        "our tickets",

        "have we",
        "did we",

        "we experienced",
        "we encountered",

        "we resolved",
        "we fixed",
    ]

    has_organizational_context = (
        _contains_any(
            q,
            organizational_terms
        )
    )

    if has_organizational_context:

        reasons.append(
            "organizational context detected"
        )

    # ==================================================================
    # INTERNAL RECORD IDS
    # ==================================================================

    internal_id_patterns = [
        r"\bSUP-\d+\b",
        r"\bCADOPS-\d+\b",
        r"\bPDM-\d+\b",
        r"\bPLMOPS-\d+\b",
        r"\bBATCH-\d+\b",
        r"\bEXEC-\d+\b",
        r"\bKB-\d+\b",
        r"\bMTG-\d+\b",
        r"\bGIT-\d+\b",
    ]

    has_internal_id = any(

        re.search(
            pattern,
            question,
            re.IGNORECASE,
        )

        for pattern
        in internal_id_patterns
    )

    if has_internal_id:

        reasons.append(
            "explicit organizational record ID detected"
        )

    # ==================================================================
    # FINAL INTENT GROUPS
    # ==================================================================

    semantic_intent = any([
        needs_semantic_history,
        needs_resolution,
        needs_root_cause,
    ])

    organizational_memory_intent = any([
        semantic_intent,
        needs_knowledge,
        has_organizational_context,
        has_internal_id,
    ])

    # ==================================================================
    # FINAL ROUTE DECISION
    # ==================================================================

    # ------------------------------------------------------------------
    # MIXED
    #
    # Quantitative analytics
    # +
    # organizational memory / historical evidence
    #
    # Example:
    #
    # "How many AUTH-401 incidents occurred,
    #  which sites were most affected,
    #  and how were these issues typically resolved?"
    #
    # needs_count      = True
    # needs_grouping   = True
    # needs_resolution = True
    #
    # Therefore:
    #
    # MIXED
    # ------------------------------------------------------------------

    if (
        analytics_intent
        and organizational_memory_intent
    ):

        route = "MIXED"

        reasons.append(
            "analytical + organizational-memory "
            "intents require MIXED routing"
        )

    # ------------------------------------------------------------------
    # ANALYTICS
    # ------------------------------------------------------------------

    elif analytics_intent:

        route = "ANALYTICS"

    # ------------------------------------------------------------------
    # KNOWLEDGE
    # ------------------------------------------------------------------

    elif needs_knowledge:

        route = "KNOWLEDGE"

    # ------------------------------------------------------------------
    # SEMANTIC
    # ------------------------------------------------------------------

    elif (
        semantic_intent
        or has_organizational_context
        or has_internal_id
    ):

        route = "SEMANTIC"

    # ------------------------------------------------------------------
    # GENERAL
    # ------------------------------------------------------------------

    else:

        route = "GENERAL"

        reasons.append(
            "no organizational-memory or "
            "analytical intent detected"
        )

    # ==================================================================
    # RETURN PLAN
    # ==================================================================

    return QueryPlan(

        route=route,

        needs_count=needs_count,

        needs_trend=needs_trend,

        needs_visualization=(
            needs_visualization
        ),

        needs_people=needs_people,

        needs_contact=needs_contact,

        needs_pareto=needs_pareto,

        needs_grouping=needs_grouping,

        needs_semantic_history=(
            needs_semantic_history
        ),

        needs_resolution=(
            needs_resolution
        ),

        needs_root_cause=(
            needs_root_cause
        ),

        needs_knowledge=(
            needs_knowledge
        ),

        reasons=reasons,
    )


# ======================================================================
# STANDALONE ROUTER TEST
# ======================================================================

if __name__ == "__main__":

    test_questions = [

        # --------------------------------------------------------------
        # GENERAL
        # --------------------------------------------------------------

        "What is machine learning?",

        # --------------------------------------------------------------
        # ANALYTICS
        # --------------------------------------------------------------

        "How many AUTH-401 incidents occurred?",

        "Show issue counts over time for ENOVIA.",

        "Which error codes contribute to 80% of incidents?",

        "Which sites had the most AUTH-401 incidents?",

        # --------------------------------------------------------------
        # SEMANTIC
        # --------------------------------------------------------------

        (
            "Have we experienced this CAD Batch "
            "authentication error before?"
        ),

        (
            "How was this CAD Batch AUTH-401 "
            "issue resolved previously?"
        ),

        # --------------------------------------------------------------
        # KNOWLEDGE
        # --------------------------------------------------------------

        (
            "What is the documented procedure "
            "for investigating ENOVIA access problems?"
        ),

        # --------------------------------------------------------------
        # MIXED
        # --------------------------------------------------------------

        (
            "How many AUTH-401 incidents occurred, "
            "which sites were most affected, "
            "and how were these issues typically resolved?"
        ),

        (
            "How many CAD Batch AUTH-401 incidents occurred "
            "and what were their historical root causes?"
        ),

        (
            "Show the monthly trend of AUTH-401 incidents "
            "and explain how these issues were usually fixed."
        ),

        (
            "Which sites had the most AUTH-401 incidents "
            "and what typically caused them?"
        ),
    ]

    print("=" * 100)
    print("CADvPDM GPT - QUERY ROUTER V4 TEST")
    print("=" * 100)

    for index, question in enumerate(
        test_questions,
        start=1
    ):

        plan = route_query(
            question
        )

        print("\n")
        print("#" * 100)

        print(
            f"TEST #{index}"
        )

        print("#" * 100)

        print(
            f"Question : {question}"
        )

        print(
            f"Route    : {plan.route}"
        )

        print(
            f"Count    : {plan.needs_count}"
        )

        print(
            f"Trend    : {plan.needs_trend}"
        )

        print(
            f"Grouping : {plan.needs_grouping}"
        )

        print(
            f"History  : {plan.needs_semantic_history}"
        )

        print(
            f"Resolution: {plan.needs_resolution}"
        )

        print(
            f"Root Cause: {plan.needs_root_cause}"
        )

        print(
            f"Knowledge : {plan.needs_knowledge}"
        )

        print(
            "Reasons:"
        )

        for reason in plan.reasons:

            print(
                f"  - {reason}"
            )