"""
CADvPDM GPT
Prompt Builder V3

Purpose:
    Build small, focused prompts for the local LLM.

Design principle:
    Do NOT send one giant universal prompt to llama3.2:3b.

    Instead:
        SEMANTIC  -> historical organizational evidence
        KNOWLEDGE -> documented procedure / methodology
        MIXED     -> analytics + organizational evidence
        GENERAL   -> normally handled directly by rag_pipeline.py

The prompt remains strictly grounded in supplied organizational evidence.
"""


def build_rag_prompt(
    question,
    context,
    mode="SEMANTIC",
    analytics_summary="",
):
    """
    Build a focused prompt based on the query route.

    Parameters
    ----------
    question : str
        Original user question.

    context : str
        Retrieved organizational evidence.

    mode : str
        GENERAL / ANALYTICS / SEMANTIC / KNOWLEDGE / MIXED

    analytics_summary : str
        Structured analytical result when available.

    Returns
    -------
    str
        Prompt for the local LLM.
    """

    mode = str(mode or "SEMANTIC").upper().strip()

    question = str(question or "").strip()

    context = str(
        context or "No organizational evidence supplied."
    ).strip()

    analytics_summary = str(
        analytics_summary
        or "No structured analytics supplied."
    ).strip()

    # ==========================================================
    # SEMANTIC
    # ==========================================================

    if mode == "SEMANTIC":
        return _build_semantic_prompt(
            question=question,
            context=context,
        )

    # ==========================================================
    # KNOWLEDGE
    # ==========================================================

    if mode == "KNOWLEDGE":
        return _build_knowledge_prompt(
            question=question,
            context=context,
        )

    # ==========================================================
    # MIXED
    # ==========================================================

    if mode == "MIXED":
        return _build_mixed_prompt(
            question=question,
            context=context,
            analytics_summary=analytics_summary,
        )

    # ==========================================================
    # ANALYTICS
    # ==========================================================

    if mode == "ANALYTICS":
        return _build_analytics_prompt(
            question=question,
            analytics_summary=analytics_summary,
        )

    # ==========================================================
    # FALLBACK
    # ==========================================================

    return _build_fallback_prompt(
        question=question,
        context=context,
    )


# ==================================================================
# SEMANTIC PROMPT
# ==================================================================

def _build_semantic_prompt(
    question,
    context,
):
    """
    Historical organizational-memory questions.

    Examples:
        - Have we experienced this before?
        - What caused this historically?
        - How was this resolved previously?
        - What fixes worked previously?
    """

    question_lower = question.lower()

    # ----------------------------------------------------------
    # Detect historical-resolution intent
    # ----------------------------------------------------------

    resolution_terms = [
        "resolved",
        "resolution",
        "fixed",
        "fix",
        "remediated",
        "remediation",
        "solved",
        "solution",
        "worked previously",
        "how was",
    ]

    resolution_intent = any(
        term in question_lower
        for term in resolution_terms
    )

    # ----------------------------------------------------------
    # Detect root-cause intent
    # ----------------------------------------------------------

    root_cause_terms = [
        "root cause",
        "root-cause",
        "caused",
        "cause",
        "why did",
        "why was",
        "reason",
    ]

    root_cause_intent = any(
        term in question_lower
        for term in root_cause_terms
    )

    # ----------------------------------------------------------
    # Detect "have we seen this before?"
    # ----------------------------------------------------------

    history_terms = [
        "experienced this before",
        "seen this before",
        "happened before",
        "occurred before",
        "previously",
        "historically",
        "similar incidents",
        "past incidents",
    ]

    history_intent = any(
        term in question_lower
        for term in history_terms
    )

    # ==========================================================
    # Historical resolution
    # ==========================================================

    if resolution_intent:

        return f"""
You are CADvPDM GPT.

The user is asking how an organizational issue was resolved
historically.

Use ONLY the organizational evidence below.

Do not use generic IT knowledge.
Do not invent facts.
Do not invent recommendations.
Do not suggest future improvements.
Do not suggest monitoring.
Do not suggest contacting support.
Do not add prevention advice.
Do not add "Additional Information".
Do not add "Related Information".
Do not say "Here is a synthesized answer".
Do not expose retrieval scores.

Answer the question directly.

Use this structure:

### Historical Resolution

State the recurring historical resolution clearly and concisely.

### Historical Root Cause

Include this section ONLY if the evidence clearly establishes the
historical root cause and it helps explain the resolution.

### Supporting Evidence

List only the most relevant organizational source IDs.

Example format:

- SUPPORT: SUP-xxxxx
- JIRA: CADOPS-xxxxx
- CONFLUENCE: KB-xxxxx

Do not list every retrieved document.
Prefer 2-4 strong supporting records.

If several records repeat the same resolution, synthesize the pattern
once instead of repeating each record.

If the evidence does not establish the answer, say so explicitly.


USER QUESTION
-------------
{question}


ORGANIZATIONAL EVIDENCE
-----------------------
{context}


FINAL INSTRUCTION
-----------------
Answer only what the user asked.

Describe what WAS DONE historically.

Do not convert historical evidence into advice about what SHOULD
be done in the future.
""".strip()

    # ==========================================================
    # Root-cause question
    # ==========================================================

    if root_cause_intent:

        return f"""
You are CADvPDM GPT.

The user is asking about the historical root cause of an
organizational issue.

Use ONLY the organizational evidence below.

Do not use generic IT knowledge.
Do not invent causes.
Do not invent recommendations.
Do not suggest future improvements.
Do not expose retrieval scores.

Use this structure:

### Historical Root-Cause Pattern

State the recurring historical root cause directly.

### Historical Resolution

Include this only when the historical resolution helps explain
the root cause.

### Supporting Evidence

List only the strongest supporting source IDs.

Prefer evidence from:
1. Support
2. Jira
3. Confluence
4. Logs

Do not repeat near-identical records.


USER QUESTION
-------------
{question}


ORGANIZATIONAL EVIDENCE
-----------------------
{context}


FINAL INSTRUCTION
-----------------
Answer concisely from historical evidence only.
""".strip()

    # ==========================================================
    # "Have we experienced this before?"
    # ==========================================================

    if history_intent:

        return f"""
You are CADvPDM GPT.

The user wants to know whether this issue has occurred previously.

Use ONLY the organizational evidence below.

If previous occurrences are clearly present, begin with:

"Yes. Historical organizational records show previous occurrences."

Then briefly summarize:

- what happened
- recurring application/error pattern
- recurring root cause, if established
- recurring historical resolution, if established

Finish with:

### Supporting Evidence

List only the strongest source IDs.

Do not provide future recommendations.
Do not provide prevention advice.
Do not suggest contacting support.
Do not expose retrieval scores.


USER QUESTION
-------------
{question}


ORGANIZATIONAL EVIDENCE
-----------------------
{context}


FINAL INSTRUCTION
-----------------
Answer the question directly and concisely.
""".strip()

    # ==========================================================
    # Generic semantic historical question
    # ==========================================================

    return f"""
You are CADvPDM GPT.

Answer the user's question using ONLY the historical organizational
evidence supplied below.

Focus on what the organization historically observed, investigated,
decided, or did.

Do not use generic IT knowledge.
Do not invent missing facts.
Do not provide recommendations unless explicitly requested.
Do not expose retrieval scores.
Do not merely summarize every document.

Synthesize recurring evidence into one concise answer.

Mention the strongest supporting source IDs.


USER QUESTION
-------------
{question}


ORGANIZATIONAL EVIDENCE
-----------------------
{context}


FINAL INSTRUCTION
-----------------
Answer only what the user asked.
""".strip()


# ==================================================================
# KNOWLEDGE PROMPT
# ==================================================================

def _build_knowledge_prompt(
    question,
    context,
):
    """
    Documented organizational knowledge.

    Examples:
        - What is the documented procedure?
        - What methodology should be followed?
        - What does the runbook say?
        - How should this be investigated according to Confluence?
    """

    return f"""
You are CADvPDM GPT.

The user is asking for documented organizational knowledge.

The supplied evidence comes primarily from internal knowledge
documents such as Confluence.

Use ONLY the documented evidence below.

Your task is to explain the documented procedure faithfully.

RULES

1. Preserve the documented sequence when one exists.

2. Do not add undocumented troubleshooting steps.

3. Do not turn this into a historical incident report.

4. Do not add generic IT recommendations.

5. Do not add:
   - Historical Resolution
   - Historical Root Cause
   - Additional Information
   - Related Information
   unless the user's question specifically asks for them.

6. Do not suggest contacting support unless the supplied
   documentation explicitly makes that part of the procedure.

7. Do not expose similarity, BM25, RRF or CrossEncoder scores.

8. Prefer Confluence evidence over incident records.

9. If several Confluence documents describe the same procedure,
   synthesize the common documented procedure once.

10. If documents disagree, briefly mention the disagreement.

11. If the retrieved evidence contains only part of the procedure,
    explicitly say that only a partial procedure was retrieved.


RESPONSE FORMAT

### Documented Procedure

Provide the documented steps as a short numbered sequence.

### Sources

List the relevant Confluence page IDs.

Do not add unnecessary sections.


USER QUESTION
-------------
{question}


DOCUMENTED ORGANIZATIONAL EVIDENCE
----------------------------------
{context}


FINAL INSTRUCTION
-----------------
Explain the documented procedure directly.

Stay faithful to the supplied documentation.
Do not invent additional steps.
""".strip()


# ==================================================================
# MIXED PROMPT
# ==================================================================

def _build_mixed_prompt(
    question,
    context,
    analytics_summary,
):
    """
    Questions requiring both structured analytics and
    organizational evidence.
    """

    return f"""
You are CADvPDM GPT.

The user asked a question requiring both:

1. structured organizational analytics
2. historical organizational evidence

Use the STRUCTURED ANALYTICS for:

- counts
- frequencies
- rankings
- top sites
- percentages
- trends
- resolver statistics

Use ORGANIZATIONAL EVIDENCE for:

- historical root causes
- historical resolutions
- procedures
- historical engineering knowledge

Never calculate organizational totals from retrieved evidence.

Never allow retrieved evidence to override structured analytics.

Do not invent recommendations.
Do not add future actions unless explicitly requested.
Do not expose retrieval scores.

Answer every material part of the user's question.

Use only the sections that are actually relevant.

Possible structure:

### Executive Summary

### Structured Findings

### Historical Root-Cause Pattern

### Historical Resolution Pattern

### Documented Operational Guidance

Do NOT mechanically include every heading.

For supporting historical claims, mention only the strongest
organizational source IDs.


USER QUESTION
-------------
{question}


STRUCTURED ANALYTICS
--------------------
{analytics_summary}


ORGANIZATIONAL EVIDENCE
-----------------------
{context}


FINAL INSTRUCTION
-----------------
Combine the analytical findings and historical evidence into one
concise answer.

Counts and rankings must come from structured analytics.

Root causes and resolutions must come from organizational evidence.

Do not add recommendations unless the user explicitly asks for them.
""".strip()


# ==================================================================
# ANALYTICS PROMPT
# ==================================================================

def _build_analytics_prompt(
    question,
    analytics_summary,
):
    """
    Structured analytical questions.
    """

    return f"""
You are CADvPDM GPT.

Answer the user's analytical question using ONLY the structured
analytics supplied below.

The structured analytics is authoritative for:

- counts
- frequencies
- rankings
- percentages
- trends
- site statistics
- resolver statistics
- SLA statistics
- resolution-time statistics
- Pareto results

Do not invent missing numbers.
Do not infer counts from retrieved documents.
Do not provide troubleshooting recommendations unless requested.

Lead with the numerical answer.

Then briefly explain the result.


USER QUESTION
-------------
{question}


STRUCTURED ANALYTICS
--------------------
{analytics_summary}


FINAL INSTRUCTION
-----------------
Answer concisely using the supplied analytics only.
""".strip()


# ==================================================================
# FALLBACK PROMPT
# ==================================================================

def _build_fallback_prompt(
    question,
    context,
):
    """
    Safe fallback for an unexpected route.
    """

    return f"""
You are CADvPDM GPT.

Answer the user's question using ONLY the organizational evidence
supplied below.

Do not invent company-specific information.
Do not add generic recommendations.
Do not expose retrieval implementation details.

If the evidence does not answer the question, say:

"The retrieved organizational evidence does not establish the
answer to this question."


USER QUESTION
-------------
{question}


ORGANIZATIONAL EVIDENCE
-----------------------
{context}


FINAL INSTRUCTION
-----------------
Give a direct and concise answer.
""".strip()


# ==================================================================
# Standalone prompt-builder test
# ==================================================================

if __name__ == "__main__":

    sample_context = """
SOURCE #1
Source Type: SUPPORT
Source ID: SUP-510289
Application: CAD Batch
Error Code: AUTH-401

Evidence:
Root Cause:
Service account credential expired or was rotated without updating
the batch runtime.

Resolution:
Renew the credential, update the secure credential store,
validate login, and re-run the failed batch.


SOURCE #2
Source Type: JIRA
Source ID: CADOPS-24270
Application: CAD Batch
Error Code: AUTH-401

Evidence:
Root Cause:
Service account credential expired or was rotated without updating
the batch runtime.

Resolution:
Renew the credential, update the secure credential store,
validate login, and re-run the failed batch.
""".strip()

    sample_question = (
        "How was the CAD Batch AUTH-401 issue "
        "resolved previously?"
    )

    prompt = build_rag_prompt(
        question=sample_question,
        context=sample_context,
        mode="SEMANTIC",
    )

    print("=" * 90)
    print("PROMPT BUILDER V3 TEST")
    print("=" * 90)
    print()
    print(prompt)