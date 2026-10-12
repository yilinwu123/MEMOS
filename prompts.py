"""All prompts used in the pipeline (A: gold bullet splitting, B: summarization, D: Stage-2 judge)."""


# ============================================================
# A2. Gold answer -> atomic gold bullets
# ============================================================

GOLD_SPLIT_SYSTEM = """You are an expert in information extraction.

Your task is to convert a GOLD ANSWER into atomic bullet points.

STRICT RULES:
- Decompose the GOLD ANSWER into atomic but complete bullet points.
- Each bullet must express one distinct factual proposition supported by the DOCUMENT and relevant to the QUERY.
- Preserve ALL information in the GOLD ANSWER. Do not omit any content.
- Do NOT invent, infer, generalize, or add information not present in the GOLD ANSWER or DOCUMENT.
- You must preserve the original wording and phrasing of the GOLD ANSWER. Do not paraphrase, add new words, rephrase, or substitute expressions unless necessary to split bullets.
- Each bullet should contain as much essential information as possible while remaining a single proposition (minimal complete unit, not fragmented pieces).
- Minimize overlap between bullets. Avoid repeating the same information across bullets.
- If two details are naturally part of the same proposition (e.g. cause + conclusion, proposal + objection), keep them in ONE bullet rather than splitting them.
- Completeness is more important than minimizing overlap, but avoid redundancy whenever possible.
- Output only bullet points.

OUTPUT FORMAT (STRICT JSON):
{
  "bullets": [
    "bullet 1",
    "bullet 2"
  ]
}

Do NOT output anything else.
"""


# The single-doc run used the same instructions without blank lines.
GOLD_SPLIT_SYSTEM_SINGLEDOC = "\n".join(l for l in GOLD_SPLIT_SYSTEM.split("\n") if l) + "\n"


def gold_split_user_prompt_multidoc(query: str, document: str, gold: str) -> str:
    return f"""
QUERY:
{query}

DOCUMENT:
{document}

GOLD ANSWER:
{gold}

TASK:
Convert the GOLD ANSWER into atomic bullet points grounded in the DOCUMENT and relevant to the QUERY.
"""


def gold_split_user_prompt_singledoc(query: str, document: str, gold: str) -> str:
    return f"""
QUERY:
{query}
DOCUMENT:
{document}
GOLD ANSWER:
{gold}
TASK:
Convert the GOLD ANSWER into atomic bullet points grounded in the DOCUMENT and relevant to the QUERY.
"""


# ============================================================
# B. Query-focused summarization (same prompt for all models)
# ============================================================

SUMMARIZE_SYSTEM_MULTIDOC = "You are a helpful assistant for query-based multi-document summarization."
SUMMARIZE_SYSTEM_SINGLEDOC = "You are a helpful assistant for query-based meeting summarization."

def summarize_prompt_multidoc(merged_query: str, context_text: str) -> str:
    return f"""
You are a professional multi-document meeting analyst.

Your task is to answer every single part and aspect contained in the Query strictly based on the provided multi-document meeting context. You must ensure no part of the query is overlooked or left unanswered.

The goal is to produce complete bullet points in the style of factual proposition extraction,
not abstract insights or high-level takeaways.

The context contains multiple documents, and each document is identified by a document ID.

You must read each document carefully, address every element in the query, and provide corresponding answers respectively.

Rules:

1. Use ONLY information explicitly stated in the documents.

2. Do NOT add assumptions, outside knowledge,
unsupported inferences, interpretations, or synthesized implications.

3. Include ONLY information relevant to the Query.

4. For each bullet, include an evidence_quote from source documents that directly supports that bullet.

Include the supporting document ID(s) in doc_ids.

If multiple documents support the same point, include all relevant document IDs.

5. Only include core meeting matters (e.g. major discussions, decisions, proposals, or preferences) related to answer the Query.
Exclude detailed technical explanations, implementation methods, and background discussions unless they directly correspond to the Query. 

6. Each bullet should contain only the essential information needed to describe the matter.
Remove secondary details, examples, and elaborations.

7. Do NOT merge multiple related discussion matters into one abstract bullet.
If the meeting contains separate discussions, decisions, proposals, or statements, write separate bullets.

8. If no relevant information is found, return exactly:
{{"bullets": []}}

9. Output MUST be valid JSON only.

10. Do NOT include explanations, notes, or markdown code fences.

Output schema:

{{
  "bullets": [
   {{
      "point": "...",
      "evidence_quote": "...",
      "doc_ids": ["..."]
    }}
  ]
}}

Query:
{merged_query}

Context:
{context_text}
""".strip()


def summarize_prompt_singledoc(query: str, transcript_text: str, doc_id: str) -> str:
    return f"""
You are a professional meeting analyst.

Your task is to answer every single part and aspect contained in the Query strictly based on the provided meeting transcript. You must ensure no part of the query is overlooked or left unanswered.

The goal is to produce complete bullet points in the style of factual proposition extraction, not abstract insights or high-level takeaways.

Rules:

1. Use ONLY information explicitly stated in the transcript.

2. Do NOT add assumptions, outside knowledge, unsupported inferences, interpretations, or synthesized implications.

3. Include ONLY information relevant to the Query.

4. For each bullet, include an evidence_quote from the transcript that directly supports that bullet. Include the Document ID in doc_ids.

5. Only include core meeting matters (e.g. major discussions, decisions, proposals, or preferences) related to answering the Query.
Exclude detailed technical explanations, implementation methods, and background discussions unless they directly correspond to the Query.

6. Each bullet should contain only the essential information needed to describe the matter.
Remove secondary details, examples, and elaborations.

7. Do NOT merge multiple related discussion matters into one abstract bullet.
If the meeting contains separate discussions, decisions, proposals, or statements, write separate bullets.

8. If no relevant information is found, return exactly:
{{"bullets": []}}

9. Output MUST be valid JSON only.

10. Do NOT include explanations, notes, or markdown code fences.

Output schema:

{{
  "bullets": [
    {{
      "point": "...",
      "evidence_quote": "...",
      "doc_ids": ["..."]
    }}
  ]
}}

Query:
{query}

Context:
This is meeting_transcript 1.
Document ID: {doc_id}

{transcript_text}
""".strip()


# ============================================================
# D. Stage 2: LLM judge for Stage-1 UNGROUNDED bullets
# ============================================================

STAGE2_BATCH_SYSTEM = """You are an expert evaluator of multi-document summarization quality.

You will be given a QUERY, the GOLD REFERENCE BULLETS, a SOURCE TRANSCRIPT, and a list of UNGROUNDED summary bullets. These bullets were generated by a summarization model and an automatic similarity check did NOT match them to any gold bullet. Most such bullets are genuinely not covered by the gold reference; a full match is uncommon and requires clear evidence.

Your task has two steps for each bullet.

STEP 1 — STRICT GOLD CHECK. Decide the bullet's gold coverage as FULL, PARTIAL, or NONE.
- First break the bullet into its claim components: the actor or speaker; the action, proposal, decision, or opinion; its object; and any status, certainty, number, name, reason, cause, or outcome it states.
- FULL: ONE single gold bullet states EVERY component with the same meaning and the same status (proposed vs. decided, tentative vs. definite), and nothing contradicts it. Numbers and names must not differ from the gold; a difference only in how the speaker is referred to (e.g. "the team" vs. "the project manager") does not by itself make it PARTIAL. Different wording is fine. A bullet that says LESS than the gold bullet but nothing beyond it is FULL only if it states that gold bullet's main point, not just a minor detail of it.
- PARTIAL: a gold bullet covers some components, but the bullet adds or changes at least one — e.g. an extra reason, cause, outcome, decision, number, name, qualifier, or a stronger certainty than the gold states — or its claim is only obtained by combining several gold bullets, or it states only a minor detail of a gold bullet.
- NONE: no gold bullet states the bullet's main claim. Sharing only the topic, meeting, speaker, or keywords is NONE.
- Only FULL may be labeled MATCHED_IN_GOLD; set "matched_gold_index" to the gold_index of the best-matching gold bullet. PARTIAL and NONE go to STEP 2.
- Before choosing FULL, check every component against the gold_quote. If any component is not stated there, or you have any doubt, the coverage is PARTIAL.
- For MATCHED_IN_GOLD, copy the supporting gold text verbatim into "gold_quote" and list any bullet component it does not state in "uncovered_components"; this list must be empty, otherwise the bullet is PARTIAL.
STEP 2 — ERROR LABEL. For bullets whose gold coverage is PARTIAL or NONE, assign the most appropriate label that explains WHY it is ungrounded. You must also provide a short rationale and an evidence quote from the transcript if applicable.

REFERENCE LABELS (suggested, but not exhaustive):
- IRRELEVANT: the bullet discusses a topic the query does not ask about, or exceeds the specific scope the query defines. If a bullet is both irrelevant AND contains other errors, prefer IRRELEVANT as the label since scope mismatch is the more fundamental problem.
- EVIDENCE_HALLUCINATION: the bullet introduces facts, names, numbers, or claims that are entirely absent from the transcript, or that directly contradict the transcript. Use this label when no transcript passage can even partially support the claim — including cases where the model has imported outside background knowledge not present in the transcript.
- UNSUPPORTED_INFERENCE - MOTIVE: the bullet adds a purpose, goal, or reason ("to X", "so that", "in order to", "for Y") not explicitly stated in the transcript.
- UNSUPPORTED_INFERENCE - RELATION: the bullet connects two facts with causal, contrastive, or comparative language ("resulting in", "rather than", "compared to", "because of") not explicitly stated in the transcript.
- UNSUPPORTED_INFERENCE - SYNTHESIS: the bullet combines scattered transcript facts into a unified conclusion or framework that the transcript does not itself draw. Use this when the individual facts are present but the overall conclusion is not.
- OVERCLAIM: the bullet (a) states as definite something the transcript only marks as uncertain or under discussion, (b) presents a discussed option as a decided outcome, or (c) exaggerates the degree or extent of something (e.g. transcript says "some people support" but bullet says "widespread support").
- OVERSPECIFICATION: the bullet adds unsupported modifiers, descriptors, or uses more specific terms than the transcript warrants, without rising to the level of a full OVERCLAIM.

Choosing between labels — priority rules:
1. If the bullet is off-topic relative to the QUERY, label IRRELEVANT regardless of other errors.
2. If the content is on-topic but the facts are fabricated or contradict the transcript, label EVIDENCE_HALLUCINATION.
3. If the facts are present in the transcript but the bullet overstates certainty or exaggerates degree, label OVERCLAIM.
4. If the facts are present but the bullet adds unsupported connective reasoning, label the most specific UNSUPPORTED_INFERENCE subtype.
5. If the facts are present but the bullet adds unwarranted specificity without overstating certainty, label OVERSPECIFICATION.
6. If none of the above fit, invent a concise descriptive label and explain it.

IMPORTANT:
- For evidence_quote: copy verbatim from the TRANSCRIPT only. Do NOT copy from the bullet. If no relevant passage exists in the transcript, use empty string.
- Be specific in rationale: cite the exact word or phrase in the bullet that causes the problem.
- Each bullet gets exactly one label — MATCHED_IN_GOLD only if its gold coverage is FULL, otherwise the one that identifies the most fundamental error.
- For MATCHED_IN_GOLD, the rationale should say which part of the gold bullet expresses the bullet's claim.

Respond ONLY with a JSON array, one object per bullet, in the same order:
[
  {
    "index": <int>,
    "gold_coverage": "<FULL, PARTIAL, or NONE>",
    "gold_quote": "<verbatim gold text if FULL, otherwise empty string>",
    "uncovered_components": ["<bullet component not stated in the gold>", ...],
    "label": "<MATCHED_IN_GOLD or your chosen error label>",
    "matched_gold_index": <gold_index as int if MATCHED_IN_GOLD, otherwise null>,
    "rationale": "<specific word or phrase in the bullet that causes the problem, <= 30 words>",
    "evidence_quote": "<exact verbatim excerpt from TRANSCRIPT only, or empty string>"
  },
  ...
]
No text outside the JSON array."""


def format_gold(gold_bullets: list) -> str:
    if not gold_bullets:
        return "(no gold bullets for this case — do not use MATCHED_IN_GOLD)"
    return "\n".join(f"[gold_index={i}] {g}" for i, g in enumerate(gold_bullets))


def stage2_batch_user_prompt(query: str, gold_bullets: list, context: str, bullets: list) -> str:
    numbered = "\n".join(f"{i+1}. [index={b['index']}] {b['text']}" for i, b in enumerate(bullets))
    return (
        f"QUERY:\n{query}\n\n"
        f"GOLD REFERENCE BULLETS:\n{format_gold(gold_bullets)}\n\n"
        f"SOURCE TRANSCRIPT:\n\"\"\"\n{context}\n\"\"\"\n\n"
        f"UNGROUNDED BULLETS:\n{numbered}\n\n"
        "Return JSON array only."
    )
