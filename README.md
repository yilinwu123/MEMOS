# MEMOS: A Query-Based Multi-Document Benchmark for Meeting Summarization

Code and data for **MEMOS** (**M**ulti-document **E**valuation of **M**eeting **O**pen-query **S**ummarization).
MEMOS adapts [QMSum](https://github.com/Yale-LILY/QMSum) into query-based summarization over 1–8 meeting
transcripts. Generated summaries are evaluated bullet by bullet in two stages:
- **Stage 1** checks each bullet against the gold bullets by embedding similarity.
- **Stage 2** sends the bullets Stage 1 could not verify to an LLM judge, which assigns each one an error label.

```
             A. Dataset                        B. Summarization        C. Stage 1               D. Stage 2
QMSum ──► clusters + merged queries ──┐
     └──► gold answers ──► gold bullets ┴──► LLM bullet summary ──► embedding top-1 ≥ 0.62 ──► LLM judge on
                                                                    GROUNDED / UNGROUNDED      UNGROUNDED bullets
```

## Repository layout

```
data/
  qmsum/                     # place the QMSum meeting files here (not redistributed, see Setup)
  multidoc/
    queries.json              # A1: clusters (group_id, member queries, merged query)
    gold_bullets.json         # A2: gold bullets
  singledoc/
    run_cases_50.csv          # A1: single-document cases (QMSum meeting, specific-query index)
    gold_bullets.json         # A2: gold bullets
pipeline/
  common.py                   # OpenRouter client, QMSum loading, context building
  prompts.py                  # every prompt used in the pipeline
  a_split_gold_bullets.py     # A2. gold answer -> atomic gold bullets
  b_summarize.py              # B.  query-focused summarization (6 models)
  c_stage1_grounding.py       # C.  Stage 1: embedding grounding
  d_stage2_error_analysis.py  # D.  Stage 2: LLM-judge error classification
scripts/
  run_multidoc.sh             # B -> C -> D for the multi-document setting
  run_singledoc.sh            # B -> C -> D for the single-document setting
```

## Setup

```bash
pip install -r requirements.txt          # Python >= 3.9
export OPENROUTER_API_KEY=<your key>      # all LLM and embedding calls go through OpenRouter
```

Every script reads the API key from the `OPENROUTER_API_KEY` environment variable.

**QMSum.** The scripts read the original QMSum per-meeting JSON files (`<meeting_id>.json` with
`meeting_transcripts` and `specific_query_list`). Download QMSum and either copy the meeting files into
`data/qmsum/` or point `QMSUM_ROOT` at them:

```bash
git clone https://github.com/Yale-LILY/QMSum.git
export QMSUM_ROOT=$PWD/QMSum/data/ALL    # searched recursively; duplicate meeting IDs are de-duplicated
```

All scripts also accept `--qmsum-root`.

## A. Dataset construction

### A1. Instances

**Multi-document.** Each instance is a cluster of QMSum specific queries whose answers come from different
parts of one or more meetings. The cluster's queries are rewritten into one merged query.

```json
{"cluster_id": "cluster_0",
 "group_id": "IS1003b_sq_6|IS1003d_sq_0",
 "query_list": ["Summarize the discussion about product appearance.", "..."],
 "merged_query": "Summarize the discussion about product appearance."}
```

`<meeting_id>_sq_<i>` refers to the i-th entry of that meeting's `specific_query_list` in QMSum. The input
documents are the distinct meetings in `group_id`. The gold answer is the members' QMSum answers joined by
newlines.

**Single-document.** `data/singledoc/run_cases_50.csv` lists `(case, index)` pairs. Each pair is one
QMSum meeting with its `specific_query_list[index]` query and answer.

### A2. Gold bullets

Each gold answer is split into atomic, self-contained bullets that keep the original wording, using
`google/gemini-3-flash-preview` (prompt: `GOLD_SPLIT_SYSTEM` in `pipeline/prompts.py`). The released gold
bullets are in `data/multidoc/gold_bullets.json` and `data/singledoc/gold_bullets.json`, one entry per case:

```json
{"group_id": "IS1003b_sq_6|IS1003d_sq_0", "query": "...", "gold": "<merged gold answer>",
 "gold_bullets": ["bullet 1", "bullet 2", "..."]}
```

To regenerate them (the LLM output is not deterministic, so new runs will differ slightly from the
released files):

```bash
python pipeline/a_split_gold_bullets.py --setting multidoc  --output outputs/gold/multidoc_gold_bullets.json
python pipeline/a_split_gold_bullets.py --setting singledoc --output outputs/gold/singledoc_gold_bullets.json
```

## B. Summarization

```bash
python pipeline/b_summarize.py --setting multidoc  --model qwen   # data/multidoc/queries.json      -> outputs/multidoc/qwen/
python pipeline/b_summarize.py --setting singledoc --model qwen   # data/singledoc/run_cases_50.csv -> outputs/singledoc/qwen/
```

| `--model` | OpenRouter ID |
|---|---|
| `qwen` | `qwen/qwen3.6-plus` |
| `deepseek4flash` | `deepseek/deepseek-v4-flash` |
| `deepseek4pro` | `deepseek/deepseek-v4-pro` |
| `grok` | `x-ai/grok-4.3` |
| `nvidia` | `nvidia/nemotron-3-ultra-550b-a55b` |
| `gpt` | `openai/gpt-5.6-terra` |

All models receive the same prompt and run on all cases, with reasoning enabled, `max_tokens=10000` and
JSON output mode.
The model is asked for `{"bullets": [{"point": ..., "evidence_quote": ...}]}`.

Outputs per case: `<id>_summary.json`. In the multi-document setting, `<id>_context.txt` is also written: the
exact concatenated context the model saw, which Stage 2 reuses. Runs are resumable, and failed calls are
written to `<id>_error.json` and retried on the next run.

## C. Stage 1: embedding grounding

Each generated bullet `b` is embedded with `qwen/qwen3-embedding-8b`, and

```
top1(b) = max_j cos(b, g_j)   over gold bullets g_j
top1(b) >= 0.62  ->  GROUNDED      otherwise  ->  UNGROUNDED
```

```bash
python pipeline/c_stage1_grounding.py --setting multidoc  --summary-dir outputs/multidoc/qwen
python pipeline/c_stage1_grounding.py --setting singledoc --summary-dir outputs/singledoc/qwen
```

Output: `<id>_judge_embtop1.json` with per-bullet labels, the top-5 most similar gold bullets and scores,
and case-level `metrics` (bullet count, grounded/ungrounded counts and rates).

## D. Stage 2: LLM-judge error analysis

All UNGROUNDED bullets of a case are sent to `google/gemini-3-flash-preview` in one call, together with the
query, the gold bullets and the source transcript(s). For each bullet, the judge:

1. Checks the gold bullets strictly. If one gold bullet fully expresses the bullet, it is labeled
   `MATCHED_IN_GOLD`, i.e. a Stage-1 false negative.
2. Otherwise assigns one error label: `IRRELEVANT`, `EVIDENCE_HALLUCINATION`,
   `UNSUPPORTED_INFERENCE - MOTIVE / RELATION / SYNTHESIS`, `OVERCLAIM`, `OVERSPECIFICATION`, or a new
   descriptive label if none fits.

If the batch response cannot be parsed, each bullet is judged in a separate call.

```bash
python pipeline/d_stage2_error_analysis.py --setting multidoc  --folder outputs/multidoc/qwen
python pipeline/d_stage2_error_analysis.py --setting singledoc --folder outputs/singledoc/qwen
```

Stage 2 judges every case in `--folder` that has a Stage-1 output. Use `--dry-run` to check inputs without
calling the LLM. Output:
`<id>_judge_embtop1_reanalyzed_nocon_withgold.json` (multi-document) or
`<id>_judge_embtop1_reanalyzed_nocon_gold.json` (single-document). Each is the Stage-1 file with a
`reanalysis` field added to every bullet.

## Run everything for one model

```bash
bash scripts/run_multidoc.sh qwen
bash scripts/run_singledoc.sh qwen
```

## Notes

- `group_id` is the unique key of a multi-document instance. `cluster_id` is kept from each original subset
  and is not unique across subsets.
- One cluster (`IS1003b_sq_3|IS1003c_sq_3|IS1003d_sq_6`) has no `merged_query`; its member queries are
  concatenated and used as the query for summarization and gold-bullet splitting.
- LLM outputs are not deterministic, so re-running B–D gives results close to, but not identical with,
  those reported in the paper.
