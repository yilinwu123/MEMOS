# Pipeline

Code for building gold bullets, generating summaries, and running the two-stage bullet-level evaluation of
MEMOS. Four scripts run in order (A2 → B → C → D) and share two modules: `common.py` and `prompts.py`.

```
gold answer ──► A2. gold bullets ──────────────────────────┐
                                                           ▼
query + transcripts ──► B. LLM summary ──► C. Stage 1 (embedding) ──► D. Stage 2 (LLM judge)
                           (bullets)         GROUNDED / UNGROUNDED     error label per UNGROUNDED bullet
```

| File | Step | What it does |
|---|---|---|
| `a_split_gold_bullets.py` | A2 | Splits each gold answer into atomic gold bullets |
| `b_summarize.py` | B | Generates a bullet-point summary for each case with one of six LLMs |
| `c_stage1_grounding.py` | C | Stage 1: labels each bullet GROUNDED / UNGROUNDED by embedding similarity to the gold bullets |
| `d_stage2_error_analysis.py` | D | Stage 2: sends UNGROUNDED bullets to an LLM judge for a gold check and an error label |
| `common.py` | – | OpenRouter client, JSON I/O, QMSum loading, context building, summary parsing |
| `prompts.py` | – | Every prompt used by A2, B, and D |

Every script takes `--setting multidoc` or `--setting singledoc`. Run the scripts from the repository root, as
in the examples below.

## Requirements

```bash
pip install -r requirements.txt           # openai, numpy; Python >= 3.9
export OPENROUTER_API_KEY=<your key>       # every LLM and embedding call goes through OpenRouter
export QMSUM_ROOT=/path/to/QMSum/data/ALL  # optional; default: data/qmsum/ (searched recursively)
```

`QMSUM_ROOT` can also be set with `--qmsum-root` on A2, B, and D. Input and output locations are defined at the
top of `common.py` (`CASES`, `GOLD_BULLETS`, `OUTPUT_ROOT`), relative to the repository root.

## A2. Gold bullets: `a_split_gold_bullets.py`

Each gold answer is decomposed into atomic, self-contained bullets that keep the original wording, using
`google/gemini-3-flash-preview`.
- **Multi-document cases**: the gold answer is the members' QMSum answers joined in document order. The query is
  the merged query, and the DOCUMENT shown to the model is the concatenated multi-document context.
- **Single-document cases**: the query, answer, and transcript come directly from QMSum.

```bash
python pipeline/a_split_gold_bullets.py --setting multidoc  --output outputs/gold/multidoc_gold_bullets.json
python pipeline/a_split_gold_bullets.py --setting singledoc --output outputs/gold/singledoc_gold_bullets.json
```

Output is one JSON list: `[{"group_id", "query", "gold", "gold_bullets": [...]}, ...]`. The file is re-saved after
every case and resumes from existing entries. A failed call, or one that returns no bullets, is not saved and is
retried on the next run.

Prompt: `GOLD_SPLIT_SYSTEM` with `gold_split_user_prompt_*` (paper: Gold Answer Decomposition Prompt). The
single-document variant uses identical text without blank lines.

## B. Summarization: `b_summarize.py`

```bash
python pipeline/b_summarize.py --setting multidoc  --model qwen   # -> outputs/multidoc/qwen/
python pipeline/b_summarize.py --setting singledoc --model qwen   # -> outputs/singledoc/qwen/
```

| `--model` | OpenRouter ID |
|---|---|
| `qwen` | `qwen/qwen3.6-plus` |
| `deepseek4flash` | `deepseek/deepseek-v4-flash` |
| `deepseek4pro` | `deepseek/deepseek-v4-pro` |
| `grok` | `x-ai/grok-4.3` |
| `nvidia` | `nvidia/nemotron-3-ultra-550b-a55b` |
| `gpt` | `openai/gpt-5.6-terra` |

- **Decoding**: every model is called with reasoning enabled (`reasoning: {enabled: true}`), JSON output mode,
  and `max_tokens=10000`. Temperature and other sampling parameters are not set, so provider defaults apply.
- **Prompt**: `summarize_prompt_multidoc` / `summarize_prompt_singledoc` (paper: Figure 2). The model returns
  `{"bullets": [{"point": ..., "evidence_quote": ..., "doc_ids": [...]}]}`.
- **Context**: each transcript is headed by `This is meeting_transcript <i>.` and `Document ID: <id>`.
  Consecutive transcripts are separated by a line of `=` (`common.build_multidoc_context`).
- **Outputs per case**:
  - `<id>_summary.json`, with the raw model output in `summary` (multi-document) or `summarization_point`
    (single-document).
  - `<id>_context.txt` (multi-document only): the exact context the model saw, which Stage 2 reuses.
- **Failures**: existing summaries are skipped. API errors, and outputs that are not a JSON object with a
  `bullets` list, are written to `<id>_error.json` and retried on the next run.

## C. Stage 1, embedding grounding: `c_stage1_grounding.py`

Every generated bullet `b` and gold bullet `g_j` is embedded with `qwen/qwen3-embedding-8b`
(L2-normalized), and

```
top1(b) = max_j cos(b, g_j)
top1(b) >= 0.62  ->  GROUNDED      otherwise  ->  UNGROUNDED (passed to Stage 2)
```

```bash
python pipeline/c_stage1_grounding.py --setting multidoc  --summary-dir outputs/multidoc/qwen
python pipeline/c_stage1_grounding.py --setting singledoc --summary-dir outputs/singledoc/qwen
```

Options: `--threshold` (default 0.62) and `--overwrite` (recompute existing outputs).

Output: `<id>_judge_embtop1.json`, containing the query, the gold bullets, case-level `metrics`, and a `bullets`
list. Case-level metrics are bullet count, grounded and ungrounded counts, and their rates; the ungrounded rate is
1 − precision. Each bullet has its `category` (GROUNDED / UNGROUNDED) and a `grounding` record:
- `best_emb`: the top-1 score.
- `evidence` and `evidence_scores`: the top-5 gold bullets and their scores.
- `gold_scores`: the cosine to every gold bullet, in gold order, used for one-to-one gold matching when
  computing recall.

A valid empty summary (`{"bullets": []}`) is kept with 0 bullets, so the case's gold bullets still count towards
recall.

## D. Stage 2, LLM judge: `d_stage2_error_analysis.py`

All UNGROUNDED bullets of a case go to `google/gemini-3-flash-preview` (`max_tokens=10000`, provider-default
sampling) in one call. The call includes the query, the gold bullets, and the source transcript:
- **Multi-document**: the transcript is `<id>_context.txt` from step B.
- **Single-document**: the transcript is the QMSum text, identical to what the summarizer saw.

For each bullet the judge:

1. **Gold check.** Labels the bullet `MATCHED_IN_GOLD` only if one gold bullet fully expresses it (coverage
   FULL), i.e. a Stage-1 false negative.
2. **Error label.** Otherwise assigns exactly one label that identifies the most fundamental error, using this
   priority:

   | Priority | Label |
   |---|---|
   | 1 | `IRRELEVANT` |
   | 2 | `EVIDENCE_HALLUCINATION` |
   | 3 | `OVERCLAIM` |
   | 4 | `UNSUPPORTED_INFERENCE - MOTIVE / RELATION / SYNTHESIS` |
   | 5 | `OVERSPECIFICATION` |

   If none of these fit, the judge gives a new descriptive label.

```bash
python pipeline/d_stage2_error_analysis.py --setting multidoc  --folder outputs/multidoc/qwen
python pipeline/d_stage2_error_analysis.py --setting singledoc --folder outputs/singledoc/qwen
```

`--dry-run` checks file discovery and inputs without calling the LLM.

Prompt: `STAGE2_BATCH_SYSTEM` with `stage2_batch_user_prompt` (paper: Error Classification Prompt).

Output: `<id>_judge_embtop1_reanalyzed_nocon_withgold.json` (multi-document) or
`<id>_judge_embtop1_reanalyzed_nocon_gold.json` (single-document). Each is the Stage-1 JSON with a `reanalysis`
field on every bullet. It is `null` for GROUNDED bullets. For the others it holds `label` and `classification`,
the judge's full output: `label`, `gold_coverage`, `gold_quote`, `uncovered_components`, `matched_gold_index`,
`rationale`, and `evidence_quote`. The script ends by
printing the label distribution and the Stage-1 miss rate (MATCHED_IN_GOLD / UNGROUNDED).

Failure handling:
- Each call is attempted up to 3 times.
- If the batch response cannot be parsed, or its bullet indices do not match the ones sent, each bullet is judged
  in its own call with the same prompt.
- A case is not written if its source transcript is missing or any bullet is still unclassified, so it is retried
  on the next run.
- Existing outputs are skipped.

## Running B → C → D for one model

```bash
bash scripts/run_multidoc.sh qwen
bash scripts/run_singledoc.sh qwen
```

## Notes

- **IDs**: `<meeting_id>_sq_<i>` is the i-th entry of that meeting's QMSum `specific_query_list`. A multi-document
  case is keyed by `group_id`, its member query IDs joined by `|`. In file names `|` becomes `__`.
- **Reproducibility**: LLM outputs are not deterministic, so re-running A2–D gives results close to, but not
  identical with, those reported in the paper.
- **Not included**: metric aggregation (micro-averaged precision, recall, F1) and the paper's additional analyses
  (dataset construction, oracle baseline, position analysis) are not part of this folder.
