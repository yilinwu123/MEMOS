"""D. Stage 2: LLM-as-a-judge for bullets that Stage 1 marked UNGROUNDED.

For every Stage-1 output (<stem>_judge_embtop1.json) in --folder, all
UNGROUNDED bullets of the case are sent to Gemini-3-Flash-Preview in a single
call together with the query, the gold bullets (data/<setting>/gold_bullets.json)
and the source transcript. The
judge first checks whether a bullet is fully expressed by one gold bullet
(MATCHED_IN_GOLD, i.e. a Stage-1 false negative); otherwise it assigns one
error label (IRRELEVANT, EVIDENCE_HALLUCINATION, UNSUPPORTED_INFERENCE - *,
OVERCLAIM, OVERSPECIFICATION, or a new label). If the batch response cannot be
parsed, each bullet is classified in its own call with the same prompt. A case
whose source transcript is missing or whose judge calls fail is not written, so
it is retried on the next run.

Source transcript:
  multidoc : <stem>_context.txt written by step B (the exact model context)
  singledoc: the meeting transcript loaded from QMSum (same text as in step B)

Output: <stem>_judge_embtop1_reanalyzed_nocon_withgold.json (multidoc) or
        <stem>_judge_embtop1_reanalyzed_nocon_gold.json (singledoc), i.e. the
        Stage-1 JSON with a "reanalysis" field added to every bullet.

Examples:
  python pipeline/d_stage2_error_analysis.py --setting multidoc --folder outputs/multidoc/qwen
  python pipeline/d_stage2_error_analysis.py --setting singledoc --folder outputs/singledoc/qwen
"""

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from common import DEFAULT_QMSUM_ROOT, GOLD_BULLETS, QMSum, get_client, load_json, save_json
from prompts import STAGE2_BATCH_SYSTEM, stage2_batch_user_prompt

MODEL = "google/gemini-3-flash-preview"
MAX_TOKENS = 10000
MAX_RETRIES = 3
RETRY_DELAY = 2
MATCHED = "MATCHED_IN_GOLD"

OUT_SUFFIX = {
    "multidoc": "_judge_embtop1_reanalyzed_nocon_withgold.json",
    "singledoc": "_judge_embtop1_reanalyzed_nocon_gold.json",
}


# ============================================================
# LLM call and JSON extraction
# ============================================================

def call_llm(client, system_prompt: str, user_prompt: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [LLM error attempt {attempt}/{MAX_RETRIES}] {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    return ""


def extract_json(text: str, kind: type) -> Optional[object]:
    """Parse `text` as a JSON list/dict, falling back to the outermost [...] / {...} span."""
    pattern = r"\[.*\]" if kind is list else r"\{.*\}"
    candidates = [text]
    match = re.search(pattern, text, re.DOTALL)
    if match:
        candidates.append(match.group())
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, kind):
                return parsed
        except Exception:
            pass
    return None


# ============================================================
# Classification
# ============================================================

def normalize_result(r: dict, n_gold: int) -> dict:
    """Canonicalize MATCHED_IN_GOLD and check that matched_gold_index is within range."""
    label = str(r.get("label", "")).strip()
    if label.upper().replace(" ", "_") == MATCHED:
        label = MATCHED
    gi = r.get("matched_gold_index")
    try:
        gi = int(gi) if gi is not None else None
    except (TypeError, ValueError):
        gi = None
    if label != MATCHED or (gi is not None and not 0 <= gi < n_gold):
        gi = None
    if label == MATCHED and n_gold == 0:
        # A case without gold bullets cannot be matched; leave it for manual inspection.
        return {**r, "label": "INVALID_MATCH_NO_GOLD", "matched_gold_index": None,
                "rationale": f"MATCHED_IN_GOLD returned but case has no gold bullets; {r.get('rationale', '')}"}
    return {**r, "label": label, "matched_gold_index": gi}


def classify(client, context: str, query: str, gold_bullets: list, bullets: list) -> dict:
    """{bullet_index: classification}. One batch call; per-bullet calls if the batch fails to parse."""
    print(f"    [Classify] batch of {len(bullets)} bullet(s) ...")
    raw = call_llm(client, STAGE2_BATCH_SYSTEM, stage2_batch_user_prompt(query, gold_bullets, context, bullets))
    result = extract_json(raw, list)
    expected = {b["index"] for b in bullets}
    if (result is not None and len(result) == len(bullets)
            and all(isinstance(r, dict) and "label" in r for r in result)
            and {r.get("index") for r in result} == expected):
        return {r["index"]: normalize_result(r, len(gold_bullets)) for r in result}

    print(f"    [Classify] batch parse failed (got {len(result) if result else 'None'} items, "
          f"expected {len(bullets)}); falling back to per-bullet calls. Raw: {raw[:200]}")
    out = {}
    for b in bullets:
        # Same judge prompt as the batch call, with a single bullet.
        raw = call_llm(client, STAGE2_BATCH_SYSTEM, stage2_batch_user_prompt(query, gold_bullets, context, [b]))
        parsed = extract_json(raw, list)
        obj = parsed[0] if parsed and len(parsed) == 1 and isinstance(parsed[0], dict) else extract_json(raw, dict)
        if obj and "label" in obj:
            obj["index"] = b["index"]
            r = normalize_result(obj, len(gold_bullets))
        else:
            r = {"index": b["index"], "label": "UNCLASSIFIED", "matched_gold_index": None,
                 "rationale": "parse error", "evidence_quote": ""}
        out[r["index"]] = r
        print(f"      bullet {b['index']}: label={r['label']}")
    return out


# ============================================================
# Inputs
# ============================================================

def load_multidoc_context(folder: Path, stem: str) -> str:
    p = folder / f"{stem}_context.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    print(f"    [WARNING] Context file not found: {p.name}")
    return ""


def load_singledoc_context(qmsum: QMSum, stem: str) -> str:
    doc_id = stem.split("_sq_")[0] if "_sq_" in stem else stem
    try:
        return qmsum.transcript_text(doc_id)
    except FileNotFoundError:
        print(f"    [WARNING] QMSum source not found for case: {doc_id}")
        return ""


def load_gold_map(path) -> dict:
    """case id (group_id / query_id) -> gold_bullets."""
    return {item["group_id"]: [str(x) for x in item.get("gold_bullets", [])] for item in load_json(path)}


def pick_gold(gold_map: dict, case_id: str, stage1_gold: list) -> tuple:
    """(source, gold_bullets) for a case; falls back to the gold stored in the Stage-1 json."""
    if case_id in gold_map:
        return "gold_bullets.json", gold_map[case_id]
    print(f"    [WARNING] {case_id} not in gold_bullets.json; using gold_bullets from the Stage-1 json")
    return "stage1_json", stage1_gold


# ============================================================
# Per-file processing
# ============================================================

def process_file(client, json_path: Path, setting: str, gold_map: dict, qmsum: Optional[QMSum],
                 dry_run: bool) -> Optional[dict]:
    print(f"\n{'=' * 60}\nProcessing: {json_path.name}")
    stem = json_path.stem.replace("_judge_embtop1", "")
    out_path = json_path.parent / f"{stem}{OUT_SUFFIX[setting]}"
    if out_path.exists() and not dry_run:
        print(f"  Already done, skipping (delete {out_path.name} to rerun).")
        return load_json(out_path)

    data = load_json(json_path)
    query = data.get("query", "")
    bullets = data.get("bullets", [])
    case_id = data.get("group_id") or data.get("query_id") or stem
    gold_src, gold_bullets = pick_gold(gold_map, case_id, data.get("gold_bullets", []))

    if setting == "multidoc":
        context = load_multidoc_context(json_path.parent, stem)
    else:
        context = load_singledoc_context(qmsum, stem)

    ungrounded = [b for b in bullets if b.get("category") == "UNGROUNDED"]
    print(f"  case id      : {case_id}")
    print(f"  query        : {query[:90]}")
    print(f"  context found: {bool(context)}")
    print(f"  gold bullets : {len(gold_bullets)}  (from {gold_src})")
    print(f"  bullets      : {len(bullets)}  UNGROUNDED: {len(ungrounded)}")

    reanalysis = {}
    if ungrounded and not context and not dry_run:
        # Without the source transcript the judge cannot label the bullets; retry on the next run.
        print("  [Error] no source transcript; case not written (rerun after restoring it).")
        return None
    if dry_run or not ungrounded:
        for b in ungrounded:
            reanalysis[b["index"]] = {"label": "DRY_RUN" if dry_run else "SKIPPED"}
    else:
        items = [{"index": b["index"], "text": b["text"]} for b in ungrounded]
        classify_map = classify(client, context, query, gold_bullets, items)

        for b in ungrounded:
            idx = b["index"]
            result = classify_map.get(idx, {"index": idx, "label": "UNCLASSIFIED", "matched_gold_index": None,
                                            "rationale": "missing from classify_map", "evidence_quote": ""})
            reanalysis[idx] = {"label": result["label"], "classification": result}
            gi = result.get("matched_gold_index")
            tag = f"(gold {gi})  " if gi is not None else ""
            print(f"    bullet {idx:>3} -> {result['label']}  {tag}{result.get('rationale', '')[:60]}")

    out_data = {**data, "gold_source": gold_src,
                "bullets": [{**b, "reanalysis": reanalysis.get(b["index"])} for b in bullets]}
    failed = [i for i, r in reanalysis.items() if r["label"] == "UNCLASSIFIED"]
    if failed and not dry_run:
        # Failed judge calls are retried on the next run instead of being saved as UNCLASSIFIED.
        print(f"\n  [Error] {len(failed)} bullet(s) unclassified {failed}; case not written (rerun to retry).")
        return None
    if not dry_run:
        save_json(out_data, out_path)
        print(f"\n  Written: {out_path.name}")
    return out_data


def print_summary(all_results: list):
    labels = Counter()
    for data in all_results:
        for b in data.get("bullets", []):
            ra = b.get("reanalysis")
            labels[ra["label"] if ra else "(GROUNDED in Stage 1)"] += 1
    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    for label, count in sorted(labels.items()):
        print(f"  {label:<42} {count:>5}")
    n_ungr = sum(c for l, c in labels.items() if l != "(GROUNDED in Stage 1)")
    if n_ungr:
        print(f"\n  Stage-1 miss rate (MATCHED_IN_GOLD / UNGROUNDED): "
              f"{labels.get(MATCHED, 0)}/{n_ungr} = {labels.get(MATCHED, 0) / n_ungr:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", choices=["multidoc", "singledoc"], required=True)
    parser.add_argument("--folder", required=True, help="folder with *_judge_embtop1.json from Stage 1")
    parser.add_argument("--dry-run", action="store_true", help="no LLM calls; only check file discovery")
    parser.add_argument("--qmsum-root", default=str(DEFAULT_QMSUM_ROOT))
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    json_files = sorted(folder.glob("*_judge_embtop1.json"))
    if not json_files:
        print(f"No *_judge_embtop1.json files found in {folder}. Exiting.")
        return

    client = None if args.dry_run else get_client()
    qmsum = QMSum(args.qmsum_root) if args.setting == "singledoc" else None
    gold_map = load_gold_map(GOLD_BULLETS[args.setting])
    print(f"Loaded gold bullets for {len(gold_map)} case(s). Processing {len(json_files)} file(s) ...")

    all_results = [process_file(client, jf, args.setting, gold_map, qmsum, args.dry_run) for jf in json_files]
    failed = sum(r is None for r in all_results)
    print_summary([r for r in all_results if r is not None])
    if failed:
        print(f"\n  {failed} case(s) not written because of missing transcripts or failed judge calls; rerun to retry.")


if __name__ == "__main__":
    main()
