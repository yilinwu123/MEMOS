"""A2. Decompose QMSum gold answers into atomic gold bullets with Gemini-3-Flash-Preview.

multidoc : for every cluster in data/multidoc/queries.json, the gold answer is
           the members' QMSum answers joined by newlines and the query is its
           merged_query (or, if it has none, its member queries joined by
           spaces, as in step B); the DOCUMENT shown to the LLM is the
           concatenated multi-doc context.
singledoc: for every (case, index) in data/singledoc/run_cases_50.csv, the gold
           answer and query are specific_query_list[index] of that meeting.

A call that fails or returns no bullets is not saved and is retried on the next run.

Output (a single JSON list; resumable, re-saved after every case):
  [{"group_id": ..., "query": ..., "gold": ..., "gold_bullets": [...]}, ...]
For singledoc, group_id is the query_id (e.g. "Bed003_sq_3").

Examples:
  python pipeline/a_split_gold_bullets.py --setting multidoc --output outputs/gold/multidoc_gold_bullets.json
  python pipeline/a_split_gold_bullets.py --setting singledoc --output outputs/gold/singledoc_gold_bullets.json
"""

import argparse
import os
import time
from typing import List

from common import CASES, DEFAULT_QMSUM_ROOT, QMSum, build_multidoc_context, doc_ids_from_group_id, \
    get_client, load_json, read_cases_csv, safe_json_load, save_json
from prompts import GOLD_SPLIT_SYSTEM, GOLD_SPLIT_SYSTEM_SINGLEDOC, gold_split_user_prompt_multidoc, \
    gold_split_user_prompt_singledoc

MODEL = "google/gemini-3-flash-preview"
SLEEP_SECONDS = 1.0


def call_llm(client, setting: str, query: str, document: str, gold: str) -> List[str]:
    if setting == "multidoc":
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": GOLD_SPLIT_SYSTEM},
                {"role": "user", "content": gold_split_user_prompt_multidoc(query, document, gold)},
            ],
            temperature=0.0,
            max_tokens=10000,
        )
    else:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": GOLD_SPLIT_SYSTEM_SINGLEDOC},
                {"role": "user", "content": gold_split_user_prompt_singledoc(query, document, gold)},
            ],
            max_tokens=10000,
            response_format={"type": "json_object"},
        )
    bullets = safe_json_load(resp.choices[0].message.content).get("bullets", [])
    return bullets if isinstance(bullets, list) else []


def iter_multidoc_cases(qmsum: QMSum, queries_path: str):
    """Yield (group_id, query, gold, document)."""
    for item in load_json(queries_path):
        group_id = str(item.get("group_id", "")).strip()
        query = str(item.get("merged_query", "") or " ".join(item.get("query_list", []))).strip()
        if not group_id or not query:
            print(f"[Skip] {group_id or item.get('cluster_id')}: missing group_id/query")
            continue
        document = build_multidoc_context(qmsum, doc_ids_from_group_id(group_id))
        yield group_id, query, qmsum.gold_answer(group_id), document


def iter_singledoc_cases(qmsum: QMSum, cases_path: str):
    for doc_id, idx in read_cases_csv(cases_path):
        query_id = f"{doc_id}_sq_{idx}"
        sq = qmsum.specific_query(query_id)
        yield query_id, sq["query"], sq["answer"], qmsum.transcript_text(doc_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", choices=["multidoc", "singledoc"], required=True)
    parser.add_argument("--output", required=True, help="output JSON file")
    parser.add_argument("--qmsum-root", default=str(DEFAULT_QMSUM_ROOT))
    args = parser.parse_args()

    client = get_client()
    qmsum = QMSum(args.qmsum_root)

    results = load_json(args.output) if os.path.exists(args.output) else []
    done = {x["group_id"] for x in results}
    print(f"[Resume] {len(done)} case(s) already done")

    cases = (iter_multidoc_cases(qmsum, CASES["multidoc"]) if args.setting == "multidoc"
             else iter_singledoc_cases(qmsum, CASES["singledoc"]))

    for n, (group_id, query, gold, document) in enumerate(cases, start=1):
        print(f"[{n}] {group_id}")
        if group_id in done:
            print("  [Skip] already done")
            continue
        if not gold:
            print("  [Skip] empty gold answer")
            continue
        try:
            bullets = [str(b).strip() for b in call_llm(client, args.setting, query, document, gold)
                       if str(b).strip()]
            if not bullets:
                raise ValueError("no gold bullets returned")
        except Exception as e:
            # Not saved, so the case is retried on the next run.
            print(f"  [LLM Error] {e}")
            continue

        results.append({"group_id": group_id, "query": query, "gold": gold, "gold_bullets": bullets})
        done.add(group_id)
        save_json(results, args.output)
        print(f"  [Saved] {len(bullets)} bullets (total {len(results)})")
        time.sleep(SLEEP_SECONDS)

    print(f"\n[Done] {len(results)} entries -> {args.output}")


if __name__ == "__main__":
    main()
