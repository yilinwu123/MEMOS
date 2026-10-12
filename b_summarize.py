"""B. Query-focused (multi-/single-document) summarization with OpenRouter models.

multidoc : one call per cluster in data/multidoc/queries.json; the model sees the
           concatenated transcripts of the cluster and the merged query.
           Writes <group>_summary.json and <group>_context.txt (the exact
           context shown to the model, reused by Stage 2).
singledoc: one call per (case, index) in data/singledoc/run_cases_50.csv; the model sees one
           transcript and its QMSum specific query.
           Writes <doc_id>_sq_<index>_summary.json.

Existing *_summary.json files are skipped, so runs are resumable. Failed
calls (API errors or output that is not a JSON object with a "bullets" list)
are written to *_error.json and retried on the next run.

Examples:
  python pipeline/b_summarize.py --setting multidoc --model qwen    # -> outputs/multidoc/qwen/
  python pipeline/b_summarize.py --setting singledoc --model qwen   # -> outputs/singledoc/qwen/
"""

import argparse
import os
import time
from typing import Dict

from common import CASES, DEFAULT_QMSUM_ROOT, OUTPUT_ROOT, QMSum, build_multidoc_context, doc_ids_from_group_id, \
    get_client, is_valid_summary, load_json, read_cases_csv, sanitize_group_id, save_json
from prompts import SUMMARIZE_SYSTEM_MULTIDOC, SUMMARIZE_SYSTEM_SINGLEDOC, summarize_prompt_multidoc, \
    summarize_prompt_singledoc

MAX_TOKENS = 10000
SLEEP_SECONDS = 1.0

# --model -> OpenRouter model identifier
MODELS: Dict[str, str] = {
    "qwen": "qwen/qwen3.6-plus",
    "deepseek4flash": "deepseek/deepseek-v4-flash",
    "deepseek4pro": "deepseek/deepseek-v4-pro",
    "grok": "x-ai/grok-4.3",
    "nvidia": "nvidia/nemotron-3-ultra-550b-a55b",
    "gpt": "openai/gpt-5.6-terra",
}


def generate_summary(client, model_id: str, setting: str, prompt: str) -> str:
    system = SUMMARIZE_SYSTEM_MULTIDOC if setting == "multidoc" else SUMMARIZE_SYSTEM_SINGLEDOC
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        extra_body={
            "reasoning": {"enabled": True},
            "response_format": {"type": "json_object"},
        },
    )
    content = response.choices[0].message.content
    if not is_valid_summary(content):
        raise ValueError(f"output is not a JSON object with a 'bullets' list: {str(content)[:200]!r}")
    return content


def clear_error(stem: str) -> None:
    """Remove the error file of an earlier failed attempt once the case succeeds."""
    if os.path.exists(f"{stem}_error.json"):
        os.remove(f"{stem}_error.json")


# ============================================================
# multidoc
# ============================================================

def run_multidoc(client, model_id: str, qmsum: QMSum, queries_path: str, output_dir: str):
    clusters = load_json(queries_path)
    print(f"[Info] cluster count: {len(clusters)}")

    for i, cluster in enumerate(clusters, start=1):
        cluster_id = cluster.get("cluster_id", f"cluster_{i}")
        group_id = cluster.get("group_id", "")
        print(f"\n[{i}/{len(clusters)}] {cluster_id}  group_id={group_id}")

        stem = os.path.join(output_dir, sanitize_group_id(group_id))
        out_path = f"{stem}_summary.json"
        if os.path.exists(out_path):
            print(f"  [Skip] already exists: {out_path}")
            continue

        try:
            query_list = cluster.get("query_list", [])
            merged_query = cluster.get("merged_query", "") or " ".join(query_list)
            doc_ids = doc_ids_from_group_id(group_id)
            context_text = build_multidoc_context(qmsum, doc_ids)
            prompt = summarize_prompt_multidoc(merged_query, context_text)
            summary = generate_summary(client, model_id, "multidoc", prompt)
        except Exception as e:
            save_json({"cluster_id": cluster_id, "group_id": group_id, "error": str(e)}, f"{stem}_error.json")
            print(f"  [Error] {e}")
            continue

        save_json({
            "cluster_id": cluster_id,
            "group_id": group_id,
            "document_ids": doc_ids,
            "query_list": query_list,
            "merged_query": merged_query,
            "summary": summary,
            "source_files": [qmsum.source_file(d) for d in doc_ids],
        }, out_path)
        with open(f"{stem}_context.txt", "w", encoding="utf-8") as f:
            f.write(context_text)
        clear_error(stem)
        print(f"  [Saved] {out_path}")
        time.sleep(SLEEP_SECONDS)


# ============================================================
# singledoc
# ============================================================

def run_singledoc(client, model_id: str, qmsum: QMSum, cases_path: str, output_dir: str):
    cases = read_cases_csv(cases_path)
    print(f"[Info] case count: {len(cases)}")

    for i, (doc_id, idx) in enumerate(cases, start=1):
        query_id = f"{doc_id}_sq_{idx}"
        print(f"[{i}/{len(cases)}] {query_id}")

        stem = os.path.join(output_dir, query_id)
        out_path = f"{stem}_summary.json"
        if os.path.exists(out_path):
            print(f"  [Skip] already exists: {out_path}")
            continue

        try:
            query = qmsum.specific_query(query_id)["query"]
            if not query:
                print("  [Warn] empty query, skipping")
                continue
            prompt = summarize_prompt_singledoc(query, qmsum.transcript_text(doc_id), doc_id)
            summary = generate_summary(client, model_id, "singledoc", prompt)
        except Exception as e:
            save_json({"query_id": query_id, "error": str(e)}, f"{stem}_error.json")
            print(f"  [Error] {e}")
            continue

        save_json({
            "source_file": qmsum.source_file(doc_id),
            "query_id": query_id,
            "query": query,
            "summarization_point": summary,
        }, out_path)
        clear_error(stem)
        print(f"  [Saved] {out_path}")
        time.sleep(SLEEP_SECONDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", choices=["multidoc", "singledoc"], required=True)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--output-dir", help="default: outputs/<setting>/<model>")
    parser.add_argument("--qmsum-root", default=str(DEFAULT_QMSUM_ROOT))
    args = parser.parse_args()

    model_id = MODELS[args.model]
    output_dir = args.output_dir or str(OUTPUT_ROOT / args.setting / args.model)
    os.makedirs(output_dir, exist_ok=True)

    client = get_client()
    qmsum = QMSum(args.qmsum_root)
    print(f"[Info] model: {model_id}  output dir: {output_dir}")

    if args.setting == "multidoc":
        run_multidoc(client, model_id, qmsum, CASES["multidoc"], output_dir)
    else:
        run_singledoc(client, model_id, qmsum, CASES["singledoc"], output_dir)
    print("\n[Done]")


if __name__ == "__main__":
    main()
