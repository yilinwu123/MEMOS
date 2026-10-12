"""C. Stage 1: embedding-based grounding of summary bullets against gold bullets.

For each generated bullet b_i, compute the cosine similarity to every gold
bullet g_j with Qwen3-Embedding-8B (via OpenRouter):

    top1(b_i) = max_j cos(b_i, g_j)
    top1 >= threshold (0.62)  -> GROUNDED
    otherwise                 -> UNGROUNDED  (passed on to Stage 2)

Reads every *_summary.json in --summary-dir, compares it with the gold bullets
in data/<setting>/gold_bullets.json, and writes <stem>_judge_embtop1.json next
to it. Existing outputs are skipped unless
--overwrite is given.

Examples:
  python pipeline/c_stage1_grounding.py --setting multidoc --summary-dir outputs/multidoc/qwen
  python pipeline/c_stage1_grounding.py --setting singledoc --summary-dir outputs/singledoc/qwen
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from common import GOLD_BULLETS, get_client, load_json, parse_summary_bullets, save_json

EMBED_MODEL = "qwen/qwen3-embedding-8b"
EMBED_BATCH_SIZE = 32
GROUNDED_THRESHOLD = 0.62
TOP_K_EVIDENCE = 5


# ============================================================
# Embeddings
# ============================================================

class Embedder:
    """L2-normalized embeddings with an in-memory cache keyed by stripped text."""

    def __init__(self, client, model: str = EMBED_MODEL, batch_size: int = EMBED_BATCH_SIZE):
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.cache: Dict[str, np.ndarray] = {}

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts, encoding_format="float")
        arr = np.array([d.embedding for d in sorted(resp.data, key=lambda x: x.index)], dtype=np.float32)
        return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12, None)

    def encode(self, texts: List[str]) -> np.ndarray:
        keys = [(t or "").strip() for t in texts]
        missing = list(dict.fromkeys(k for k in keys if k not in self.cache))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start: start + self.batch_size]
            for key, emb in zip(batch, self._embed_batch(batch)):
                self.cache[key] = emb
        return np.vstack([self.cache[k] for k in keys]).astype(np.float32)


def embedding_grounding(embedder: Embedder, gold_bullets: List[str], bullets: List[str],
                        threshold: float) -> List[Dict[str, Any]]:
    if not bullets:
        return []
    if not gold_bullets:
        return [{"index": i, "label": "UNGROUNDED", "best_emb": 0.0, "best_text": "",
                 "evidence": [], "evidence_scores": [], "gold_scores": [],
                 "rationale": "no gold bullets available"}
                for i in range(len(bullets))]

    gold_emb = embedder.encode(gold_bullets)
    bullet_emb = embedder.encode(bullets)

    results = []
    for i in range(len(bullets)):
        sims = gold_emb @ bullet_emb[i]
        top_idx = np.argsort(sims)[::-1][:min(TOP_K_EVIDENCE, len(gold_bullets))]
        top_scores = [float(sims[j]) for j in top_idx]
        top_texts = [gold_bullets[j] for j in top_idx]
        best_emb, best_text = top_scores[0], top_texts[0]

        results.append({
            "index": i,
            "label": "GROUNDED" if best_emb >= threshold else "UNGROUNDED",
            "best_emb": round(best_emb, 4),
            "best_text": best_text,
            "evidence": top_texts,
            "evidence_scores": [round(x, 4) for x in top_scores],
            # cosine to every gold bullet (gold order), for one-to-one gold matching in recall
            "gold_scores": [round(float(x), 4) for x in sims],
            "rationale": (
                f"top1={best_emb:.3f}, threshold={threshold}, "
                f"top1_text='{best_text[:80]}', "
                f"top_scores={[round(x, 3) for x in top_scores]}"
            ),
        })
    return results


# ============================================================
# Gold bullets
# ============================================================

def load_gold_map(path) -> Dict[str, Dict[str, Any]]:
    """group_id (multidoc) or query_id (singledoc) -> {query, gold, gold_bullets}."""
    result: Dict[str, Dict[str, Any]] = {}
    for item in load_json(path):
        group_id = str(item.get("group_id", "")).strip()
        if not group_id:
            continue
        result[group_id] = {
            "query": str(item.get("query", "")).strip(),
            "gold": str(item.get("gold", "")).strip(),
            "gold_bullets": [str(x).strip() for x in item.get("gold_bullets", []) if str(x).strip()],
        }
    return result


# ============================================================
# Per-case evaluation
# ============================================================

def evaluate_one_case(embedder: Embedder, summary_path: Path, gold_map: Dict[str, Dict[str, Any]],
                      setting: str, threshold: float, overwrite: bool) -> Optional[Dict[str, Any]]:
    stem = summary_path.name[: -len("_summary.json")]
    out_path = summary_path.parent / f"{stem}_judge_embtop1.json"
    if out_path.exists() and not overwrite:
        return None

    record = load_json(summary_path)
    if setting == "multidoc":
        case_id = str(record.get("group_id", "")).strip()
        summary_bullets = parse_summary_bullets(record.get("summary", ""))
        header = {"group_id": case_id}
    else:
        case_id = str(record.get("query_id", "")).strip() or stem
        summary_bullets = parse_summary_bullets(record.get("summarization_point", ""))
        header = {"query_id": case_id, "source_file": str(record.get("source_file", ""))}

    if not summary_bullets:
        # A valid empty output ({"bullets": []}) is kept with 0 bullets so that the
        # case still counts towards recall; unparseable outputs never reach here (step B).
        print("  [Warn] summary has no bullets")
    gold_info = gold_map.get(case_id)
    if not gold_info:
        raise ValueError(f"No gold bullets for {case_id}")

    grounding = embedding_grounding(embedder, gold_info["gold_bullets"], summary_bullets, threshold)
    bullets = [{"index": i, "text": b, "grounding": g, "category": g["label"]}
               for i, (b, g) in enumerate(zip(summary_bullets, grounding))]

    total = len(bullets)
    grounded = sum(1 for b in bullets if b["category"] == "GROUNDED")
    output = {
        **header,
        "query": gold_info["query"],
        "gold": gold_info["gold"],
        "gold_bullets": gold_info["gold_bullets"],
        "metrics": {
            "bullet_count": total,
            "grounded_count": grounded,
            "ungrounded_count": total - grounded,
            "grounded_rate": grounded / max(1, total),
            "ungrounded_rate": (total - grounded) / max(1, total),
            "threshold": threshold,
        },
        "bullets": bullets,
    }
    save_json(output, out_path)
    print(f"  [Saved] {out_path.name} | grounded={grounded} ungrounded={total - grounded}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", choices=["multidoc", "singledoc"], required=True)
    parser.add_argument("--summary-dir", required=True, help="folder with *_summary.json from step B")
    parser.add_argument("--threshold", type=float, default=GROUNDED_THRESHOLD)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    embedder = Embedder(get_client())
    gold_map = load_gold_map(GOLD_BULLETS[args.setting])
    summary_files = sorted(Path(args.summary_dir).glob("*_summary.json"))
    print(f"Stage 1 | {len(summary_files)} files | embed={EMBED_MODEL} | top1 >= {args.threshold} -> GROUNDED\n")

    done = skipped = 0
    errors = []
    for i, path in enumerate(summary_files, start=1):
        print(f"[{i}/{len(summary_files)}] {path.name}")
        try:
            if evaluate_one_case(embedder, path, gold_map, args.setting, args.threshold, args.overwrite) is None:
                skipped += 1
            else:
                done += 1
        except Exception as e:
            print(f"  [Error] {e}")
            errors.append((path.name, str(e)))

    print(f"\nDone={done}  Skipped={skipped}  Errors={len(errors)}")
    for name, err in errors:
        print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
