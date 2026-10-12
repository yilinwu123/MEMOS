"""Shared helpers: OpenRouter client, JSON I/O, QMSum loading, context building."""

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_ROOT = REPO_ROOT / "outputs"
DEFAULT_QMSUM_ROOT = Path(os.getenv("QMSUM_ROOT", str(DATA_DIR / "qmsum")))

# Benchmark instances and gold bullets of each setting.
#   multidoc : queries.json lists the clusters (group_id, query_list, merged_query)
#   singledoc: run_cases_50.csv lists (case, index) = (QMSum meeting, specific-query index)
CASES = {
    "multidoc": DATA_DIR / "multidoc" / "queries.json",
    "singledoc": DATA_DIR / "singledoc" / "run_cases_50.csv",
}
GOLD_BULLETS = {
    "multidoc": DATA_DIR / "multidoc" / "gold_bullets.json",
    "singledoc": DATA_DIR / "singledoc" / "gold_bullets.json",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ============================================================
# OpenRouter client
# ============================================================

def get_client():
    """All LLM / embedding calls go through OpenRouter's OpenAI-compatible API."""
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Please set the OPENROUTER_API_KEY environment variable.")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


# ============================================================
# JSON I/O
# ============================================================

def load_json(path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path) -> None:
    d = os.path.dirname(str(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_json_load(s: str) -> Any:
    """Parse a JSON string, tolerating a surrounding ```json ... ``` fence."""
    s = (s or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return json.loads(s)


# ============================================================
# IDs
#   query_id : "<doc_id>_sq_<i>"  -> i-th entry of the doc's specific_query_list
#   group_id : query_ids joined by "|", e.g. "IS1003b_sq_6|IS1003d_sq_0"
# ============================================================

def sanitize_group_id(group_id: str) -> str:
    """group_id -> file-name stem, e.g. "A_sq_1|B_sq_0" -> "A_sq_1__B_sq_0"."""
    name = group_id.replace("|", "__")
    return re.sub(r'[\\/:*?"<>]+', "_", name)


def split_query_id(query_id: str) -> Tuple[str, int]:
    if "_sq_" not in query_id:
        raise ValueError(f"Invalid query_id: {query_id}")
    doc_id, idx = query_id.rsplit("_sq_", 1)
    return doc_id, int(idx)


def doc_ids_from_group_id(group_id: str) -> List[str]:
    """Distinct document IDs of a group, in order of first appearance."""
    doc_ids: List[str] = []
    for part in group_id.split("|"):
        doc_id, _ = split_query_id(part)
        if doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


def read_cases_csv(csv_path) -> List[Tuple[str, int]]:
    """Read a single-doc case list (columns: case,index) -> [(doc_id, sq_index), ...]."""
    cases = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2 or row[0].startswith("#"):
                continue
            idx_raw = row[1].strip()
            if not idx_raw.lstrip("-").isdigit():  # header
                continue
            cases.append((row[0].strip(), int(idx_raw)))
    return cases


# ============================================================
# QMSum
# ============================================================

class QMSum:
    """Index of QMSum meeting files (<doc_id>.json), searched recursively.

    If the same doc_id appears more than once (e.g. both under ALL/ and a
    domain folder), the shallowest path wins.
    """

    def __init__(self, root=DEFAULT_QMSUM_ROOT):
        self.root = Path(root)
        self.paths: Dict[str, Path] = {}
        for path in sorted(self.root.rglob("*.json"), key=lambda p: (len(p.parts), str(p))):
            self.paths.setdefault(path.stem, path)
        if not self.paths:
            raise SystemExit(
                f"No QMSum meeting files found under {self.root}. "
                "See README.md (Setup) for how to obtain QMSum."
            )
        self._cache: Dict[str, Dict[str, Any]] = {}

    def source_file(self, doc_id: str) -> str:
        return self.path(doc_id).relative_to(self.root).as_posix()

    def path(self, doc_id: str) -> Path:
        if doc_id not in self.paths:
            raise FileNotFoundError(f"Cannot find {doc_id}.json under {self.root}")
        return self.paths[doc_id]

    def meeting(self, doc_id: str) -> Dict[str, Any]:
        if doc_id not in self._cache:
            self._cache[doc_id] = load_json(self.path(doc_id))
        return self._cache[doc_id]

    def transcript_text(self, doc_id: str) -> str:
        return transcript_list_to_text(self.meeting(doc_id).get("meeting_transcripts", []))

    def specific_query(self, query_id: str) -> Dict[str, str]:
        """{"query": ..., "answer": ...} for a query_id."""
        doc_id, idx = split_query_id(query_id)
        sql = self.meeting(doc_id).get("specific_query_list", [])
        if not 0 <= idx < len(sql):
            raise IndexError(f"{query_id}: index out of range (len={len(sql)})")
        item = sql[idx]
        return {
            "query": str(item.get("query", "")).strip(),
            "answer": str(item.get("answer", "")).strip(),
        }

    def gold_answer(self, group_id: str) -> str:
        """Merged gold answer of a group: the members' QMSum answers joined by newlines."""
        answers = [self.specific_query(q)["answer"] for q in group_id.split("|")]
        return "\n".join(a for a in answers if a).strip()


def transcript_list_to_text(meeting_transcripts: List[Dict[str, Any]]) -> str:
    """[{"speaker": "Grad C", "content": "Nice ."}, ...] -> "Grad C: Nice .\\n..." """
    lines = []
    for turn in meeting_transcripts:
        speaker = str(turn.get("speaker", "")).strip()
        content = str(turn.get("content", "")).strip()
        if speaker and content:
            lines.append(f"{speaker}: {content}")
        elif content:
            lines.append(content)
    return "\n".join(lines)


def build_multidoc_context(qmsum: QMSum, doc_ids: List[str]) -> str:
    """Concatenate the transcripts of a group, each headed by its index and document ID,
    with a "=====" separator line between consecutive documents."""
    chunks = []
    for i, doc_id in enumerate(doc_ids, start=1):
        chunks.append(
            f"This is meeting_transcript {i}.\n"
            f"Document ID: {doc_id}\n\n"
            f"{qmsum.transcript_text(doc_id)}"
        )
    return ("\n\n" + "=" * 100 + "\n\n").join(chunks)


# ============================================================
# Summary parsing
# ============================================================

def is_valid_summary(summary_field: Any) -> bool:
    """True if a raw model output is a JSON object with a "bullets" list (possibly empty)."""
    try:
        obj = safe_json_load(summary_field) if isinstance(summary_field, str) else summary_field
    except Exception:
        return False
    return isinstance(obj, dict) and isinstance(obj.get("bullets"), list)


def parse_summary_bullets(summary_field: Any) -> List[str]:
    """Extract the "point" strings from a model's raw JSON summary output."""
    if summary_field is None:
        return []
    if isinstance(summary_field, dict):
        obj = summary_field
    elif isinstance(summary_field, str):
        if not summary_field.strip():
            return []
        try:
            obj = safe_json_load(summary_field)
        except Exception:
            return []
    else:
        return []
    if not isinstance(obj, dict):
        return []

    bullets = []
    for item in obj.get("keypoints") or obj.get("bullets") or []:
        if isinstance(item, dict):
            point = str(item.get("point", "")).strip()
            if point:
                bullets.append(point)
    return bullets
