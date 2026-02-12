#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run 100 questions against your RAG API and export results to CSV.

Usage:
  python -m eval.run_100 \
    --api_base http://localhost:8000 \
    --input_jsonl data/samples/100_questions_rag_mistral.jsonl \
    --out_csv data/logs/run_100_results.csv

Optional:
  --token "Bearer XXX"  (or just "XXX", script will add Bearer)
  --timeout 60
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


REFUSAL_PATTERNS = [
    r"\bje ne sais pas\b",
    r"\bje n'ai pas\b.*\bcontexte\b",
    r"\bpas (assez|de) (d'informations|contexte)\b",
    r"\bje ne peux pas\b",
    r"\bimpossible\b.*\b(répondre|déterminer)\b",
    r"\bje n'ai pas trouvé\b",
    r"\baucun document\b",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--api_base", required=True, help="Base URL, e.g. http://localhost:8000")
    p.add_argument("--input_jsonl", required=True, help="Path to JSONL questions file")
    p.add_argument("--out_csv", required=True, help="Path to output CSV")
    p.add_argument("--token", default="", help="Auth token (optional). Use 'Bearer xxx' or 'xxx'.")
    p.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")
    p.add_argument("--sleep_ms", type=int, default=0, help="Sleep between requests (ms)")
    return p.parse_args()


def norm_api_base(api_base: str) -> str:
    return api_base.rstrip("/")


def build_headers(token: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = (token or "").strip()
    if token:
        if not token.lower().startswith("bearer "):
            token = "Bearer " + token
        headers["Authorization"] = token
    return headers


def detect_refusal(answer: str) -> bool:
    """Heuristic: detect refusal-like answers."""
    if not answer:
        return True
    a = answer.strip().lower()
    if len(a) < 10:
        return True
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, a, flags=re.IGNORECASE):
            return True
    return False


def extract_answer_and_citations(resp_json: Any) -> Tuple[str, List[str]]:
    """
    Try to be robust to different API formats.
    Supported-ish:
      - {"answer": "...", "citations": [...]}
      - {"text": "...", "sources": [...]}
      - {"answer": "...", "context": [{"source":...}, ...]}
      - plain string
    """
    if isinstance(resp_json, str):
        return resp_json, []

    if not isinstance(resp_json, dict):
        return json.dumps(resp_json, ensure_ascii=False), []

    answer = resp_json.get("answer") or resp_json.get("text") or resp_json.get("response") or ""
    citations: List[str] = []

    if isinstance(resp_json.get("citations"), list):
        citations = [str(x) for x in resp_json["citations"]]
    elif isinstance(resp_json.get("sources"), list):
        citations = [str(x) for x in resp_json["sources"]]
    elif isinstance(resp_json.get("context"), list):
        # context might be list of dicts
        for c in resp_json["context"]:
            if isinstance(c, dict):
                citations.append(str(c.get("source") or c.get("id") or c.get("title") or c))
            else:
                citations.append(str(c))

    return answer, citations


def call_ask(api_base: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int) -> Tuple[int, Any, float]:
    url = norm_api_base(api_base) + "/ask"
    t0 = time.perf_counter()
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # parse json if possible
    content_type = (r.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        try:
            return r.status_code, r.json(), dt_ms
        except Exception:
            return r.status_code, r.text, dt_ms
    else:
        # some APIs return text
        return r.status_code, r.text, dt_ms


def main():
    args = parse_args()
    api_base = args.api_base
    headers = build_headers(args.token)

    in_path = Path(args.input_jsonl)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    total = 0
    refuse_expected = 0
    refuse_ok = 0
    answered_expected = 0
    answered_ok = 0
    citation_present_when_answer = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            total += 1

            qid = item.get("id", f"Q{total:03d}")
            question = item["question"]
            filters = item.get("filters", {})
            should_refuse = bool(item.get("should_refuse", False))
            tags = ",".join(item.get("tags", []))

            payload = {
                "question": question,
                "filters": filters,
            }

            status, body, latency_ms = call_ask(api_base, headers, payload, args.timeout)

            if status >= 400:
                answer = ""
                citations = []
                error = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
                did_refuse = True  # treat errors as refusals for scoring, but keep error visible
            else:
                error = ""
                answer, citations = extract_answer_and_citations(body)
                did_refuse = detect_refusal(answer)

            # scoring for refusal behaviour
            if should_refuse:
                refuse_expected += 1
                ok = 1 if did_refuse else 0
                refuse_ok += ok
            else:
                answered_expected += 1
                ok = 1 if (not did_refuse) else 0
                answered_ok += ok
                if citations:
                    citation_present_when_answer += 1

            rows.append({
                "id": qid,
                "question": question,
                "tags": tags,
                "should_refuse": should_refuse,
                "did_refuse_detected": did_refuse,
                "refusal_ok": (1 if should_refuse and did_refuse else 0) if should_refuse else (1 if (not should_refuse and not did_refuse) else 0),
                "http_status": status,
                "latency_ms": round(latency_ms, 1),
                "citations_count": len(citations),
                "citations": " | ".join(citations)[:2000],
                "answer": (answer or "")[:4000],
                "error": (error or "")[:2000],
            })

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

    # write csv
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # summary
    def pct(a: int, b: int) -> float:
        return (100.0 * a / b) if b else 0.0

    print("=== Résumé run_100 ===")
    print(f"Total questions: {total}")
    print(f"Questions 'should_refuse': {refuse_expected} | refus OK: {refuse_ok} ({pct(refuse_ok, refuse_expected):.1f}%)")
    print(f"Questions 'should_answer': {answered_expected} | réponses OK (non-refus): {answered_ok} ({pct(answered_ok, answered_expected):.1f}%)")
    print(f"Réponses (non-refus) avec citations: {citation_present_when_answer}/{answered_ok} ({pct(citation_present_when_answer, answered_ok):.1f}%)")
    print(f"CSV écrit: {out_path}")


if __name__ == "__main__":
    main()
