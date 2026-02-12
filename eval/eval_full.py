#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full evaluation script for RAG assistant.

Measures:
- Refusal accuracy (should_refuse vs did_refuse)
- Answer rate (non-refusal when should_answer)
- Retrieval coverage gap rate (should_answer but refused)
- Citation rate
- Latency (avg/median/p95)
- Score per tag
- Global score

Important:
This script does NOT measure legal correctness. It measures behavior + robustness.
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


# Better refusal patterns (more robust)
REFUSAL_PATTERNS = [
    r"\bje ne sais pas\b",
    r"\bje n'ai pas\b.*\bcontexte\b",
    r"\bpas (assez|de) (d'informations|contexte)\b",
    r"\bje ne peux pas\b",
    r"\bimpossible\b.*\b(répondre|déterminer)\b",
    r"\baucun document\b",
    r"\bje ne peux pas répondre\b",
    r"\bje ne peux pas répondre de façon fiable\b",
    r"\bje ne peux pas répondre avec certitude\b",
    r"\bdans la base documentaire\b",
]


def detect_refusal(answer: str) -> bool:
    if not answer:
        return True
    a = answer.strip().lower()
    if len(a) < 15:
        return True
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, a, flags=re.IGNORECASE):
            return True
    return False


def call_api(api_base: str, payload: Dict[str, Any], timeout: int) -> Tuple[int, Dict[str, Any], float, str]:
    url = api_base.rstrip("/") + "/ask"
    t0 = time.perf_counter()
    error = ""

    try:
        r = requests.post(url, json=payload, timeout=timeout)
        latency = (time.perf_counter() - t0) * 1000.0
        try:
            data = r.json()
        except Exception:
            data = {"answer": r.text}
        return r.status_code, data, latency, error

    except requests.exceptions.Timeout as e:
        latency = (time.perf_counter() - t0) * 1000.0
        return 0, {"answer": ""}, latency, f"TIMEOUT: {e}"

    except requests.exceptions.RequestException as e:
        latency = (time.perf_counter() - t0) * 1000.0
        return 0, {"answer": ""}, latency, f"REQUEST_ERROR: {e}"


def extract_answer_and_citations(data: Dict[str, Any]) -> Tuple[str, List[str]]:
    answer = data.get("answer") or data.get("text") or data.get("response") or ""
    citations: List[str] = []

    if isinstance(data.get("citations"), list):
        citations = [str(x) for x in data["citations"]]

    # some APIs may include "context"/"sources"
    if isinstance(data.get("sources"), list):
        citations.extend([str(x) for x in data["sources"]])

    if isinstance(data.get("context"), list):
        for c in data["context"]:
            if isinstance(c, dict):
                citations.append(str(c.get("source") or c.get("id") or c.get("title") or ""))
            else:
                citations.append(str(c))

    # dedupe
    citations = [c for c in dict.fromkeys(citations) if c]
    return answer, citations


def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[max(0, min(k, len(s) - 1))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_base", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out_csv", default="", help="Optional CSV output path")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    out_csv = Path(args.out_csv) if args.out_csv else None
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    refusal_expected = 0
    refusal_correct = 0

    answer_expected = 0
    answered = 0
    coverage_gap = 0  # should_answer but refused OR no context

    citation_ok = 0

    timeouts_or_errors = 0
    latencies: List[float] = []

    per_tag_stats: Dict[str, Dict[str, int]] = {}

    # CSV setup
    csv_writer = None
    csv_file = None
    if out_csv:
        csv_file = out_csv.open("w", encoding="utf-8", newline="")
        fieldnames = [
            "id","question","tags","should_refuse","http_status","latency_ms",
            "did_refuse","refusal_ok","citations_count","error","answer_preview"
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            total += 1

            qid = item.get("id", f"Q{total:03d}")
            should_refuse = bool(item.get("should_refuse", False))
            tags = item.get("tags", [])
            question = item["question"]

            payload = {"question": question, "filters": item.get("filters", {})}

            status, data, latency, error = call_api(args.api_base, payload, args.timeout)
            latencies.append(latency)
            if error or status == 0 or status >= 500:
                timeouts_or_errors += 1

            answer, citations = extract_answer_and_citations(data)
            did_refuse = detect_refusal(answer)

            # init tag stats
            for tag in tags:
                per_tag_stats.setdefault(tag, {"total": 0, "correct": 0})
                per_tag_stats[tag]["total"] += 1

            ok = (should_refuse and did_refuse) or ((not should_refuse) and (not did_refuse))
            for tag in tags:
                if ok:
                    per_tag_stats[tag]["correct"] += 1

            if should_refuse:
                refusal_expected += 1
                if did_refuse:
                    refusal_correct += 1
            else:
                answer_expected += 1
                if not did_refuse:
                    answered += 1
                    if citations:
                        citation_ok += 1
                else:
                    # not a hallucination: it's a coverage/retrieval gap
                    coverage_gap += 1

            if csv_writer:
                csv_writer.writerow({
                    "id": qid,
                    "question": question,
                    "tags": ",".join(tags),
                    "should_refuse": should_refuse,
                    "http_status": status,
                    "latency_ms": round(latency, 1),
                    "did_refuse": did_refuse,
                    "refusal_ok": 1 if ok else 0,
                    "citations_count": len(citations),
                    "error": error,
                    "answer_preview": (answer or "")[:250].replace("\n", " "),
                })

    if csv_file:
        csv_file.close()

    refusal_acc = 100 * refusal_correct / refusal_expected if refusal_expected else 0.0
    answer_rate = 100 * answered / answer_expected if answer_expected else 0.0
    coverage_gap_rate = 100 * coverage_gap / answer_expected if answer_expected else 0.0
    citation_rate = 100 * citation_ok / answered if answered else 0.0

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    med_latency = percentile(latencies, 50)
    p95_latency = percentile(latencies, 95)

    # Global score (simple & explainable)
    # - refusal accuracy is important (safety)
    # - answer_rate is coverage (utility)
    # - citation_rate is trust
    # - penalize errors/timeouts and coverage gaps
    error_rate = 100 * timeouts_or_errors / total if total else 0.0

    global_score = (
        0.35 * refusal_acc +
        0.35 * answer_rate +
        0.20 * citation_rate +
        0.10 * (100 - error_rate)
    )

    print("\n=== ÉVALUATION COMPLÈTE ===")
    print(f"Total questions: {total}")
    print(f"Refusal accuracy (should_refuse): {refusal_acc:.1f}% ({refusal_correct}/{refusal_expected})")
    print(f"Answer rate (should_answer): {answer_rate:.1f}% ({answered}/{answer_expected})")
    print(f"Coverage gap rate (should_answer but refused): {coverage_gap_rate:.1f}% ({coverage_gap}/{answer_expected})")
    print(f"Citation rate (among answered): {citation_rate:.1f}% ({citation_ok}/{answered})")
    print(f"Errors/Timeouts: {error_rate:.1f}% ({timeouts_or_errors}/{total})")

    print("\n=== LATENCE ===")
    print(f"Average latency: {avg_latency:.0f} ms")
    print(f"Median latency:  {med_latency:.0f} ms")
    print(f"P95 latency:     {p95_latency:.0f} ms")

    print(f"\nGLOBAL SCORE: {global_score:.1f} / 100")

    print("\n=== SCORE PAR CATÉGORIE ===")
    for tag, stats in sorted(per_tag_stats.items(), key=lambda x: x[0]):
        score = 100 * stats["correct"] / stats["total"] if stats["total"] else 0.0
        print(f"{tag}: {score:.1f}% ({stats['correct']}/{stats['total']})")

    print("\n=== DIAGNOSTIC ===")
    if global_score > 85:
        print("🔥 Système très solide (niveau production/proche)")
    elif global_score > 70:
        print("✅ Bon prototype académique (robuste)")
    elif global_score > 55:
        print("⚠️ Correct mais améliorable (couverture/docs/retrieval)")
    else:
        print("🚨 Fragile (erreurs/timeouts ou faible couverture)")


if __name__ == "__main__":
    main()
