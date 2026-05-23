from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _recall_at_k(relevant_pages, retrieved_pages):
    if not relevant_pages:
        return 1.0
    hits = sum(1 for p in relevant_pages if p in retrieved_pages)
    return hits / len(relevant_pages)


def _answer_contains_truth(answer, ground_truth):
    answer_lower = answer.lower()
    truth_tokens = ground_truth.lower().split()
    return sum(1 for t in truth_tokens if t in answer_lower) / len(truth_tokens) >= 0.6


def run_eval(pdf_path, qa_path, top_k=4):
    from src.pipeline import RAGPipeline
    import asyncio

    pipeline = RAGPipeline()

    logger.info(f"Индексирую {pdf_path} ...")
    t0 = time.time()
    asyncio.run(pipeline.index(pdf_path))
    index_time = time.time() - t0
    logger.info(f"Индексирование: {index_time:.1f} сек")

    with open(qa_path, encoding="utf-8") as f:
        qa_pairs = json.load(f)

    results = []
    total_latency = 0.0

    for i, item in enumerate(qa_pairs):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        relevant_pages = item.get("relevant_pages", [])

        t0 = time.time()
        answer, sources = asyncio.run(pipeline.ask(question))
        latency = time.time() - t0
        total_latency += latency

        retrieved_pages = [s.page for s in sources]
        recall = _recall_at_k(relevant_pages, retrieved_pages)
        contains = _answer_contains_truth(answer, ground_truth) if ground_truth else None

        result = {
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "retrieved_pages": retrieved_pages,
            "recall@k": recall,
            "answer_match": contains,
            "latency_sec": round(latency, 2),
        }
        results.append(result)
        logger.info(
            f"[{i+1}/{len(qa_pairs)}] recall={recall:.2f} match={contains} latency={latency:.2f}s | {question[:60]}"
        )

    avg_recall = sum(r["recall@k"] for r in results) / len(results)
    match_results = [r["answer_match"] for r in results if r["answer_match"] is not None]
    avg_match = sum(match_results) / len(match_results) if match_results else None
    avg_latency = total_latency / len(results)

    summary = {
        "index_time_sec": round(index_time, 1),
        "num_questions": len(results),
        "avg_recall_at_k": round(avg_recall, 3),
        "avg_answer_match": round(avg_match, 3) if avg_match is not None else None,
        "avg_latency_sec": round(avg_latency, 2),
        "results": results,
    }

    out_path = Path("data/eval_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=== ИТОГ ===")
    logger.info(f"Recall@{top_k}:     {avg_recall:.3f}")
    if avg_match is not None:
        logger.info(f"Answer match: {avg_match:.3f}")
    logger.info(f"Avg latency:  {avg_latency:.2f} сек")
    logger.info(f"Результаты → {out_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    run_eval(args.pdf, args.qa, args.top_k)