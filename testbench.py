#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import json
import os
import time
from pathlib import Path

from gptcache.adapter import openai
from gptcache import cache, Config
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.onnx import OnnxModelEvaluation  # noqa: F401 (kept for reference)
from gptcache.embedding import Onnx as EmbeddingOnnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation


class WrapEvaluation(SearchDistanceEvaluation):
    def evaluation(self, src_dict, cache_dict, **kwargs):
        return super().evaluation(src_dict, cache_dict, **kwargs)

    def range(self):
        return super().range()

def run(args):
    # --------------------------------------------------
    # 1  Load / truncate mock dataset
    # --------------------------------------------------
    with open(args.test_data, "r") as fp:
        mock_data = json.load(fp)

    if args.max_points > 0:
        mock_data = mock_data[: args.max_points]

    # --------------------------------------------------
    # 2  Build cache manager
    # --------------------------------------------------
    embedding_onnx = EmbeddingOnnx()
    sqlite_file = "sqlite.db"
    faiss_file = "faiss.index"
    has_data = Path(sqlite_file).is_file() and Path(faiss_file).is_file()

    cache_base   = CacheBase("sqlite")
    vector_base  = VectorBase("faiss", dimension=embedding_onnx.dimension)
    data_manager = get_data_manager(cache_base, vector_base,
                                    max_size=args.cache_size)

    cache.init(
        embedding_func=embedding_onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=WrapEvaluation(),
        config=Config(similarity_threshold=args.sim_threshold),
    )
    cache.set_openai_key()

    # Attach IDs to dataset rows ---------------------------------------------
    for idx, pair in enumerate(mock_data):
        pair["id"] = str(idx)

    # --------------------------------------------------
    # 3  Populate cache once
    # --------------------------------------------------
    if not has_data:
        print("Inserting data into cache …")
        t0 = time.time()
        questions, answers = map(
            list, zip(*((p["origin"], p["id"]) for p in mock_data))
        )
        cache.import_data(questions=questions, answers=answers)
        print(f"Finished insert – {time.time() - t0:.2f}s")

    # --------------------------------------------------
    # 4  Query loop
    # --------------------------------------------------
    total_time = 0.0
    hit_pos, hit_neg, fail = 0, 0, 0

    for pair in mock_data:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": pair["similar"]},
        ]
        try:
            t0 = time.time()
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
            )
            answer = openai.get_message_from_openai_answer(res)
            if answer == pair["id"]:
                hit_pos += 1
            else:
                hit_neg += 1
            elapsed = time.time() - t0
            total_time += elapsed
            print(f"Query latency: {elapsed:.2f}s")
        except Exception as e:
            fail += 1
            print("⚠️  request failed:", e)

    # --------------------------------------------------
    # 5  Report
    # --------------------------------------------------
    n_queries = len(mock_data) if len(mock_data) else 1
    print(f"\n=== Benchmark summary ({n_queries} queries) ===")
    print(f"avg latency:        {total_time / n_queries:.2f}s")
    print(f"cache hits (👍):     {hit_pos}")
    print(f"cache misses (👎):   {hit_neg}")
    print(f"failures:           {fail}")
    print(f"avg embedding time: {cache.report.average_embedding_time():.4f}s")
    print(f"avg search time:    {cache.report.average_search_time():.4f}s")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPTCache benchmark for QQP-style similarity data"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=(    Path(__file__).resolve().parent
                / "GPTCache"
                / "examples"
                / "benchmark"
                / "similiar_qqp.json"),
        help="Path to test-set JSON file",
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=10_000,
        help="Maximum records kept in the SQLite/Faiss cache",
    )
    parser.add_argument(
        "--sim_threshold",
        type=float,
        default=0.70,
        help="Similarity threshold ∈ [0,1] used by GPTCache",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="Number of datapoints to test (0 = all)",
    )

    run(parser.parse_args())