#!/usr/bin/env python3

import argparse
import json
import time
import random
from pathlib import Path

from gptcache.adapter import openai
from gptcache import cache, Config
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.embedding import Onnx as EmbeddingOnnx
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from gptcache.similarity_evaluation.onnx import OnnxModelEvaluation

import logging
logging.getLogger().setLevel(logging.INFO)

class WrapEvaluation(OnnxModelEvaluation):

    def __init__(self):
        super().__init__()

    def evaluation(self, src_dict, cache_dict, **kwargs):
        """Evaluate the similarity score of pair.

        :param src_dict: the query dictionary to evaluate with cache.
        :type src_dict: Dict
        :param cache_dict: the cache dictionary.
        :type cache_dict: Dict

        :return: evaluation score.
        """
        try:
            src_question = src_dict["question"]
            cache_question = cache_dict["question"]
            if src_question.lower() == cache_question.lower():
                return 1
            res = self.inference(src_question, [cache_question])
            return res
        except Exception:  # pylint: disable=W0703
            return 0
    # def range(self):
    #     return 0.0, 1.0

def run(args):
    # 1. Load dataset
    with open(args.test_data, "r") as fp:
        full_data = json.load(fp)

    # Assign IDs
    for idx, pair in enumerate(full_data):
        pair["id"] = str(idx)

    # Optional truncation
    if args.max_points > 0:
        full_data = full_data[: args.max_points]

    # 2. Initialize GPTCache
    embedding_onnx = EmbeddingOnnx()
    sqlite_file = "sqlite.db"
    faiss_file = "faiss.index"
    has_data = Path(sqlite_file).is_file() and Path(faiss_file).is_file()

    cache_base = CacheBase("sqlite")
    vector_base = VectorBase("faiss", dimension=embedding_onnx.dimension)
    data_manager = get_data_manager(cache_base, vector_base, max_size=args.cache_size)

    cache.init(
        embedding_func=embedding_onnx.to_embeddings,
        data_manager=data_manager,
        similarity_evaluation=SearchDistanceEvaluation(),
        config=Config(similarity_threshold=args.sim_threshold),
    )
    cache.set_openai_key()

    # 3. Insert all data into cache if needed
    if not has_data:
        print("Inserting data into cache …")
        t0 = time.time()
        questions, answers = map(
            list, zip(*((p["origin"], p["id"]) for p in full_data))
        )
        cache.import_data(questions=questions, answers=answers)
        print(f"Finished insert – {time.time() - t0:.2f}s")

    # 4. Prepare test set
    positives = [d for d in full_data if d.get("label", 1) == 1]
    negatives = [d for d in full_data if d.get("label", 1) == 0]
    if args.n_test_samples == 0:
        n_total = len(full_data)
    else:
        n_total = min(args.n_test_samples, len(full_data))
    n_pos = int(n_total * args.sample)
    n_neg = n_total - n_pos
    if n_pos > len(positives) or n_neg > len(negatives):
        print("⚠️  Not enough positive/negative examples to match requested ratio. Using random sampling.")
        sampled_data = random.sample(full_data, n_total)
    else:
        sampled_data = random.sample(positives, n_pos) + random.sample(negatives, n_neg)
        random.shuffle(sampled_data)

    print(f"Testing on {len(sampled_data)} queries "
          f"({len([d for d in sampled_data if d.get('label') == 1])} positive, "
          f"{len([d for d in sampled_data if d.get('label') == 0])} negative)")

    # 5. Query loop
    total_time = 0.0
    hit_pos, hit_neg, cache_miss, fail = 0, 0, 0, 0
    results = []

    for pair in sampled_data:
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
            elapsed = time.time() - t0
            total_time += elapsed

            answer = openai.get_message_from_openai_answer(res)

            status = ""
            if res.get("gptcache", False) == False:
                cache_miss += 1
                is_hit = False
                status = "cache miss"
                print("Cache miss")
            else:
                breakpoint()
                is_hit = (answer == pair["id"])
                if is_hit:
                    hit_pos += 1
                    status = "positive cache hit"
                    print("Cache hit")
                else:
                    hit_neg += 1
                    status = "negative cache hit"
                    print("Neg hit")

            results.append({
                "id": pair["id"],
                "query": pair["similar"],
                "hit": is_hit,
                "status": status,
                "label": pair.get("label", None),
                "latency_sec": round(elapsed, 4),
            })
            print(f"Query latency: {elapsed:.2f}s")

        except Exception as e:
            fail += 1
            results.append({
                "id": pair.get("id", "N/A"),
                "query": pair.get("similar", ""),
                "hit": False,
                "latency_sec": None,
                "label": pair.get("label", None),
                "error": str(e),
            })
            print("⚠️  request failed:", e)

    # 6. Summary
    n_queries = len(sampled_data) if sampled_data else 1
    summary = {
        "n_queries": n_queries,
        "cache_hit_positive": hit_pos,
        "cache_hit_negative": hit_neg,
        "cache_misses": cache_miss,
        "fail_count": fail,
        "avg_latency_sec": round(total_time / n_queries, 4),
        "avg_embedding_time_sec": round(cache.report.average_embedding_time(), 4),
        "avg_search_time_sec": round(cache.report.average_search_time(), 4),
        "similarity_threshold": args.sim_threshold,
        "cache_size": args.cache_size,
        "max_points": args.max_points,
        "n_test_samples": args.n_test_samples,
        "sample_frac_positive": args.sample,
        "results": results,
    }

    print(f"\n=== Benchmark summary ===")
    for k, v in summary.items():
        if k != "results":
            print(f"{k:30s}: {v}")

    output_path = Path(args.output).resolve()
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n📁 Results saved to {output_path}")

import ipdb
if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        parser = argparse.ArgumentParser(
            description="GPTCache benchmark with sampled QQP-style test set"
        )
        parser.add_argument("--test_data", type=str, required=True, help="Path to dataset JSON file")
        parser.add_argument("--cache_size", type=int, default=10_000, help="Cache capacity")
        parser.add_argument("--sim_threshold", type=float, default=0.70, help="Similarity threshold [0,1]")
        parser.add_argument("--max_points", type=int, default=0, help="Max records to load from dataset (0 = all)")
        parser.add_argument("--sample", type=float, default=0.5, help="Fraction of positives in the test set (∈ [0,1])")
        parser.add_argument("--n_test_samples", type=int, default=1160, help="Exact number of test queries to sample")
        parser.add_argument("--output", type=str, default="gptcache_benchmark_results.json", help="JSON output path")

        run(parser.parse_args())
