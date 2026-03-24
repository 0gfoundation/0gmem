"""Trace where target messages are lost in the retrieval pipeline.

Usage:
    PYTHONPATH=src python3 scripts/trace_pipeline.py
"""

import json
import sys

from openai import OpenAI

from zerogmem.evaluation.locomo import LoCoMoEvaluator

# (question, keywords_expected_in_answer_message, conv_id)
TRACE_QUESTIONS = [
    ("What personality traits might Melanie say Caroline has?",
     ["thoughtful", "authentic", "real", "driven", "drive"], "conv-26"),
]


def main():
    llm_client = OpenAI()

    # Group by conv
    by_conv: dict[str, list] = {}
    for q, kw, cid in TRACE_QUESTIONS:
        by_conv.setdefault(cid, []).append((q, kw))

    for conv_id, questions in by_conv.items():
        print(f"\n{'='*70}")
        print(f"Loading {conv_id}...")
        print(f"{'='*70}")

        evaluator = LoCoMoEvaluator(
            llm_client=llm_client,
            llm_model="gpt-5.2",
            use_cache=True,
            use_bm25=True,
        )
        evaluator.load_dataset("data/locomo/locomo10.json")

        # Find and ingest the right conversation
        conv = evaluator.conversations.get(conv_id)
        if not conv:
            print(f"  Conversation {conv_id} not found!")
            continue

        evaluator._reset_memory()
        evaluator.ingest_conversation(conv)
        sys.stdout.flush()

        for question_text, keywords in questions:
            print(f"\n{'─'*70}")
            print(f"Q: {question_text}")
            print(f"Keywords: {keywords}")
            print(f"{'─'*70}")
            sys.stdout.flush()

            trace = evaluator.retriever.trace_retrieval(question_text, keywords)

            for stage in [
                "per_strategy",
                "post_rrf",
                "post_entity_scoring",
                "post_cross_encoder",
                "post_llm_rerank",
                "post_attention_filter",
                "final_top_k",
            ]:
                info = trace[stage]
                if stage == "per_strategy":
                    print(f"\n  [{stage}]")
                    for sname, sinfo in info.items():
                        matches = sinfo["matches"]
                        if matches:
                            print(f"    {sname} ({sinfo['total']} total): "
                                  f"{len(matches)} match(es)")
                            for m in matches[:3]:
                                print(f"      rank={m['rank']} score={m['score']:.4f} "
                                      f"[{m['source']}] {m['content'][:80]}")
                        else:
                            print(f"    {sname} ({sinfo['total']} total): no matches")
                else:
                    matches = info["matches"]
                    tag = f"{len(matches)} match(es)" if matches else "NO MATCHES"
                    print(f"\n  [{stage}] ({info['total']} total): {tag}")
                    for m in matches[:3]:
                        print(f"    rank={m['rank']} score={m['score']:.4f} "
                              f"[{m['source']}] {m['content'][:80]}")
            sys.stdout.flush()

    evaluator.save_cache()


if __name__ == "__main__":
    main()
