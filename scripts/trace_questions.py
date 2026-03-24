"""Trace specific questions to verify fixes and check for regressions."""
import logging, os, sys
import openai
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.basicConfig(level=logging.WARNING)
logging.getLogger("zerogmem.retriever.query_planner").setLevel(logging.DEBUG)

from zerogmem.evaluation.locomo import LoCoMoEvaluator
from zerogmem.memory.manager import MemoryConfig
from zerogmem.retriever.retriever import RetrieverConfig

CONV_ID = "conv-26"
QUESTIONS = [
    ("Q75", "How many children does Melanie have?", "3", "single_hop"),
    ("Q69", "What personality traits might Melanie say Caroline has?", "Thoughtful, authentic, driven", "multi_hop"),
    ("Q178", "Is Oscar Melanie's pet?", "No", "adversarial"),
    ("Q33", "When did Caroline go to a pride parade during the summer?", "The week before 3 July 2023", "temporal"),
    ("Q59", "Would Caroline be considered religious?", "Somewhat, but not extremely religious", "multi_hop"),
    ("Q68", "How long has Melanie been practicing art?", "Since 2016", "temporal"),
    ("Q188", "Did Melanie and Alex attend the same school?", "No", "adversarial"),
    ("Q133", "What precautionary sign did Melanie see at the cafe?", "A sign stating that someone is not being able to leave", "open_domain"),
]

def main():
    llm_client = openai.OpenAI()
    evaluator = LoCoMoEvaluator(
        data_path="data/locomo/locomo10.json",
        llm_client=llm_client, use_bm25=True, use_cache=True,
        retriever_config=RetrieverConfig(use_query_planner=True),
        memory_config=MemoryConfig(use_llm_fact_extraction=True),
        ingestion_cache_dir=".cache/ingestion",
    )
    evaluator.load_dataset()
    conv = evaluator.conversations[CONV_ID]
    print("Loading/ingesting...")
    evaluator.ingest_or_load_cache(conv)
    evaluator._conversation_year = evaluator._extract_conversation_year(conv)
    print(f"Done. {len(evaluator.memory.graph.memories)} memories\n")

    results = []
    for qid, question, gold, category in QUESTIONS:
        print(f"{'='*60}")
        print(f"{qid}: {question}")
        print(f"GOLD: {gold}")
        print(f"{'='*60}")

        answer, meta = evaluator.retriever.answer(question, category=category)
        context = meta.get("context", "")
        print(f"\nCONTEXT ({len(context)} chars):")
        print(context[:2000])
        print(f"\nANSWER: '{answer}'")
        print(f"GOLD:   '{gold}'")

        # Score using same logic as full eval (F1 >= 0.4, then LLM judge fallback)
        f1 = evaluator._compute_f1(answer, gold)
        is_correct = f1 >= 0.4
        llm_judged = False
        if not is_correct and evaluator.llm_client:
            llm_judged = evaluator._llm_judge(question, answer, gold)
            if llm_judged:
                is_correct = True

        status = "PASS" if is_correct else "FAIL"
        suffix = f"  (F1={f1:.2f})" + (" [LLM judge]" if llm_judged else "")
        print(f"STATUS: {status}{suffix}")
        results.append((qid, status, answer, gold, f1))
        print()

    print("=" * 60)
    print("SUMMARY:")
    for qid, status, answer, gold, f1 in results:
        print(f"  {qid}: {status}  F1={f1:.2f}  (answer='{answer[:50]}', gold='{gold}')")
    correct = sum(1 for _, s, *_ in results if s == "PASS")
    print(f"\n  {correct}/{len(results)} correct")

if __name__ == "__main__":
    main()
