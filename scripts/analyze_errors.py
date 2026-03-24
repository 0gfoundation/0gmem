#!/usr/bin/env python3
"""Analyze all wrongly answered questions from 10-conv consolidation evaluation."""

import json
import re
from collections import Counter, defaultdict

RESULTS_PATH = "results/10conv_consolidation.json"

def classify_error(q):
    """Classify error type based on question, expected, and predicted answers."""
    question = q["question"].lower()
    expected = q["expected"].lower().strip()
    predicted = q["predicted"].lower().strip()
    category = q["category"]

    # --- Adversarial / Negation failure ---
    if category == "adversarial":
        # Adversarial questions expect "not mentioned" / "unknown" / "unanswerable"
        if any(kw in expected for kw in ["not mentioned", "unknown", "unanswerable", "not enough information"]):
            if not any(kw in predicted for kw in ["not mentioned", "unknown", "unanswerable", "not enough information", "no information", "not specified", "not explicitly"]):
                return "negation_failure"
        return "negation_failure"

    # --- Retrieval failure: system says unknown/not mentioned when info exists ---
    not_found_phrases = [
        "not mentioned", "not explicitly mentioned", "unknown", "unanswerable",
        "not enough information", "no information", "not specified",
        "not explicitly stated", "i don't have", "cannot determine",
        "no specific", "not provided", "not directly mentioned",
        "not clear", "cannot be determined", "there is no", "there's no",
        "does not mention", "doesn't mention", "no mention",
        "not discussed", "not indicated", "not available"
    ]
    if any(phrase in predicted for phrase in not_found_phrases):
        # Make sure expected is actually a real answer
        if not any(phrase in expected for phrase in not_found_phrases):
            return "retrieval_failure"

    # --- Counting error ---
    if any(kw in question for kw in ["how many", "how much", "number of", "count"]):
        # Check if both are numeric
        exp_nums = re.findall(r'\d+', expected)
        pred_nums = re.findall(r'\d+', predicted)
        if exp_nums and pred_nums and exp_nums != pred_nums:
            return "counting_error"
        if exp_nums and not pred_nums:
            return "counting_error"

    # --- Temporal confusion ---
    # Check if both answers have dates/times but they differ
    date_patterns = [
        r'\b\d{4}\b',  # years
        r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        r'\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
    ]
    if category == "temporal":
        exp_years = re.findall(r'\b(20\d{2})\b', expected)
        pred_years = re.findall(r'\b(20\d{2})\b', predicted)
        if exp_years and pred_years and set(exp_years) != set(pred_years):
            return "temporal_confusion"
        # Check month differences
        months = ["january", "february", "march", "april", "may", "june",
                   "july", "august", "september", "october", "november", "december"]
        exp_months = [m for m in months if m in expected]
        pred_months = [m for m in months if m in predicted]
        if exp_months and pred_months and set(exp_months) != set(pred_months):
            return "temporal_confusion"

    # --- Wrong entity: answer mentions a completely different person/thing ---
    # Extract potential entity names (capitalized words) from expected and predicted
    # Simple heuristic: if predicted contains proper nouns not in expected
    exp_orig = q["expected"]
    pred_orig = q["predicted"]

    # Extract capitalized multi-word names
    exp_names = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', exp_orig))
    pred_names = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', pred_orig))

    # Filter out common words that happen to be capitalized
    common_caps = {"The", "This", "That", "These", "Those", "Yes", "No", "Not", "She", "He",
                   "They", "Her", "His", "Its", "Their", "Based", "According", "However",
                   "Also", "Both", "Each", "Every", "Some", "Many", "Most", "Several",
                   "About", "After", "Before", "During", "Since", "Until", "While",
                   "January", "February", "March", "April", "May", "June", "July",
                   "August", "September", "October", "November", "December",
                   "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
    exp_names -= common_caps
    pred_names -= common_caps

    if pred_names and exp_names and not pred_names.intersection(exp_names):
        # Predicted has names, expected has names, but no overlap
        # Check if it's a clear entity swap
        if len(pred_names) >= 1 and len(exp_names) >= 1:
            return "wrong_entity"

    # --- Incomplete answer ---
    # If expected has multiple parts and predicted only covers some
    if "and" in expected or "," in expected:
        exp_parts = re.split(r'\s*(?:,|and)\s*', expected)
        exp_parts = [p.strip() for p in exp_parts if len(p.strip()) > 2]
        if len(exp_parts) >= 2:
            matches = sum(1 for p in exp_parts if p.lower() in predicted)
            if 0 < matches < len(exp_parts):
                return "incomplete"

    # --- Inference gap: multi-hop or requires world knowledge ---
    if category == "multi_hop":
        return "inference_gap"

    # --- Wrong fact: same entity context but wrong attribute ---
    # If there's some overlap in content but the key fact differs
    exp_words = set(expected.split())
    pred_words = set(predicted.split())
    if len(exp_words) > 2 and len(pred_words) > 2:
        overlap = exp_words.intersection(pred_words)
        if len(overlap) >= 2 and len(overlap) < len(exp_words) * 0.5:
            return "wrong_fact"

    # --- Hallucination: predicted answer is specific but completely off ---
    if len(predicted) > 10 and not any(phrase in predicted for phrase in not_found_phrases):
        # Has a specific answer but it's wrong
        if len(exp_words.intersection(pred_words)) <= 1:
            return "hallucination"

    # Default classification based on content
    if len(predicted) > 5 and len(expected) > 5:
        return "wrong_fact"

    return "other"


def main():
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    # Print summary
    summary = data["summary"]
    cat_scores = data["category_scores"]
    print("=" * 100)
    print("10-CONV CONSOLIDATION ERROR ANALYSIS")
    print("=" * 100)
    print(f"\nTotal: {summary['total_questions']} questions, {summary['correct_count']} correct, "
          f"{summary['total_questions'] - summary['correct_count']} wrong, "
          f"Accuracy: {summary['accuracy']:.2%}")
    print(f"\nCategory breakdown:")
    for cat, s in sorted(cat_scores.items()):
        wrong = s['total'] - s['correct']
        print(f"  {cat:15s}: {s['correct']:3d}/{s['total']:3d} ({s['accuracy']:.2%}) — {wrong} wrong")

    # Extract wrong answers
    wrong = [r for r in data["detailed_results"] if not r["is_correct"]]
    print(f"\nTotal wrong answers to analyze: {len(wrong)}")

    # Classify each error
    errors = []
    for q in wrong:
        etype = classify_error(q)
        errors.append({
            "conversation_id": q["conversation_id"],
            "question_id": q["question_id"],
            "category": q["category"],
            "question": q["question"],
            "expected": q["expected"],
            "predicted": q["predicted"],
            "f1_score": q.get("f1_score", 0),
            "llm_judged": q.get("llm_judged", False),
            "error_type": etype
        })

    # ==========================================
    # SECTION 1: Error type distribution
    # ==========================================
    print("\n" + "=" * 100)
    print("SECTION 1: ERROR TYPE DISTRIBUTION")
    print("=" * 100)

    type_counts = Counter(e["error_type"] for e in errors)
    for etype, count in type_counts.most_common():
        pct = count / len(errors) * 100
        print(f"  {etype:25s}: {count:4d} ({pct:5.1f}%)")

    # ==========================================
    # SECTION 2: Category x Error Type cross-tab
    # ==========================================
    print("\n" + "=" * 100)
    print("SECTION 2: CATEGORY x ERROR TYPE CROSS-TABULATION")
    print("=" * 100)

    categories = sorted(set(e["category"] for e in errors))
    error_types = [et for et, _ in type_counts.most_common()]

    # Header
    header = f"{'Category':15s}"
    for et in error_types:
        header += f" | {et:>15s}"
    header += f" | {'TOTAL':>6s}"
    print(header)
    print("-" * len(header))

    for cat in categories:
        cat_errors = [e for e in errors if e["category"] == cat]
        cat_type_counts = Counter(e["error_type"] for e in cat_errors)
        row = f"{cat:15s}"
        for et in error_types:
            row += f" | {cat_type_counts.get(et, 0):>15d}"
        row += f" | {len(cat_errors):>6d}"
        print(row)

    # Total row
    row = f"{'TOTAL':15s}"
    for et in error_types:
        row += f" | {type_counts[et]:>15d}"
    row += f" | {len(errors):>6d}"
    print("-" * len(header))
    print(row)

    # ==========================================
    # SECTION 3: Conversations with highest error rates
    # ==========================================
    print("\n" + "=" * 100)
    print("SECTION 3: CONVERSATIONS WITH HIGHEST ERROR RATES")
    print("=" * 100)

    # Count total and wrong per conversation
    conv_total = Counter(r["conversation_id"] for r in data["detailed_results"])
    conv_wrong = Counter(e["conversation_id"] for e in errors)

    conv_stats = []
    for conv_id in sorted(conv_total.keys()):
        total = conv_total[conv_id]
        wrong_count = conv_wrong.get(conv_id, 0)
        rate = wrong_count / total * 100
        conv_stats.append((conv_id, wrong_count, total, rate))

    conv_stats.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Conv ID':12s} | {'Wrong':>6s} | {'Total':>6s} | {'Error%':>7s} | Error types")
    print("-" * 90)
    for conv_id, wrong_count, total, rate in conv_stats:
        conv_errors = [e for e in errors if e["conversation_id"] == conv_id]
        conv_etypes = Counter(e["error_type"] for e in conv_errors)
        etype_str = ", ".join(f"{et}={c}" for et, c in conv_etypes.most_common(5))
        print(f"{conv_id:12s} | {wrong_count:>6d} | {total:>6d} | {rate:>6.1f}% | {etype_str}")

    # ==========================================
    # SECTION 4: Detailed error listing by category
    # ==========================================
    print("\n" + "=" * 100)
    print("SECTION 4: DETAILED ERROR LISTING BY CATEGORY")
    print("=" * 100)

    for cat in ["adversarial", "temporal", "multi_hop", "single_hop", "open_domain"]:
        cat_errors = [e for e in errors if e["category"] == cat]
        if not cat_errors:
            continue
        print(f"\n{'─' * 100}")
        print(f"CATEGORY: {cat.upper()} ({len(cat_errors)} errors)")
        print(f"{'─' * 100}")

        for e in sorted(cat_errors, key=lambda x: (x["conversation_id"], int(x["question_id"]))):
            print(f"\n  [{e['conversation_id']}] Q{e['question_id']} | Error: {e['error_type']} | F1: {e['f1_score']:.2f}")
            print(f"  Question:  {e['question']}")
            print(f"  Expected:  {e['expected'][:200]}")
            print(f"  Predicted: {e['predicted'][:200]}")

    # ==========================================
    # SECTION 5: Top failure patterns
    # ==========================================
    print("\n" + "=" * 100)
    print("SECTION 5: TOP FAILURE PATTERNS")
    print("=" * 100)

    # Pattern 1: Retrieval failures by category
    print("\n--- Pattern Analysis: Retrieval Failures ---")
    ret_failures = [e for e in errors if e["error_type"] == "retrieval_failure"]
    if ret_failures:
        rf_by_cat = Counter(e["category"] for e in ret_failures)
        for cat, count in rf_by_cat.most_common():
            print(f"  {cat}: {count}")

    # Pattern 2: Wrong entity by category
    print("\n--- Pattern Analysis: Wrong Entity ---")
    we_errors = [e for e in errors if e["error_type"] == "wrong_entity"]
    if we_errors:
        we_by_cat = Counter(e["category"] for e in we_errors)
        for cat, count in we_by_cat.most_common():
            print(f"  {cat}: {count}")

    # Pattern 3: Temporal confusion specifics
    print("\n--- Pattern Analysis: Temporal Confusion ---")
    tc_errors = [e for e in errors if e["error_type"] == "temporal_confusion"]
    if tc_errors:
        print(f"  Total: {len(tc_errors)}")
        for e in tc_errors[:10]:
            print(f"  [{e['conversation_id']}] Q{e['question_id']}: expected '{e['expected'][:50]}' got '{e['predicted'][:50]}'")

    # Pattern 4: Counting errors
    print("\n--- Pattern Analysis: Counting Errors ---")
    count_errors = [e for e in errors if e["error_type"] == "counting_error"]
    if count_errors:
        print(f"  Total: {len(count_errors)}")
        for e in count_errors:
            print(f"  [{e['conversation_id']}] Q{e['question_id']}: '{e['question'][:60]}' expected='{e['expected'][:30]}' got='{e['predicted'][:30]}'")

    # Pattern 5: Inference gap details
    print("\n--- Pattern Analysis: Inference Gaps ---")
    ig_errors = [e for e in errors if e["error_type"] == "inference_gap"]
    if ig_errors:
        print(f"  Total: {len(ig_errors)}")
        for e in ig_errors[:15]:
            print(f"  [{e['conversation_id']}] Q{e['question_id']}: '{e['question'][:70]}'")
            print(f"    Expected: {e['expected'][:80]}")
            print(f"    Got:      {e['predicted'][:80]}")

    # Pattern 6: High F1 near-misses (F1 > 0.3 but marked wrong)
    print("\n--- Pattern Analysis: Near-Misses (F1 > 0.3 but wrong) ---")
    near_misses = [e for e in errors if e["f1_score"] > 0.3]
    near_misses.sort(key=lambda x: x["f1_score"], reverse=True)
    print(f"  Total near-misses: {len(near_misses)}")
    for e in near_misses[:20]:
        print(f"  [{e['conversation_id']}] Q{e['question_id']} F1={e['f1_score']:.2f} | {e['error_type']}")
        print(f"    Q: {e['question'][:80]}")
        print(f"    Exp: {e['expected'][:80]}")
        print(f"    Got: {e['predicted'][:80]}")

    # Pattern 7: LLM judged errors (was marked wrong even after LLM judge)
    print("\n--- Pattern Analysis: LLM-Judged Still Wrong ---")
    llm_judged_wrong = [e for e in errors if e.get("llm_judged")]
    print(f"  Total LLM-judged but still wrong: {len(llm_judged_wrong)}")
    for e in llm_judged_wrong[:10]:
        print(f"  [{e['conversation_id']}] Q{e['question_id']} | {e['error_type']} | F1={e['f1_score']:.2f}")
        print(f"    Q: {e['question'][:80]}")
        print(f"    Exp: {e['expected'][:60]}")
        print(f"    Got: {e['predicted'][:60]}")

    # ==========================================
    # SECTION 6: Summary statistics
    # ==========================================
    print("\n" + "=" * 100)
    print("SECTION 6: FINAL SUMMARY")
    print("=" * 100)

    print(f"\nTotal errors: {len(errors)} / {summary['total_questions']} ({len(errors)/summary['total_questions']:.1%})")
    print(f"\nError type ranking:")
    for i, (etype, count) in enumerate(type_counts.most_common(), 1):
        pct = count / len(errors) * 100
        print(f"  {i}. {etype}: {count} ({pct:.1f}%)")

    print(f"\nCategory error rates:")
    for cat in ["multi_hop", "single_hop", "open_domain", "temporal", "adversarial"]:
        s = cat_scores[cat]
        wrong_count = s['total'] - s['correct']
        print(f"  {cat:15s}: {wrong_count:3d} wrong / {s['total']:3d} total ({1-s['accuracy']:.1%} error rate)")

    print(f"\nTop 3 worst conversations:")
    for conv_id, wrong_count, total, rate in conv_stats[:3]:
        conv_errors = [e for e in errors if e["conversation_id"] == conv_id]
        conv_etypes = Counter(e["error_type"] for e in conv_errors)
        print(f"  {conv_id}: {wrong_count}/{total} wrong ({rate:.1f}%) — top errors: {dict(conv_etypes.most_common(3))}")

    # Save detailed errors to JSON for further analysis
    output = {
        "total_errors": len(errors),
        "error_type_distribution": dict(type_counts.most_common()),
        "errors_by_category": {},
        "errors_by_conversation": {},
        "all_errors": errors
    }

    for cat in categories:
        cat_errors = [e for e in errors if e["category"] == cat]
        cat_type_counts = Counter(e["error_type"] for e in cat_errors)
        output["errors_by_category"][cat] = {
            "total": len(cat_errors),
            "by_type": dict(cat_type_counts.most_common())
        }

    for conv_id in sorted(conv_total.keys()):
        conv_errors = [e for e in errors if e["conversation_id"] == conv_id]
        conv_type_counts = Counter(e["error_type"] for e in conv_errors)
        output["errors_by_conversation"][conv_id] = {
            "total_questions": conv_total[conv_id],
            "total_errors": len(conv_errors),
            "error_rate": len(conv_errors) / conv_total[conv_id],
            "by_type": dict(conv_type_counts.most_common())
        }

    with open("results/10conv_error_analysis.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed error analysis saved to results/10conv_error_analysis.json")


if __name__ == "__main__":
    main()
