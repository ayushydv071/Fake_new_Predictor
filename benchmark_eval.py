"""
Benchmark Evaluation Script for Fake News Predictor
---------------------------------------------------
Evaluates pipeline performance on 15 benchmark test cases.
Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
"""

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Direct pipeline imports
from pipeline.input_layer import process_input
from pipeline.claim_extractor import extract_claims
from pipeline.query_generator import generate_queries
from pipeline.web_search import search_web
from pipeline.evidence_collector import collect_evidence
from pipeline.semantic_similarity import compute_similarity
from pipeline.evidence_ranker import rank_evidence
from pipeline.stance_detector import detect_stance
from pipeline.credibility_scorer import attach_credibility
from pipeline.hybrid_scorer import compute_final_score
from pipeline.aggregator import aggregate_results
from pipeline.verdict import generate_verdict
from pipeline.explainer import generate_explanation

TEST_CASES = [
    {"text": "NASA launched the James Webb Space Telescope in December 2021.", "expected": "REAL"},
    {"text": "The World Health Organization declared COVID-19 a global pandemic in March 2020.", "expected": "REAL"},
    {"text": "France won the FIFA World Cup in 2018.", "expected": "REAL"},
    {"text": "Japan experienced a major earthquake in March 2011 causing a tsunami.", "expected": "REAL"},
    {"text": "Apple introduced the first iPhone in 2007.", "expected": "REAL"},
    {"text": "Barack Obama was elected as the 44th President of the United States.", "expected": "REAL"},
    {"text": "The Eiffel Tower is located in Paris France.", "expected": "REAL"},
    
    {"text": "Scientists confirm the moon is made entirely of green cheddar cheese.", "expected": "LIKELY FAKE"},
    {"text": "Drinking boiling water completely cures diabetes within 24 hours.", "expected": "LIKELY FAKE"},
    {"text": "Aliens officially landed in Times Square New York in 2025 and signed a peace treaty.", "expected": "LIKELY FAKE"},
    {"text": "The US government passed a law making it illegal to own domestic cats in 2024.", "expected": "LIKELY FAKE"},
    {"text": "Drinking liquid bleach daily guarantees 100 percent immunity to all viral infections.", "expected": "LIKELY FAKE"},
    {"text": "A massive meteor hit London in 2024 destroying the entire city completely.", "expected": "LIKELY FAKE"},
    {"text": "Consuming raw garlic eliminates all need for sleep forever.", "expected": "LIKELY FAKE"},

    {"text": "Extraterrestrial life forms will be discovered by scientists in the year 2030.", "expected": "UNCERTAIN"},
]


def run_pipeline(text: str) -> str:
    """Run the pipeline on a single text string and return the predicted verdict."""
    try:
        sentences = process_input(text=text)
        if not sentences:
            return "UNCERTAIN"

        claims = extract_claims(sentences)
        if not claims:
            claims = [sentences[0]]

        primary_claim = claims[0]
        queries = generate_queries([primary_claim])
        search_results = search_web(queries)

        if not search_results:
            return "UNCERTAIN"

        evidence_list = collect_evidence(search_results, mode="fast")
        if not evidence_list:
            return "UNCERTAIN"

        evidence_list = compute_similarity(primary_claim, evidence_list)
        evidence_list = rank_evidence(evidence_list)

        if not evidence_list:
            return "UNCERTAIN"

        evidence_list = detect_stance(primary_claim, evidence_list)
        evidence_list = attach_credibility(evidence_list)
        evidence_list = compute_final_score(evidence_list)

        agg = aggregate_results(evidence_list)
        verdict_result = generate_verdict(
            score=agg["final_score"],
            support_count=agg["support_count"],
            refute_count=agg["refute_count"],
        )

        llm_result = generate_explanation(
            claim=primary_claim,
            evidence=evidence_list,
            verdict=verdict_result["verdict"],
        )

        return llm_result["verdict"]

    except Exception as e:
        print(f"[Benchmark] Pipeline error for '{text[:30]}...': {e}")
        return "UNCERTAIN"


def evaluate():
    y_true = []
    y_pred = []

    print("==================================================")
    print("RUNNING PIPELINE BENCHMARK EVALUATION")
    print("==================================================\n")

    for i, test in enumerate(TEST_CASES, 1):
        text = test["text"]
        expected = test["expected"]
        print(f"[{i}/{len(TEST_CASES)}] Testing: \"{text}\"")
        predicted = run_pipeline(text)
        print(f"    Expected: {expected} | Predicted: {predicted}\n")

        y_true.append(expected)
        y_pred.append(predicted)

    labels = ["REAL", "LIKELY FAKE", "UNCERTAIN"]
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("==================================================")
    print("BENCHMARK METRICS SUMMARY")
    print("==================================================")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")

    print("Confusion Matrix (Labels: REAL, LIKELY FAKE, UNCERTAIN):")
    print("           Pred REAL  Pred FAKE  Pred UNCERTAIN")
    for label, row in zip(labels, cm):
        print(f"True {label:<11} {row[0]:<10} {row[1]:<10} {row[2]:<14}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    evaluate()
