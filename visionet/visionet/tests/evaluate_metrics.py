"""
VeriLabel AI - Performance Metrics & Precision/Recall Evaluator
Team: Brahmastra | GITAM University
Implements evaluation metrics defined in HLD Section 11:
- Precision
- Recall
- F1 Score
- Accuracy & Specificity
- Average Processing Latency (ms)
"""

import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from verilabel.pipeline import VeriLabelPipeline
from verilabel.config import SAMPLE_DATA_DIR


def evaluate_system_performance():
    """Evaluates Precision, Recall, and Processing Latency against ground-truth defect labels."""
    pipeline = VeriLabelPipeline()
    labels_dir = SAMPLE_DATA_DIR / "labels"

    # Ground truth test cases definition
    # defect_rules: set of rule IDs that are INTENTIONALLY failed (defects)
    test_cases = [
        {
            "name": "TC01 - Fully Compliant",
            "file": "TC01_compliant.png",
            "expected_defects": set(),
        },
        {
            "name": "TC02 - Missing Expiry Date",
            "file": "TC02_missing_expiry.png",
            "expected_defects": {"MAND-003"},
        },
        {
            "name": "TC03 - Invalid UDI Checksum",
            "file": "TC03_invalid_udi.png",
            "expected_defects": {"UDI-002"},
        },
        {
            "name": "TC04 - Missing Sterile Symbol",
            "file": "TC04_missing_symbol.png",
            "expected_defects": {"SYM-001"},
        },
        {
            "name": "TC05 - Missing Warning Text",
            "file": "TC05_missing_warnings.png",
            "expected_defects": {"MAND-005"},
        },
        {
            "name": "TC06 - Valid Barcode",
            "file": "TC01_compliant.png",
            "expected_defects": set(),
        }
    ]

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    latencies = []

    print("\n" + "=" * 78)
    print(" VeriLabel AI - Regulatory Compliance Model Evaluation (HLD Section 11)")
    print("=" * 78)

    for tc in test_cases:
        path = labels_dir / tc["file"]
        if not path.exists():
            print(f"Skipping {tc['name']}: file not found at {path}")
            continue

        t0 = time.perf_counter()
        result = pipeline.process_label(path, save_to_db=False)
        elapsed = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed)

        # Observed failed rule IDs
        observed_defects = {c.rule_id for c in result.report.checks if not c.passed}
        expected = tc["expected_defects"]

        # Classification counts
        all_rules = {c.rule_id for c in result.report.checks}
        tp = len(expected.intersection(observed_defects))
        fp = len(observed_defects - expected)
        fn = len(expected - observed_defects)
        tn = len(all_rules - (expected.union(observed_defects)))

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        verdict_str = result.report.verdict.value
        score_str = f"{result.report.compliance_score:.1f}%"
        print(f"[{tc['name']:<30}] Verdict: {verdict_str:<16} Score: {score_str:<7} Time: {elapsed:>6.1f}ms | Defects Found: {list(observed_defects)}")

    # Aggregate Metrics Calculation
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    print("-" * 78)
    print(" PERFORMANCE EVALUATION SUMMARY")
    print("-" * 78)
    print(f"  • True Positives (Actual defects caught)     : {total_tp}")
    print(f"  • False Positives (False alarms)             : {total_fp}")
    print(f"  • False Negatives (Defects missed)           : {total_fn}")
    print(f"  • True Negatives (Compliant checks passed)   : {total_tn}")
    print(f"  -------------------------------------------------------------")
    print(f"  • PRECISION                                  : {precision * 100:.2f}%")
    print(f"  • RECALL (Defect Sensitivity)                : {recall * 100:.2f}%")
    print(f"  • F1-SCORE                                   : {f1 * 100:.2f}%")
    print(f"  • OVERALL ACCURACY                           : {accuracy * 100:.2f}%")
    print(f"  • AVERAGE PROCESSING LATENCY                 : {avg_latency:.2f} ms")
    print("=" * 78 + "\n")

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency
    }


if __name__ == "__main__":
    evaluate_system_performance()
