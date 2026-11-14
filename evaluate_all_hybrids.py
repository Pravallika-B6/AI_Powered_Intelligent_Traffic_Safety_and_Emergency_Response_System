"""
Unified Benchmark Script: Runs, Trains, and Evaluates all 4 Hybrid Models on data1.
Generates a quantitative comparison table, saves results to JSON/CSV, and plots a comparison chart.
"""

import json
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import config
from hybrid1_transcnn import train_and_eval_hybrid1
from hybrid2_yolo_san import train_and_eval_hybrid2
from hybrid3_waveconv import train_and_eval_hybrid3
from hybrid4_deep_forest import train_and_eval_hybrid4

def run_all_benchmarks(epochs=5):
    print("=" * 80)
    print("STARTING COMPREHENSIVE BENCHMARK: 4 HYBRID MODELS ON ACCIDENT DATASET")
    print("=" * 80)

    results = []

    # 1. Hybrid 1: TransCNN
    try:
        m1 = train_and_eval_hybrid1(epochs=epochs)
        m1["primary_fusion"] = "Cross-Attention"
        results.append(m1)
    except Exception as e:
        print(f"[ERROR in Hybrid 1]: {e}")

    # 2. Hybrid 2: YOLO-SAN
    try:
        m2 = train_and_eval_hybrid2(epochs=epochs)
        m2["primary_fusion"] = "Spatial Gating"
        results.append(m2)
    except Exception as e:
        print(f"[ERROR in Hybrid 2]: {e}")

    # 3. Hybrid 3: WaveConv
    try:
        m3 = train_and_eval_hybrid3(epochs=epochs)
        m3["primary_fusion"] = "Wavelet Dual-Domain"
        results.append(m3)
    except Exception as e:
        print(f"[ERROR in Hybrid 3]: {e}")

    # 4. Hybrid 4: Deep-Forest
    try:
        m4 = train_and_eval_hybrid4()
        m4["primary_fusion"] = "Embedding Stacking"
        results.append(m4)
    except Exception as e:
        print(f"[ERROR in Hybrid 4]: {e}")

    # Create summary DataFrame
    df = pd.DataFrame(results)
    if not df.empty:
        # Reorder and format columns
        display_df = pd.DataFrame({
            "Model Name": df["model_name"],
            "Primary Fusion": df["primary_fusion"],
            "Test Accuracy (%)": (df["test_acc"] * 100).round(2),
            "Precision (%)": (df["precision"] * 100).round(2),
            "Recall (%)": (df["recall"] * 100).round(2),
            "F1-Score (%)": (df["f1"] * 100).round(2),
            "Latency (ms)": df["latency_ms"].round(2),
            "Train Time (s)": df["training_time_s"].round(1)
        })

        print("\n" + "=" * 90)
        print("FINAL QUANTITATIVE COMPARISON OF PROPOSED HYBRIDS")
        print("=" * 90)
        print(display_df.to_string(index=False))
        print("=" * 90 + "\n")

        # Save to CSV and JSON
        csv_path = config.OUTPUT_DIR / "hybrid_comparison_table.csv"
        json_path = config.OUTPUT_DIR / "hybrid_benchmark_results.json"
        display_df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to:\n  - {csv_path}\n  - {json_path}")

        # Generate comparison chart
        try:
            plt.figure(figsize=(10, 5))
            x = range(len(display_df))
            plt.bar([i - 0.2 for i in x], display_df["Test Accuracy (%)"], width=0.4, label="Accuracy (%)", color="#1f77b4")
            plt.bar([i + 0.2 for i in x], display_df["F1-Score (%)"], width=0.4, label="F1-Score (%)", color="#ff7f0e")
            plt.xticks(x, display_df["Model Name"], rotation=15)
            plt.ylabel("Percentage (%)")
            plt.ylim([70, 105])
            plt.title("Accident Detection: Hybrid Models Comparative Performance")
            plt.legend()
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()

            chart_path = config.OUTPUT_DIR / "hybrid_comparison_chart.png"
            plt.savefig(chart_path, dpi=300)
            plt.close()
            print(f"  - Comparison chart: {chart_path}")
        except Exception as pe:
            print(f"[Chart Generation Note]: {pe}")

    return results

if __name__ == "__main__":
    run_all_benchmarks(epochs=5)
