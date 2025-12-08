import json
import sys
import pandas as pd

def generate_report():
    # Load Current Metrics
    with open("metrics.json", "r") as f:
        current = json.load(f)
    
    # Load Baseline (Prod) Metrics if exists
    baseline = None
    try:
        with open("prod_metrics.json", "r") as f:
            baseline = json.load(f)
    except FileNotFoundError:
        pass

    # --- Bagian 1: Model & Parameter ---
    report = "## 🤖 Model Evaluation Report\n\n"
    report += "### ⚙️ Configuration\n"
    report += f"**Current Model:** `{current['metrics']['model_name']}`\n"
    report += f"**Params:** `{current['metrics']['parameters']}`\n"
    
    # --- Bagian 2: Metrics Table ---
    report += "\n### 📊 Metrics Comparison\n"
    report += "| Metric | Main (Baseline) | PR (Candidate) | Delta |\n"
    report += "|---|---|---|---|\n"
    
    # Helper untuk format baris
    def add_row(name, curr_val, base_val):
        delta_str = "-"
        base_str = "N/A"
        if base_val is not None:
            base_str = f"{base_val:.4f}"
            delta = curr_val - base_val
            icon = "🟢" if delta >= 0 else "🔴"
            delta_str = f"{icon} {delta:+.4f}"
        
        return f"| {name} | {base_str} | {curr_val:.4f} | {delta_str} |\n"

    # Accuracy
    acc_base = baseline['metrics']['accuracy'] if baseline else None
    report += add_row("Accuracy", current['metrics']['accuracy'], acc_base)
    
    # F1 Scores per label
    for label, score in current['metrics']['f1_scores'].items():
        base_score = baseline['metrics']['f1_scores'].get(label) if baseline else None
        report += add_row(f"F1-Score ({label})", score, base_score)

    # --- Bagian 3: Inference Check ---
    report += "\n### 🕵️ Inference Sanity Check (5 Samples)\n"
    report += "| Expected | Predicted (PR) | Match? | Text |\n"
    report += "|---|---|---|---|\n"
    
    for item in current['inference']:
        match_icon = "✅" if item['match'] else "❌"
        # Truncate text jika terlalu panjang
        text_short = (item['text'][:75] + '..') if len(item['text']) > 75 else item['text']
        report += f"| **{item['expected']}** | {item['predicted']} | {match_icon} | {text_short} |\n"
        
    # Write report to file
    with open("report.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    generate_report()
