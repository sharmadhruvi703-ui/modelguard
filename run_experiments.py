"""Run ModelGuard's reliability experiment suite and create a Markdown report."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import pandas as pd
from data.generate_data import generate_dataset
from src.evaluation import train_and_evaluate, split_data, METRICS
from src.metrics import robustness_score, degradation
from src.perturbations import PERTURBATIONS

SEVERITIES = [0.05, 0.15, 0.30]

def run_experiments(df, seed=42):
    train, test = split_data(df, seed)
    baseline = train_and_evaluate(train, test, seed)
    rows = []
    for model, metrics in baseline.items():
        for metric, value in metrics.items(): rows.append({"model": model, "scenario": "clean", "severity": 0.0, "metric": metric, "value": value})
    for scenario, function in PERTURBATIONS.items():
        for severity in SEVERITIES:
            perturbed = function(test, severity=severity, seed=seed)
            for model, metrics in train_and_evaluate(train, perturbed, seed).items():
                for metric, value in metrics.items(): rows.append({"model": model, "scenario": scenario, "severity": severity, "metric": metric, "value": value})
    return baseline, pd.DataFrame(rows)

def save_charts(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    f1 = results[results.metric == "f1"]
    plt.figure(figsize=(12, 6))
    for model in f1.model.unique():
        subset = f1[f1.model == model]
        for scenario in subset.scenario.unique():
            s = subset[subset.scenario == scenario]
            plt.plot(s.severity, s.value, marker="o", label=f"{model} / {scenario}")
    plt.xlabel("Perturbation severity (0 = clean)"); plt.ylabel("F1 score"); plt.title("Model performance under data problems"); plt.grid(alpha=.25); plt.legend(fontsize=7, ncol=2); plt.tight_layout(); plt.savefig(output_dir / "f1_degradation.png", dpi=160); plt.close()
    summary = f1[f1.scenario != "clean"].groupby("scenario").apply(lambda x: x.value.mean(), include_groups=False).sort_values()
    plt.figure(figsize=(9, 5)); summary.plot(kind="barh", color="#2a6f97"); plt.xlabel("Mean F1 across models and severities"); plt.title("Average impact by perturbation"); plt.tight_layout(); plt.savefig(output_dir / "perturbation_impact.png", dpi=160); plt.close()

def write_report(baseline, results, report_path):
    f1 = results[results.metric == "f1"]
    scores = {}
    for model in f1.model.unique():
        base = baseline[model]["f1"]
        scores[model] = robustness_score(base, f1[(f1.model == model) & (f1.scenario != "clean")].value.tolist())
    damage = f1[f1.scenario != "clean"].groupby("scenario").value.mean()
    worst = damage.idxmin()
    lines = ["# ModelGuard Reliability Report", "", "This report was generated programmatically from the experiment results. It describes an engineering exploration on synthetic customer data, not a universal claim about model behavior.", "", "## Baseline results", "", "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |", "|---|---:|---:|---:|---:|---:|"]
    for model, vals in baseline.items(): lines.append(f"| {model} | {vals['accuracy']:.3f} | {vals['precision']:.3f} | {vals['recall']:.3f} | {vals['f1']:.3f} | {vals['roc_auc']:.3f} |")
    lines += ["", "## Perturbations and degradation", "", "The table below reports F1 degradation from each model's clean baseline. Positive values indicate a drop.", "", "| Scenario | Severity | Logistic regression degradation | Random forest degradation |", "|---|---:|---:|---:|"]
    for scenario in PERTURBATIONS:
        for sev in SEVERITIES:
            vals=[]
            for model in baseline:
                current=f1[(f1.model==model)&(f1.scenario==scenario)&(f1.severity==sev)].value.iloc[0]
                vals.append(degradation(baseline[model]['f1'], current))
            lines.append(f"| {scenario} | {sev:.0%} | {vals[0]:.3f} | {vals[1]:.3f} |")
    lines += ["", "## Model comparison", "", f"The most damaging perturbation by mean F1 was **{worst}** in this run. Results vary with the chosen seed, generated data, and model settings.", "", "## Robustness score", "", "This project-defined score is `100 × mean(perturbed F1) / baseline F1`, averaged across all tested perturbations and severities. It is a compact comparison aid, not a calibrated reliability guarantee.", "", "| Model | Robustness score |", "|---|---:|"]
    for model, score in scores.items(): lines.append(f"| {model} | {score:.1f}/100 |")
    lines += ["", "## Key observations", "", f"- The baseline F1 values were measured before the controlled perturbations.", f"- The ranking of perturbations is dataset- and model-dependent; **{worst}** had the lowest mean F1 here.", "- Preprocessing handled missing values and unseen categories without changing the experiment code.", "- Duplicate rows and missing features represent data-pipeline problems rather than changes to the underlying customer behavior.", "", "## Limitations", "", "The data is synthetic, the perturbations are isolated rather than compositional, and the robustness score only summarizes this experiment grid. Production monitoring, temporal validation, calibration, and causal analysis are outside the scope of this small project."]
    report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--data", type=Path, default=Path("data/customer_churn.csv")); parser.add_argument("--seed", type=int, default=42); args=parser.parse_args()
    df = pd.read_csv(args.data) if args.data.exists() else generate_dataset(seed=args.seed)
    baseline, results = run_experiments(df, args.seed)
    out=Path("experiments/results"); out.mkdir(parents=True, exist_ok=True); results.to_csv(out / "metrics.csv", index=False); (out / "baseline.json").write_text(json.dumps(baseline, indent=2)); save_charts(results, out); write_report(baseline, results, Path("reports/model_reliability_report.md"))
    print("Baseline:", json.dumps(baseline, indent=2)); print("Wrote experiments/results/metrics.csv and reports/model_reliability_report.md")

if __name__ == "__main__": main()
