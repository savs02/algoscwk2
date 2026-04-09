import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".mplconfig"))
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "outputs" / "evaluation"
ANALYSIS_DIR = ROOT / "outputs" / "analysis"
PLOTS_DIR = ANALYSIS_DIR / "plots"


def ensure_dirs():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str):
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def aggregate(df: pd.DataFrame, by, metric="f1"):
    grouped = df.groupby(by, as_index=False)[metric].agg(["mean", "std"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped


SKETCH_ORDER = ["CMS", "CU-CMS", "CS"]
SKETCH_COLORS = {
    "CMS": "#0f4c5c",
    "CU-CMS": "#e36414",
    "CS": "#5f0f40",
}
SKETCH_LINESTYLES = {
    "CMS": "-",
    "CU-CMS": "--",
    "CS": ":",
}
SKETCH_MARKERS = {
    "CMS": "o",
    "CU-CMS": "s",
    "CS": "^",
}
SKETCH_OFFSETS = {
    "CMS": -0.06,
    "CU-CMS": 0.0,
    "CS": 0.06,
}



def plot_baseline_vs_improved():
    df = pd.read_csv(EVAL_DIR / "baseline_vs_improved.csv")
    if df.empty:
        return df

    def get_approach_label(row):
        dt = row["detector"]
        sk = row["sketch"]
        if dt == "baseline_raw_l1":
            return f"Baseline ({sk})"
        return f"Improved ({sk})"

    df["approach"] = df.apply(get_approach_label, axis=1)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="anomaly_type", y="f1", hue="approach", palette="Set2")
    plt.ylim(0, 1.05)
    plt.title("Baseline vs Improved Detectors by Anomaly Type")
    plt.ylabel("F1 Score")
    plt.xlabel("Anomaly Type")
    plt.legend(title="Detector & Sketch")
    plt.tight_layout()
    savefig("baseline_vs_improved.png")
    return df

def plot_threshold_sweep():
    df = pd.read_csv(EVAL_DIR / "threshold_sweep.csv")
    summary = aggregate(df, ["threshold"])
    plt.figure(figsize=(8, 4.8))
    plt.plot(summary["threshold"], summary["mean"], color="#0f4c5c",
             marker="o", linewidth=2.5, markersize=7)
    plt.fill_between(summary["threshold"],
                     summary["mean"] - summary["std"],
                     summary["mean"] + summary["std"],
                     alpha=0.18,
                     color="#0f4c5c")
    plt.ylim(0, 1.05)
    plt.ylabel("Validation F1")
    plt.xlabel("Normalized L1 Threshold")
    plt.title("Threshold Selection")
    plt.grid(axis="y", alpha=0.25)
    savefig("threshold_sweep.png")
    return summary


def draw_sketch_lines(summary, x_col, title, out_name, y_label):
    x_values = sorted(summary[x_col].unique())
    x_positions = {value: idx for idx, value in enumerate(x_values)}

    plt.figure(figsize=(8.4, 4.8))
    ax = plt.gca()
    for sketch in SKETCH_ORDER:
        sketch_df = summary[summary["sketch"] == sketch].sort_values(x_col)
        xs = [x_positions[v] + SKETCH_OFFSETS[sketch] for v in sketch_df[x_col]]
        ys = sketch_df["mean"].tolist()
        yerr = sketch_df["std"].tolist()
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            label=sketch,
            color=SKETCH_COLORS[sketch],
            linestyle=SKETCH_LINESTYLES[sketch],
            marker=SKETCH_MARKERS[sketch],
            linewidth=2.2,
            markersize=6.5,
            capsize=3,
            elinewidth=1.0,
            markeredgecolor="#1f1f1f",
            markeredgewidth=0.7,
        )

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels([str(v) for v in x_values])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(title="Sketch", ncol=3, frameon=True, loc="upper center",
              bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    savefig(out_name)


def plot_line_sweep(csv_name, x_col, title, out_name):
    df = pd.read_csv(EVAL_DIR / csv_name)
    summary = aggregate(df, ["sketch", x_col])
    draw_sketch_lines(summary, x_col, title, out_name, "Mean F1")
    return summary


def plot_bin_scheme():
    df = pd.read_csv(EVAL_DIR / "bin_scheme_comparison.csv")
    summary = aggregate(df, ["sketch", "scheme"])
    draw_sketch_lines(summary, "scheme", "Uniform vs Logarithmic Bins",
                      "bin_scheme_comparison.png", "Mean F1")
    return summary


def plot_hash_rotation():
    df = pd.read_csv(EVAL_DIR / "hash_rotation.csv")
    plt.figure(figsize=(8.4, 4.8))
    ax = plt.gca()
    x_values = list(df["flow"].astype(str).unique())
    x_positions = {value: idx for idx, value in enumerate(x_values)}
    for sketch in SKETCH_ORDER:
        sketch_df = df[df["sketch"] == sketch].copy()
        sketch_df["flow"] = sketch_df["flow"].astype(str)
        sketch_df = sketch_df.set_index("flow").reindex(x_values).reset_index()
        xs = [x_positions[v] + SKETCH_OFFSETS[sketch] for v in sketch_df["flow"]]
        ys = sketch_df["mean_l1"].fillna(0.0).to_list()
        yerr = sketch_df["std_l1"].fillna(0.0).to_list()
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            capsize=3,
            color=SKETCH_COLORS[sketch],
            label=sketch,
            linestyle=SKETCH_LINESTYLES[sketch],
            marker=SKETCH_MARKERS[sketch],
            linewidth=2.2,
            markersize=6.5,
            elinewidth=1.0,
            markeredgecolor="#1f1f1f",
            markeredgewidth=0.7,
        )

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(x_values)
    ax.set_ylabel("Mean Residual L1")
    ax.set_xlabel("Flow")
    ax.set_title("Hash Rotation Residuals")
    ax.legend(title="Sketch", ncol=3, frameon=True, loc="upper center",
              bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    savefig("hash_rotation.png")
    return df


def plot_classifier_accuracy():
    df = pd.read_csv(EVAL_DIR / "classifier_accuracy_vs_n.csv")
    summary = aggregate(df, ["sketch", "n_packets"], metric="accuracy")
    draw_sketch_lines(
        summary,
        "n_packets",
        "Classifier Robustness",
        "classifier_accuracy_vs_n.png",
        "Mean Classification Accuracy",
    )
    return summary


def plot_confusion():
    df = pd.read_csv(EVAL_DIR / "classifier_confusion_matrix.csv")
    if df.empty:
        return df
    pivot = df.pivot(index="expected", columns="predicted", values="count").fillna(0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues")
    plt.title("Classifier Confusion Matrix (N=1000 aggregate)")
    savefig("classifier_confusion_matrix.png")
    return df


def write_summary(threshold_summary, bins_summary, snapshots_summary, zipf_summary,
                  sensitivity_summary, bin_scheme_summary, hash_rotation_df,
                  classifier_summary):
    baseline = pd.read_csv(EVAL_DIR / "baseline_vs_improved.csv")
    best_threshold_row = threshold_summary.loc[threshold_summary["mean"].idxmax()]
    bins_best = bins_summary.sort_values("mean", ascending=False).iloc[0]
    snap_worst = snapshots_summary.sort_values("mean", ascending=True).iloc[0]
    zipf_worst = zipf_summary.sort_values("mean", ascending=True).iloc[0]
    sensitivity_low = sensitivity_summary.sort_values("magnitude").iloc[0]
    scheme_gap = bin_scheme_summary.groupby("scheme", as_index=False)["mean"].mean()
    classifier_low = classifier_summary.sort_values("n_packets").iloc[0]
    cu_hash = hash_rotation_df[hash_rotation_df["sketch"] == "CU-CMS"]

    lines = [
        "# Evaluation Summary",
        "",
        "## Detector",
        f"- Baseline raw L1 F1: {baseline.loc[baseline['detector'] == 'baseline_raw_l1', 'f1'].iloc[0]:.3f}",
        f"- Improved normalized detector best threshold: {best_threshold_row['threshold']:.2f}",
        f"- Best validation mean F1: {best_threshold_row['mean']:.3f}",
        f"- Held-out improved F1 (all sketches): {baseline.loc[baseline['detector'] == 'improved_normalized', 'f1'].mean():.3f}",
        "",
        "## Sweeps",
        f"- Best bins setting in current sweep: {int(bins_best['bins'])} bins for {bins_best['sketch']} with mean F1 {bins_best['mean']:.3f}",
        f"- Weakest snapshot setting: K={int(snap_worst['snapshots'])} for {snap_worst['sketch']} with mean F1 {snap_worst['mean']:.3f}",
        f"- Hardest Zipf regime: alpha={zipf_worst['zipf_alpha']:.1f} with mean F1 {zipf_worst['mean']:.3f}",
        f"- Lowest sensitivity point: magnitude={sensitivity_low['magnitude']:.1f} with mean F1 {sensitivity_low['mean']:.3f}",
        "",
        "## Structure",
        f"- Mean F1 by scheme: "
        f"log={scheme_gap.loc[scheme_gap['scheme'] == 'logarithmic', 'mean'].iloc[0]:.3f}, "
        f"uniform={scheme_gap.loc[scheme_gap['scheme'] == 'uniform', 'mean'].iloc[0]:.3f}",
        f"- CU-CMS hash-rotation residuals remain zero in this experiment: "
        f"max mean L1={cu_hash['mean_l1'].max():.1f}",
        "",
        "## Classifier",
        f"- Lowest packet regime ({int(classifier_low['n_packets'])} packets) mean accuracy: {classifier_low['mean']:.3f}",
        f"- Highest packet regime mean accuracy: "
        f"{classifier_summary.sort_values('n_packets').iloc[-1]['mean']:.3f}",
        "",
    ]

    (ANALYSIS_DIR / "summary.md").write_text("\n".join(lines))


def main():
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.facecolor": "#fcfbf7",
        "axes.facecolor": "#fcfbf7",
        "axes.edgecolor": "#2f2f2f",
        "axes.titleweight": "bold",
        "axes.labelcolor": "#1f1f1f",
        "xtick.color": "#1f1f1f",
        "ytick.color": "#1f1f1f",
        "grid.color": "#d9d4c7",
        "legend.facecolor": "#fffdf8",
        "legend.edgecolor": "#cccccc",
    })

    baseline_df = plot_baseline_vs_improved()
    threshold_summary = plot_threshold_sweep()
    savefig_name_bins = "appendix_bins_sweep.png"
    bins_summary = plot_line_sweep("bins_sweep.csv", "bins", "Bins Sweep", savefig_name_bins)
    snapshots_summary = plot_line_sweep(
        "snapshots_sweep.csv", "snapshots", "Snapshots Sweep", "appendix_snapshots_sweep.png"
    )
    zipf_summary = plot_line_sweep(
        "zipf_sweep.csv", "zipf_alpha", "Zipf Sweep", "appendix_zipf_sweep.png"
    )
    sensitivity_summary = plot_line_sweep(
        "sensitivity_sweep.csv", "magnitude", "Sensitivity Sweep", "sensitivity_sweep.png"
    )
    bin_scheme_summary = plot_bin_scheme()
    hash_rotation_df = plot_hash_rotation()
    classifier_summary = plot_classifier_accuracy()
    plot_confusion()
    write_summary(
        threshold_summary,
        bins_summary,
        snapshots_summary,
        zipf_summary,
        sensitivity_summary,
        bin_scheme_summary,
        hash_rotation_df,
        classifier_summary,
    )
    print(f"Wrote plots to {PLOTS_DIR}")
    print(f"Wrote summary to {ANALYSIS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
