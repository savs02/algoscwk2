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

    df["detector_label"] = df["detector"].map({
        "baseline_raw_l1":     "Baseline (raw L1 > 300)",
        "improved_normalized": "Improved (normalised L1)",
    }).fillna(df["detector"])

    plt.figure(figsize=(9, 5.2))
    sns.barplot(
        data=df,
        x="sketch",
        y="f1",
        hue="detector_label",
        order=SKETCH_ORDER,
        errorbar="sd",
        capsize=0.12,
        err_kws={"linewidth": 1.2},
        palette={"Baseline (raw L1 > 300)": "#0f4c5c",
                 "Improved (normalised L1)": "#e36414"},
    )
    plt.ylim(0, 1.05)
    n_seeds = df["seed"].nunique()
    plt.title(f"Baseline vs Improved Detector "
              f"(mean ± std across {n_seeds} held-out seeds)")
    plt.ylabel("F1 Score")
    plt.xlabel("Sketch")
    plt.legend(title="Detector", loc="lower right")
    plt.grid(axis="y", alpha=0.25)
    savefig("baseline_vs_improved.png")
    return df

def plot_memory_sweep():
    path = EVAL_DIR / "memory_sweep.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return df

    # Aggregate across seeds per (sketch, width). Keep memory_bytes for x-axis.
    summary = (df.groupby(["sketch", "width", "memory_bytes"], as_index=False)
                 .agg(mean=("f1", "mean"), std=("f1", "std")))
    summary["std"] = summary["std"].fillna(0.0)
    summary["memory_kb"] = summary["memory_bytes"] / 1024.0

    plt.figure(figsize=(12.5, 6.5))
    ax = plt.gca()
    for sketch in SKETCH_ORDER:
        sub = summary[summary["sketch"] == sketch].sort_values("width")
        if sub.empty:
            continue
        ax.errorbar(
            sub["memory_kb"],
            sub["mean"],
            yerr=sub["std"],
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

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Total memory (KB, log scale) — width × depth × bins × snapshots × 4B")
    ax.set_ylabel("F1 Score")
    ax.set_title("Memory vs F1 Tradeoff (5 held-out seeds, improved detector)")
    ax.legend(title="Sketch", loc="lower right")
    ax.grid(which="both", axis="both", alpha=0.25)
    ax.set_axisbelow(True)

    # Annotate each point with its width for readability.
    widths_seen = sorted(summary["width"].unique())
    for w in widths_seen:
        row = summary[(summary["sketch"] == "CMS") & (summary["width"] == w)]
        if not row.empty:
            ax.annotate(
                f"w={w}",
                xy=(row["memory_kb"].iloc[0], row["mean"].iloc[0]),
                xytext=(0, -14),
                textcoords="offset points",
                fontsize=8,
                ha="center",
                color="#555",
            )

    savefig("memory_sweep.png")
    return summary

def plot_per_type_breakdown():
    path = EVAL_DIR / "per_type_breakdown.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return df

    df["detector_label"] = df["detector"].map({
        "baseline_raw_l1":     "Baseline",
        "improved_normalized": "Improved",
    }).fillna(df["detector"])

    type_order = ["SuddenSpike", "GradualRamp", "PeriodicBurst",
                  "Spread", "Disappearance"]
    type_order = [t for t in type_order if t in df["anomaly_type"].unique()]
    n_seeds = df["seed"].nunique()

    # --- Main plot: recall per anomaly type, averaged across sketches AND seeds ---
    plt.figure(figsize=(11, 5.2))
    sns.barplot(
        data=df,
        x="anomaly_type",
        y="recall",
        hue="detector_label",
        order=type_order,
        errorbar="sd",
        capsize=0.1,
        err_kws={"linewidth": 1.2},
        palette={"Baseline": "#0f4c5c", "Improved": "#e36414"},
    )
    plt.ylim(0, 1.05)
    plt.title(f"Per-Anomaly-Type Recall "
              f"(mean ± std across 3 sketches × {n_seeds} seeds)")
    plt.ylabel("Recall")
    plt.xlabel("Anomaly Type")
    plt.legend(title="Detector", loc="lower right")
    plt.grid(axis="y", alpha=0.25)
    savefig("per_type_recall.png")

    # --- Faceted plot: per sketch, still averaged across seeds ---
    g = sns.catplot(
        data=df,
        x="anomaly_type",
        y="recall",
        hue="detector_label",
        col="sketch",
        col_order=SKETCH_ORDER,
        order=type_order,
        kind="bar",
        height=4.2,
        aspect=1.05,
        errorbar="sd",
        capsize=0.12,
        err_kws={"linewidth": 1.0},
        palette={"Baseline": "#0f4c5c", "Improved": "#e36414"},
        legend_out=False,
    )
    g.set(ylim=(0, 1.05))
    g.set_axis_labels("Anomaly Type", "Recall")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(25)
            label.set_horizontalalignment("right")
        ax.grid(axis="y", alpha=0.25)
    g.fig.suptitle(f"Per-Type Recall by Sketch (mean ± std across {n_seeds} seeds)",
                   y=1.03)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "per_type_recall_by_sketch.png",
                dpi=180, bbox_inches="tight")
    plt.close(g.fig)

    return df

def plot_threshold_sweep():
    df = pd.read_csv(EVAL_DIR / "threshold_sweep.csv")
    # Average over sketch and seed at each threshold.
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
    plt.ylabel("Validation F1 (mean over 3 sketches × 2 seeds)")
    plt.xlabel("Normalized L1 Threshold")
    plt.title("Threshold Selection")
    plt.grid(axis="y", alpha=0.25)
    savefig("threshold_sweep.png")
    return summary


def draw_sketch_lines(summary, x_col, title, out_name, y_label, ylim=(0, 1.05), legend_kwargs=None):
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
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_title(title)
    if legend_kwargs is None:
        legend_kwargs = dict(title="Sketch", ncol=3, frameon=True, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.legend(**legend_kwargs)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    savefig(out_name)


def plot_line_sweep(csv_name, x_col, title, out_name, ylim=(0, 1.05), legend_kwargs=None):
    df = pd.read_csv(EVAL_DIR / csv_name)
    summary = aggregate(df, ["sketch", x_col])
    draw_sketch_lines(summary, x_col, title, out_name, "Mean F1", ylim=ylim, legend_kwargs=legend_kwargs)
    return summary


def plot_bin_scheme():
    df = pd.read_csv(EVAL_DIR / "bin_scheme_comparison.csv")
    summary = aggregate(df, ["sketch", "scheme"])
    
    plt.figure(figsize=(8.4, 4.8))
    sns.barplot(
        data=df,
        x="scheme",
        y="f1",
        hue="sketch",
        hue_order=SKETCH_ORDER,
        errorbar="sd",
        capsize=0.1,
        err_kws={"linewidth": 1.2},
        palette=SKETCH_COLORS,
    )
    plt.ylim(0, 1.05)
    plt.ylabel("Mean F1")
    plt.xlabel("Scheme")
    plt.title("Uniform vs Logarithmic Bins")
    plt.legend(title="Sketch", loc="upper left", fontsize="small", title_fontsize="small")
    plt.grid(axis="y", alpha=0.25)
    plt.gca().set_axisbelow(True)
    savefig("bin_scheme_comparison.png")
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
    ax.legend(title="Sketch", loc="upper left", fontsize="small", title_fontsize="small")
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
                  classifier_summary, per_type_df, memory_summary):
    baseline = pd.read_csv(EVAL_DIR / "baseline_vs_improved.csv")
    best_threshold_row = threshold_summary.loc[threshold_summary["mean"].idxmax()]
    bins_best = bins_summary.sort_values("mean", ascending=False).iloc[0]
    snap_worst = snapshots_summary.sort_values("mean", ascending=True).iloc[0]
    zipf_worst = zipf_summary.sort_values("mean", ascending=True).iloc[0]
    sensitivity_low = sensitivity_summary.sort_values("magnitude").iloc[0]
    scheme_gap = bin_scheme_summary.groupby("scheme", as_index=False)["mean"].mean()
    classifier_low = classifier_summary.sort_values("n_packets").iloc[0]
    cu_hash = hash_rotation_df[hash_rotation_df["sketch"] == "CU-CMS"]

    all_rows = baseline[baseline["anomaly_type"] == "ALL"]
    n_seeds = all_rows["seed"].nunique()
    baseline_f1 = all_rows.loc[all_rows["detector"] == "baseline_raw_l1", "f1"].mean()
    baseline_std = all_rows.loc[all_rows["detector"] == "baseline_raw_l1", "f1"].std()
    improved_f1 = all_rows.loc[all_rows["detector"] == "improved_normalized", "f1"].mean()
    improved_std = all_rows.loc[all_rows["detector"] == "improved_normalized", "f1"].std()

    lines = [
        "# Evaluation Summary",
        "",
        f"## Detector ({n_seeds} held-out seeds, all 5 anomalies together)",
        f"- Baseline raw L1: F1 = {baseline_f1:.3f} ± {baseline_std:.3f}",
        f"- Improved normalized: F1 = {improved_f1:.3f} ± {improved_std:.3f}",
        f"- Best validation threshold: {best_threshold_row['threshold']:.2f} "
        f"(val mean F1 {best_threshold_row['mean']:.3f})",
        "",
    ]

    if per_type_df is not None and not per_type_df.empty:
        improved_per_type = (per_type_df[per_type_df["detector"] == "improved_normalized"]
                             .groupby("anomaly_type")["recall"].mean()
                             .sort_values())
        baseline_per_type = (per_type_df[per_type_df["detector"] == "baseline_raw_l1"]
                             .groupby("anomaly_type")["recall"].mean()
                             .sort_values())
        lines += [
            "## Per-type recall (mean across 3 sketches)",
        ]
        if not baseline_per_type.empty:
            lines.append(
                f"- Baseline hardest type: {baseline_per_type.index[0]} "
                f"(recall {baseline_per_type.iloc[0]:.3f})"
            )
        if not improved_per_type.empty:
            lines.append(
                f"- Improved hardest type: {improved_per_type.index[0]} "
                f"(recall {improved_per_type.iloc[0]:.3f})"
            )
        lines.append("")
    if memory_summary is not None and not memory_summary.empty:
        # Find smallest width where mean F1 >= 0.9 (the "minimum viable memory")
        viable = (memory_summary[memory_summary["mean"] >= 0.9]
                  .sort_values("memory_bytes"))
        if not viable.empty:
            first = viable.iloc[0]
            lines += [
                f"- Minimum viable memory (mean F1 >= 0.9): "
                f"w={int(first['width'])}, {first['memory_bytes']/1024:.1f} KB "
                f"({first['sketch']}, F1={first['mean']:.3f})",
            ]

    lines += [
        "## Sweeps",
        f"- Best bins setting in current sweep: {int(bins_best['bins'])} bins for "
        f"{bins_best['sketch']} with mean F1 {bins_best['mean']:.3f}",
        f"- Weakest snapshot setting: K={int(snap_worst['snapshots'])} for "
        f"{snap_worst['sketch']} with mean F1 {snap_worst['mean']:.3f}",
        f"- Hardest Zipf regime: alpha={zipf_worst['zipf_alpha']:.1f} with mean F1 "
        f"{zipf_worst['mean']:.3f}",
        f"- Lowest sensitivity point: magnitude={sensitivity_low['magnitude']:.1f} "
        f"with mean F1 {sensitivity_low['mean']:.3f}",
        "",
        "## Structure",
        f"- Mean F1 by scheme: "
        f"log={scheme_gap.loc[scheme_gap['scheme'] == 'logarithmic', 'mean'].iloc[0]:.3f}, "
        f"uniform={scheme_gap.loc[scheme_gap['scheme'] == 'uniform', 'mean'].iloc[0]:.3f}",
        f"- CU-CMS hash-rotation residuals remain zero in this experiment: "
        f"max mean L1={cu_hash['mean_l1'].max():.1f}",
        "",
        "## Classifier",
        f"- Lowest packet regime ({int(classifier_low['n_packets'])} packets) "
        f"mean accuracy: {classifier_low['mean']:.3f}",
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
    per_type_df = plot_per_type_breakdown()
    threshold_summary = plot_threshold_sweep()
    
    sweep_legend_kwargs = dict(title="Sketch", loc="lower left", fontsize="small", title_fontsize="small")
    
    savefig_name_bins = "appendix_bins_sweep.png"
    bins_summary = plot_line_sweep("bins_sweep.csv", "bins", "Bins Sweep", savefig_name_bins, ylim=None, legend_kwargs=sweep_legend_kwargs)
    snapshots_summary = plot_line_sweep(
        "snapshots_sweep.csv", "snapshots", "Snapshots Sweep", "appendix_snapshots_sweep.png", ylim=None, legend_kwargs=sweep_legend_kwargs
    )
    zipf_legend_kwargs = dict(title="Sketch", loc="lower right", fontsize="small", title_fontsize="small")
    zipf_summary = plot_line_sweep(
        "zipf_sweep.csv", "zipf_alpha", "Zipf Sweep", "appendix_zipf_sweep.png", ylim=None, legend_kwargs=zipf_legend_kwargs
    )
    sensitivity_summary = plot_line_sweep(
        "sensitivity_sweep.csv", "magnitude", "Sensitivity Sweep", "sensitivity_sweep.png", ylim=None
    )
    epoch_summary = plot_line_sweep(
        "epoch_sweep.csv", "epoch_size", "Epoch Size Sweep", "epoch_sweep.png", ylim=None
    )

    memory_summary = plot_memory_sweep()   # add this line
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
        per_type_df,
        memory_summary,
    )
    print(f"Wrote plots to {PLOTS_DIR}")
    print(f"Wrote summary to {ANALYSIS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
