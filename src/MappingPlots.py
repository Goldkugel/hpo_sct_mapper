import os

import matplotlib
matplotlib.use("Agg")   # no display on the cluster
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

# ==========================================================================
# MappingPlots.py
#
# Tables and plots of the evaluation (called by MappingErrorAnalysis):
#   (1) generation_metrics : precision, recall and F1 per generation method,
#                            alone and cumulative (run order)
#   (2) generation_counts  : candidate pairs, HPO classes and new pairs per
#                            generation method, alone and cumulative
#   (3) filter_metrics     : precision, recall and F1 per filter step and
#                            cumulative, plus LLM-accepted and final
#   (4) filter_counts      : pairs per filter step by outcome, and HPO
#                            classes per step
# Each plot is written as PDF (for the paper) and PNG, each table as CSV.
#
# Only pandas, numpy and matplotlib are used, so the module can be tested
# without the pipeline packages. All counts are unique (hpo_id, sct_id)
# pairs; precision counts every pair, also of HPO classes without gold
# pairs, so it is a lower bound.
# ==========================================================================

# Okabe-Ito colours (colour-blind safe).
COLOR_METRIC = {"precision": "#0072B2", "recall": "#E69F00", "f1": "#009E73"}
COLOR_COUNT  = {"pairs": "#0072B2", "hpo_classes": "#E69F00", "new_pairs": "#CC79A7"}
COLOR_OUTCOME = {
    "accepted":  "#009E73",
    "conflict":  "#D55E00",
    "review":    "#56B4E9",
    "removed":   "#999999",
}
OUTCOME_LABELS = {
    "accepted": "accepted (code 10)",
    "conflict": "source-collapse conflict (codes 1–7)",
    "review":   "fallback candidate (codes 8–9)",
    "removed":  "domain mismatch (code 0)",
}
METRIC_LABELS = {"precision": "Precision", "recall": "Recall", "f1": "F1"}
COUNT_LABELS  = {"pairs": "Pairs", "hpo_classes": "HPO classes", "new_pairs": "New pairs"}

# Validity codes (as in MappingUtils; repeated so the module stays
# independent of the pipeline packages).
CODE_DOMAIN    = 0
CODES_CONFLICT = range(1, 8)
CODES_REVIEW   = (8, 9)
CODE_ACCEPTED  = 10


def pair_metrics(pairs: set, gold_pairs: set) -> dict:
    """Counts and precision, recall and F1 of a set of pairs."""
    tp        = len(pairs & gold_pairs)
    precision = tp / len(pairs) if pairs else 0.0
    recall    = tp / len(gold_pairs) if gold_pairs else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(
        pairs=len(pairs), hpo_classes=len({hpo for hpo, _ in pairs}), correct=tp,
        precision=precision, recall=recall, f1=f1,
    )


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def generation_tables(
    gen_rows: pd.DataFrame, id_col: str, val_col: str, attr_col: str,
    methods: list, method_labels: dict, gold_pairs: set,
) -> tuple:
    """
    Per generation method (in run order): the method alone, the pairs it
    adds to the methods before it, and the cumulative union.
    Returns (alone, cumulative) DataFrames.
    """
    alone_rows, cumul_rows = [], []
    cumulative = set()
    for method in methods:
        rows  = gen_rows[gen_rows[attr_col] == method]
        pairs = set(zip(rows[id_col], rows[val_col]))
        new   = pairs - cumulative
        cumulative |= pairs
        label = method_labels.get(method, method)
        alone_rows.append(dict(method=method, label=label, **pair_metrics(pairs, gold_pairs),
                               new_pairs=len(new)))
        cumul_rows.append(dict(method=method, label=label, **pair_metrics(cumulative, gold_pairs)))
    return pd.DataFrame(alone_rows), pd.DataFrame(cumul_rows)


def outcome_of(code) -> str:
    """Outcome group of a validity code."""
    if code == CODE_ACCEPTED:
        return "accepted"
    if code == CODE_DOMAIN:
        return "removed"
    if code in CODES_REVIEW:
        return "review"
    return "conflict"


def filter_tables(
    pair_table: pd.DataFrame, id_col: str, val_col: str,
    validity_col: str, step_col: str, steps: list,
    llm_pairs: set, gold_pairs: set,
) -> tuple:
    """
    pair_table: one row per generated pair with its original validity code
    and the step that set it (NA if not reached by any step).
    steps: [(step id, label, code)] in processing order.

    Per step: the pairs the step classified, split by outcome; metrics of
    the pairs it accepted (acceptance steps) or marked for review (fallback
    steps). Pairs accepted in a step and demoted later count for the step
    whose code they carry, i.e. as conflicts of their accepting step.

    Cumulative: auto-accepted pairs of the steps so far; from the first
    fallback step on, the pairs passed to the LLM (codes 1–10) of the steps
    so far; then LLM-accepted and final (auto-accepted + LLM-accepted).
    Returns (per_step, cumulative) DataFrames.
    """
    classified = pair_table[pair_table[step_col].notna()].copy()
    classified["outcome"] = classified[validity_col].map(outcome_of)

    def pairs_of(df) -> set:
        return set(zip(df[id_col], df[val_col]))

    per_step_rows, cumul_rows = [], []
    done_steps = []
    fallback_seen = False
    for step_id, label, code in steps:
        done_steps.append(step_id)
        rows = classified[classified[step_col] == step_id]
        by_outcome = rows["outcome"].value_counts()

        is_fallback = code in CODES_REVIEW
        fallback_seen |= is_fallback
        if code == CODE_DOMAIN:
            output = rows                                   # removed pairs
        elif is_fallback:
            output = rows[rows["outcome"] == "review"]
        else:
            output = rows[rows["outcome"] == "accepted"]
        m = pair_metrics(pairs_of(output), gold_pairs)

        per_step_rows.append(dict(
            step=step_id, label=label,
            output=("removed" if code == CODE_DOMAIN else "review" if is_fallback else "accepted"),
            classified_pairs=len(rows.drop_duplicates([id_col, val_col])),
            classified_hpo_classes=rows[id_col].nunique(),
            **{f"pairs_{o}": int(by_outcome.get(o, 0)) for o in COLOR_OUTCOME},
            **m,
        ))

        if code == CODE_DOMAIN:
            continue
        so_far = classified[classified[step_col].isin(done_steps)]
        if fallback_seen:
            pool, kind = so_far[so_far["outcome"] != "removed"], "auto-accepted + review (codes 1–10)"
        else:
            pool, kind = so_far[so_far["outcome"] == "accepted"], "auto-accepted"
        cumul_rows.append(dict(step=step_id, label=f"≤ {label}", set=kind,
                               **pair_metrics(pairs_of(pool), gold_pairs)))

    auto = pairs_of(classified[classified["outcome"] == "accepted"])
    cumul_rows.append(dict(step="llm", label="LLM-accepted", set="LLM-accepted",
                           **pair_metrics(llm_pairs, gold_pairs)))
    cumul_rows.append(dict(step="final", label="Final alignment", set="auto-accepted ∪ LLM-accepted",
                           **pair_metrics(auto | llm_pairs, gold_pairs)))
    return pd.DataFrame(per_step_rows), pd.DataFrame(cumul_rows)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def _grouped_bars(ax, table: pd.DataFrame, columns: list, colors: dict, labels: dict,
                  log_y: bool = False, fmt: str = "{:.2f}") -> None:
    """Grouped bars of table[columns] per row, values written above the bars."""
    n_groups, n_bars = len(table), len(columns)
    width = 0.8 / n_bars
    x = np.arange(n_groups)
    for i, col in enumerate(columns):
        values = table[col].astype(float).to_numpy()
        heights = np.where(values > 0, values, np.nan) if log_y else values
        bars = ax.bar(x + (i - (n_bars - 1) / 2) * width, heights, width,
                      color=colors[col], label=labels[col])
        for bar, value in zip(bars, values):
            y = bar.get_height() if not np.isnan(bar.get_height()) else (1 if log_y else 0)
            ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, y),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", rotation=90, fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(table["label"], rotation=30, ha="right", fontsize=8)
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=n_bars, loc="upper left")


def _save(fig, out_dir: str, name: str) -> list:
    """Write fig as PDF and PNG; returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, f"{name}.{ext}") for ext in ("pdf", "png")]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def _metric_figure(left: pd.DataFrame, right: pd.DataFrame, titles: tuple, subtitle: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, table, title in zip(axes, (left, right), titles):
        _grouped_bars(ax, table, list(COLOR_METRIC), COLOR_METRIC, METRIC_LABELS)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("Value")
    fig.suptitle(subtitle, fontsize=9)
    fig.tight_layout()
    return fig


def plot_generation(alone: pd.DataFrame, cumulative: pd.DataFrame, gold_total: int,
                    out_dir: str) -> list:
    """Plots (1) and (2) and their tables; returns the written paths."""
    subtitle = (f"Candidate generation; reference: {gold_total:,} pairs of the expert's "
                f"judgement (precision is a lower bound)")
    paths = []

    fig = _metric_figure(alone, cumulative,
                         ("Each method alone", "Cumulative (run order)"), subtitle)
    paths += _save(fig, out_dir, "generation_metrics")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    _grouped_bars(axes[0], alone, ["pairs", "hpo_classes", "new_pairs"], COLOR_COUNT,
                  COUNT_LABELS, log_y=True, fmt="{:,.0f}")
    _grouped_bars(axes[1], cumulative, ["pairs", "hpo_classes"], COLOR_COUNT,
                  COUNT_LABELS, log_y=True, fmt="{:,.0f}")
    top = max(alone[["pairs", "hpo_classes"]].to_numpy().max(),
              cumulative["pairs"].max(), 1)
    for ax, title in zip(axes, ("Each method alone", "Cumulative (run order)")):
        ax.set_ylim(1, top * 30)
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel("Count (log scale)")
    fig.suptitle(subtitle + "\nNew pairs: not found by the methods run before it", fontsize=9)
    fig.tight_layout()
    paths += _save(fig, out_dir, "generation_counts")

    paths += _write_csv(alone, out_dir, "generation_metrics_alone")
    paths += _write_csv(cumulative, out_dir, "generation_metrics_cumulative")
    return paths


def plot_filter(per_step: pd.DataFrame, cumulative: pd.DataFrame, gold_total: int,
                out_dir: str) -> list:
    """Plots (3) and (4) and their tables; returns the written paths."""
    subtitle = (f"Alignment set building; reference: {gold_total:,} pairs of the expert's "
                f"judgement (precision is a lower bound)")
    paths = []

    # (3) Domain filter left out of the left panel: its output are removed pairs.
    output_steps = per_step[per_step["output"] != "removed"]
    fig = _metric_figure(
        output_steps, cumulative,
        ("Each step", "Cumulative, then LLM review"),
        subtitle + "\nLeft: pairs accepted (steps 2–8) or marked for review (steps 9–10). "
                   "Right: auto-accepted up to step 8, from step 9 on all pairs passed to the LLM.",
    )
    paths += _save(fig, out_dir, "filter_metrics")

    # (4) Pairs by outcome (stacked) and HPO classes per step.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(per_step))
    bottom = np.zeros(len(per_step))
    for outcome, color in COLOR_OUTCOME.items():
        values = per_step[f"pairs_{outcome}"].to_numpy(dtype=float)
        axes[0].bar(x, values, 0.7, bottom=bottom, color=color, label=OUTCOME_LABELS[outcome])
        bottom += values
    for xi, total in zip(x, bottom):
        axes[0].annotate(f"{total:,.0f}", (xi, total), xytext=(0, 2), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7)
    axes[0].set_title("Pairs classified by each step", fontsize=11)
    axes[0].set_ylabel("Pairs")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper center")

    bars = axes[1].bar(x, per_step["classified_hpo_classes"], 0.7, color=COLOR_COUNT["hpo_classes"])
    for bar in bars:
        axes[1].annotate(f"{bar.get_height():,.0f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    axes[1].set_title("HPO classes involved in each step", fontsize=11)
    axes[1].set_ylabel("HPO classes")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(per_step["label"], rotation=30, ha="right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    fig.suptitle(subtitle + "\nDemoted pairs count as conflicts of the step that accepted them",
                 fontsize=9)
    fig.tight_layout()
    paths += _save(fig, out_dir, "filter_counts")

    paths += _write_csv(per_step, out_dir, "filter_metrics_per_step")
    paths += _write_csv(cumulative, out_dir, "filter_metrics_cumulative")
    return paths


def _write_csv(table: pd.DataFrame, out_dir: str, name: str) -> list:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    table.round(4).to_csv(path, sep=";", index=False)
    return [path]
