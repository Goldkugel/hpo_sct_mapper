import os
import re

import matplotlib
matplotlib.use("Agg")   # no display on the cluster
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd

# ==========================================================================
# MappingPlots.py
#
# Tables and plots of the evaluation (called by MappingEvaluation):
#   (1) generation_metrics : precision, recall and F1 per generation method
#   (2) generation_counts  : candidate pairs and HPO classes per generation
#                            method (new pairs only in the CSV)
#   (3) filter_metrics     : precision, recall and F1 per filter step and
#                            cumulative, plus LLM-accepted and final
#   (4) filter_counts      : pairs per filter step by outcome, and HPO
#                            classes per step
#   (3b) filter_cumulative_metrics : line plot of precision, recall and F1
#                            of the cumulative set after each filter step,
#                            then LLM-accepted and final
#   (4b) filter_cumulative_counts  : line plot of pairs, HPO classes, pairs
#                            accepted by the expert and pairs rejected by the
#                            pipeline of that cumulative set
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
# Precision, recall and F1 are plotted in percent (F1, the harmonic mean of
# two percentages, is a percentage as well).
METRIC_AXIS_LABEL = "Score (%)"
COLOR_COUNT["correct"]  = "#009E73"
COLOR_COUNT["rejected"] = "#D55E00"
COUNT_LABELS  = {"pairs": "Pairs", "hpo_classes": "HPO classes", "new_pairs": "New pairs",
                 "correct": "Accepted by the expert", "rejected": "Rejected by the pipeline"}

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
    'rejected' are the classified pairs not in the set: after a step, the
    pairs of the steps so far (domain filter included) left out of the set;
    after the LLM review (LLM-accepted and final), all classified pairs not
    in the final alignment: the domain mismatches and the reviewed pairs
    (codes 1–9) the LLM did not accept.
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
        kept = pairs_of(pool)
        # Without the step number: the x-axis is in processing order.
        cumul_rows.append(dict(step=step_id, label=re.sub(r"^\d+\s+", "", label), set=kind,
                               **pair_metrics(kept, gold_pairs),
                               rejected=len(pairs_of(so_far) - kept)))

    auto     = pairs_of(classified[classified["outcome"] == "accepted"])
    final    = auto | llm_pairs
    cumul_rows.append(dict(step="llm", label="LLM-accepted", set="LLM-accepted",
                           **pair_metrics(llm_pairs, gold_pairs),
                           rejected=len(pairs_of(classified) - final)))
    cumul_rows.append(dict(step="final", label="Final alignment", set="auto-accepted ∪ LLM-accepted",
                           **pair_metrics(final, gold_pairs),
                           rejected=len(pairs_of(classified) - final)))
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


def _grid(ax) -> None:
    """Light horizontal and vertical grid lines behind the data."""
    ax.grid(True, which="major", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)


def _percent(table: pd.DataFrame) -> pd.DataFrame:
    """Copy of table with precision, recall and F1 in percent."""
    table = table.copy()
    for col in COLOR_METRIC:
        if col in table:
            table[col] = table[col] * 100
    return table


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
        _grouped_bars(ax, _percent(table), list(COLOR_METRIC), COLOR_METRIC, METRIC_LABELS,
                      fmt="{:.1f}")
        ax.set_ylim(0, 112)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(METRIC_AXIS_LABEL)
        ax.tick_params(axis="y", labelleft=True)
        _grid(ax)
    fig.suptitle(subtitle, fontsize=11)
    fig.tight_layout()
    return fig


def plot_generation(alone: pd.DataFrame, cumulative: pd.DataFrame, gold_total: int,
                    out_dir: str) -> list:
    """Plots (1) and (2) and their tables; returns the written paths."""
    paths = []

    # (1) Each method alone (the cumulative values are only written as CSV).
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    _grouped_bars(ax, _percent(alone), list(COLOR_METRIC), COLOR_METRIC, METRIC_LABELS,
                  fmt="{:.1f}")
    ax.set_ylim(0, 112)
    ax.set_title("Metrics per generation method", fontsize=11)
    ax.set_ylabel(METRIC_AXIS_LABEL)
    _grid(ax)
    fig.tight_layout()
    paths += _save(fig, out_dir, "generation_metrics")

    # (2)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    _grouped_bars(ax, alone, ["pairs", "hpo_classes"], COLOR_COUNT,
                  COUNT_LABELS, log_y=True, fmt="{:,.0f}")
    ax.set_ylim(1, max(alone[["pairs", "hpo_classes"]].to_numpy().max(), 1) * 30)
    ax.set_title("Candidates per generation method", fontsize=11)
    ax.set_ylabel("Count (log scale)")
    _grid(ax)
    fig.tight_layout()
    paths += _save(fig, out_dir, "generation_counts")

    paths += _write_csv(alone, out_dir, "generation_metrics_alone")
    paths += _write_csv(cumulative, out_dir, "generation_metrics_cumulative")
    return paths


def plot_filter(per_step: pd.DataFrame, cumulative: pd.DataFrame, gold_total: int,
                out_dir: str) -> list:
    """Plots (3) and (4) and their tables; returns the written paths."""
    paths = []

    # (3) Domain filter left out of the left panel: its output are removed pairs.
    output_steps = per_step[per_step["output"] != "removed"]
    fig = _metric_figure(
        output_steps, cumulative,
        ("Each step", "Cumulative, then LLM review"),
        "Metrics of the filter steps",
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
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        _grid(ax)
    fig.suptitle("Pairs and HPO classes per filter step", fontsize=11)
    fig.tight_layout()
    paths += _save(fig, out_dir, "filter_counts")

    paths += _plot_filter_cumulative(cumulative, out_dir)

    paths += _write_csv(per_step, out_dir, "filter_metrics_per_step")
    paths += _write_csv(cumulative, out_dir, "filter_metrics_cumulative")
    return paths


# Candidate positions of a vertical value label next to a point of a line
# plot: (dx, dy) in points, horizontal and vertical alignment. Earlier
# candidates are preferred when several are equally free.
LABEL_CANDIDATES = [c for d in (5, 13, 21) for c in (
    ((0, d), "center", "bottom"),     # above
    ((0, -d), "center", "top"),       # below
    ((d, 3), "left", "bottom"),       # right, upwards
    ((d, -3), "left", "top"),         # right, downwards
    ((-d, 3), "right", "bottom"),     # left, upwards
    ((-d, -3), "right", "top"),       # left, downwards
)]


def _place_value_labels(fig, ax, x: np.ndarray, series: dict, colors: dict, fmt: str,
                        separators: list, marker_size: float) -> None:
    """
    Writes every value of the line plot vertically next to its point, at the
    candidate position with the fewest collisions. Penalties, highest first:
    outside the axes, on another label or the legend, on a marker, on a
    dashed separator, per sampled line point inside the label.
    series: {column: values}; separators: x positions of the dashed lines.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    axes_box = ax.get_window_extent(renderer)
    legend = ax.get_legend()
    legend_box = legend.get_window_extent(renderer) if legend else None
    marker_r = marker_size / 2 * fig.dpi / 72 + 1

    points = {col: to_px(np.column_stack([x, values])) for col, values in series.items()}
    samples = []
    for pts in points.values():
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            t = np.linspace(0, 1, 40)[:, None]
            samples.append(np.hstack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)]))
    samples = np.vstack(samples) if samples else np.empty((0, 2))
    markers = np.vstack(list(points.values()))
    separator_px = [to_px((b, 0))[0] for b in separators]

    def label(text, xy, offset, ha, va, color, **kwargs):
        return ax.annotate(text, xy, xytext=offset, textcoords="offset points", rotation=90,
                           ha=ha, va=va, fontsize=6.5, color=color, zorder=5, **kwargs)

    placed = []
    for i in range(len(x)):
        for col, values in series.items():
            text, xy = fmt.format(values[i]), (x[i], values[i])
            best = None
            for rank, (offset, ha, va) in enumerate(LABEL_CANDIDATES):
                t = label(text, xy, offset, ha, va, colors[col])
                bb = t.get_window_extent(renderer).expanded(1.05, 1.05)
                t.remove()
                near = lambda a, r=0: ((np.abs(a[:, 0] - np.clip(a[:, 0], bb.x0, bb.x1)) <= r) &
                                       (np.abs(a[:, 1] - np.clip(a[:, 1], bb.y0, bb.y1)) <= r))
                score = (
                    100 * (bb.x0 < axes_box.x0 or bb.x1 > axes_box.x1 or
                           bb.y0 < axes_box.y0 or bb.y1 > axes_box.y1)
                    + 30 * sum(bb.overlaps(o) for o in placed)
                    + 30 * (legend_box is not None and bb.overlaps(legend_box))
                    + 10 * int(near(markers, marker_r).sum())
                    + 5 * sum(bb.x0 <= b <= bb.x1 for b in separator_px)
                    + int(near(samples).sum())
                    + 0.3 * rank
                )
                if best is None or score < best[0]:
                    best = (score, offset, ha, va, bb)
            _, offset, ha, va, bb = best
            label(text, xy, offset, ha, va, colors[col],
                  bbox=dict(boxstyle="square,pad=0.05", facecolor="white", edgecolor="none"))
            placed.append(bb)


def _plot_filter_cumulative(cumulative: pd.DataFrame, out_dir: str) -> list:
    """
    Plots (3b) and (4b): line plots of the metrics and counts of the
    cumulative set after each filter step, then LLM-accepted and final.
    Dashed lines separate the groups of the cumulative table's 'set' column
    (auto-accepted; pairs passed to the LLM; LLM-accepted; final). Every
    value is written next to its point. Returns the written paths.
    """
    x = np.arange(len(cumulative))
    sets = cumulative["set"].tolist()
    separators = [i - 0.5 for i in range(1, len(sets)) if sets[i] != sets[i - 1]]
    width = max(6.5, 0.75 * len(cumulative) + 2)
    marker_size = 5

    def line_plot(columns, colors, labels, fmt, title, ylabel, name, ylim=None) -> list:
        fig, ax = plt.subplots(figsize=(width, 4.8))
        table = _percent(cumulative) if columns == list(COLOR_METRIC) else cumulative
        series = {col: table[col].to_numpy(dtype=float) for col in columns}
        for col, values in series.items():
            ax.plot(x, values, marker="o", markersize=marker_size, linewidth=1.8,
                    color=colors[col], label=labels[col], zorder=3)
        ax.set_ylim(*(ylim or (0, max(cumulative[columns].to_numpy().max(), 1) * 1.2)))
        if not ylim:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        for b in separators:
            ax.axvline(b, color="#666666", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cumulative["label"], rotation=30, ha="right", fontsize=8)
        ax.set_xlim(-0.5, len(cumulative) - 0.3)
        _grid(ax)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, ncol=len(columns), loc="upper left")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        # After the layout, so the label positions are measured on the final axes.
        _place_value_labels(fig, ax, x, series, colors, fmt, separators, marker_size)
        return _save(fig, out_dir, name)

    paths = []
    paths += line_plot(list(COLOR_METRIC), COLOR_METRIC, METRIC_LABELS, "{:.1f}",
                       "Metrics after each filter step", METRIC_AXIS_LABEL,
                       "filter_cumulative_metrics", ylim=(0, 108))
    paths += line_plot(["pairs", "hpo_classes", "correct", "rejected"], COLOR_COUNT,
                       COUNT_LABELS, "{:,.0f}", "Counts after each filter step", "Count",
                       "filter_cumulative_counts")
    return paths


def _write_csv(table: pd.DataFrame, out_dir: str, name: str) -> list:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    table.round(4).to_csv(path, sep=";", index=False)
    return [path]


# --------------------------------------------------------------------------
# (5) Flow of the pairs through the pipeline (Sankey diagram)
# --------------------------------------------------------------------------

COLOR_FLOW = {
    "start":    "#4D4D4D",
    "removed":  COLOR_OUTCOME["removed"],
    "accepted": COLOR_OUTCOME["accepted"],
    "conflict": COLOR_OUTCOME["conflict"],
    "review":   COLOR_OUTCOME["review"],
    "final":    "#009E73",
    "rejected": "#BBBBBB",
}


def pipeline_flow_tables(
    universe: set, pairs_by_outcome: dict, llm_pairs: set, gold_pairs: set,
) -> tuple:
    """
    Nodes and flows of the pipeline from the pairs the filter classified
    (codes 0–10) over the outcome groups to the final alignment.
    pairs_by_outcome: {"removed", "accepted", "conflict", "review"} -> pairs.
    Auto-accepted pairs stay in the alignment whatever the LLM decides.
    Returns (nodes, flows) DataFrames.
    """
    classified = set().union(*pairs_by_outcome.values())
    conflict_final = pairs_by_outcome["conflict"] & llm_pairs
    review_final   = pairs_by_outcome["review"] & llm_pairs
    final    = pairs_by_outcome["accepted"] | conflict_final | review_final
    rejected = (pairs_by_outcome["conflict"] | pairs_by_outcome["review"]) - final

    node_sets = [
        ("start",    0, "Classified by the filter",          classified),
        ("removed",  1, "Domain mismatch (code 0)",           pairs_by_outcome["removed"]),
        ("accepted", 1, "Auto-accepted (code 10)",            pairs_by_outcome["accepted"]),
        ("conflict", 1, "Source-collapse conflicts (1–7)",    pairs_by_outcome["conflict"]),
        ("review",   1, "Fallback candidates (8–9)",          pairs_by_outcome["review"]),
        ("final",    2, "Final alignment",                    final),
        ("rejected", 2, "Rejected by the LLM",                rejected),
    ]
    nodes = pd.DataFrame([
        dict(node=key, column=col, label=label, pairs=len(s), correct=len(s & gold_pairs),
             hpo_classes=len({h for h, _ in s}))
        for key, col, label, s in node_sets
    ])
    flow_sets = [
        ("start", "removed",  pairs_by_outcome["removed"]),
        ("start", "accepted", pairs_by_outcome["accepted"]),
        ("start", "conflict", pairs_by_outcome["conflict"]),
        ("start", "review",   pairs_by_outcome["review"]),
        ("accepted", "final",    pairs_by_outcome["accepted"]),
        ("conflict", "final",    conflict_final),
        ("conflict", "rejected", pairs_by_outcome["conflict"] - conflict_final),
        ("review",   "final",    review_final),
        ("review",   "rejected", pairs_by_outcome["review"] - review_final),
    ]
    flows = pd.DataFrame([
        dict(source=src, target=dst, pairs=len(s), correct=len(s & gold_pairs))
        for src, dst, s in flow_sets
    ])
    not_reached = universe - classified
    nodes.attrs["universe"]            = len(universe)
    nodes.attrs["universe_correct"]    = len(universe & gold_pairs)
    nodes.attrs["not_reached"]         = len(not_reached)
    nodes.attrs["not_reached_correct"] = len(not_reached & gold_pairs)
    return nodes, flows


def _band(ax, x0: float, y0: tuple, x1: float, y1: tuple, color: str) -> None:
    """Filled band between the vertical spans y0 at x0 and y1 at x1 (cubic curves)."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    xm = (x0 + x1) / 2
    verts = [
        (x0, y0[1]), (xm, y0[1]), (xm, y1[1]), (x1, y1[1]),
        (x1, y1[0]), (xm, y1[0]), (xm, y0[0]), (x0, y0[0]), (x0, y0[1]),
    ]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=0.35))


def plot_pipeline_flow(nodes: pd.DataFrame, flows: pd.DataFrame, gold_total: int,
                       out_dir: str) -> list:
    """Sankey diagram (5) and its tables; returns the written paths."""
    node_w, gap = 0.035, 0.035
    xs = {0: 0.0, 1: 0.45, 2: 0.9}
    scale = 1.0 / max(nodes.groupby("column")["pairs"].sum().max(), 1)

    # Node spans, stacked from the top of each column.
    span = {}
    for col, group in nodes.groupby("column"):
        n_gaps = max(len(group) - 1, 0)
        height = group["pairs"].sum() * scale + n_gaps * gap
        top = 0.5 + height / 2
        for _, row in group.iterrows():
            h = row["pairs"] * scale
            span[row["node"]] = (top - h, top)
            top -= h + gap

    fig, ax = plt.subplots(figsize=(12, 6.5))
    out_cursor = {k: v[1] for k, v in span.items()}    # next free y (from top) for outgoing
    in_cursor  = {k: v[1] for k, v in span.items()}    # next free y for incoming
    node_col   = dict(zip(nodes["node"], nodes["column"]))
    for _, f in flows.iterrows():
        h = f["pairs"] * scale
        s0, s1 = out_cursor[f["source"]] - h, out_cursor[f["source"]]
        t0, t1 = in_cursor[f["target"]] - h, in_cursor[f["target"]]
        out_cursor[f["source"]] = s0
        in_cursor[f["target"]]  = t0
        if h > 0:
            _band(ax, xs[node_col[f["source"]]] + node_w, (s0, s1),
                  xs[node_col[f["target"]]], (t0, t1), COLOR_FLOW[f["target"]])

    for _, n in nodes.iterrows():
        y0, y1 = span[n["node"]]
        x = xs[n["column"]]
        color = COLOR_FLOW[n["node"]]
        ax.add_patch(plt.Rectangle((x, y0), node_w, y1 - y0, facecolor=color, alpha=0.45,
                                   edgecolor="none"))
        # Correct pairs as the solid lower part of the node.
        h_correct = n["correct"] * scale
        ax.add_patch(plt.Rectangle((x, y0), node_w, h_correct, facecolor=color, edgecolor="none"))
        precision = n["correct"] / n["pairs"] if n["pairs"] else 0.0
        text = (f"{n['label']}\n{n['pairs']:,} pairs, {n['hpo_classes']:,} HPO classes\n"
                f"{n['correct']:,} correct (precision {precision:.1%})")
        if n["column"] == 0:
            ax.text(x - 0.01, (y0 + y1) / 2, text, ha="right", va="center", fontsize=8)
        else:
            ax.text(x + node_w + 0.01, (y0 + y1) / 2, text, ha="left", va="center", fontsize=8)

    ax.set_xlim(-0.32, 1.25)
    ax.set_ylim(-0.08, 1.08)
    ax.axis("off")
    fig.suptitle("Flow of the candidate pairs through the filter", fontsize=11)
    paths = _save(fig, out_dir, "pipeline_flow")
    paths += _write_csv(nodes, out_dir, "pipeline_flow_nodes")
    paths += _write_csv(flows, out_dir, "pipeline_flow_links")
    return paths


# --------------------------------------------------------------------------
# (6) Overlap of the generation methods (UpSet plot)
# --------------------------------------------------------------------------

def method_overlap_table(pair_methods: dict, methods: list, gold_pairs: set) -> pd.DataFrame:
    """
    pair_methods: pair -> set of the generation methods that found it.
    One row per method combination (exclusive: found by exactly these
    methods), with candidate pairs, correct pairs and precision.
    """
    counts = {}
    for pair, found_by in pair_methods.items():
        key = tuple(m for m in methods if m in found_by)
        total, correct = counts.get(key, (0, 0))
        counts[key] = (total + 1, correct + (pair in gold_pairs))
    rows = [
        dict(combination=" + ".join(key), n_methods=len(key), pairs=total, correct=correct,
             precision=correct / total if total else 0.0, **{m: m in key for m in methods})
        for key, (total, correct) in counts.items()
    ]
    return (pd.DataFrame(rows)
            .sort_values(["correct", "pairs"], ascending=False)
            .reset_index(drop=True))


def plot_method_overlap(table: pd.DataFrame, methods: list, method_labels: dict,
                        gold_total: int, out_dir: str, top: int = 15) -> list:
    """UpSet plot (6) of the top combinations by correct pairs; returns the written paths."""
    methods = [m for m in methods if table[m].any()]
    shown = table.head(top)
    x = np.arange(len(shown))

    fig = plt.figure(figsize=(12, 6.5))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 4], height_ratios=[3, 1.6],
                            wspace=0.02, hspace=0.05)
    ax_bar  = fig.add_subplot(grid[0, 1])
    ax_dots = fig.add_subplot(grid[1, 1], sharex=ax_bar)
    ax_set  = fig.add_subplot(grid[1, 0], sharey=ax_dots)

    bars = ax_bar.bar(x, shown["correct"], 0.6, color=COLOR_METRIC["precision"])
    for bar, (_, row) in zip(bars, shown.iterrows()):
        ax_bar.annotate(f"{row['correct']:,}\nP {row['precision']:.1%}\nof {row['pairs']:,}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5)
    ax_bar.set_ylabel("Correct pairs")
    ax_bar.set_ylim(0, shown["correct"].max() * 1.3 if len(shown) else 1)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_bar.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    _grid(ax_bar)

    y = np.arange(len(methods))[::-1]
    for xi, (_, row) in zip(x, shown.iterrows()):
        member = [row[m] for m in methods]
        ax_dots.scatter([xi] * len(methods), y, s=40,
                        c=["#333333" if m else "#DDDDDD" for m in member], zorder=2)
        ys = [yi for yi, m in zip(y, member) if m]
        if len(ys) > 1:
            ax_dots.plot([xi, xi], [min(ys), max(ys)], color="#333333", lw=1.5, zorder=1)
    ax_dots.set_yticks(y)
    ax_dots.tick_params(axis="y", labelleft=False)
    ax_dots.tick_params(axis="x", bottom=False, labelbottom=False)
    for spine in ax_dots.spines.values():
        spine.set_visible(False)
    ax_dots.tick_params(axis="y", left=False)

    # Correct pairs per method (all its pairs, not exclusive).
    per_method = [int(table.loc[table[m], "correct"].sum()) for m in methods]
    ax_set.barh(y, per_method, 0.5, color=COLOR_METRIC["recall"])
    ax_set.set_yticks(y)
    ax_set.set_yticklabels([method_labels.get(m, m) for m in methods], fontsize=8)
    ax_set.tick_params(axis="y", left=False)
    ax_set.set_xlabel("Correct pairs per method", fontsize=8)
    ax_set.set_xlim(max(per_method + [1]) * 1.35, 0)
    ax_set.spines[["top", "left", "right"]].set_visible(False)
    ax_set.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_set.grid(True, axis="x", color="#DDDDDD", linewidth=0.6)
    ax_set.set_axisbelow(True)
    for yi, value in zip(y, per_method):
        ax_set.annotate(f"{value:,}", (value, yi), xytext=(-2, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=7)

    fig.suptitle("Correct pairs per combination of generation methods", fontsize=11)
    paths = _save(fig, out_dir, "method_overlap")
    paths += _write_csv(table, out_dir, "method_overlap")
    return paths


def method_agreement_table(overlap: pd.DataFrame) -> pd.DataFrame:
    """
    Candidate pairs grouped by the number of generation methods that found
    them, with correct pairs and precision (from method_overlap_table).
    """
    table = (overlap.groupby("n_methods", as_index=False)[["pairs", "correct"]].sum()
             .sort_values("n_methods").reset_index(drop=True))
    table["precision"] = table["correct"] / table["pairs"].where(table["pairs"] > 0)
    table["precision"] = table["precision"].fillna(0.0)
    return table


def plot_method_agreement(table: pd.DataFrame, out_dir: str) -> list:
    """Plot (6b): precision by the number of methods that found a pair; returns the written paths."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(table))
    bars = ax.bar(x, table["precision"] * 100, 0.6, color=COLOR_METRIC["precision"])
    for bar, (_, row) in zip(bars, table.iterrows()):
        ax.annotate(f"{row['precision'] * 100:.1f}\n{int(row['correct']):,} of {int(row['pairs']):,}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(n)) for n in table["n_methods"]])
    ax.set_xlabel("Number of generation methods that found the pair")
    ax.set_ylabel("Precision (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Precision by number of generation methods", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    _grid(ax)
    fig.tight_layout()
    paths = _save(fig, out_dir, "method_agreement")
    paths += _write_csv(table, out_dir, "method_agreement")
    return paths


# --------------------------------------------------------------------------
# (7) Is-a distance of the aligned pairs to the nearest correct target
# --------------------------------------------------------------------------

COLOR_RELATION = {"correct": "#009E73", "broader": "#0072B2", "narrower": "#E69F00",
                  "lateral": "#999999"}


def gold_distance_table(pair_info: dict, sets: dict, max_distance: int = 5) -> pd.DataFrame:
    """
    pair_info: pair -> (distance or None, nearest correct target, relation)
    for pairs of expert-judged HPO classes. sets: label -> pairs.
    One row per set and distance bucket (0 … max_distance+), with the
    relation counts, the share of the set's measured pairs, and the
    cumulative share. Pairs without a distance (correct target not in
    SNOMED CT) are counted in "unmeasured".
    """
    buckets = [str(d) for d in range(max_distance)] + [f"{max_distance}+"]
    rows = []
    for label, pairs in sets.items():
        info = [pair_info[p] for p in pairs if p in pair_info]
        measured = [(d, rel) for d, _, rel in info if d is not None]
        total = len(measured)
        cumulative = 0
        for bucket in buckets:
            in_bucket = [rel for d, rel in measured
                         if (str(d) if d < max_distance else f"{max_distance}+") == bucket]
            cumulative += len(in_bucket)
            row = dict(set=label, distance=bucket, pairs=len(in_bucket),
                       share=len(in_bucket) / total if total else 0.0,
                       cumulative_share=cumulative / total if total else 0.0,
                       measured=total, unmeasured=len(info) - total)
            for rel in ("broader", "narrower", "lateral"):
                row[rel] = sum(1 for r in in_bucket if r == rel)
            row["correct"] = len(in_bucket) if bucket == "0" else 0
            rows.append(row)
    return pd.DataFrame(rows)


def plot_gold_distance(table: pd.DataFrame, gold_total: int, out_dir: str) -> list:
    """Stacked bars (7), one panel per set; returns the written paths."""
    sets = list(dict.fromkeys(table["set"]))
    fig, axes = plt.subplots(1, len(sets), figsize=(4 * len(sets), 4.8), sharey=True,
                             squeeze=False)
    for ax, label in zip(axes[0], sets):
        rows = table[table["set"] == label].reset_index(drop=True)
        x = np.arange(len(rows))
        bottom = np.zeros(len(rows))
        for rel, color in COLOR_RELATION.items():
            values = (rows[rel] / rows["measured"].replace(0, np.nan)).fillna(0).to_numpy() * 100
            ax.bar(x, values, 0.65, bottom=bottom, color=color, label=rel)
            bottom += values
        for xi, top, cum in zip(x, bottom, rows["cumulative_share"]):
            ax.annotate(f"{cum:.0%}", (xi, top), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(rows["distance"])
        ax.set_xlabel("Is-a distance to the nearest correct target")
        measured = int(rows["measured"].iloc[0]) if len(rows) else 0
        ax.set_title(f"{label}\n({measured:,} pairs of expert-judged HPO classes)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 105)
        ax.tick_params(axis="y", labelleft=True)
        _grid(ax)
    axes[0][0].set_ylabel("Share of pairs (%)")
    axes[0][-1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Is-a distance to the nearest correct target", fontsize=11)
    fig.tight_layout()
    paths = _save(fig, out_dir, "gold_distance")
    paths += _write_csv(table, out_dir, "gold_distance")
    return paths
