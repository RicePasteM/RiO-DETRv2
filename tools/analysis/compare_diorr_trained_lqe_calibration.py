#!/usr/bin/env python3
"""Compare trained boundary-only and angle-aware LQE models."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


MODELS = ("boundary_run1", "boundary_run2", "angle_aware")
BASELINES = ("boundary_run1", "boundary_run2")
NAMES = {
    "boundary_run1": "Boundary LQE (run 1)",
    "boundary_run2": "Boundary LQE (run 2)",
    "angle_aware": "Angle-aware LQE",
}
COLORS = {
    "boundary_run1": "#F58518",
    "boundary_run2": "#E45756",
    "angle_aware": "#2A9D8F",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary-only", required=True)
    parser.add_argument("--boundary-only-rep2", required=True)
    parser.add_argument("--angle-aware", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-images", type=int, default=11738)
    parser.add_argument("--total-gt", type=int, default=124438)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def load_data(args):
    sources = {
        "boundary_run1": (args.boundary_only, "boundary_lqe"),
        "boundary_run2": (args.boundary_only_rep2, "boundary_lqe"),
        "angle_aware": (args.angle_aware, "full_lqe"),
    }
    data = {}
    for name, (path, prefix) in sources.items():
        archive = np.load(path)
        data[name] = {
            key: archive[f"{prefix}_{key}"]
            for key in ("score", "tp50", "tp75", "q50")
        }
        expected = args.num_images * args.topk
        if any(len(value) != expected for value in data[name].values()):
            raise ValueError(f"{name} does not contain {expected} predictions")
    return data


def fixed_ece(scores, targets, num_bins):
    ids = np.minimum((scores * num_bins).astype(np.int64), num_bins - 1)
    score_sum = np.bincount(ids, weights=scores, minlength=num_bins)
    target_sum = np.bincount(ids, weights=targets, minlength=num_bins)
    return float(np.abs(score_sum - target_sum).sum() / len(scores))


def nll(scores, targets):
    scores = np.clip(scores.astype(np.float64), 1e-7, 1.0 - 1e-7)
    targets = targets.astype(np.float64)
    return float(
        -np.mean(targets * np.log(scores) + (1.0 - targets) * np.log1p(-scores))
    )


def brier(scores, targets):
    return float(np.mean((scores.astype(np.float64) - targets) ** 2))


def aurc(scores, targets):
    order = np.argsort(-scores, kind="stable")
    errors = 1.0 - targets[order].astype(np.float64)
    return float(
        np.mean(np.cumsum(errors) / np.arange(1, len(errors) + 1))
    )


def summarize(item, total_gt, num_bins):
    score = item["score"]
    tp50 = item["tp50"].astype(np.float64)
    tp75 = item["tp75"].astype(np.float64)
    q50 = item["q50"].astype(np.float64)
    return {
        "ece50": fixed_ece(score, tp50, num_bins),
        "nll50": nll(score, tp50),
        "brier50": brier(score, tp50),
        "aurc50": aurc(score, tp50),
        "ece75": fixed_ece(score, tp75, num_bins),
        "nll75": nll(score, tp75),
        "brier75": brier(score, tp75),
        "aurc75": aurc(score, tp75),
        "quality_ece50": fixed_ece(score, q50, num_bins),
        "quality_brier50": brier(score, q50),
        "confidence_quality_spearman": float(
            spearmanr(score, q50).statistic
        ),
        "recall50": float(tp50.sum() / total_gt),
        "recall75": float(tp75.sum() / total_gt),
    }


def per_image_additive(item, key, metric, num_images, topk):
    scores = item["score"].reshape(num_images, topk).astype(np.float64)
    targets = item[key].reshape(num_images, topk).astype(np.float64)
    if metric == "nll":
        scores = np.clip(scores, 1e-7, 1.0 - 1e-7)
        values = -(
            targets * np.log(scores) + (1.0 - targets) * np.log1p(-scores)
        )
    elif metric == "brier":
        values = (scores - targets) ** 2
    else:
        raise ValueError(metric)
    return values.mean(axis=1)


def per_image_bin_stats(item, key, num_images, topk, num_bins):
    scores = item["score"].reshape(num_images, topk).astype(np.float64)
    targets = item[key].reshape(num_images, topk).astype(np.float64)
    ids = np.minimum((scores * num_bins).astype(np.int64), num_bins - 1)
    stats = np.zeros((num_images, num_bins, 2), dtype=np.float64)
    for bin_index in range(num_bins):
        mask = ids == bin_index
        stats[:, bin_index, 0] = np.where(mask, scores, 0.0).sum(axis=1)
        stats[:, bin_index, 1] = np.where(mask, targets, 0.0).sum(axis=1)
    return stats


def paired_bootstrap(data, args):
    specs = {
        "ece50": ("tp50", "ece"),
        "nll50": ("tp50", "nll"),
        "brier50": ("tp50", "brier"),
        "ece75": ("tp75", "ece"),
        "nll75": ("tp75", "nll"),
        "brier75": ("tp75", "brier"),
        "quality_ece50": ("q50", "ece"),
        "quality_brier50": ("q50", "brier"),
    }
    prepared = {}
    for metric_name, (target, kind) in specs.items():
        prepared[metric_name] = {}
        for model in MODELS:
            if kind == "ece":
                prepared[metric_name][model] = per_image_bin_stats(
                    data[model],
                    target,
                    args.num_images,
                    args.topk,
                    args.num_bins,
                )
            else:
                prepared[metric_name][model] = per_image_additive(
                    data[model],
                    target,
                    kind,
                    args.num_images,
                    args.topk,
                )

    rng = np.random.default_rng(args.seed)
    deltas = {
        baseline: {
            metric_name: np.empty(args.bootstrap, dtype=np.float64)
            for metric_name in specs
        }
        for baseline in BASELINES
    }
    batch_size = 10
    for start in range(0, args.bootstrap, batch_size):
        size = min(batch_size, args.bootstrap - start)
        indices = rng.integers(
            0, args.num_images, size=(size, args.num_images), dtype=np.int32
        )
        for metric_name, (_target, kind) in specs.items():
            values = {}
            for model in MODELS:
                source = prepared[metric_name][model]
                if kind == "ece":
                    aggregate = source[indices].sum(axis=1)
                    values[model] = (
                        np.abs(aggregate[:, :, 0] - aggregate[:, :, 1]).sum(axis=1)
                        / (args.num_images * args.topk)
                    )
                else:
                    values[model] = source[indices].mean(axis=1)
            for baseline in BASELINES:
                deltas[baseline][metric_name][start : start + size] = (
                    values["angle_aware"] - values[baseline]
                )

    return {
        baseline: {
            name: {
                "delta_angle_minus_boundary_mean": float(values.mean()),
                "ci95": [
                    float(np.percentile(values, 2.5)),
                    float(np.percentile(values, 97.5)),
                ],
                "probability_angle_aware_is_lower": float(np.mean(values < 0)),
            }
            for name, values in baseline_deltas.items()
        }
        for baseline, baseline_deltas in deltas.items()
    }


def risk_curve(item, target_key, points=500):
    order = np.argsort(-item["score"], kind="stable")
    errors = 1.0 - item[target_key][order].astype(np.float64)
    risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    counts = np.unique(
        np.clip(
            np.rint(np.geomspace(1, len(errors), points)).astype(np.int64),
            1,
            len(errors),
        )
    )
    return counts / len(errors), risk[counts - 1]


def plot_risk(output_dir, data):
    fig, axes = plt.subplots(
        1, 2, figsize=(10.2, 4.2), constrained_layout=True
    )
    curves = {}
    for model in MODELS:
        coverage, risk = risk_curve(data[model], "tp50")
        curves[model] = (coverage, risk)
        axes[0].plot(
            coverage,
            risk,
            label=NAMES[model],
            color=COLORS[model],
            linewidth=1.8,
        )
    axes[0].set(
        xlabel="Coverage",
        ylabel="Selective risk (1 - precision@0.50)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    coverage = curves["angle_aware"][0]
    baseline_mean_risk = np.mean(
        [curves[model][1] for model in BASELINES], axis=0
    )
    risk_delta = (curves["angle_aware"][1] - baseline_mean_risk) * 1000.0
    axes[1].axhline(0.0, color="#777777", linestyle="--", linewidth=1)
    axes[1].plot(coverage, risk_delta, color=COLORS["angle_aware"], linewidth=1.8)
    axes[1].fill_between(
        coverage, 0.0, risk_delta, color=COLORS["angle_aware"], alpha=0.15
    )
    axes[1].set(
        xlabel="Coverage",
        ylabel=r"$\Delta$ risk (angle-aware - baseline mean) $\times 10^3$",
        xlim=(0, 1),
    )
    axes[1].grid(alpha=0.2)
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"trained_ablation_risk_coverage.{extension}", dpi=220)
    plt.close(fig)


def reliability_points(scores, targets, num_bins):
    ids = np.minimum((scores * num_bins).astype(np.int64), num_bins - 1)
    rows = []
    for bin_index in range(num_bins):
        mask = ids == bin_index
        if mask.any():
            rows.append((float(scores[mask].mean()), float(targets[mask].mean())))
    return np.asarray(rows)


def plot_reliability(output_dir, data, num_bins):
    fig, axes = plt.subplots(
        1, 2, figsize=(10.2, 4.2), constrained_layout=True
    )
    for axis, target_key, ylabel in (
        (axes[0], "tp50", "Empirical precision"),
        (axes[1], "q50", "Mean matched OBB quality"),
    ):
        axis.plot(
            [0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1
        )
        for model in MODELS:
            points = reliability_points(
                data[model]["score"], data[model][target_key], num_bins
            )
            axis.plot(
                points[:, 0],
                points[:, 1],
                marker="o",
                markersize=3,
                linewidth=1.6,
                label=NAMES[model],
                color=COLORS[model],
            )
        axis.set(
            xlabel="Mean confidence",
            ylabel=ylabel,
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"trained_ablation_reliability.{extension}", dpi=220
        )
    plt.close(fig)


def write_tables(output_dir, summaries):
    fields = ["model"] + list(next(iter(summaries.values())).keys())
    with (output_dir / "trained_ablation_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in MODELS:
            writer.writerow({"model": model, **summaries[model]})

    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Score & ECE$_{50}\\downarrow$ & NLL$_{50}\\downarrow$ & "
        "Brier$_{50}\\downarrow$ & AURC$_{50}\\downarrow$ & "
        "Q-ECE$\\downarrow$ & $\\rho\\uparrow$ \\\\",
        "\\midrule",
    ]
    for model in MODELS:
        item = summaries[model]
        lines.append(
            f"{NAMES[model]} & {item['ece50']:.4f} & {item['nll50']:.4f} & "
            f"{item['brier50']:.4f} & {item['aurc50']:.4f} & "
            f"{item['quality_ece50']:.4f} & "
            f"{item['confidence_quality_spearman']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output_dir / "trained_ablation_table.tex").write_text(
        "\n".join(lines) + "\n"
    )


def write_appendix_draft(output_dir, summaries, bootstrap):
    ours = summaries["angle_aware"]

    def baseline_mean(key):
        return float(np.mean([summaries[model][key] for model in BASELINES]))

    def baseline_range(key):
        values = [summaries[model][key] for model in BASELINES]
        return min(values), max(values)

    def reduction(key):
        base = baseline_mean(key)
        return 100.0 * (base - ours[key]) / base

    ece_base = baseline_mean("ece50")
    nll_base = baseline_mean("nll50")
    brier_base = baseline_mean("brier50")
    quality_ece_base = baseline_mean("quality_ece50")
    aurc_range = baseline_range("aurc50")
    rho_range = baseline_range("confidence_quality_spearman")
    ci1 = bootstrap["boundary_run1"]["ece50"]["ci95"]
    ci2 = bootstrap["boundary_run2"]["ece50"]["ci95"]

    text = f"""### Calibration over all retained predictions

To remove the selection conditions used in the conditional localization
analysis, we additionally evaluate all {11738 * 300:,} predictions retained by
the standard top-300 postprocessor on the DIOR-R test set. We do not filter
predictions by score, class correctness, or overlap before evaluation.
Predictions are matched to ground truth using class-aware, confidence-ordered,
one-to-one OBB matching. A prediction is positive at threshold $t$ only if it
has the correct class and an OBB IoU of at least $t$ with an unmatched ground
truth instance; unmatched, duplicate, and misclassified predictions are
negative.

We compare two independent runs of the trained boundary-only LQE ablation with
the final angle-aware LQE model. The two configurations differ by only 113
parameters. Relative to the mean of the two boundary-only runs, adding angle
reliability reduces ECE50 from {ece_base:.4f} to {ours['ece50']:.4f}
({reduction('ece50'):.1f}% relative), NLL50 from {nll_base:.4f} to
{ours['nll50']:.4f} ({reduction('nll50'):.1f}%), and Brier score from
{brier_base:.4f} to {ours['brier50']:.4f}
({reduction('brier50'):.1f}%). Quality-aware ECE, where the target is matched
OBB IoU for true positives and zero otherwise, decreases from
{quality_ece_base:.4f} to {ours['quality_ece50']:.4f}. Paired image-level
bootstrap gives 95% intervals of [{ci1[0]:.4f}, {ci1[1]:.4f}] and
[{ci2[0]:.4f}, {ci2[1]:.4f}] for the ECE50 difference against the two
boundary-only runs, respectively.

The ranking metrics are effectively unchanged: AURC50 of the angle-aware model
is {ours['aurc50']:.4f}, compared with a boundary-only range of
[{aurc_range[0]:.4f}, {aurc_range[1]:.4f}], and its confidence-quality
Spearman correlation is {ours['confidence_quality_spearman']:.4f}, compared
with [{rho_range[0]:.4f}, {rho_range[1]:.4f}]. These results support a scoped
interpretation: distribution statistics provide useful localization
reliability cues and reduce absolute calibration error, but should not be
interpreted as a fully calibrated posterior uncertainty model. Completely
missed objects have no associated prediction-level score and remain outside
the scope of this analysis.
"""
    (output_dir / "appendix_draft.md").write_text(text)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(args)
    summaries = {
        model: summarize(data[model], args.total_gt, args.num_bins)
        for model in MODELS
    }
    bootstrap = paired_bootstrap(data, args)
    result = {
        "setup": {
            "num_images": args.num_images,
            "topk": args.topk,
            "detections_per_model": args.num_images * args.topk,
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.seed,
        },
        "metrics": summaries,
        "paired_image_bootstrap": bootstrap,
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    write_tables(output_dir, summaries)
    write_appendix_draft(output_dir, summaries, bootstrap)
    plot_risk(output_dir, data)
    plot_reliability(output_dir, data, args.num_bins)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
