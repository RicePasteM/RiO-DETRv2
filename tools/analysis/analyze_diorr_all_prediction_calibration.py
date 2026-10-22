#!/usr/bin/env python3
"""Evaluate confidence calibration on every retained DIOR-R prediction.

The script extracts three score variants from one forward pass:
  1. raw classification logits,
  2. logits after boundary LQE,
  3. logits after boundary and angle LQE.

Each variant independently selects the standard top-k query/class pairs. No
score threshold is applied. Predictions are matched to GT with class-aware,
confidence-ordered, one-to-one OBB matching.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VARIANTS = ("raw", "boundary_lqe", "full_lqe")
DISPLAY_NAMES = {
    "raw": "Classification",
    "boundary_lqe": "Boundary LQE",
    "full_lqe": "Full LQE",
}
COLORS = {
    "raw": "#4C78A8",
    "boundary_lqe": "#F58518",
    "full_lqe": "#2A9D8F",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rio-diorr/riov2_hgnetv2_x_diorr.yml",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/riov2_hgnetv2_x_diorr/best_stg1.pth",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/analysis/riov2_hgnetv2_x_diorr_all_prediction_calibration",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument("--disable-angle-lqe", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def load_model_and_data(args):
    from engine.core import YAMLConfig

    cfg = YAMLConfig(args.config, resume=args.checkpoint)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    if args.disable_angle_lqe:
        cfg.yaml_cfg["RTDETRTransformerv2OBBFDR"]["angle_aware_lqe"] = False
    cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = args.batch_size
    cfg.yaml_cfg["val_dataloader"]["num_workers"] = args.workers

    model = cfg.model
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "ema" in checkpoint:
        state_dict = checkpoint["ema"]["module"]
        weight_source = "ema.module"
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
        weight_source = "model"
    else:
        state_dict = checkpoint
        weight_source = "checkpoint"
    incompatible = model.load_state_dict(state_dict, strict=True)
    print(
        f"Loaded {args.checkpoint} from {weight_source}; "
        f"incompatible={incompatible}",
        flush=True,
    )
    model.to(args.device).eval()
    return cfg, model, cfg.val_dataloader, checkpoint, weight_source


def register_score_hooks(model):
    decoder = model.decoder
    layer_index = decoder.eval_idx
    boundary_lqe = decoder.decoder.lqe_layers[layer_index]
    angle_lqe = decoder.decoder.angle_lqe
    captured = {}

    def save(name):
        def hook(_module, _inputs, output):
            captured[name] = output.detach()

        return hook

    handles = [
        decoder.dec_score_head[layer_index].register_forward_hook(save("raw")),
    ]
    if angle_lqe is None:
        def save_boundary_and_full(_module, _inputs, output):
            captured["boundary_lqe"] = output.detach()
            captured["full_lqe"] = output.detach()

        handles.append(boundary_lqe.register_forward_hook(save_boundary_and_full))
    else:
        handles.extend(
            [
                boundary_lqe.register_forward_hook(save("boundary_lqe")),
                angle_lqe.register_forward_hook(save("full_lqe")),
            ]
        )
    return captured, handles, layer_index, angle_lqe is not None


def select_topk(logits, boxes, topk):
    batch_size, num_queries, num_classes = logits.shape
    flat_scores = logits.sigmoid().flatten(1)
    topk = min(topk, flat_scores.shape[1])
    scores, flat_indices = torch.topk(
        flat_scores, topk, dim=1, largest=True, sorted=True
    )
    labels = flat_indices.remainder(num_classes)
    query_indices = torch.div(
        flat_indices, num_classes, rounding_mode="floor"
    )
    selected_boxes = boxes.gather(
        1, query_indices.unsqueeze(-1).expand(batch_size, topk, boxes.shape[-1])
    )
    return scores, labels, selected_boxes


def greedy_match(iou, det_labels, gt_labels, threshold):
    num_detections = len(det_labels)
    tp = np.zeros(num_detections, dtype=np.uint8)
    quality = np.zeros(num_detections, dtype=np.float32)
    if iou.shape[1] == 0:
        return tp, quality

    matched = np.zeros(iou.shape[1], dtype=bool)
    for det_index, label in enumerate(det_labels):
        candidates = np.flatnonzero((gt_labels == label) & ~matched)
        if candidates.size == 0:
            continue
        candidate_ious = iou[det_index, candidates]
        local_best = int(candidate_ious.argmax())
        best_gt = int(candidates[local_best])
        best_iou = float(candidate_ious[local_best])
        if best_iou >= threshold:
            matched[best_gt] = True
            tp[det_index] = 1
            quality[det_index] = best_iou
    return tp, quality


def fixed_width_bins(scores, targets, num_bins):
    rows = []
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    for index in range(num_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == num_bins - 1:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)
        count = int(mask.sum())
        rows.append(
            {
                "bin": index,
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_score": float(scores[mask].mean()) if count else None,
                "mean_target": float(targets[mask].mean()) if count else None,
            }
        )
    return rows


def expected_calibration_error(scores, targets, num_bins):
    total = len(scores)
    return float(
        sum(
            row["count"]
            / total
            * abs(row["mean_score"] - row["mean_target"])
            for row in fixed_width_bins(scores, targets, num_bins)
            if row["count"]
        )
    )


def adaptive_calibration_error(scores, targets, num_bins):
    order = np.argsort(scores, kind="stable")
    chunks = np.array_split(order, num_bins)
    return float(
        sum(
            len(chunk)
            / len(scores)
            * abs(float(scores[chunk].mean()) - float(targets[chunk].mean()))
            for chunk in chunks
            if len(chunk)
        )
    )


def binary_nll(scores, targets):
    scores = np.clip(scores.astype(np.float64), 1e-7, 1.0 - 1e-7)
    targets = targets.astype(np.float64)
    return float(
        -np.mean(targets * np.log(scores) + (1.0 - targets) * np.log1p(-scores))
    )


def brier_score(scores, targets):
    return float(np.mean((scores.astype(np.float64) - targets) ** 2))


def aurc(scores, targets):
    order = np.argsort(-scores, kind="stable")
    errors = 1.0 - targets[order].astype(np.float64)
    risks = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    value = float(risks.mean())
    error_rate = float(errors.mean())
    accuracy = 1.0 - error_rate
    optimal = error_rate + accuracy * math.log(max(accuracy, 1e-12))
    return value, float(value - optimal)


def correlation(x, y):
    spearman = spearmanr(x, y).statistic
    pearson = pearsonr(x, y).statistic
    return {
        "spearman": float(spearman),
        "pearson": float(pearson),
    }


def risk_coverage_curve(scores, targets, points=400):
    order = np.argsort(-scores, kind="stable")
    errors = 1.0 - targets[order].astype(np.float64)
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    counts = np.unique(
        np.clip(
            np.rint(np.geomspace(1, len(errors), points)).astype(np.int64),
            1,
            len(errors),
        )
    )
    return {
        "coverage": counts.astype(np.float64) / len(errors),
        "risk": cumulative_risk[counts - 1],
        "count": counts,
    }


def summarize_variant(data, total_gt, num_bins):
    scores = data["score"]
    tp50 = data["tp50"].astype(np.float64)
    tp75 = data["tp75"].astype(np.float64)
    q50 = data["q50"].astype(np.float64)
    aurc50, eaurc50 = aurc(scores, tp50)
    aurc75, eaurc75 = aurc(scores, tp75)
    return {
        "detections": int(len(scores)),
        "gt_instances": int(total_gt),
        "mean_score": float(scores.mean()),
        "precision50_all_topk": float(tp50.mean()),
        "precision75_all_topk": float(tp75.mean()),
        "recall50": float(tp50.sum() / total_gt),
        "recall75": float(tp75.sum() / total_gt),
        "ece50": expected_calibration_error(scores, tp50, num_bins),
        "aece50": adaptive_calibration_error(scores, tp50, num_bins),
        "nll50": binary_nll(scores, tp50),
        "brier50": brier_score(scores, tp50),
        "aurc50": aurc50,
        "eaurc50": eaurc50,
        "ece75": expected_calibration_error(scores, tp75, num_bins),
        "aece75": adaptive_calibration_error(scores, tp75, num_bins),
        "nll75": binary_nll(scores, tp75),
        "brier75": brier_score(scores, tp75),
        "aurc75": aurc75,
        "eaurc75": eaurc75,
        "quality_ece50": expected_calibration_error(scores, q50, num_bins),
        "quality_brier50": brier_score(scores, q50),
        "confidence_quality_correlation": correlation(scores, q50),
    }


def write_metric_table(path, summaries):
    fields = [
        "variant",
        "detections",
        "precision50_all_topk",
        "recall50",
        "ece50",
        "aece50",
        "nll50",
        "brier50",
        "aurc50",
        "eaurc50",
        "ece75",
        "nll75",
        "brier75",
        "aurc75",
        "quality_ece50",
        "quality_brier50",
        "confidence_quality_spearman",
        "confidence_quality_pearson",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant in VARIANTS:
            summary = summaries[variant]
            row = {field: summary.get(field) for field in fields}
            row["variant"] = variant
            row["confidence_quality_spearman"] = summary[
                "confidence_quality_correlation"
            ]["spearman"]
            row["confidence_quality_pearson"] = summary[
                "confidence_quality_correlation"
            ]["pearson"]
            writer.writerow(row)


def write_appendix_table(path, summaries):
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Score & ECE$_{50}\\downarrow$ & NLL$_{50}\\downarrow$ & "
        "Brier$_{50}\\downarrow$ & AURC$_{50}\\downarrow$ & "
        "Q-ECE$\\downarrow$ & $\\rho$(Conf., Quality)$\\uparrow$ \\\\",
        "\\midrule",
    ]
    for variant in VARIANTS:
        item = summaries[variant]
        lines.append(
            f"{DISPLAY_NAMES[variant]} & "
            f"{item['ece50']:.4f} & {item['nll50']:.4f} & "
            f"{item['brier50']:.4f} & {item['aurc50']:.4f} & "
            f"{item['quality_ece50']:.4f} & "
            f"{item['confidence_quality_correlation']['spearman']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def write_curve_csv(path, curves):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "coverage", "risk", "count"])
        for variant in VARIANTS:
            curve = curves[variant]
            for coverage, risk, count in zip(
                curve["coverage"], curve["risk"], curve["count"]
            ):
                writer.writerow([variant, coverage, risk, count])


def plot_reliability(output_dir, all_data, num_bins, target_key, suffix, ylabel):
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    for variant in VARIANTS:
        data = all_data[variant]
        rows = fixed_width_bins(data["score"], data[target_key], num_bins)
        x = [row["mean_score"] for row in rows if row["count"]]
        y = [row["mean_target"] for row in rows if row["count"]]
        ax.plot(
            x,
            y,
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=DISPLAY_NAMES[variant],
            color=COLORS[variant],
        )
    ax.set(xlabel="Mean confidence", ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"reliability_{suffix}.{extension}", dpi=220)
    plt.close(fig)


def plot_risk_coverage(output_dir, curves):
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    for variant in VARIANTS:
        curve = curves[variant]
        ax.plot(
            curve["coverage"],
            curve["risk"],
            linewidth=1.8,
            label=DISPLAY_NAMES[variant],
            color=COLORS[variant],
        )
    ax.set(
        xlabel="Coverage",
        ylabel="Selective risk (1 - precision@0.50)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"risk_coverage_tp50.{extension}", dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    cfg, model, loader, checkpoint, weight_source = load_model_and_data(args)
    captured, handles, layer_index, angle_lqe_enabled = register_score_hooks(model)
    storage = {
        variant: {"score": [], "tp50": [], "tp75": [], "q50": []}
        for variant in VARIANTS
    }
    total_gt = 0
    processed_images = 0

    try:
        with torch.inference_mode():
            for batch_index, (samples, targets) in enumerate(loader):
                if args.max_images is not None and processed_images >= args.max_images:
                    break
                samples = samples.to(args.device)
                captured.clear()
                outputs = model(samples)
                missing = set(VARIANTS) - set(captured)
                if missing:
                    raise RuntimeError(f"Hooks did not capture: {sorted(missing)}")
                if not torch.allclose(
                    captured["full_lqe"], outputs["pred_logits"], atol=1e-6, rtol=1e-5
                ):
                    raise RuntimeError("Full-LQE hook output differs from model output")

                boxes = outputs["pred_boxes"].detach()
                selections = {
                    variant: select_topk(captured[variant], boxes, args.topk)
                    for variant in VARIANTS
                }
                batch_size = samples.shape[0]
                if args.max_images is not None:
                    batch_size = min(
                        batch_size, args.max_images - processed_images
                    )

                for sample_index in range(batch_size):
                    gt_boxes = targets[sample_index]["boxes"].to(
                        args.device, dtype=torch.float32
                    )
                    gt_labels = (
                        targets[sample_index]["labels"].cpu().numpy().astype(np.int64)
                    )
                    total_gt += len(gt_labels)

                    selected_boxes = torch.cat(
                        [
                            selections[variant][2][sample_index]
                            for variant in VARIANTS
                        ],
                        dim=0,
                    ).float()
                    selected_boxes = selected_boxes.clone()
                    selected_boxes[:, 2:4].clamp_(min=1e-6)
                    selected_boxes[:, 4].mul_(math.pi)
                    gt_for_iou = gt_boxes.clone()
                    if len(gt_for_iou):
                        gt_for_iou[:, 2:4].clamp_(min=1e-6)
                        gt_for_iou[:, 4].mul_(math.pi)
                        iou = (
                            box_iou_rotated(
                                selected_boxes.contiguous(),
                                gt_for_iou.contiguous(),
                                clockwise=True,
                            )
                            .cpu()
                            .numpy()
                        )
                    else:
                        iou = np.zeros(
                            (selected_boxes.shape[0], 0), dtype=np.float32
                        )

                    offset = 0
                    for variant in VARIANTS:
                        scores, labels, selected = selections[variant]
                        count = selected.shape[1]
                        variant_iou = iou[offset : offset + count]
                        variant_labels = (
                            labels[sample_index].cpu().numpy().astype(np.int64)
                        )
                        tp50, q50 = greedy_match(
                            variant_iou, variant_labels, gt_labels, 0.50
                        )
                        tp75, _ = greedy_match(
                            variant_iou, variant_labels, gt_labels, 0.75
                        )
                        storage[variant]["score"].append(
                            scores[sample_index].float().cpu().numpy()
                        )
                        storage[variant]["tp50"].append(tp50)
                        storage[variant]["tp75"].append(tp75)
                        storage[variant]["q50"].append(q50)
                        offset += count

                processed_images += batch_size
                if processed_images % 160 == 0 or batch_index == 0:
                    elapsed = time.time() - started
                    print(
                        f"Processed {processed_images} images "
                        f"({elapsed / 60.0:.1f} min)",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()

    all_data = {
        variant: {
            key: np.concatenate(chunks)
            for key, chunks in variant_storage.items()
        }
        for variant, variant_storage in storage.items()
    }
    expected_count = processed_images * args.topk
    for variant in VARIANTS:
        if len(all_data[variant]["score"]) != expected_count:
            raise RuntimeError(
                f"{variant} has {len(all_data[variant]['score'])} rows; "
                f"expected {expected_count}"
            )

    summaries = {
        variant: summarize_variant(all_data[variant], total_gt, args.num_bins)
        for variant in VARIANTS
    }
    curves = {
        variant: risk_coverage_curve(
            all_data[variant]["score"], all_data[variant]["tp50"]
        )
        for variant in VARIANTS
    }

    checkpoint_epoch = checkpoint.get("last_epoch")
    metadata = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_weights": weight_source,
        "images": processed_images,
        "gt_instances": total_gt,
        "decoder_layer": layer_index,
        "angle_lqe_enabled": angle_lqe_enabled,
        "topk_per_image_per_variant": args.topk,
        "score_threshold": None,
        "matching": (
            "class-aware, confidence-ordered, one-to-one OBB matching "
            "independently at IoU 0.50 and 0.75"
        ),
        "quality_definition": "q50 = matched IoU for TP50, otherwise 0",
        "ece_bins": args.num_bins,
        "ece_binning": "equal-width",
        "aece_binning": "equal-count",
        "elapsed_seconds": time.time() - started,
        "metrics": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    write_metric_table(output_dir / "metrics.csv", summaries)
    write_appendix_table(output_dir / "appendix_table.tex", summaries)
    write_curve_csv(output_dir / "risk_coverage_tp50.csv", curves)

    reliability_rows = []
    for variant in VARIANTS:
        for target_key in ("tp50", "tp75", "q50"):
            for row in fixed_width_bins(
                all_data[variant]["score"],
                all_data[variant][target_key],
                args.num_bins,
            ):
                reliability_rows.append(
                    {"variant": variant, "target": target_key, **row}
                )
    with (output_dir / "reliability_bins.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "target",
                "bin",
                "lower",
                "upper",
                "count",
                "mean_score",
                "mean_target",
            ],
        )
        writer.writeheader()
        writer.writerows(reliability_rows)

    plot_reliability(
        output_dir, all_data, args.num_bins, "tp50", "tp50", "Empirical precision"
    )
    plot_reliability(
        output_dir,
        all_data,
        args.num_bins,
        "q50",
        "quality50",
        "Mean matched OBB quality",
    )
    plot_risk_coverage(output_dir, curves)

    if args.save_predictions:
        arrays = {}
        for variant in VARIANTS:
            for key, value in all_data[variant].items():
                arrays[f"{variant}_{key}"] = value
        np.savez_compressed(output_dir / "all_predictions_top300.npz", **arrays)

    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Results written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
