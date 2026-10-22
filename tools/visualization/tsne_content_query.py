"""
t-SNE / UMAP Visualization of Decoder Content Query Features (Q_content).

Extracts the last decoder layer's output features for TP detections,
then plots two t-SNE maps:
  A) colored by GT category   — shows category clustering
  B) colored by GT angle θ    — shows angle gradient within clusters
  C) per-category subplots    — highlights intra-class angle gradient
"""

import os
import sys
import argparse
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.core import YAMLConfig, yaml_utils
from engine.misc import dist_utils
from engine.solver._solver import remove_module_prefix
from engine.data.dataset.fair1m_dataset import FAIR1M_CLASSES


# ---------------------------------------------------------------------------
# OBB IoU utilities (copied from analyze_tp_angle_evolution.py)
# ---------------------------------------------------------------------------

def obb_iou_single(box1, box2):
    cx1, cy1, w1, h1, a1 = [float(x) for x in box1]
    cx2, cy2, w2, h2, a2 = [float(x) for x in box2]

    scale = 1000.0
    cx1 *= scale; cy1 *= scale; w1 *= scale; h1 *= scale
    cx2 *= scale; cy2 *= scale; w2 *= scale; h2 *= scale

    angle1 = a1 * 180.0
    angle2 = a2 * 180.0

    rect1 = ((cx1, cy1), (max(w1, 1e-6), max(h1, 1e-6)), angle1)
    rect2 = ((cx2, cy2), (max(w2, 1e-6), max(h2, 1e-6)), angle2)

    try:
        ret, inter_pts = cv2.rotatedRectangleIntersection(rect1, rect2)
        if inter_pts is None or ret == cv2.INTERSECT_NONE:
            inter_area = 0.0
        else:
            inter_area = float(cv2.contourArea(inter_pts))
    except Exception:
        inter_area = 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def obb_iou_matrix(boxes1, boxes2):
    n1, n2 = boxes1.shape[0], boxes2.shape[0]
    if n1 == 0 or n2 == 0:
        return np.zeros((n1, n2), dtype=np.float32)
    ious = np.zeros((n1, n2), dtype=np.float32)
    for i in range(n1):
        for j in range(n2):
            ious[i, j] = obb_iou_single(boxes1[i], boxes2[j])
    return ious


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.update:
        update_dict = yaml_utils.parse_cli(args.update)
        for k, v in update_dict.items():
            if k in cfg.yaml_cfg:
                if isinstance(cfg.yaml_cfg[k], dict):
                    cfg.yaml_cfg[k].update(v)
            else:
                cfg.yaml_cfg[k] = v

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    # ---- load model --------------------------------------------------------
    ckpt_path = args.resume
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    model_state = state["ema"]["module"] if "ema" in state else state["model"]
    model_state = remove_module_prefix(model_state)
    cfg.model.load_state_dict(model_state, strict=False)

    model = cfg.model.to(device)
    model.eval()

    # ---- register hook on the last decoder layer ---------------------------
    obb_decoder = model.decoder.decoder          # TransformerDecoderOBB
    eval_idx = obb_decoder.eval_idx
    last_layer = obb_decoder.layers[eval_idx]

    captured_features = []

    def hook_fn(_module, _input, output):
        captured_features.append(output.detach().cpu())

    hook_handle = last_layer.register_forward_hook(hook_fn)

    # ---- build val dataloader ----------------------------------------------
    val_loader = cfg.val_dataloader
    val_loader = dist_utils.warp_loader(val_loader, shuffle=False)

    # ---- iterate and collect -----------------------------------------------
    all_features = []
    all_categories = []
    all_angles = []

    score_thr = args.score_threshold
    iou_thr = args.iou_threshold
    max_feat = args.max_features

    print(f"Device: {device}")
    print(f"Score threshold: {score_thr}, IoU threshold: {iou_thr}")
    print(f"Max features to collect: {max_feat}")
    print(f"Decoder eval layer index: {eval_idx}")
    print("Extracting features …")

    with torch.no_grad():
        for batch_idx, (samples, targets) in enumerate(val_loader):
            if len(all_features) >= max_feat:
                break

            samples = samples.to(device)
            targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]

            captured_features.clear()
            outputs = model(samples, targets_dev)

            content_feats = captured_features[0]       # [bs, nq, dim]
            pred_logits = outputs["pred_logits"].cpu()  # [bs, nq, C]
            pred_boxes = outputs["pred_boxes"].cpu()    # [bs, nq, 5]

            bs = pred_logits.shape[0]

            for img_i in range(bs):
                if len(all_features) >= max_feat:
                    break

                gt_boxes = targets[img_i]["boxes"].numpy()
                gt_labels = targets[img_i]["labels"].numpy()
                if len(gt_boxes) == 0:
                    continue

                scores = pred_logits[img_i].sigmoid()        # [nq, C]
                max_scores, max_labels = scores.max(dim=-1)   # [nq]

                valid = max_scores > score_thr
                v_idx = torch.where(valid)[0]
                if len(v_idx) == 0:
                    continue

                v_boxes = pred_boxes[img_i, v_idx].numpy()
                v_labels = max_labels[v_idx].numpy()
                v_scores = max_scores[v_idx].numpy()

                order = np.argsort(-v_scores)
                v_idx = v_idx[order]
                v_boxes = v_boxes[order]
                v_labels = v_labels[order]

                iou_mat = obb_iou_matrix(v_boxes, gt_boxes)

                matched_gt = set()
                for p_i in range(len(v_idx)):
                    if len(all_features) >= max_feat:
                        break
                    row = iou_mat[p_i]
                    if row.max() < iou_thr:
                        continue
                    best_gt = int(row.argmax())
                    if best_gt in matched_gt:
                        continue
                    if v_labels[p_i] != gt_labels[best_gt]:
                        continue

                    matched_gt.add(best_gt)
                    q_idx = v_idx[p_i].item()
                    all_features.append(content_feats[img_i, q_idx].numpy())
                    all_categories.append(int(gt_labels[best_gt]))
                    all_angles.append(float(gt_boxes[best_gt, 4]))

            if (batch_idx + 1) % 20 == 0:
                print(f"  batch {batch_idx + 1:>5d} | collected {len(all_features)} features")

    hook_handle.remove()
    n_total = len(all_features)
    print(f"\nTotal TP features collected: {n_total}")

    if n_total < 50:
        print("Too few features – try lowering --score-threshold or processing more images.")
        return

    features = np.stack(all_features)           # [N, dim]
    categories = np.array(all_categories)        # [N]
    angles_deg = np.array(all_angles) * 180.0    # [0, 180]

    print("\nPer-category counts:")
    for c in np.unique(categories):
        print(f"  [{c:>2d}] {FAIR1M_CLASSES[c]:<25s}: {(categories == c).sum()}")

    # ---- dimensionality reduction ------------------------------------------
    print(f"\nRunning {args.method.upper()} …")
    if args.method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42,
                       perplexity=min(30, n_total - 1),
                       n_iter=1000, learning_rate="auto", init="pca")
    else:
        try:
            import umap
        except ImportError:
            print("umap-learn not installed, falling back to t-SNE")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42,
                           perplexity=min(30, n_total - 1),
                           n_iter=1000, learning_rate="auto", init="pca")
        else:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15,
                                min_dist=0.1, metric="cosine")

    emb_2d = reducer.fit_transform(features)

    # ---- plotting ----------------------------------------------------------
    print("Plotting …")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    os.makedirs(args.output_dir, exist_ok=True)
    tag = args.method

    # ===================== Figure A: by category ============================
    fig, ax = plt.subplots(figsize=(18, 14))
    unique_cats = np.unique(categories)
    n_cats = len(unique_cats)
    cmap_cat = plt.cm.get_cmap("tab20" if n_cats <= 20 else "gist_ncar", n_cats)

    for i, c in enumerate(unique_cats):
        mask = categories == c
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=[cmap_cat(i)], s=12, alpha=0.7, label=FAIR1M_CLASSES[c],
                   edgecolors="none")

    ax.set_title(f"Content Query {tag.upper()} — Colored by Category", fontsize=16)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
              markerscale=2.5, ncol=2, framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    path_a = os.path.join(args.output_dir, f"{tag}_by_category.png")
    fig.savefig(path_a, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path_a}")
    plt.close(fig)

    # ===================== Figure B: by angle (global) ======================
    fig, ax = plt.subplots(figsize=(15, 12))
    norm = Normalize(vmin=0, vmax=180)
    sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1],
                    c=angles_deg, cmap="hsv", norm=norm,
                    s=12, alpha=0.7, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Angle θ (degrees)", shrink=0.8)
    ax.set_title(f"Content Query {tag.upper()} — Colored by Angle θ", fontsize=16)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    path_b = os.path.join(args.output_dir, f"{tag}_by_angle.png")
    fig.savefig(path_b, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path_b}")
    plt.close(fig)

    # ===================== Figure C: per-category angle =====================
    top_cats = sorted(unique_cats,
                      key=lambda c: (categories == c).sum(), reverse=True)
    n_show = min(12, len(top_cats))
    top_cats = top_cats[:n_show]

    n_cols = 4
    n_rows = math.ceil(n_show / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, c in enumerate(top_cats):
        r, col = divmod(idx, n_cols)
        ax = axes[r, col]
        mask = categories == c
        sc = ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                        c=angles_deg[mask], cmap="hsv", norm=norm,
                        s=20, alpha=0.8, edgecolors="none")
        ax.set_title(f"{FAIR1M_CLASSES[c]} (n={mask.sum()})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, label="θ°", shrink=0.8)

    for idx in range(n_show, n_rows * n_cols):
        r, col = divmod(idx, n_cols)
        axes[r, col].axis("off")

    fig.suptitle(f"Per-Category {tag.upper()} with Angle Coloring", fontsize=15, y=1.01)
    plt.tight_layout()
    path_c = os.path.join(args.output_dir, f"{tag}_per_category_angle.png")
    fig.savefig(path_c, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path_c}")
    plt.close(fig)

    # ===================== Figure D: side-by-side summary ===================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12))

    for i, c in enumerate(unique_cats):
        mask = categories == c
        ax1.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                    c=[cmap_cat(i)], s=10, alpha=0.6, label=FAIR1M_CLASSES[c],
                    edgecolors="none")
    ax1.set_title("(A) Colored by Category", fontsize=14)
    ax1.legend(bbox_to_anchor=(0, -0.08), loc="upper left", fontsize=6,
               markerscale=2, ncol=5, framealpha=0.9)
    ax1.set_xticks([]); ax1.set_yticks([])

    sc2 = ax2.scatter(emb_2d[:, 0], emb_2d[:, 1],
                      c=angles_deg, cmap="hsv", norm=norm,
                      s=10, alpha=0.6, edgecolors="none")
    plt.colorbar(sc2, ax=ax2, label="θ°", shrink=0.8)
    ax2.set_title("(B) Colored by Angle θ", fontsize=14)
    ax2.set_xticks([]); ax2.set_yticks([])

    fig.suptitle(f"Decoder Content Query {tag.upper()} Embedding (N={n_total})",
                 fontsize=16, y=1.02)
    plt.tight_layout()
    path_d = os.path.join(args.output_dir, f"{tag}_summary.png")
    fig.savefig(path_d, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path_d}")
    plt.close(fig)

    # ---- save raw data for later analysis ----------------------------------
    np.savez(os.path.join(args.output_dir, "tsne_data.npz"),
             features=features, emb_2d=emb_2d,
             categories=categories, angles_deg=angles_deg,
             class_names=np.array(FAIR1M_CLASSES))
    print(f"  Saved raw data: {os.path.join(args.output_dir, 'tsne_data.npz')}")

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="t-SNE / UMAP visualization of decoder content query features")
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--resume", "-r", type=str, required=True)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="./outputs/tsne_analysis")
    parser.add_argument("--max-features", type=int, default=10000,
                        help="max number of TP features to collect")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "umap"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--update", "-u", nargs="+")
    parser.add_argument("--print-method", type=str, default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)

    args = parser.parse_args()
    main(args)
