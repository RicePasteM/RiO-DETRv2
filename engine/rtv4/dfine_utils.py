"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import math

import torch
import torch.nn.functional as F
from .box_ops import box_xyxy_to_cxcywh


def weighting_function(reg_max, up, reg_scale, deploy=False):
    """
    Generates the non-uniform Weighting Function W(n) for bounding box regression.

    Args:
        reg_max (int): Max number of the discrete bins.
        up (Tensor): Controls upper bounds of the sequence,
                     where maximum offset is ±up * H / W.
        reg_scale (float): Controls the curvature of the Weighting Function.
                           Larger values result in flatter weights near the central axis W(reg_max/2)=0
                           and steeper weights at both ends.
        deploy (bool): If True, uses deployment mode settings.

    Returns:
        Tensor: Sequence of Weighting Function.
    """
    if deploy:
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-(step) ** i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
        return torch.tensor(values, dtype=up.dtype, device=up.device)
    else:
        upper_bound1 = abs(up[0]) * abs(reg_scale)
        upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-(step) ** i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
        return torch.cat(values, 0)


def wrap_angle_01(angle):
    """Wrap normalized angles to [0, 1)."""
    wrapped = torch.fmod(angle, 1.0)
    return torch.where(wrapped < 0, wrapped + 1.0, wrapped)


def wrap_angle_half(angle):
    """Wrap normalized angle residuals to [-0.5, 0.5)."""
    return wrap_angle_01(angle + 0.5) - 0.5


def decode_circular_angle_delta(logits):
    """Decode circular logits into the shortest normalized angle residual."""
    num_bins = logits.shape[-1]
    probs = F.softmax(logits, dim=-1)
    bin_ids = torch.arange(num_bins, device=logits.device, dtype=logits.dtype)
    phase = bin_ids * (2.0 * math.pi / num_bins)
    sin_mean = (probs * torch.sin(phase)).sum(dim=-1, keepdim=True)
    cos_mean = (probs * torch.cos(phase)).sum(dim=-1, keepdim=True)
    return torch.atan2(sin_mean, cos_mean) / (2.0 * math.pi)


def decode_linear_angle_delta(logits, max_delta):
    """Decode logits over a bounded linear residual interval."""
    if max_delta <= 0:
        raise ValueError(f'max_delta must be positive, got {max_delta}')

    values = torch.linspace(
        -max_delta,
        max_delta,
        logits.shape[-1],
        dtype=logits.dtype,
        device=logits.device,
    )
    return (F.softmax(logits, dim=-1) * values).sum(dim=-1, keepdim=True)


def angle_to_circular_targets(ref_angle, target_angle, num_bins):
    """Encode target angles as circular residual soft labels on uniform bins."""
    delta = wrap_angle_half(target_angle - ref_angle).reshape(-1)
    delta_unit = wrap_angle_01(delta)
    position = delta_unit * num_bins

    left_unwrapped = torch.floor(position).long()
    weight_right = position - left_unwrapped.to(position.dtype)
    weight_left = 1.0 - weight_right
    left = left_unwrapped % num_bins
    right = (left + 1) % num_bins

    return left, right, weight_left, weight_right


def angle_to_linear_targets(ref_angle, target_angle, num_bins, max_delta):
    """Encode shortest residuals on a bounded, non-circular local grid."""
    if num_bins < 2:
        raise ValueError(f'num_bins must be at least 2, got {num_bins}')
    if max_delta <= 0:
        raise ValueError(f'max_delta must be positive, got {max_delta}')

    delta = wrap_angle_half(target_angle - ref_angle).reshape(-1)
    delta = delta.clamp(min=-max_delta, max=max_delta)
    position = (delta + max_delta) * ((num_bins - 1) / (2.0 * max_delta))

    left = torch.floor(position).long().clamp(min=0, max=num_bins - 1)
    right = (left + 1).clamp(max=num_bins - 1)
    weight_right = position - left.to(position.dtype)
    weight_right = torch.where(left == right, torch.zeros_like(weight_right), weight_right)
    weight_left = 1.0 - weight_right

    return left, right, weight_left, weight_right


def translate_gt(gt, reg_max, reg_scale, up):
    """
    Decodes bounding box ground truth (GT) values into distribution-based GT representations.

    This function maps continuous GT values into discrete distribution bins, which can be used
    for regression tasks in object detection models. It calculates the indices of the closest
    bins to each GT value and assigns interpolation weights to these bins based on their proximity
    to the GT value.

    Args:
        gt (Tensor): Ground truth bounding box values, shape (N, ).
        reg_max (int): Maximum number of discrete bins for the distribution.
        reg_scale (float): Controls the curvature of the Weighting Function.
        up (Tensor): Controls the upper bounds of the Weighting Function.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - indices (Tensor): Index of the left bin closest to each GT value, shape (N, ).
            - weight_right (Tensor): Weight assigned to the right bin, shape (N, ).
            - weight_left (Tensor): Weight assigned to the left bin, shape (N, ).
    """
    gt = gt.reshape(-1)
    function_values = weighting_function(reg_max, up, reg_scale)

    # Find the closest left-side indices for each value
    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
    mask = diffs <= 0
    closest_left_indices = torch.sum(mask, dim=1) - 1

    # Calculate the weights for the interpolation
    indices = closest_left_indices.float()

    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)

    valid_idx_mask = (indices >= 0) & (indices < reg_max)
    valid_indices = indices[valid_idx_mask].long()

    # Obtain distances
    left_values = function_values[valid_indices]
    right_values = function_values[valid_indices + 1]

    left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
    right_diffs = torch.abs(right_values - gt[valid_idx_mask])

    # Valid weights
    weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]

    # Invalid weights (out of range)
    invalid_idx_mask_neg = (indices < 0)
    weight_right[invalid_idx_mask_neg] = 0.0
    weight_left[invalid_idx_mask_neg] = 1.0
    indices[invalid_idx_mask_neg] = 0.0

    invalid_idx_mask_pos = (indices >= reg_max)
    weight_right[invalid_idx_mask_pos] = 1.0
    weight_left[invalid_idx_mask_pos] = 0.0
    indices[invalid_idx_mask_pos] = reg_max - 0.1

    return indices, weight_right, weight_left


def distance2bbox(points, distance, reg_scale):
    """
    Decodes edge-distances into bounding box coordinates.

    Args:
        points (Tensor): (B, N, 4) or (N, 4) format, representing [x, y, w, h],
                         where (x, y) is the center and (w, h) are width and height.
        distance (Tensor): (B, N, 4) or (N, 4), representing distances from the
                           point to the left, top, right, and bottom boundaries.

        reg_scale (float): Controls the curvature of the Weighting Function.

    Returns:
        Tensor: Bounding boxes in (N, 4) or (B, N, 4) format [cx, cy, w, h].
    """
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    return box_xyxy_to_cxcywh(bboxes)


def _rotate_points(points, angle):
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    x = points[..., 0]
    y = points[..., 1]
    return torch.stack([x * cosa - y * sina, x * sina + y * cosa], dim=-1)


def distance2obb_local(ref_obbs, distance, target_angle, reg_scale):
    """
    Decodes local-frame edge distances into [cx, cy, w, h].

    The four distances are interpreted in the predicted target box's local
    frame, while the reference box still provides the center anchor and width /
    height normalization used by FDR.
    """
    reg_scale = abs(reg_scale)
    ref_xy = ref_obbs[..., :2]
    ref_wh = ref_obbs[..., 2:4].clamp(min=1e-16)
    target_angle = target_angle[..., 0] * math.pi

    left = (0.5 * reg_scale + distance[..., 0]) * (ref_wh[..., 0] / reg_scale)
    top = (0.5 * reg_scale + distance[..., 1]) * (ref_wh[..., 1] / reg_scale)
    right = (0.5 * reg_scale + distance[..., 2]) * (ref_wh[..., 0] / reg_scale)
    bottom = (0.5 * reg_scale + distance[..., 3]) * (ref_wh[..., 1] / reg_scale)

    local_center = torch.stack([(right - left) * 0.5, (bottom - top) * 0.5], dim=-1)
    global_center = ref_xy + _rotate_points(local_center, target_angle)
    wh = torch.stack([left + right, top + bottom], dim=-1)
    return torch.cat([global_center, wh], dim=-1)


def obb2distance_local(ref_obbs, target_obbs, reg_max, reg_scale, up, eps=0.1):
    """
    Encodes target [cx, cy, w, h, angle] as distances in the target box's
    local frame, measured from the reference center point.
    """
    reg_scale = abs(reg_scale)
    ref_xy = ref_obbs[..., :2]
    ref_wh = ref_obbs[..., 2:4].clamp(min=1e-16)
    target_angle = target_obbs[..., 4] * math.pi

    delta = target_obbs[..., :2] - ref_xy
    local_delta = _rotate_points(delta, -target_angle)
    target_wh = target_obbs[..., 2:4]

    left = (target_wh[..., 0] * 0.5 - local_delta[..., 0]) / (ref_wh[..., 0] / reg_scale + 1e-16) - 0.5 * reg_scale
    top = (target_wh[..., 1] * 0.5 - local_delta[..., 1]) / (ref_wh[..., 1] / reg_scale + 1e-16) - 0.5 * reg_scale
    right = (target_wh[..., 0] * 0.5 + local_delta[..., 0]) / (ref_wh[..., 0] / reg_scale + 1e-16) - 0.5 * reg_scale
    bottom = (target_wh[..., 1] * 0.5 + local_delta[..., 1]) / (ref_wh[..., 1] / reg_scale + 1e-16) - 0.5 * reg_scale

    four_lens = torch.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = translate_gt(four_lens, reg_max, reg_scale, up)
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max - eps)
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()


def bbox2distance(points, bbox, reg_max, reg_scale, up, eps=0.1):
    """
    Converts bounding box coordinates to distances from a reference point.

    Args:
        points (Tensor): (n, 4) [x, y, w, h], where (x, y) is the center.
        bbox (Tensor): (n, 4) bounding boxes in "xyxy" format.
        reg_max (float): Maximum bin value.
        reg_scale (float): Controling curvarture of W(n).
        up (Tensor): Controling upper bounds of W(n).
        eps (float): Small value to ensure target < reg_max.

    Returns:
        Tensor: Decoded distances.
    """
    reg_scale = abs(reg_scale)
    left   = (points[:, 0] - bbox[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    top    = (points[:, 1] - bbox[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    right  = (bbox[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale + 1e-16) - 0.5 * reg_scale
    bottom = (bbox[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale + 1e-16) - 0.5 * reg_scale
    four_lens = torch.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = translate_gt(four_lens, reg_max, reg_scale, up)
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max-eps)
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()
