"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.distributed
import torch.nn.functional as F
import torchvision

import copy

from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .dfine_utils import angle_to_circular_targets, angle_to_linear_targets, obb2distance_local
from .utils_obb import kld_loss

from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized
from ..core import register

import logging

_logger = logging.getLogger(__name__)


@register()
class RTv4OBBCriterion(nn.Module):
    """ This class computes the loss for RT-DETRv4.
    """
    __share__ = ['num_classes', ]
    __inject__ = ['matcher', ]

    def __init__(self, \
                 matcher,
                 weight_dict,
                 losses,
                 alpha=0.2,
                 gamma=2.0,
                 num_classes=80,
                 reg_max=32,
                 boxes_weight_format=None,
                 use_uni_set=True,
                 distill_adaptive_params=None,
                 ):
        """Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_classes: number of object categories, omitting the special no-object category.
            boxes_weight_format: format for boxes weight (iou, ).
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.boxes_weight_format = boxes_weight_format
        self.alpha = alpha
        self.gamma = gamma
        self.use_uni_set = use_uni_set
        self.reg_max = reg_max
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.num_pos, self.num_neg = None, None
        self.distill_adaptive_params = distill_adaptive_params

    def loss_distillation(self, outputs, targets, indices, num_boxes, **kwargs):
        student_feature_map = outputs.get('student_distill_output')
        teacher_feature_map = outputs.get('teacher_encoder_output')

        if student_feature_map is None or teacher_feature_map is None:
            device = next(
                (v.device for v in outputs.values() if isinstance(v, torch.Tensor)),
                torch.device('cpu'),
            )
            return {
                'loss_distill': torch.tensor(
                    0.0,
                    device=device,
                    requires_grad=True,
                )
            }

        if student_feature_map.shape[1] != teacher_feature_map.shape[1]:
            _logger.error(
                "[RTv4OBBCriterion] Feature dimension mismatch! "
                f"Student: {student_feature_map.shape[1]}, "
                f"Teacher: {teacher_feature_map.shape[1]}"
            )
            raise ValueError(
                "Feature dimension mismatch between student and teacher for distillation loss."
            )

        student_height, student_width = student_feature_map.shape[2:]
        teacher_height, teacher_width = teacher_feature_map.shape[2:]
        if (student_height, student_width) != (teacher_height, teacher_width):
            _logger.warning(
                "[RTv4OBBCriterion] Resizing teacher feature map from "
                f"{teacher_height}x{teacher_width} to student's "
                f"{student_height}x{student_width} for distillation."
            )
            teacher_feature_map = F.interpolate(
                teacher_feature_map,
                size=(student_height, student_width),
                mode='bilinear',
                align_corners=False,
            )

        student_output = student_feature_map.flatten(2).permute(0, 2, 1)
        teacher_output = teacher_feature_map.flatten(2).permute(0, 2, 1)
        student_output = F.normalize(student_output, p=2, dim=-1)
        teacher_output = F.normalize(teacher_output, p=2, dim=-1)
        cosine_similarity = F.cosine_similarity(
            student_output,
            teacher_output,
            dim=-1,
        )
        return {'loss_distill': (1 - cosine_similarity).mean()}

    def _get_distillation_weight_for_epoch(self) -> float:
        return self.weight_dict.get('loss_distill', 0.0)


    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].to(src_logits.dtype)
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {'loss_focal': loss}

    def _get_obb_quality(self, pred_boxes, target_boxes):
        quality = 1.0 - kld_loss(
            pred_boxes,
            target_boxes,
            fun='log1p',
            tau=1.0,
            sqrt=False,
        )
        return torch.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0).clamp_(min=0.0, max=1.0)

    def _get_ddf_query_weights(self, outputs, idx, quality):
        teacher_logits = outputs.get('teacher_logits')
        if teacher_logits is None:
            return None, None

        with torch.no_grad():
            query_weights = teacher_logits.detach().sigmoid().max(dim=-1)[0]
            query_mask = torch.zeros_like(query_weights, dtype=torch.bool)
            if idx[0].numel() > 0:
                query_mask[idx] = True
                query_weights[idx] = quality.reshape_as(query_weights[idx]).to(query_weights.dtype)
        return query_weights.detach(), query_mask

    def _balanced_ddf_loss(self, loss, mask, outputs):
        if 'is_dn' not in outputs or self.num_pos is None or self.num_neg is None:
            batch_scale = 8 / outputs['pred_boxes'].shape[0]
            self.num_pos = (mask.sum().to(loss.dtype) * batch_scale).sqrt()
            self.num_neg = ((~mask).sum().to(loss.dtype) * batch_scale).sqrt()

        loss_pos = loss[mask].mean() if mask.any() else loss.sum() * 0.0
        loss_neg = loss[~mask].mean() if (~mask).any() else loss.sum() * 0.0
        denom = (self.num_pos + self.num_neg).clamp(min=1e-6)
        return (loss_pos * self.num_pos + loss_neg * self.num_neg) / denom

    def _distribution_ddf_loss(self, pred_logits, teacher_logits, query_weights, query_mask, num_bins, repeat_factor, outputs, T=5):
        if query_weights is None or query_mask is None or teacher_logits is None:
            return None

        # Keep the loss schema identical across DDP ranks for empty/equal DN outputs.
        if torch.equal(pred_logits, teacher_logits):
            return pred_logits.sum() * 0.0

        pred = pred_logits.reshape(-1, num_bins)
        teacher = teacher_logits.detach().reshape(-1, num_bins)
        dist_weights = query_weights.unsqueeze(-1)
        dist_mask = query_mask.unsqueeze(-1)
        if repeat_factor > 1:
            dist_weights = dist_weights.repeat(1, 1, repeat_factor)
            dist_mask = dist_mask.repeat(1, 1, repeat_factor)
        dist_weights = dist_weights.reshape(-1).to(pred.dtype).detach()
        dist_mask = dist_mask.reshape(-1)

        loss = dist_weights * (T ** 2) * (
            nn.KLDivLoss(reduction='none')(
                F.log_softmax(pred / T, dim=1),
                F.softmax(teacher / T, dim=1),
            )
        ).sum(-1)
        return self._balanced_ddf_loss(loss, dist_mask, outputs)

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs['pred_boxes'][idx].detach()
            target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
            values = self._get_obb_quality(src_boxes, target_boxes)

        src_logits = outputs['pred_logits']
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].to(src_logits.dtype)

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = values.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {'loss_vfl': loss}

    def loss_l1(self, outputs, targets, indices, num_boxes, **kwargs):
        """Compute the L1 regression loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        
        # Split xywh and angle
        src_xywh = src_boxes[..., :4]
        src_angle = src_boxes[..., 4:5]
        
        tgt_xywh = target_boxes[..., :4]
        tgt_angle = target_boxes[..., 4:5]
        
        # xywh use standard L1
        loss_xywh = F.l1_loss(src_xywh, tgt_xywh, reduction='none')

        # [Safety] 确保 target 也在 0-1 之间 (防止数据增强溢出)
        tgt_angle = torch.fmod(tgt_angle, 1.0)
        tgt_angle = torch.where(tgt_angle < 0, tgt_angle + 1.0, tgt_angle)
        
        # angle use periodic L1
        diff_angle = torch.abs(src_angle - tgt_angle)
        loss_angle = torch.min(diff_angle, 1.0 - diff_angle)
        
        # Combine losses
        loss_l1 = torch.cat([loss_xywh, loss_angle], dim=-1)
        
        losses['loss_l1'] = loss_l1.sum() / num_boxes

        if target_boxes.shape[0] == 0:
            losses['loss_l1'] += outputs['pred_boxes'].sum() * 0.0

        return losses


    def loss_kld(self, outputs, targets, indices, num_boxes, boxes_weight=None, **kwargs):
        """Compute the KLD loss (replacing GIoU)
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        # Replace GIoU with GDLoss
        loss_kld = kld_loss(src_boxes, target_boxes, fun='log1p', tau=1.0, sqrt=False)
        loss_kld = loss_kld if boxes_weight is None else loss_kld * boxes_weight

        losses['loss_kld'] = loss_kld.sum() / num_boxes

        if target_boxes.shape[0] == 0:
            losses['loss_kld'] += outputs['pred_boxes'].sum() * 0.0

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, **kwargs):
        losses = {}
        if 'pred_corners' not in outputs:
            return losses

        idx = self._get_src_permutation_idx(indices)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        pred_corners = outputs['pred_corners'][idx].reshape(-1, (self.reg_max + 1))
        ref_points = outputs['ref_points'][idx].detach()
        with torch.no_grad():
            if self.fgl_targets_dn is None and 'is_dn' in outputs:
                self.fgl_targets_dn = obb2distance_local(
                    ref_points,
                    target_boxes,
                    self.reg_max,
                    outputs['reg_scale'],
                    outputs['up'],
                )
            if self.fgl_targets is None and 'is_dn' not in outputs:
                self.fgl_targets = obb2distance_local(
                    ref_points,
                    target_boxes,
                    self.reg_max,
                    outputs['reg_scale'],
                    outputs['up'],
                )

        target_corners, weight_right, weight_left = self.fgl_targets_dn if 'is_dn' in outputs else self.fgl_targets
        with torch.no_grad():
            kld_quality = self._get_obb_quality(outputs['pred_boxes'][idx].detach(), target_boxes)
        weight_targets = kld_quality.unsqueeze(-1).repeat(1, 4).reshape(-1)

        losses['loss_fgl'] = self.unimodal_distribution_focal_loss(
            pred_corners,
            target_corners,
            weight_right,
            weight_left,
            weight_targets,
            avg_factor=num_boxes,
        )

        if 'loss_ddf' in self.weight_dict and 'teacher_corners' in outputs:
            query_weights, query_mask = self._get_ddf_query_weights(outputs, idx, kld_quality)
            loss_ddf = self._distribution_ddf_loss(
                outputs['pred_corners'],
                outputs['teacher_corners'],
                query_weights,
                query_mask,
                self.reg_max + 1,
                4,
                outputs,
            )
            if loss_ddf is not None:
                losses['loss_ddf'] = loss_ddf
        return losses

    def loss_angle_local(self, outputs, targets, indices, num_boxes, **kwargs):
        losses = {}
        if 'pred_angle_logits' not in outputs or 'ref_angles' not in outputs:
            return losses

        idx = self._get_src_permutation_idx(indices)
        pred_angle_logits = outputs['pred_angle_logits'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ref_angles = outputs['ref_angles'][idx]

        if pred_angle_logits.numel() == 0:
            losses['loss_angle_fgl'] = outputs['pred_angle_logits'].sum() * 0.0
            return losses

        with torch.no_grad():
            if outputs.get('angle_circular', True):
                label_left, label_right, weight_left, weight_right = angle_to_circular_targets(
                    ref_angles,
                    target_boxes[..., 4:5],
                    pred_angle_logits.shape[-1],
                )
            else:
                label_left, label_right, weight_left, weight_right = angle_to_linear_targets(
                    ref_angles,
                    target_boxes[..., 4:5],
                    pred_angle_logits.shape[-1],
                    outputs['angle_max_residual'],
                )
            kld_quality = self._get_obb_quality(outputs['pred_boxes'][idx].detach(), target_boxes)

        losses['loss_angle_fgl'] = self.circular_distribution_focal_loss(
            pred_angle_logits,
            label_left,
            label_right,
            weight_left,
            weight_right,
            weight=kld_quality,
            avg_factor=num_boxes,
        )

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers. """
        results = []
        for indices_aux in indices_aux_list:
            indices = [(torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                       for idx1, idx2 in zip(indices.copy(), indices_aux.copy())]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results


    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'l1': self.loss_l1,
            'kld': self.loss_kld,
            'focal': self.loss_labels_focal,
            'vfl': self.loss_labels_vfl,
            'local': self.loss_local,
            'angle_local': self.loss_angle_local,
            'distill': self.loss_distillation,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)['indices']
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

        # Get the matching union set across all decoder layers.
        if 'aux_outputs' in outputs:
            indices_aux_list, cached_indices, cached_indices_enc = [], [], []
            aux_outputs_list = outputs['aux_outputs']
            if 'pre_outputs' in outputs:
                aux_outputs_list = outputs['aux_outputs'] + [outputs['pre_outputs']]
            for i, aux_outputs in enumerate(aux_outputs_list):
                indices_aux = self.matcher(aux_outputs, targets)['indices']
                cached_indices.append(indices_aux)
                indices_aux_list.append(indices_aux)
            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                indices_enc = self.matcher(aux_outputs, targets)['indices']
                cached_indices_enc.append(indices_enc)
                indices_aux_list.append(indices_enc)
            indices_go = self._get_go_indices(indices, indices_aux_list)

            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float,
                                           device=next(iter(outputs.values())).device)
            if is_dist_available_and_initialized():
                torch.distributed.all_reduce(num_boxes_go)
            num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else:
            assert 'aux_outputs' in outputs, ''

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses, main loss
        losses = {}
        for loss_name in self.losses:
            # TODO, indices and num_box are different from RT-DETRv2
            if loss_name == 'distill':
                l_dict = self.get_loss(loss_name, outputs, targets, None, None, **kwargs)
                if 'loss_distill' in l_dict and l_dict['loss_distill'] != 0:
                    distill_weight = self._get_distillation_weight_for_epoch()
                    l_dict['loss_distill'] = l_dict['loss_distill'] * distill_weight
            else:
                use_uni_set = self.use_uni_set and (loss_name in ['l1', 'kld', 'local', 'angle_local'])
                indices_in = indices_go if use_uni_set else indices
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss_name, outputs, targets, indices_in)
                l_dict = self.get_loss(loss_name, outputs, targets, indices_in, num_boxes_in, **meta)
                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    if loss == 'distill':
                        continue
                    # TODO, indices and num_box are different from RT-DETRv2
                    if 'local' in self.losses and 'pred_corners' in aux_outputs:
                        aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                    use_uni_set = self.use_uni_set and (loss in ['l1', 'kld', 'local', 'angle_local'])
                    indices_in = indices_go if use_uni_set else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of auxiliary traditional head output at first decoder layer. just for dfine
        if 'pre_outputs' in outputs:
            aux_outputs = outputs['pre_outputs']
            for loss in self.losses:
                if loss == 'distill':
                    continue
                # TODO, indices and num_box are different from RT-DETRv2
                use_uni_set = self.use_uni_set and (loss in ['l1', 'kld', 'local', 'angle_local'])
                indices_in = indices_go if use_uni_set else cached_indices[-1]
                num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_in)
                l_dict = self.get_loss(loss, aux_outputs, targets, indices_in, num_boxes_in, **meta)

                l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                l_dict = {k + '_pre': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # In case of encoder auxiliary losses.
        if 'enc_aux_outputs' in outputs:
            assert 'enc_meta' in outputs, ''
            class_agnostic = outputs['enc_meta']['class_agnostic']
            if class_agnostic:
                orig_num_classes = self.num_classes
                self.num_classes = 1
                enc_targets = copy.deepcopy(targets)
                for t in enc_targets:
                    t['labels'] = torch.zeros_like(t["labels"])
            else:
                enc_targets = targets

            for i, aux_outputs in enumerate(outputs['enc_aux_outputs']):
                for loss in self.losses:
                    if loss == 'distill':
                        continue
                    # TODO, indices and num_box are different from RT-DETRv2
                    use_uni_set = self.use_uni_set and (loss == 'l1' or loss == 'kld' or loss == 'local' or loss == 'angle_local')
                    indices_in = indices_go if use_uni_set else cached_indices_enc[i]
                    num_boxes_in = num_boxes_go if use_uni_set else num_boxes
                    meta = self.get_loss_meta_info(loss, aux_outputs, enc_targets, indices_in)
                    l_dict = self.get_loss(loss, aux_outputs, enc_targets, indices_in, num_boxes_in, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            if class_agnostic:
                self.num_classes = orig_num_classes

        # In case of cdn auxiliary losses.
        if 'dn_outputs' in outputs:
            assert 'dn_meta' in outputs, ''
            indices_dn = self.get_cdn_matched_indices(outputs['dn_meta'], targets)
            if 'dn_target_idx' in outputs['dn_meta']:
                dn_num_boxes = sum(indices.numel() for indices in outputs['dn_meta']['dn_target_idx'])
                dn_num_boxes = torch.as_tensor(
                    [dn_num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
                )
                if is_dist_available_and_initialized():
                    torch.distributed.all_reduce(dn_num_boxes)
                dn_num_boxes = torch.clamp(dn_num_boxes / get_world_size(), min=1).item()
            else:
                dn_num_boxes = num_boxes * outputs['dn_meta']['dn_num_group']
                dn_num_boxes = 1 if dn_num_boxes == 0 else dn_num_boxes

            for i, aux_outputs in enumerate(outputs['dn_outputs']):
                if 'local' in self.losses and 'pred_corners' in aux_outputs:
                    aux_outputs['is_dn'] = True
                    aux_outputs['up'], aux_outputs['reg_scale'] = outputs['up'], outputs['reg_scale']
                for loss in self.losses:
                    if loss == 'distill':
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary traditional head output at first decoder layer, just for dfine
            if 'dn_pre_outputs' in outputs:
                aux_outputs = outputs['dn_pre_outputs']
                for loss in self.losses:
                    if loss == 'distill':
                        continue
                    meta = self.get_loss_meta_info(loss, aux_outputs, targets, indices_dn)
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_dn, dn_num_boxes, **meta)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + '_dn_pre': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # For debugging Objects365 pre-train.
        losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in losses.items()}
        return losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs['pred_boxes'][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if self.boxes_weight_format == 'iou':
            iou, _ = box_iou(box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes))
            iou = torch.diag(iou)
        elif self.boxes_weight_format == 'giou':
            iou = torch.diag(generalized_box_iou( \
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)))
        else:
            raise AttributeError()

        if loss in ('boxes',):
            meta = {'boxes_weight': iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        dn_target_idx = dn_meta.get("dn_target_idx")
        num_gts = [len(t['labels']) for t in targets]
        device = targets[0]['labels'].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if dn_target_idx is not None:
                gt_idx = dn_target_idx[i].to(device=device, dtype=torch.int64)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            elif num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device), \
                                         torch.zeros(0, dtype=torch.int64, device=device)))

        return dn_match_indices

    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)

    def unimodal_distribution_focal_loss(self, pred, label, weight_right, weight_left, weight=None, reduction='sum',
                                         avg_factor=None):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left.reshape(-1) \
               + F.cross_entropy(pred, dis_right, reduction='none') * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def circular_distribution_focal_loss(self, pred, label_left, label_right, weight_left, weight_right, weight=None,
                                         reduction='sum', avg_factor=None):
        loss = F.cross_entropy(pred, label_left, reduction='none') * weight_left.reshape(-1) \
               + F.cross_entropy(pred, label_right, reduction='none') * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs['aux_outputs']) + 1 if 'aux_outputs' in outputs else 1
        step = .5 / (num_layers - 1)
        opt_list = [.5 + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list
