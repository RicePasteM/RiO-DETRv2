"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
Modifications Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import math 
import copy 
import functools
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 
from typing import List

from .denoising_obb import get_contrastive_denoising_training_group_obb
from .utils import bias_init_with_prob, get_activation, inverse_sigmoid, coordinate_to_encoding_4
from .utils import deformable_attention_core_func_v2
from .obb_matcher import HungarianMatcherOBB
from .dfine_utils import (
    decode_circular_angle_delta,
    decode_linear_angle_delta,
    distance2obb_local,
    weighting_function,
    wrap_angle_01,
)
import os

from ..core import register

__all__ = ['RTDETRTransformerv2OBB', 'RTDETRTransformerv2OBBFDR']


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class IntegralOBB(nn.Module):
    def __init__(self, reg_max=32):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        out_shape = x.shape[:-1] + (4,)
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        return F.linear(x, project.to(x.device)).reshape(out_shape)


class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max, act='relu'):
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        b, n, _ = pred_corners.shape
        prob = F.softmax(pred_corners.reshape(b, n, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(b, n, -1))
        return scores + quality_score


class AngleLQE(nn.Module):
    def __init__(self, num_angle_layers, hidden_dim=16, num_mlp_layers=2, act='relu'):
        super().__init__()
        self.num_angle_layers = num_angle_layers
        self.angle_conf = MLP(
            num_angle_layers + 1,
            hidden_dim,
            1,
            num_mlp_layers,
            act=act,
        )
        init.constant_(self.angle_conf.layers[-1].bias, 0)
        init.constant_(self.angle_conf.layers[-1].weight, 0)

    def forward(self, scores, angle_entropies, pred_boxes):
        if len(angle_entropies) != self.num_angle_layers:
            raise ValueError(
                f'Expected {self.num_angle_layers} angle entropy tensors, got {len(angle_entropies)}'
            )
        wh = pred_boxes[..., 2:4].detach().abs().clamp_min(1e-6)
        anisotropy = 1.0 - wh.min(dim=-1, keepdim=True).values / wh.max(dim=-1, keepdim=True).values
        stat = torch.cat([*(entropy.detach() for entropy in angle_entropies), anisotropy], dim=-1)
        return scores + self.angle_conf(stat)


class QueryPosHeadRiO(nn.Module):
    def __init__(self, hidden_dim, num_feats, act='relu'):
        super().__init__()
        self.num_feats = num_feats
        self.ref_point_head = MLP(hidden_dim * 2, hidden_dim, hidden_dim, 2, act=act)

    def forward(self, reference_points: torch.Tensor):
        query_sine_embed = coordinate_to_encoding_4(reference_points[..., :4], num_feats=self.num_feats)
        return self.ref_point_head(query_sine_embed)


class MSDeformableAttention(nn.Module):
    def __init__(
        self, 
        embed_dim=256, 
        num_heads=8, 
        num_levels=4, 
        num_points=4, 
        method='default',
        offset_scale=0.5,
        value_shape='default',
        use_value_proj=True,
        use_output_proj=True,
    ):
        """Multi-Scale Deformable Attention
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list
        
        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method
        self.use_value_proj = use_value_proj
        self.use_output_proj = use_output_proj

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim) if use_value_proj else None
        self.output_proj = nn.Linear(embed_dim, embed_dim) if use_output_proj else None

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, 
                                                    method=self.method, value_shape=value_shape) 

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        if self.value_proj is not None:
            init.xavier_uniform_(self.value_proj.weight)
            init.constant_(self.value_proj.bias, 0)
        if self.output_proj is not None:
            init.xavier_uniform_(self.output_proj.weight)
            init.constant_(self.output_proj.bias, 0)


    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                value,
                value_spatial_shapes: List[int],
                value_mask: torch.Tensor=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]

        if isinstance(value, (list, tuple)):
            processed_value = value
        else:
            Len_v = value.shape[1]
            processed_value = self.value_proj(value) if self.value_proj is not None else value
            if value_mask is not None:
                processed_value = processed_value * value_mask.to(processed_value.dtype).unsqueeze(-1)
            processed_value = processed_value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))

        reference_points = reference_points.expand(-1, -1, self.num_levels, -1)

        ref_expanded = []
        for i, n in enumerate(self.num_points_list):
            ref_l = reference_points[:, :, i:i+1, :]
            ref_l = ref_l.unsqueeze(2).repeat(1, 1, 1, n, 1)
            ref_expanded.append(ref_l)
        ref_expanded = torch.cat(ref_expanded, dim=3)

        num_points_scale = self.num_points_scale.to(device=query.device, dtype=query.dtype).view(1, 1, 1, -1, 1)

        angle = ref_expanded[..., 4:] * math.pi
        angle = angle.expand(-1, -1, self.num_heads, -1, -1)
        head_angle_offset = torch.zeros(self.num_heads, device=angle.device, dtype=angle.dtype)
        head_angle_offset[self.num_heads // 2:] = math.pi / 2
        angle = angle + head_angle_offset.view(1, 1, self.num_heads, 1, 1)
        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        rot = torch.cat([cosa, -sina, sina, cosa], dim=-1)
        rot = rot.reshape(*rot.shape[:-1], 2, 2)
        wh = (ref_expanded[..., 2:4] * self.offset_scale).expand(-1, -1, self.num_heads, -1, -1)
        rotated_points = torch.einsum('...ij,...j->...i', rot, wh)
        offset = sampling_offsets * num_points_scale * rotated_points
        sampling_locations = ref_expanded[..., :2] + offset

        output = self.ms_deformable_attn_core(
            processed_value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
            self.num_points_list,
        )

        if self.output_proj is not None:
            output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default',
                 value_shape='default',
                 ):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method, value_shape=value_shape)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                target,
                reference_points,
                memory,
                memory_spatial_shapes,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(\
            self.with_pos_embed(target, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes, 
            memory_mask)
        target = target + self.dropout2(target2)
        target = self.norm2(target)

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target)

        return target


class Gate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class LQEIdentity(nn.Module):
    def forward(self, scores, pred_corners=None):
        return scores


class TransformerDecoderLayerDFineOBB(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default',
                 layer_scale=None):
        super().__init__()
        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformableAttention(
            d_model,
            n_head,
            n_levels,
            n_points,
            method=cross_attn_method,
            use_value_proj=False,
            use_output_proj=False,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.gateway = Gate(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                target,
                reference_points,
                value,
                memory_spatial_shapes,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        target2 = self.cross_attn(
            self.with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            memory_spatial_shapes,
        )
        target = self.gateway(target, self.dropout2(target2))

        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))

        return target


class TransformerDecoderOBB(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoderOBB, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.angle_scales = [1.0 / (1.5 ** i) for i in range(num_layers)]

    def forward(self,
                target,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None):
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)

        output = target
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(output, ref_points_input, memory, memory_spatial_shapes, attn_mask, memory_mask, query_pos_embed)

            layer_output = bbox_head[i](output)
            
            # Split coordinates and angle
            delta_xywh = layer_output[..., :4]
            delta_angle = layer_output[..., 4:5]

            def get_new_bbox(d_xywh, d_angle, ref_xywh, ref_angle):
                # Coordinates update (keep original logic)
                new_xywh = F.sigmoid(d_xywh + inverse_sigmoid(ref_xywh))
                
                # Angle update (Tanh + Linear Add + Modulo)
                angle_scale_factor = self.angle_scales[i]
                delta_angle_bounded = torch.tanh(d_angle) * angle_scale_factor
                raw_new_angle = ref_angle + delta_angle_bounded
                
                # Periodic normalization to [0, 1)
                new_angle = torch.fmod(raw_new_angle, 1.0)
                new_angle = torch.where(new_angle < 0, new_angle + 1.0, new_angle)
                
                return torch.cat([new_xywh, new_angle], dim=-1)

            # Update for next layer input (detached ref points)
            inter_ref_bbox = get_new_bbox(
                delta_xywh, delta_angle,
                ref_points_detach[..., :4], ref_points_detach[..., 4:5]
            )

            # Check for environment variable override
            force_all_layers = os.environ.get('RTDETR_RETURN_ALL_LAYERS', '0') == '1'

            if self.training or force_all_layers:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    loss_bbox = get_new_bbox(
                        delta_xywh, delta_angle,
                        ref_points[..., :4], ref_points[..., 4:5]
                    )
                    dec_out_bboxes.append(loss_bbox)

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


class TransformerDecoderOBBFDR(nn.Module):
    def __init__(self,
                 hidden_dim,
                 decoder_layer,
                 decoder_layer_wide,
                 num_layers,
                 num_head,
                 reg_max,
                 reg_scale,
                 angle_bins,
                 up,
                 eval_idx=-1,
                 layer_scale=1,
                 use_dfine_decoder_attention=True,
                 angle_fdr_mode='iterative_circular',
                 angle_fdr_scale_factor=1.5,
                 angle_aware_lqe=False,
                 act='relu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.reg_max = reg_max
        self.reg_scale = reg_scale
        self.angle_bins = angle_bins
        self.up = up
        self.use_dfine_decoder_attention = use_dfine_decoder_attention
        if angle_fdr_mode not in ('iterative_circular', 'coarse_to_fine'):
            raise ValueError(f'Unsupported angle_fdr_mode: {angle_fdr_mode}')
        if angle_fdr_scale_factor < 1.0:
            raise ValueError('angle_fdr_scale_factor must be at least 1.0')
        self.angle_fdr_mode = angle_fdr_mode
        self.angle_fdr_scale_factor = angle_fdr_scale_factor
        if angle_fdr_mode == 'coarse_to_fine':
            self.angle_max_residuals = tuple(
                0.5 / (angle_fdr_scale_factor ** i) for i in range(num_layers)
            )
            self.angle_circular_layers = (True,) + (False,) * (num_layers - 1)
        else:
            self.angle_max_residuals = (0.5,) * num_layers
            self.angle_circular_layers = (True,) * num_layers
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)])
        self.angle_lqe = (
            AngleLQE(self.eval_idx + 1, act=act) if angle_aware_lqe else None
        )

    def _update_angle(self, angle_logits, ref_angle, layer_idx):
        if self.angle_circular_layers[layer_idx]:
            delta_angle = decode_circular_angle_delta(angle_logits)
        else:
            delta_angle = decode_linear_angle_delta(
                angle_logits,
                self.angle_max_residuals[layer_idx],
            )
        return wrap_angle_01(ref_angle + delta_angle)

    def _normalized_angle_entropy(self, angle_logits):
        detached_logits = angle_logits.detach()
        log_prob = F.log_softmax(detached_logits, dim=-1)
        entropy = -(log_prob.exp() * log_prob).sum(dim=-1, keepdim=True)
        return entropy / math.log(self.angle_bins)

    def _split_reg_output(self, reg_output):
        bbox_dim = 4 * (self.reg_max + 1)
        return reg_output[..., :bbox_dim], reg_output[..., bbox_dim:bbox_dim + self.angle_bins]

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(value, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[:self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([LQEIdentity() for _ in range(self.eval_idx)] + [self.lqe_layers[self.eval_idx]])

    @staticmethod
    def _sanitize_xywh(boxes):
        return boxes.clamp(min=1e-6, max=1.0 - 1e-6)

    def forward(self,
                target,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                bbox_head,
                angle_head,
                score_head,
                query_pos_head,
                pre_bbox_head,
                integral,
                up,
                reg_scale,
                attn_mask=None,
                memory_mask=None,
                mixed_reg_head=False):
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        dec_out_angle_logits = []
        dec_out_angle_refs = []
        angle_entropies = []

        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, memory_spatial_shapes) \
            if self.use_dfine_decoder_attention else memory
        ref_points_detach = F.sigmoid(ref_points_unact)
        if not hasattr(self, 'project'):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        pre_bboxes, pre_scores = None, None
        ref_obb_initial = None
        force_all_layers = os.environ.get('RTDETR_RETURN_ALL_LAYERS', '0') == '1'

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)
            if self.use_dfine_decoder_attention:
                query_pos_embed = query_pos_embed.clamp(min=-10, max=10)

            if self.use_dfine_decoder_attention and i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, memory_spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(output, ref_points_input, value, memory_spatial_shapes, attn_mask, memory_mask, query_pos_embed)

            if mixed_reg_head:
                bbox_logits, angle_logits = self._split_reg_output(bbox_head[i](output + output_detach))
            else:
                bbox_logits = bbox_head[i](output + output_detach)
                angle_logits = angle_head[i](output)
            if self.angle_lqe is not None and i <= self.eval_idx:
                angle_entropies.append(self._normalized_angle_entropy(angle_logits))

            if i == 0:
                pre_xywh = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach[..., :4]))
                pre_angle = self._update_angle(angle_logits, ref_points_detach[..., 4:5], i)
                pre_bboxes = torch.cat([self._sanitize_xywh(pre_xywh), pre_angle], dim=-1)
                pre_scores = score_head[0](output)
                ref_obb_initial = pre_bboxes.detach()

            inter_angle = self._update_angle(angle_logits, ref_points_detach[..., 4:5], i)
            pred_corners = bbox_logits + pred_corners_undetach
            inter_xywh = distance2obb_local(
                ref_obb_initial,
                integral(pred_corners, project),
                inter_angle,
                reg_scale,
            )
            inter_ref_bbox = torch.cat([inter_xywh, inter_angle], dim=-1)

            scores = self.lqe_layers[i](score_head[i](output), pred_corners)
            if self.angle_lqe is not None and i == self.eval_idx:
                scores = self.angle_lqe(scores, angle_entropies, inter_ref_bbox)
            if self.training or force_all_layers or i == self.eval_idx:
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_obb_initial)
                dec_out_angle_logits.append(angle_logits)
                dec_out_angle_refs.append(ref_points_detach[..., 4:5])

                if not (self.training or force_all_layers):
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        if self.training or os.environ.get('RTDETR_RETURN_ALL_LAYERS', '0') == '1':
            return (
                torch.stack(dec_out_bboxes),
                torch.stack(dec_out_logits),
                torch.stack(dec_out_pred_corners),
                torch.stack(dec_out_refs),
                torch.stack(dec_out_angle_logits),
                torch.stack(dec_out_angle_refs),
                pre_bboxes,
                pre_scores,
            )

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


@register()
class RTDETRTransformerv2OBB(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=True,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2, 
                 aux_loss=True, 
                 cross_attn_method='default', 
                 query_select_method='default',
                 value_shape='reshape',
                 mlp_act='relu',
                 query_pos_method='default',
                 anchor_aspect_ratio=1.0,
                 ):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.anchor_aspect_ratio = anchor_aspect_ratio

        assert query_select_method in ('default', 'one2many', 'agnostic'), ''
        assert cross_attn_method in ('default', 'discrete'), ''
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method
        self.query_pos_method = query_pos_method

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method, value_shape=value_shape)
        self.decoder = TransformerDecoderOBB(hidden_dim, decoder_layer, num_layers, eval_idx)

        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0: 
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        if query_pos_method == 'as_reg':
            self.query_pos_head = MLP(5, hidden_dim, hidden_dim, 3, act=mlp_act)
            print("     ### Query Position Embedding@{} ###     ".format(query_pos_method))
        elif query_pos_method == 'rio':
            num_feats = hidden_dim // 2
            self.query_pos_head = QueryPosHeadRiO(hidden_dim, num_feats, act=mlp_act)
        else:
            self.query_pos_head = MLP(5, 2 * hidden_dim, hidden_dim, 2, act=mlp_act)

        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),
            ('norm', nn.LayerNorm(hidden_dim,)),
        ]))

        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 5, 3, act=mlp_act)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 5, 3, act=mlp_act) for _ in range(num_layers)
        ])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)

        self._reset_parameters()
        
    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for _cls, _reg in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(_cls.bias, bias)
            init.constant_(_reg.layers[-1].weight, 0)
            init.constant_(_reg.layers[-1].bias, 0)
        
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        if hasattr(self.query_pos_head, 'ref_point_head'):
            init.xavier_uniform_(self.query_pos_head.ref_point_head.layers[0].weight)
            init.xavier_uniform_(self.query_pos_head.ref_point_head.layers[1].weight)
        else:
            init.xavier_uniform_(self.query_pos_head.layers[0].weight)
            init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m in self.input_proj:
            init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim
        
        self.logging_matcher = HungarianMatcherOBB(weight_dict={'cost_class': 1, 'cost_bbox': 5, 'cost_giou': 2}, use_focal_loss=True)
        self.log_step = 0

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            
            aspect_ratio = self.anchor_aspect_ratio
            
            # 构造宽高
            # 形状: [1, h, w, 2]
            valid_wh = torch.stack([
                torch.ones_like(grid_x) * 1.0,           # w 比例为 1
                torch.ones_like(grid_y) * aspect_ratio   # h 比例
            ], dim=-1).unsqueeze(0) * grid_size * (2.0 ** lvl)
            
            lvl_anchors = torch.concat([grid_xy, valid_wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask


    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           denoising_logits=None,
                           denoising_bbox_unact=None):

        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask

        # memory = torch.where(valid_mask, memory, 0)
        # TODO fix type error for onnx export 
        memory = valid_mask.to(memory.dtype) * memory  

        output_memory :torch.Tensor = self.enc_output(memory)
        enc_outputs_logits :torch.Tensor = self.enc_score_head(output_memory)

        # Expand anchors to 5 dim: (cx, cy, w, h, angle)
        anchors_5d = torch.cat([anchors, torch.zeros_like(anchors[..., :1])], dim=-1)
        enc_outputs_coord_unact :torch.Tensor = self.enc_bbox_head(output_memory) + anchors_5d
        enc_outputs_coord_unact = torch.clamp(enc_outputs_coord_unact, min=-20.0, max=20.0)

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_bbox_unact = \
            self._select_topk(output_memory, enc_outputs_logits, enc_outputs_coord_unact, self.num_queries)
            
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).expand(memory.shape[0], -1, -1)
        else:
            content = enc_topk_memory.detach()
            
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)
        
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_coords_unact: torch.Tensor, topk: int):
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
        
        topk_ind: torch.Tensor

        topk_coords = outputs_coords_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_coords_unact.shape[-1]))
        
        topk_logits = outputs_logits.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))
        
        topk_memory = memory.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_coords


    def forward(self, feats, targets=None):
        # input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(feats)
        
        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group_obb(targets, \
                    self.num_classes, 
                    self.num_queries, 
                    self.denoising_class_embed, 
                    num_denoising=self.num_denoising, 
                    label_noise_ratio=self.label_noise_ratio, 
                    box_noise_scale=self.box_noise_scale, )
            if dn_meta is not None:
                dn_meta['num_denoising_groups'] = dn_meta.get('dn_num_group', 0)
                dn_meta['num_denoising_queries'] = dn_meta['dn_num_split'][0]
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

        force_all_layers = os.environ.get('RTDETR_RETURN_ALL_LAYERS', '0') == '1'

        if (self.training and self.aux_loss) or force_all_layers:
            # self._log_decoder_output(out_logits[-1], out_bboxes[-1], targets)
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:
                out['dn_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        return out


    @torch.no_grad()
    def _log_decoder_output(self, outputs_logits, outputs_bboxes, targets):
        if targets is None:
            return

        # Prepare outputs for matcher
        outputs = {'pred_logits': outputs_logits, 'pred_boxes': outputs_bboxes}
        
        # Run matcher
        # Indices is list of (src_idx, tgt_idx) tuples
        match_result = self.logging_matcher(outputs, targets)
        indices = match_result['indices']

        log_file = "decoder_metrics.log"
        self.log_step += 1
        
        with open(log_file, "a") as f:
            f.write(f"Step: {self.log_step}\n")
            
            batch_size = outputs_logits.shape[0]
            for b in range(batch_size):
                src_idx, tgt_idx = indices[b]
                
                # 1. Top 10 matched queries
                matched_logits = outputs_logits[b, src_idx]
                matched_bboxes = outputs_bboxes[b, src_idx]
                target_classes = targets[b]['labels'][tgt_idx]
                
                matched_scores = matched_logits.sigmoid() 
                
                matched_conf = []
                for i in range(len(src_idx)):
                    cls_idx = target_classes[i]
                    if cls_idx < matched_scores.shape[1]:
                        score = matched_scores[i, cls_idx].item()
                    else:
                        score = 0.0 # Should not happen if labels are correct
                    matched_conf.append(score)
                
                matched_info = list(zip(matched_conf, matched_bboxes.tolist(), src_idx.tolist()))
                matched_info.sort(key=lambda x: x[0], reverse=True)
                top10_matched = matched_info[:10]
                
                f.write(f"  Batch {b} - Top 10 Matched Queries (Score, BBox, QueryIdx):\n")
                for score, bbox, q_idx in top10_matched:
                    bbox_str = ", ".join([f"{x:.4f}" for x in bbox])
                    f.write(f"    Query {q_idx}: Score={score:.4f}, BBox=[{bbox_str}]\n")

                # 2. Top 10 Classification Queries (highest max score)
                scores = outputs_logits[b].sigmoid().max(dim=-1)[0] # [num_queries]
                topk_scores, topk_indices = torch.topk(scores, 10)
                
                topk_bboxes = outputs_bboxes[b, topk_indices]
                
                f.write(f"  Batch {b} - Top 10 Classification Queries (Score, BBox, QueryIdx):\n")
                for i in range(len(topk_indices)):
                    score = topk_scores[i].item()
                    bbox = topk_bboxes[i].tolist()
                    q_idx = topk_indices[i].item()
                    bbox_str = ", ".join([f"{x:.4f}" for x in bbox])
                    f.write(f"    Query {q_idx}: Score={score:.4f}, BBox=[{bbox_str}]\n")

                # 3. Ground Truths
                f.write(f"  Batch {b} - Ground Truths (Label, BBox):\n")
                tgt_labels = targets[b]['labels']
                tgt_boxes = targets[b]['boxes']
                for t_l, t_b in zip(tgt_labels, tgt_boxes):
                    t_b_list = t_b.tolist()
                    t_b_str = ", ".join([f"{x:.4f}" for x in t_b_list])
                    f.write(f"    Label {t_l.item()}: BBox=[{t_b_str}]\n")

            f.write("\n")

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]


@register()
class RTDETRTransformerv2OBBFDR(RTDETRTransformerv2OBB):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self, reg_max=32, reg_scale=4.0, angle_bins=36, layer_scale=1,
                 use_dfine_decoder_attention=True, mixed_reg_head=False,
                 angle_fdr_mode='iterative_circular', angle_fdr_scale_factor=1.5,
                 angle_aware_lqe=False,
                 **kwargs):
        mlp_act = kwargs.get('mlp_act', 'relu')
        dim_feedforward = kwargs.get('dim_feedforward', 1024)
        dropout = kwargs.get('dropout', 0.)
        activation = kwargs.get('activation', 'relu')
        num_points = kwargs.get('num_points', 4)
        super().__init__(**kwargs)

        self.reg_max = reg_max
        self.angle_bins = angle_bins
        self.mixed_reg_head = mixed_reg_head
        self.angle_fdr_mode = angle_fdr_mode
        self.angle_fdr_scale_factor = angle_fdr_scale_factor
        self.angle_aware_lqe = angle_aware_lqe
        self.use_dfine_decoder_attention = use_dfine_decoder_attention
        self.layer_scale = layer_scale if use_dfine_decoder_attention else 1
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        scaled_dim = round(self.layer_scale * self.hidden_dim)

        if use_dfine_decoder_attention:
            decoder_layer = TransformerDecoderLayerDFineOBB(
                self.hidden_dim,
                self.nhead,
                dim_feedforward,
                dropout,
                activation,
                self.num_levels,
                num_points,
                cross_attn_method=self.cross_attn_method,
            )
            decoder_layer_wide = TransformerDecoderLayerDFineOBB(
                self.hidden_dim,
                self.nhead,
                dim_feedforward,
                dropout,
                activation,
                self.num_levels,
                num_points,
                cross_attn_method=self.cross_attn_method,
                layer_scale=self.layer_scale,
            )
        else:
            decoder_layer = TransformerDecoderLayer(
                self.hidden_dim,
                self.nhead,
                dim_feedforward,
                dropout,
                activation,
                self.num_levels,
                num_points,
                cross_attn_method=self.cross_attn_method,
                value_shape='reshape',
            )
            decoder_layer_wide = decoder_layer
        self.decoder = TransformerDecoderOBBFDR(
            self.hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            self.num_layers,
            self.nhead,
            self.reg_max,
            self.reg_scale,
            self.angle_bins,
            self.up,
            eval_idx=self.decoder.eval_idx,
            layer_scale=self.layer_scale,
            use_dfine_decoder_attention=use_dfine_decoder_attention,
            angle_fdr_mode=angle_fdr_mode,
            angle_fdr_scale_factor=angle_fdr_scale_factor,
            angle_aware_lqe=angle_aware_lqe,
            act=mlp_act,
        )
        self.eval_idx = self.decoder.eval_idx
        self.angle_max_residuals = self.decoder.angle_max_residuals
        self.angle_circular_layers = self.decoder.angle_circular_layers
        self.dec_score_head = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.num_classes) for _ in range(self.eval_idx + 1)]
            + [nn.Linear(scaled_dim, self.num_classes) for _ in range(self.num_layers - self.eval_idx - 1)]
        )
        bbox_reg_dim = 4 * (self.reg_max + 1)
        if self.mixed_reg_head:
            mixed_reg_dim = bbox_reg_dim + self.angle_bins
            self.dec_reg_head = nn.ModuleList(
                [MLP(self.hidden_dim, self.hidden_dim, mixed_reg_dim, 3, act=mlp_act) for _ in range(self.eval_idx + 1)]
                + [MLP(scaled_dim, scaled_dim, mixed_reg_dim, 3, act=mlp_act)
                   for _ in range(self.num_layers - self.eval_idx - 1)]
            )
            self.dec_bbox_head = None
            self.dec_angle_head = None
        else:
            self.dec_reg_head = None
            self.dec_bbox_head = nn.ModuleList(
                [MLP(self.hidden_dim, self.hidden_dim, bbox_reg_dim, 3, act=mlp_act) for _ in range(self.eval_idx + 1)]
                + [MLP(scaled_dim, scaled_dim, bbox_reg_dim, 3, act=mlp_act)
                   for _ in range(self.num_layers - self.eval_idx - 1)]
            )
            self.dec_angle_head = nn.ModuleList(
                [MLP(self.hidden_dim, self.hidden_dim, self.angle_bins, 3, act=mlp_act) for _ in range(self.eval_idx + 1)]
                + [MLP(scaled_dim, scaled_dim, self.angle_bins, 3, act=mlp_act)
                   for _ in range(self.num_layers - self.eval_idx - 1)]
            )
        self.pre_bbox_head = MLP(self.hidden_dim, self.hidden_dim, 4, 3, act=mlp_act)
        self.integral = IntegralOBB(self.reg_max)
        self._reset_fdr_parameters()

    def _reset_fdr_parameters(self):
        bias = bias_init_with_prob(0.01)
        if self.mixed_reg_head:
            for _cls, _reg in zip(self.dec_score_head, self.dec_reg_head):
                init.constant_(_cls.bias, bias)
                init.constant_(_reg.layers[-1].weight, 0)
                init.constant_(_reg.layers[-1].bias, 0)
        else:
            for _cls, _reg, _angle in zip(self.dec_score_head, self.dec_bbox_head, self.dec_angle_head):
                init.constant_(_cls.bias, bias)
                init.constant_(_reg.layers[-1].weight, 0)
                init.constant_(_reg.layers[-1].bias, 0)
                init.constant_(_angle.layers[-1].weight, 0)
                init.constant_(_angle.layers[-1].bias, 0)
        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)

    def convert_to_deploy(self):
        self.decoder.convert_to_deploy()
        self.dec_score_head = nn.ModuleList(
            [nn.Identity()] * self.eval_idx + [self.dec_score_head[self.eval_idx]]
        )
        if self.mixed_reg_head:
            self.dec_reg_head = nn.ModuleList(
                [self.dec_reg_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_reg_head))]
            )
        else:
            self.dec_bbox_head = nn.ModuleList(
                [self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_bbox_head))]
            )
            self.dec_angle_head = nn.ModuleList(
                [self.dec_angle_head[i] if i <= self.eval_idx else nn.Identity() for i in range(len(self.dec_angle_head))]
            )

    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)

        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group_obb(targets, \
                    self.num_classes,
                    self.num_queries,
                    self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=self.box_noise_scale, )
            if dn_meta is not None:
                dn_meta['num_denoising_groups'] = dn_meta.get('dn_num_group', 0)
                dn_meta['num_denoising_queries'] = dn_meta['dn_num_split'][0]
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        force_all_layers = os.environ.get('RTDETR_RETURN_ALL_LAYERS', '0') == '1'
        reg_head = self.dec_reg_head if self.mixed_reg_head else self.dec_bbox_head
        angle_head = None if self.mixed_reg_head else self.dec_angle_head
        if self.training or force_all_layers:
            out_bboxes, out_logits, out_corners, out_refs, out_angle_logits, out_angle_refs, pre_bboxes, pre_logits = self.decoder(
                init_ref_contents,
                init_ref_points_unact,
                memory,
                spatial_shapes,
                reg_head,
                angle_head,
                self.dec_score_head,
                self.query_pos_head,
                self.pre_bbox_head,
                self.integral,
                self.up,
                self.reg_scale,
                attn_mask=attn_mask,
                mixed_reg_head=self.mixed_reg_head)
        else:
            out_bboxes, out_logits = self.decoder(
                init_ref_contents,
                init_ref_points_unact,
                memory,
                spatial_shapes,
                reg_head,
                angle_head,
                self.dec_score_head,
                self.query_pos_head,
                self.pre_bbox_head,
                self.integral,
                self.up,
                self.reg_scale,
                attn_mask=attn_mask,
                mixed_reg_head=self.mixed_reg_head)

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta['dn_num_split'], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta['dn_num_split'], dim=1)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta['dn_num_split'], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta['dn_num_split'], dim=2)
            dn_out_angle_logits, out_angle_logits = torch.split(out_angle_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_angle_refs, out_angle_refs = torch.split(out_angle_refs, dn_meta['dn_num_split'], dim=2)

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}

        if self.training or force_all_layers:
            out['pred_corners'] = out_corners[-1]
            out['ref_points'] = out_refs[-1]
            out['pred_angle_logits'] = out_angle_logits[-1]
            out['ref_angles'] = out_angle_refs[-1]
            out['angle_max_residual'] = self.angle_max_residuals[-1]
            out['angle_circular'] = self.angle_circular_layers[-1]
            out['up'] = self.up
            out['reg_scale'] = self.reg_scale

        if (self.training and self.aux_loss) or force_all_layers:
            out['aux_outputs'] = self._set_aux_loss2(
                out_logits[:-1],
                out_bboxes[:-1],
                out_corners[:-1],
                out_refs[:-1],
                out_angle_logits[:-1],
                out_angle_refs[:-1],
                self.angle_max_residuals[:-1],
                self.angle_circular_layers[:-1],
                teacher_corners=out_corners[-1],
                teacher_logits=out_logits[-1],
            )
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['pre_outputs'] = {'pred_logits': pre_logits, 'pred_boxes': pre_bboxes}
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:
                out['dn_outputs'] = self._set_aux_loss2(
                    dn_out_logits,
                    dn_out_bboxes,
                    dn_out_corners,
                    dn_out_refs,
                    dn_out_angle_logits,
                    dn_out_angle_refs,
                    self.angle_max_residuals,
                    self.angle_circular_layers,
                    teacher_corners=dn_out_corners[-1],
                    teacher_logits=dn_out_logits[-1],
                )
                out['dn_pre_outputs'] = {'pred_logits': dn_pre_logits, 'pred_boxes': dn_pre_bboxes}
                out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss2(
        self,
        outputs_class,
        outputs_coord,
        outputs_corners,
        outputs_ref,
        outputs_angle_logits=None,
        outputs_ref_angles=None,
        angle_max_residuals=None,
        angle_circular_layers=None,
        teacher_corners=None,
        teacher_logits=None,
    ):
        return [
            {
                'pred_logits': a,
                'pred_boxes': b,
                'pred_corners': c,
                'ref_points': d,
                **({
                    'pred_angle_logits': e,
                    'ref_angles': f,
                    'angle_max_residual': g,
                    'angle_circular': h,
                } if e is not None and f is not None else {}),
                **({'teacher_corners': teacher_corners, 'teacher_logits': teacher_logits}
                   if teacher_corners is not None and teacher_logits is not None else {}),
            }
            for a, b, c, d, e, f, g, h in zip(
                outputs_class,
                outputs_coord,
                outputs_corners,
                outputs_ref,
                outputs_angle_logits if outputs_angle_logits is not None else [None] * len(outputs_class),
                outputs_ref_angles if outputs_ref_angles is not None else [None] * len(outputs_class),
                angle_max_residuals if angle_max_residuals is not None else [0.5] * len(outputs_class),
                angle_circular_layers if angle_circular_layers is not None else [True] * len(outputs_class),
            )
        ]
