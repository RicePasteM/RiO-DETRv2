"""AI-TOD-R evaluator with the official COCO-style AP interpolation."""

from ...core import register
from .dota_evaluator import DotaEvaluator


@register()
class AITODREvaluator(DotaEvaluator):
    """Evaluate AI-TOD-R with geometric OBB IoU and 101-point AP."""

    def __init__(self, dataset, iou_types=None, iou_thrs=None):
        super().__init__(
            dataset=dataset,
            iou_types=iou_types,
            iou_thrs=iou_thrs,
            ap_mode='coco101',
        )
