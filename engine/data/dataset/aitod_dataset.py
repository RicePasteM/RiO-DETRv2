import os

from .dota_dataset import DOTADataset
from ...core import register


AITOD_CLASSES = (
    'airplane',
    'bridge',
    'storage-tank',
    'ship',
    'swimming-pool',
    'vehicle',
    'person',
    'wind-mill',
)


@register()
class AITODDataset(DOTADataset):
    """AI-TOD-R dataset in DOTA text format."""

    CLASSES = AITOD_CLASSES

    def __init__(
        self,
        img_folder,
        ann_folder,
        transforms=None,
        return_masks=False,
        exclude_ids=None,
    ):
        super().__init__(
            img_folder,
            ann_folder,
            transforms=transforms,
            return_masks=return_masks,
        )

        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

        def normalize_image_id(image_id):
            image_id = os.path.basename(str(image_id))
            return os.path.splitext(image_id)[0] if image_id.lower().endswith(image_extensions) else image_id

        excluded = {normalize_image_id(image_id) for image_id in (exclude_ids or [])}
        if excluded:
            self.img_ids = [image_id for image_id in self.img_ids if image_id not in excluded]
