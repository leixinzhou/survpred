import torch

from models.base import BaseModule


class SegLoss(BaseModule):
    """
    Implements the segmentation loss.
    """
    def __init__(self):
        # type: () -> None
        """
        Class constructor.
        """
        super(SegLoss, self).__init__()
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, x, x_r):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of ref segmentations.
        :param x_r: the batch of prediction segmentations.
        :return: the mean seg loss (averaged along the batch axis).
        """
        return self.loss_fn(x_r, x)
