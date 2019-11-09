import torch

from models.base import BaseModule


class SegCELoss(BaseModule):
    """
    Implements the segmentation loss.
    """
    def __init__(self):
        # type: () -> None
        """
        Class constructor.
        """
        super(SegLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x, x_r):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of ref segmentations.
        :param x_r: the batch of prediction segmentations.
        :return: the mean seg loss (averaged along the batch axis).
        """
        return self.loss_fn(x_r, x)

class SegDiceLoss(BaseModule):
    """
    Implements the segmentation loss.
    """
    def __init__(self, smooth=1.):
        # type: () -> None
        """
        Class constructor.
        """
        super(SegDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, x, x_r):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of ref segmentations.
        :param x_r: the batch of prediction segmentations.
        :return: the mean seg loss (averaged along the batch axis).
        """
        x_r = torch.sigmoid(x_r).unsqueeze(1)
        x = x.unsqueeze(1)

        dice_array = torch.stack([1. - 2. * (torch.sum(x_r[i,] * x[i,]) + self.smooth) / (torch.sum(x_r[i,]) + \
                            torch.sum(x[i,]) + self.smooth) for i in range(x.size(0))])
        return torch.mean(dice_array)


