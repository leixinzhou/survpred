import numpy as np
import SimpleITK as sitk 
import torch
from torchvision import datasets
from trochvision import transforms

class IOWA96(datasets):
    """
    Model the IOWA96 dataset.
    """
    def __init__(self, img_seg_dir, gt_dir):
        """
        Class constructor.

        :param img_seg_dir: the folder in which images and seg gt locate.
        :param gt_dir: the file of survival gt 
        """
        super(IOWA96, self).__init__()

        self.img_seg_dir = img_seg_dir
        self.gt_dir = gt_dir

        