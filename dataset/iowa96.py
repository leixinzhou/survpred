import numpy as np
import SimpleITK as sitk 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import csv
import random, os

TR_NB = 48
VAL_NB = 5
ROOT_DIR = "./Iowa_96_cases_binary_selected/"
IMG_NAME = "InputCTPrimary_ROI.nii.gz"
GT_NAME = "GTV_Primary_ROI_CT_ST.nii.gz"
OUTPUT_DIR = "./IOWA96_np/"

surv_list = []
dead_list = []

with open('Binary_96_selected.csv') as csvfile:
    reader  = csv.reader(csvfile)
    next(reader)
    for row in reader:
        if row[-1] == '1':
            surv_list.append(row[0])
        else:
            dead_list.append(row[0])
print("Surv case # :", len(surv_list))
print("Dead case # :", len(dead_list))

random.seed(0)
random.shuffle(surv_list)
random.shuffle(dead_list)

tr_case = surv_list[:TR_NB]
val_case = surv_list[TR_NB:TR_NB+VAL_NB]

class IOWA96(Dataset):
    """
    Model the IOWA96 dataset.
    """
    def __init__(self, img_dir, seg_gt_dir):
        """
        Class constructor.

        :param img_dir: the img numpy file
        :param seg_gt_dir: the segmentation gt numpy file
        """
        super(IOWA96, self).__init__()

        self.img_dir = img_dir
        self.seg_gt_dir = seg_gt_dir

        self.img = np.load(self.img_dir)
        self.seg_gt = np.load(self.seg_gt_dir)

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, i):
        tmp_img = np.clip(self.img[i,], -500, 200)
        tmp_img -= np.mean(tmp_img)
        tmp_img /= np.std(tmp_img)
        tmp_gt = self.seg_gt[i,]
        tmp_img = torch.from_numpy(tmp_img.astype(np.float32)).unsqueeze(0)
        tmp_gt = torch.from_numpy(tmp_gt.astype(np.float32)).unsqueeze(0)

        return tmp_img, tmp_gt


if __name__ == "__main__":
    img_dir = "data/iowa_surv/IOWA96_np/tr_img.npy"
    gt_dir = "data/iowa_surv/IOWA96_np/tr_seg_gt.npy"
    dataset = IOWA96(img_dir, gt_dir)
    loader = DataLoader(dataset, shuffle=True)
    for _, (img, gt) in enumerate(loader):
        fig, axes = plt.subplots(1,2)
        pos = axes[0].imshow(img.squeeze(), cmap='gray')
        fig.colorbar(pos, ax=axes[0])
        pos = axes[1].imshow(gt.squeeze())
        fig.colorbar(pos, ax=axes[1])
        plt.show()
        break


        