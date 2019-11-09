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
ROOT_DIR = "/home/leizhou/Documents/outlier_detect/survpred/data/iowa_surv/"
IMG_NAME = "InputCTPrimary_ROI.nii.gz"
GT_NAME = "GTV_Primary_ROI_CT_ST.nii.gz"

surv_list = []
dead_list = []

with open(os.path.join(ROOT_DIR, 'Binary_96_selected.csv')) as csvfile:
    reader  = csv.reader(csvfile)
    next(reader)
    for row in reader:
        if row[-1] == '1':
            surv_list.append(row[0])
        else:
            dead_list.append(row[0])
# print("Surv case # :", len(surv_list))
# print("Dead case # :", len(dead_list))

random.seed(0)
random.shuffle(surv_list)
random.shuffle(dead_list)

tr_case = surv_list[:TR_NB]
val_case = surv_list[TR_NB:TR_NB+VAL_NB]

class IOWA96(Dataset):
    """
    Model the IOWA96 dataset.
    """
    def __init__(self, train=True):
        """
        Class constructor.

        :param train: training dataset or not
        """
        super(IOWA96, self).__init__()

        self.train = train
        if self.train:
            self.case_list = tr_case
        else:
            self.case_list = val_case

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, i):

        img = os.path.join(ROOT_DIR, 'Iowa_96_cases_binary_selected', self.case_list[i], IMG_NAME)
        gt = os.path.join(ROOT_DIR, 'Iowa_96_cases_binary_selected', self.case_list[i], GT_NAME)
        img = sitk.GetArrayFromImage(sitk.ReadImage(img))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gt))
        img = np.clip(img, -500, 200)
        img -= np.mean(img)
        img /= np.std(img)
        rn = random.uniform(0,1)
        if self.train:
            if rn > 0.66:
                img = np.rot90(img, axes=(1,2))
                gt = np.rot90(gt, axes=(1,2))
            elif rn < 0.33:
                img = np.rot90(img, axes=(2,1))
                gt = np.rot90(gt, axes=(2,1))
            rn = random.uniform(0,1)
            if rn > 0.5:
                img = np.fliplr(img)
                gt = np.fliplr(gt)
            rn = random.uniform(0,1)
            if rn > 0.5:
                img = np.flipud(img)
                gt = np.flipud(gt)
        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        gt = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0)

        return img, gt


if __name__ == "__main__":
    dataset = IOWA96()
    loader = DataLoader(dataset, shuffle=False)
    for _, (img, gt) in enumerate(loader):
        fig, axes = plt.subplots(1,2)
        pos = axes[0].imshow(img.squeeze()[0,], cmap='gray')
        fig.colorbar(pos, ax=axes[0])
        pos = axes[1].imshow(gt.squeeze()[0,])
        fig.colorbar(pos, ax=axes[1])
        plt.show()
        break


        