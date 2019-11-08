import argparse
from argparse import Namespace
from torch.utils.data import DataLoader
import torch

from models import LSAIOWA96
from dataset import IOWA96
# from result_helpers import OneClassResultHelper
# from result_helpers import VideoAnomalyDetectionResultHelper
# from utils import set_random_seed
import matplotlib.pyplot as plt
from vis import show_img_contour

def test_iowa96():
    # type: () -> None
    """
    Performs One-class classification tests on iowa96
    """

    # Build dataset and model
    val_dataset = IOWA96(img_dir="data/iowa_surv/IOWA96_np/val_img.npy", 
                        seg_gt_dir="data/iowa_surv/IOWA96_np/val_seg_gt.npy")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    # network
    net = LSAIOWA96(input_shape=(1, 96, 96), code_length=64, cpd_channels=100).cuda()
    net.load_state_dict(torch.load("run/no_aug/model_best.pth.tar")["state_dict"])
    net.eval()
    for step, (img, seg_gt) in enumerate(val_loader, 1):
        img, seg_gt = img.cuda(), seg_gt.cuda()
        x_r, z, z_dist = net(img)
        x_r = x_r > 0.
        
        fig, axes = plt.subplots(1,3, figsize=(8,4))
        img_np = img.squeeze().detach().cpu().numpy()
        seg_gt_np = seg_gt.squeeze().detach().cpu().numpy()
        seg_pred_np = x_r.squeeze().detach().cpu().numpy()
        for i in range(3):
            axes[i].set(xticks=[], yticks=[])
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title('Image')
        axes[1].set_title('GT')
        show_img_contour(img_np, seg_gt_np, axes[1], 'r')
        axes[2].set_title('Pred')
        show_img_contour(img_np, seg_pred_np, axes[2], 'r')

        plt.savefig('run/no_aug/vis/val/'+str(step)+".png", dpi=150)
        plt.close()


# Entry point
if __name__ == '__main__':
    test_iowa96()
