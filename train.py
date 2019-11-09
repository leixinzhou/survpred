import argparse
from argparse import Namespace
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
import numpy as np
import os, torch

from models import LSAIOWA96
from dataset import IOWA96
from models.loss_functions import LSALoss

PATH = "run/with_autoreg_e-1/"
if not os.path.exists(PATH):
    os.makedirs(PATH)

def save_checkpoint(states,  path, filename='model_best.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint_name = os.path.join(path,  filename)
    torch.save(states, checkpoint_name)


def train_iowa96():
    """
    Train on iowa96.
    """
    tr_dataset = IOWA96(train=True)
    val_dataset = IOWA96(train=False)
    tr_loader = DataLoader(tr_dataset, batch_size=12, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=0)
    # network
    net = LSAIOWA96(input_shape=(1, 48, 96, 96), code_length=64, cpd_channels=100).cuda()
    # optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = LSALoss(cpd_channels=100, lam=0.1)
    # training tracker
    writer = SummaryWriter(PATH)

    best_loss = np.float("inf")
    for epoch in range(1000):
        net.train()
        epoch_loss = 0
        epoch_dice_loss = 0
        epoch_reg_loss = 0
        for step, (img, seg_gt) in enumerate(tr_loader, 1):
            img, seg_gt = img.cuda(), seg_gt.cuda()
            x_r, z, z_dist = net(img)
            loss = loss_fn(x=seg_gt, x_r=x_r, z=z, z_dist=z_dist)
            epoch_loss += loss.detach().cpu().numpy()
            epoch_dice_loss += loss_fn.reconstruction_loss
            epoch_reg_loss += loss_fn.autoregression_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss /= step
        epoch_dice_loss /= step
        epoch_reg_loss /= step
        print("epoch: ", epoch, "tr loss: ", "%.5e" % epoch_loss, "tr dice_loss: ", "%.5e" % epoch_dice_loss, \
                        "tr reg_loss: ", "%.5e" % epoch_reg_loss)
        writer.add_scalar("tr_loss", epoch_loss, epoch)
        writer.add_scalar("tr_dice_loss", epoch_dice_loss, epoch)
        writer.add_scalar("tr_reg_loss", epoch_reg_loss, epoch)

        net.eval()
        epoch_loss = 0
        epoch_dice_loss = 0
        epoch_reg_loss = 0
        for step, (img, seg_gt) in enumerate(val_loader, 1):
            img, seg_gt = img.cuda(), seg_gt.cuda()
            x_r, z, z_dist = net(img)
            loss = loss_fn(x=seg_gt, x_r=x_r, z=z, z_dist=z_dist)
            epoch_loss += loss.detach().cpu().numpy()
            epoch_dice_loss += loss_fn.reconstruction_loss
            epoch_reg_loss += loss_fn.autoregression_loss
        epoch_loss /= step
        epoch_dice_loss /= step
        epoch_reg_loss /= step
        print("epoch: ", epoch, "val loss: ", "%.5e" % epoch_loss, "val dice_loss: ", "%.5e" % epoch_dice_loss, \
                        "val reg_loss: ", "%.5e" % epoch_reg_loss)
        writer.add_scalar("val_loss", epoch_loss, epoch)
        writer.add_scalar("val_dice_loss", epoch_dice_loss, epoch)
        writer.add_scalar("val_reg_loss", epoch_reg_loss, epoch)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict()
                },
                path=PATH,
            )

if __name__ == "__main__":
    train_iowa96()
        
