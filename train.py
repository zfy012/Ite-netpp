import argparse
import os
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from dataset import ImagePool, Normalize, ToTensor
from loss import BCEDiceLoss
from Net import IterUnetPP
from utils import get_logger
from medpy.metric.binary import dc
import numpy as np
import nibabel as nib


train_image_dir = os.path.join(r"F:\3dite-unet++\dataset", 'train_set', 'image')
train_label_dir = os.path.join(r"F:\3dite-unet++\dataset", 'train_set', 'label')
valid_image_dir = os.path.join(r"F:\3dite-unet++\dataset", 'valid_set', 'image')
valid_label_dir = os.path.join(r"F:\3dite-unet++\dataset", 'valid_set', 'label')


# load data
Transfrom = transforms.Compose([Normalize(), ToTensor()])
train_set = ImagePool(train_image_dir, train_label_dir, transform=Transfrom, mode='train')
valid_set = ImagePool(valid_image_dir, valid_label_dir, transform=Transfrom, mode='valid')

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)

deep_supervision = False
out_predict = True
# build model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IterUnetPP(in_shape=[240, 240, 48], in_channel=1, out_channel=1, seg_channel=1, deep_supervision=deep_supervision).to(device)

# prepare optimizer
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = BCEDiceLoss()
train_maps, valid_maps = {}, {}
writer = SummaryWriter()
logger = get_logger("train_logger.log")
logger.info("start training...")

ite_num = 4
ep_num = 3
stop_threshold = 500

while True:
    for ite in range(ite_num):
        for ep in range(ep_num):
            total_threshold = 0
            model.train()
            train_loss = 0.0
            for idx, batch in enumerate(train_loader):
                image = batch['img'].to(device)
                label = batch['seg'].to(device)

                optimizer.zero_grad()
                if ite == 0:
                    label_map = torch.ones((240, 240, 48), dtype=torch.float32).sub_(0.5).view(-1, 1, 240, 240, 48).to(device)
                else:
                    label_map = torch.tensor(train_maps[idx]).view(-1, 1, 240, 240, 48).to(device)
                loss = 0.0
                if deep_supervision:
                    d1, d2, d3, d4 = model(image, label_map)
                    loss += criterion(d1, label)
                    loss += criterion(d2, label)
                    loss += criterion(d3, label)
                    loss += criterion(d4, label)
                    loss /= 4.
                else:
                    d0 = model(image, label_map)
                    loss = criterion(d0, label)

                loss.backward()
                optimizer.step()
                if ep == ep_num - 1:
                    if deep_supervision:
                       out = torch.sigmoid(d4).squeeze().detach().cpu().numpy()
                    else:
                       out = torch.sigmoid(d0).squeeze().detach().cpu().numpy()
                    train_maps[idx] = out
                torch.cuda.empty_cache()

                train_loss += loss.item()
                logger.info("Epoch:[{:0>5d}/{:0>5d}/{:0>5d}]\t train loss={:.5f}\t".format(idx + 1, ep + 1, ite + 1, loss.item()))

            train_loss /= len(train_loader)
            writer.add_scalar('Loss/train', train_loss, ite*ep_num+ep)

            model.eval()
            valid_loss = 0.0
            for idx, batch in enumerate(valid_loader):
                image = batch['img'].to(device)
                label = batch['seg'].to(device)

                with torch.no_grad():
                    if ite == 0:
                        label_map = torch.ones((240, 240, 48), dtype=torch.float32).sub_(0.5).view(-1, 1, 240, 240, 48).to(device)
                    else:
                        label_map = torch.tensor(valid_maps[idx]).view(-1, 1, 240, 240, 48).to(device)

                    loss = 0.0
                    if deep_supervision:
                        d1, d2, d3, d4 = model(image, label_map)
                        loss += criterion(d1, label)
                        loss += criterion(d2, label)
                        loss += criterion(d3, label)
                        loss += criterion(d4, label)
                        loss /= 4.
                    else:
                        d0 = model(image, label_map)
                        loss = criterion(d0, label)

                    if ep == ep_num - 1:
                        if deep_supervision:
                            out = torch.sigmoid(d4).squeeze().detach().cpu().numpy()
                        else:
                            out = torch.sigmoid(d0).squeeze().detach().cpu().numpy()
                        valid_maps[idx] = out
                        if ite != 0:
                            for idx in valid_maps.keys():
                                threshold = np.sum(np.fabs(np.subtract(iter_map[idx], valid_maps[idx])))
                                total_threshold += threshold
                            print("total threshold is {}".format(total_threshold))

                            if total_threshold < stop_threshold:
                                print("iteration is over, total threshold = {}".format(total_threshold))
                                logger.info("finish training....")
                                break
                        iter_map = copy.deepcopy(valid_maps)

                    torch.cuda.empty_cache()

                    valid_loss += loss.item()
                    logger.info("Epoch:[{:0>5d}/{:0>5d}/{:0>5d}]\t valid loss={:.5f}\t".format(idx + 1, ep + 1, ite + 1, loss.item()))

            valid_loss /= len(valid_loader)
            writer.add_scalar('Loss/valid', valid_loss, ite*ep_num+ep)




    # save model


    model_name = os.path.join(r"F:\3dite-unet++\model", '%05d.ckpt' % (ite + 1))
    torch.save(model.state_dict(), model_name)





