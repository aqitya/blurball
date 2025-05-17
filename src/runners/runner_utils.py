import os
import os.path as osp
import time
import wandb
import logging
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch
from torch import nn

from utils import save_checkpoint, AverageMeter

log = logging.getLogger(__name__)


def train_epoch(
    epoch, model, train_loader, loss_criterion, optimizer, device, current_step
):
    batch_loss = AverageMeter()
    model.train()
    t_start = time.time()
    for batch_idx, (imgs, hms) in enumerate(
        tqdm(train_loader, desc="[(TRAIN) Epoch {}]".format(epoch))
    ):
        for scale, hm in hms.items():
            hms[scale] = hm.to(device)

        optimizer.zero_grad()

        preds = model(imgs)
        loss = loss_criterion(preds, hms)
        loss.backward()
        optimizer.step()

        batch_loss.update(loss.item(), preds[0].size(0))
        current_step += 1
    t_elapsed = time.time() - t_start

    # wandb.log({"train/loss": loss.item()}, step=current_step)
    log.info(
        "(TRAIN) Epoch {epoch} Loss:{batch_loss.avg:.6f} Time:{time:.1f}(sec)".format(
            epoch=epoch, batch_loss=batch_loss, time=t_elapsed
        )
    )
    return {"epoch": epoch, "loss": batch_loss.avg, "current_step": current_step}


@torch.no_grad()
def test_epoch(
    epoch, model, dataloader, loss_criterion, device, cfg, current_step, vis_dir=None
):
    batch_loss = AverageMeter()
    model.eval()

    t_start = time.time()
    for batch_idx, (imgs, hms, trans, xys_gt, visis_gt, img_paths) in enumerate(
        tqdm(dataloader, desc="[(TEST) Epoch {}]".format(epoch))
    ):
        imgs = imgs.to(device)
        for scale, hm in hms.items():
            hms[scale] = hm.to(device)
        preds = model(imgs)
        loss = loss_criterion(preds, hms)
        batch_loss.update(loss.item(), preds[0].size(0))
    t_elapsed = time.time() - t_start

    log.info(
        "(TEST) Epoch {epoch} Loss:{batch_loss.avg:.6f} Time:{time:.1f}(sec)".format(
            epoch=epoch, batch_loss=batch_loss, time=t_elapsed
        )
    )
    return {"epoch": epoch, "loss": batch_loss.avg}
