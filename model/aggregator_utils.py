import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def _train_ga_regressor(model,
                        train_loader,
                        max_lr,
                        epochs,
                        device):
    # Optimize
    model.train()
    criteria = nn.L1Loss()
    criteria.to(device)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        betas=(0.9, 0.999),
        lr=max_lr,
        weight_decay=1e-4
    )
    lr_schedular = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.02
    )
    losses, loss_run_avg = [], []

    for epo in range(1, epochs + 1):
        for idx, (data, geo_proximity) in enumerate(train_loader):
            input_tensor = data.to(device)
            optimizer.zero_grad()

            gt = input_tensor[:, -1:, -1:].clone()  # [bs, 1, 1]
            input_tensor[:, -1:, -1:] = torch.nan  # mask the target y
            pred_y = model(input_tensor, geo_proximity)

            loss = criteria(gt, pred_y)
            loss_run_avg.append(loss.item())
            loss.backward()
            optimizer.step()
            lr_schedular.step()

            if idx % 100 == 0:
                step_loss = sum(loss_run_avg) / len(loss_run_avg)
                losses.append(step_loss)
                loss_run_avg = []
                logger.info(f'Epoch: {epo:2}/{epochs}  |  '
                            f'Step: {idx:3}/{len(train_loader)}  |  '
                            f'loss_step_avg: {step_loss:.4f}  |  '
                            f'lr: {lr_schedular.get_last_lr()[0]:.4f}  |  '
                            f'abf: {model._perceiver.attn_bias_factor.item():.4f}')


def _test_ga_regressor(model,
                       test_loader,
                       device,
                       n_estimate=8,
                       get_std=False):
    # Predict
    model.eval()
    criteria = nn.L1Loss()
    criteria.to(device)
    predictions = torch.zeros(size=(n_estimate, len(test_loader)))
    with torch.no_grad():
        for i_est in range(n_estimate):
            for i_data, (data, geo_proximity) in tqdm(enumerate(test_loader), desc='Inferencing'):
                input_tensor = data.to(device)

                input_tensor[:, -1:, -1:] = torch.nan  # (pseudo) ground truth position
                pred_y = model(input_tensor, geo_proximity)

                predictions[i_est, i_data] = pred_y.item()
    
    if get_std:
        return predictions.mean(dim=0), predictions.std(dim=0)
    else:
        return predictions.mean(dim=0)
