#! /usr/bin/env python3
import numpy as np
import json
from omegaconf import OmegaConf, DictConfig
from torch import nn
import sys
import logging
import cv2

import torch.utils
import torch.utils.data
sys.path.append('src')
from hydra.utils import instantiate
from matplotlib import pyplot as plt
import torch
from csi_sign_language.data_utils.ph14.evaluator_sclite import Pheonix14Evaluator
from csi_sign_language.data.datamodule.ph14 import Ph14DataModule

from einops import rearrange
from csi_sign_language.models.slr_model import SLRModel
import hydra
import os
import json
from datetime import datetime
import click
from torchmetrics.text import WordErrorRate
from lightning import LightningModule, Trainer
from lightning import Callback
import math

feat_map = None
grads = None

def regist_hooks_vit(model: LightningModule):

    def reshape(x):
        patch_token = x[:, 1:]
        B, N, C = patch_token.shape
        # breakpoint()
        # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
        h = int(math.sqrt(N))
        w = int(math.sqrt(N))
        return patch_token.reshape(B, h, w, -1).transpose(0, 3, 1, 2)

    target_layer: torch.nn.Module = model.backbone.encoder.vitpool.layers[-2]
    # target_layer: torch.nn.Module = model.backbone.encoder.vit.layers[-1]
    def fhook(m, argvit, output):
        print(output.shape)
        global feat_map
        feat_map = reshape(output.clone().detach().cpu().numpy())
    def bhook(m, g_in, g_out):
        print(len(g_out))
        print(g_out[0].shape)
        global grads
        grads = reshape(g_out[0].clone().detach().cpu().numpy())

    target_layer.register_backward_hook(bhook)
    target_layer.register_forward_hook(fhook)

def regist_hooks_x3d(model: LightningModule):
    def reshape(x):
        return rearrange(x, 'n c t h w -> (n t) c h w')
    target_layer: torch.nn.Module = model.backbone.encoder.x3d.res_stages[-1]
    def fhook(m, args, output):
        print(output.shape)
        global feat_map
        feat_map = reshape(output.clone().detach().cpu().numpy())
    def bhook(m, g_in, g_out):
        print(len(g_out))
        print(g_out[0].shape)
        global grads
        grads = reshape(g_out[0].clone().detach().cpu().numpy())

    target_layer.register_backward_hook(bhook)
    target_layer.register_forward_hook(fhook)

def regist_hooks(model: LightningModule):
    target_layer: torch.nn.Module = model.backbone.encoder.resnet.layer4
    def fhook(m, args, output):
        print(output.shape)
        global feat_map
        feat_map = output.clone().detach().cpu().numpy()
    def bhook(m, g_in, g_out):
        print(len(g_out))
        print(g_out[0].shape)
        global grads
        grads = g_out[0].clone().detach().cpu().numpy()

    target_layer.register_backward_hook(bhook)
    target_layer.register_forward_hook(fhook)

@click.option('--config', '-c', default='outputs/vitpose_trans_best/config.yaml')
@click.option('-ckpt', '--checkpoint', default='outputs/vitpose_trans_best/val-wer=24.ckpt')
@click.option('--ph14_root', default='dataset/phoenix2014-release')
@click.option('--ph14_lmdb_root', default='preprocessed/ph14_lmdb')
@click.option('--index', default=2)
@click.command()
def main(config, checkpoint, ph14_root, ph14_lmdb_root, index):

    current_time = datetime.now()
    file_name = os.path.basename(__file__)
    save_dir = os.path.join('outputs', file_name[:-3], current_time.strftime("%Y-%m-%d_%H-%M-%S"))
    cfg = OmegaConf.load(config)

    dm = Ph14DataModule(ph14_lmdb_root, batch_size=1, num_workers=6, train_shuffle=True, val_transform=instantiate(cfg.transforms.test), test_transform=instantiate(cfg.transforms.test))
    dataset = dm.test_set
    single_dataset = torch.utils.data.Subset(dataset, [index])
    single_loader = torch.utils.data.DataLoader(single_dataset, batch_size=1, num_workers=6, collate_fn=dm.collate_fn)

    model = SLRModel.load_from_checkpoint(checkpoint, cfg=cfg, map_location='cpu', ctc_search_type='beam', strict=False)
    model.set_post_process(dm.get_post_process())
    
    t = Trainer(
        accelerator='gpu',
        strategy='ddp',
        max_steps=1,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        precision=32,
        callbacks=[CB()]
    )
    
    result = t.predict(model, single_loader)[0][0]

    single_data = single_dataset[0]
    single_data['gloss'] = torch.tensor(dataset.vocab(result[1]), dtype=torch.int64)
    single_data['gloss_label'] = result[1]
    single_loader = torch.utils.data.DataLoader([single_data], batch_size=1, collate_fn=dm.collate_fn)
    
    regist_hooks_vit(model=model)

    t.fit(model, single_loader)
        
    # cam = compute_grad_cam_batch(feat_map, grads)
    cam = compute_grad_cam_plus_plus_batch(feat_map, grads)

    
    #bgr2rgb
    original_images = np.uint8(single_data['video'].numpy().transpose(0, 2, 3, 1)*255.)
    original_images = original_images[:, :, :, ::-1]
    visualize_and_save_grad_cam_images(original_images, cam, save_dir)
    with open(os.path.join(save_dir, '_info.json'), 'w') as f:
        info = dict(
            cfg=config,
            ckpt=checkpoint,
            test_index=index,
            id=result[0],
            hyp=result[1],
            gt=result[2],
        )
        json.dump(info, f, indent=4)
    
def visualize_and_save_grad_cam_images(original_images, grad_cam_heatmap_batch, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Superimpose Grad-CAM heatmaps on the original images in the batch
    for i, (original_image, grad_cam_heatmap) in enumerate(zip(original_images, grad_cam_heatmap_batch)):
        # Resize Grad-CAM heatmap to match the input image size
        grad_cam_heatmap_resized = cv2.resize(grad_cam_heatmap, (original_image.shape[1], original_image.shape[0]))

        # Apply colormap for visualization
        grad_cam_heatmap_colored = cv2.applyColorMap(np.uint8(255 * grad_cam_heatmap_resized), cv2.COLORMAP_JET)

        # Superimpose Grad-CAM heatmap on the original image
        output_image = cv2.addWeighted(original_image, 0.6, grad_cam_heatmap_colored, 0.4, 0)

        # Save the output image
        output_path = os.path.join(output_dir, f"grad_cam_image_{i}.jpg")
        cv2.imwrite(output_path, output_image)


class CB(Callback):
    
    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx) -> None:
        pl_module.eval()
        
def compute_grad_cam_batch(feature_map_batch, grads_batch):
    # Global average pooling of gradients
    weights = np.mean(grads_batch, axis=(2, 3))

    # Weighted combination of feature maps
    grad_cam_batch = np.einsum('ijkl,ij->ikl', feature_map_batch, weights)

    # ReLU operation
    grad_cam_batch = np.maximum(grad_cam_batch, 0)

    # Normalize the heatmap
    grad_cam_batch /= np.max(grad_cam_batch, axis=(1, 2), keepdims=True)

    return grad_cam_batch

def compute_grad_cam_plus_plus_batch(feature_map_batch, grads_batch):
    # Global average pooling of gradients
    weights_pos = np.maximum(0.0, np.sum(grads_batch, axis=(2, 3)))  # Positive gradients
    weights_neg = np.maximum(0.0, -np.sum(grads_batch, axis=(2, 3)))  # Negative gradients

    # Weighted combination of feature maps for positive gradients
    grad_cam_pos = np.einsum('ijkl,ij->ikl', feature_map_batch, weights_pos)

    # Weighted combination of feature maps for negative gradients
    grad_cam_neg = np.einsum('ijkl,ij->ikl', feature_map_batch, weights_neg)

    # Combine positive and negative contributions
    grad_cam_plus_plus_batch = np.maximum(grad_cam_pos, 0) + np.minimum(grad_cam_neg, 0)

    # Normalize the heatmap for each sample in the batch
    max_vals = np.max(grad_cam_plus_plus_batch, axis=(1, 2), keepdims=True)
    grad_cam_plus_plus_batch /= (max_vals + 1e-8)

    return grad_cam_plus_plus_batch


main()