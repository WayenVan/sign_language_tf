import torch
import hydra
import sys

sys.path.append("src")
from hydra.utils import instantiate
from csi_sign_language.models.slr_model import SLRModel
from lightning import Trainer
from lightning.pytorch import LightningModule
from lightning.pytorch import callbacks
import socket
from lightning.pytorch.callbacks import Callback


class DebugCallback(Callback):
    def on_train_start(self, trainer, pl_module: LightningModule):
        # for name, param in pl_module.named_parameters():
        #     print(name)
        return

    def on_train_batch_start(
        self, trainer, pl_module: LightningModule, batch, batch_idx
    ):
        # print('on_train_batch_start')
        # _, _, T, _, _ = batch['video'].shape
        # pl_module.log('t_length', T, logger=None, on_step=True, on_epoch=False, prog_bar=True)
        # print(T)
        return

    def on_before_optimizer_step(
        self, trainer: Trainer, pl_module: LightningModule, optimizer
    ) -> None:
        # for name, param in pl_module.named_parameters():
        #     if name == 'backbone.encoder.vit.heatmap_headers.10.conv.conv.weight':
        #         print(param.grad)
        return


def test_model():
    hydra.initialize_config_dir("/root/projects/sign_language_transformer/configs")
    cfg = hydra.compose("run/train/resnet_distill_v2_efficient")
    # cfg = hydra.compose('run/train/dual')
    index = 0
    print(socket.gethostname())

    datamodule = instantiate(cfg.datamodule)
    vocab = datamodule.get_vocab()

    lightning_module = SLRModel(cfg, vocab)
    # lightning_module = SLRModel.load_from_checkpoint('outputs/train/2024-08-05_20-40-03/epoch=4_wer-val=89.29_lr=1.00e-04_loss=7.62.ckpt', cfg=cfg)
    lightning_module.set_post_process(datamodule.get_post_process())

    t = Trainer(
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        # strategy='deepspeed_stage_2',
        # max_steps=100,
        devices=getattr(cfg, "devices", [1]),
        logger=False,
        enable_checkpointing=False,
        precision=16,
        callbacks=[callbacks.RichProgressBar(), DebugCallback()],
    )

    t.fit(lightning_module, datamodule)
    # t.validate(lightning_module, datamodule.val_dataloader())
    return


if __name__ == "__main__":
    test_model()
