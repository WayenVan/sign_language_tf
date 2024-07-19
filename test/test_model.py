import torch
import hydra
import sys
sys.path.append('src')
from hydra.utils import instantiate
from omegaconf import OmegaConf
from csi_sign_language.models.slr_model import SLRModel
from lightning import Trainer
from lightning.pytorch import callbacks
from mmpretrain.models.backbones.levit import LeViT
def test_model():
    hydra.initialize_config_dir('/home/jingyan/Documents/sign_language_transformer/configs')
    cfg = hydra.compose('run/train/vifi_clip')
    index = 0

    datamodule = instantiate(cfg.datamodule)
    vocab = datamodule.get_vocab()

    lightning_module = SLRModel(cfg, vocab)
    lightning_module.set_post_process(datamodule.get_post_process())
    
    t = Trainer(
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_true',
        max_steps=3,
        devices=[1],
        logger=False,
        enable_checkpointing=False,
        precision=16,
        callbacks=[callbacks.RichProgressBar()]
    )
    
    t.fit(lightning_module, datamodule)
    return



if __name__ == "__main__":
    test_model()
    
    