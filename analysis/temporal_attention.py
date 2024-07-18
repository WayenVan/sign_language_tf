from omegaconf import OmegaConf, DictConfig
import numpy as np 
from torch import nn
import re
import sys
sys.path.append('src')
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from csi_sign_language.data_utils.ph14.evaluator_sclite import Pheonix14Evaluator
from csi_sign_language.data.datamodule.ph14 import Ph14DataModule

from csi_sign_language.models.slr_model import SLRModel
import os
from datetime import datetime
import click

from torch.nn import MultiheadAttention

@click.option('--config', '-c', default='outputs/train/2024-05-17_22-32-27/config.yaml')
@click.option('-ckpt', '--checkpoint', default='outputs/train/2024-05-17_22-32-27/epoch=56_wer-val=28.05_lr=0.00e+00_loss=0.00.ckpt')
# @click.option('--config', '-c', default='outputs/train/2024-05-14_22-07-53/config.yaml')
# @click.option('-ckpt', '--checkpoint', default='outputs/train/2024-05-14_22-07-53/epoch=54_wer-val=23.25_lr=0.00e+00_loss=0.00.ckpt')
@click.option('--ph14_lmdb_root', default='dataset/preprocessed/ph14_lmdb')
@click.option('--mode', default='test')
@click.option('--device', default='cuda:1')
@click.command()
def main(config, checkpoint, ph14_lmdb_root, mode, device):

    if mode not in ['val', 'test']:
        raise NotImplementedError()

    current_time = datetime.now()
    file_name = os.path.basename(__file__)
    save_dir = os.path.join('outputs', file_name[:-3], current_time.strftime("%Y-%m-%d_%H-%M-%S"))
    cfg = OmegaConf.load(config)
    
    dm = Ph14DataModule(ph14_lmdb_root, batch_size=1, num_workers=6, train_shuffle=True, val_transform=instantiate(cfg.transforms.test), test_transform=instantiate(cfg.transforms.test))
    model = SLRModel.load_from_checkpoint(checkpoint, cfg=cfg, map_location='cpu', ctc_search_type='beam', strict=False)
    model.set_post_process(dm.get_post_process())

    for p in model.parameters():
        assert p.isnan().any() != True
        assert p.isinf().any() != True
    
    for n, m in model.named_modules():
        print(n)
        if re.match(r'backbone.decoder.tf.trans_decoder.layers.[0-9].self_attn', n):
            m.register_forward_hook(get_attention)
    
    test_loader = dm.test_dataloader()
    
    i = 0
    for test_data in iter(test_loader):
        i+=1
        if i == 4:
            break
    # test_data = next(iter(test_loader))
    
    x = test_data['video'].to(device)
    t_length = test_data['video_length'].to(device)
    model.to(device)
    
    output, hyp = model(x, t_length)
    print(len(attn))
    
    print(hyp)
    print(test_data['gloss_label'])
    print(dm.get_post_process().process(hyp, test_data['gloss_label']))
    

    
    #draw heatmap
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.subplots_adjust(wspace=0.05, hspace=0.15)
    for i in range(2):
        for j in range(3):
            axes[i][j].set_title('output {}'.format(i*2+j))
            axes[i][j].yaxis.set_major_locator(ticker.MultipleLocator(1))
            axes[i][j].xaxis.set_major_locator(ticker.MultipleLocator(1))
            axes[i][j].imshow(X=attn[i*2+j][0], vmin=0., vmax=0.25)
            print(np.std(a=attn[i*2+j], axis=-1))
            axes[i][j].tick_params(axis='x', labelrotation=90)
            axes[i][j].grid(True)

    plt.savefig('resources/attention.pdf')

attn = []
def get_attention(m, input, output):
    attn.append(output[1].detach().cpu().numpy())

    
    
if __name__ == '__main__':
    main() 
    

