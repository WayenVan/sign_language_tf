import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import os
import sys
import matplotlib.pyplot as plt
sys.path.append('src')

def main():
    device = 'cuda'
    check_point_path = 'outputs/train_transformer'
    cfg = OmegaConf.load(os.path.join(check_point_path, '.hydra/config.yaml'))
    dataset = instantiate(cfg.data.val_set)
    model: torch.nn.Module = instantiate(cfg.model, vocab=dataset.get_vocab())
    model.to(device)

    data = dataset[1]
    id=[data['id']]
    video=data['video'].unsqueeze(0).permute(1, 0, 2, 3, 4).to(device)
    gloss=data['gloss'].unsqueeze(0).to(device)
    video_length=torch.tensor([len(data['video'])], dtype=torch.int64).to(device)
    gloss_length=torch.tensor([len(data['gloss'])], dtype=torch.int64).to(device)
    

    for name, _ in model.backbone.named_modules():
        print(name, _.__class__.__name__)

    result = []
    def f(module, input, output):
        result.append(output)

    for name, module in model.backbone.named_modules():
        if 'resnet.layer1.0.bn2' in name:
            # if module.__class__.__name__ == 'ReLU':
            module.register_forward_hook(f)

    model.inference(video, video_length)
    output1 = result[0].squeeze(0).detach().cpu().numpy()[0]
    # output2 = result[1].squeeze(0).detach().cpu().numpy()
    
    fx, ax = plt.subplots(8, 8)
    for i in range(64):
        ax[i//8][i%8].imshow(output1.transpose(1, 2, 0)[:, : ,i])
    # ax[1].imshow(output2.transpose(1, 0))
    plt.show()

if __name__ == "__main__":
    main()
    

