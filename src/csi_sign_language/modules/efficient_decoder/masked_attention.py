import torch
from torch import tensor
import random

@torch.no_grad()
def make_diagonal_mask(Lq, Lk, device, k=4):
    assert Lq == Lk
    mask = torch.eye(Lq, dtype=torch.int64, device=device)
    
    indices = tuple(range(Lq))

    l = k
    while(l <= Lk - 1):
        off = Lk-l
        mask[indices[0:off], indices[-off:]] = 1
        mask[indices[-off:], indices[0:off]] = 1
        l += k
    
    
    return mask


@torch.no_grad()
def make_random_mask_bucket(Lq, Lk, device, bucket_size=4):
    def split_list_into_groups(lst, group_size):
        return [lst[i:i + group_size] for i in range(0, len(lst), group_size)]
    
    groups = split_list_into_groups(list(range(Lk)), bucket_size)
    sampled_index = [random.choice(group) for group in groups]

    mask = torch.zeros(Lq, Lk, device=device, dtype=torch.int64)
    mask[:, sampled_index] = 1

    return mask
    

def plot_mask(mask, file_path):
    import matplotlib.pyplot as plt
    if len(mask.shape) == 2:
        plt.imshow(mask.cpu().numpy())
        plt.savefig(file_path)
    elif len(mask.shape) == 3:
        plt.imshow(mask[0].cpu().numpy())
        plt.savefig(file_path)
    
if __name__ == '__main__':
    mask = make_diagonal_mask(40, 40, 'cuda:1')
    plot_mask(mask, 'resources/mask.pdf')
    print( mask.float().mean())

