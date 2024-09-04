import torch
import random
from torch import nn
from einops import rearrange, einsum, repeat
# from ..components.drop_path import DropPath
# from .masked_attention import (
#     make_diagonal_mask,
#     make_random_mask_bucket,
#     plot_mask,
# )
# from .my_scale_dot_product import MyScaledDotProduct
# from xformers.components import MultiHeadDispatch
# from xformers.components.attention.attention_mask import AttentionMask
# from xformers.components.input_projection import InputProjection
# from xformers.components.feedforward import MLP
# from xformers.components.activations import Activation


# class DiagonalMaskGenerator:
#     def __init__(self, step) -> None:
#         self.step = step
#
#     def __call__(self, Lq, Lk, device):
#         return make_diagonal_mask(Lq, Lk, device, k=self.step)
#
#
# class RandomBucketMaskGenerator:
#     def __init__(self, bucket_size) -> None:
#         self.bucket_size = bucket_size
#
#     def __call__(self, Lq, Lk, device):
#         return make_random_mask_bucket(Lq, Lk, device, bucket_size=self.bucket_size)
#
#
# class SparseAttention(nn.Module):
#     """
#     Thes module implements the sparse attention by xformers api,
#     apply different attention mask, when the mask is lower than 30%
#     will be automatically switch to sparse calculation
#     """
#
#     def __init__(self, d_model, num_heads, mask_generator, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#
#         self.attn = MultiHeadDispatch(
#             dim_model=d_model,
#             num_heads=num_heads,
#             attention=MyScaledDotProduct(),
#             use_rotary_embeddings=False,
#         )
#         self.mask_generator = mask_generator
#
#     def forward(self, q, k, v, key_length=None):
#         # [t n c]
#
#         N = q.shape[1]
#         Lq = q.shape[0]
#         Lk = k.shape[0]
#         Lv = v.shape[0]
#         assert Lk == Lv
#
#         with torch.no_grad():
#             mask = self.mask_generator(Lq, Lk, q.device)
#             if key_length is not None:
#                 mask = repeat(mask, "q k -> n q k", n=N).contiguous()
#                 for i in range(N):
#                     mask[i, :, : key_length[i]] = 1
#             mask = mask.bool()
#
#         q, k, v = tuple(rearrange(x, "t n c -> n t c") for x in (q, k, v))
#
#         if q.device.type != "cpu":
#             with torch.cuda.device(q.device):
#                 y = self.attn(q, k, v, att_mask=mask)
#         else:
#             y = self.attn(q, k, v, att_mask=mask)
#
#         y = rearrange(y, "n t c -> t n c")
#         return y


class BucketRandomAttention(nn.Module):
    """
    This Module divdes the temporal dimension into buckets and only samples one element from each bucket.
    Implemented by torch.nn.Multihead, which is native self-attention, was faster
    """

    def __init__(self, d_model, num_heads, bucket_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
        )
        self.bucket_size = bucket_size

    @staticmethod
    def split_list_into_groups(lst, group_size):
        return [lst[i : i + group_size] for i in range(0, len(lst), group_size)]

    @staticmethod
    def modify_key_length(key_length, sampled_index):
        """
        @param key_length: [b]
        @param sampled_index: [L]
        """
        sampled_index = rearrange(sampled_index, "l -> 1 l")
        key_length = rearrange(key_length, "b -> b 1")

        ret = sampled_index < key_length
        ret = ret.int()
        ret = torch.sum(ret, dim=1, keepdim=False)
        return ret

    def sample_index(self, Lk, device):
        groups = self.split_list_into_groups(list(range(Lk)), self.bucket_size)
        sampled_index = [random.choice(group) for group in groups]
        sampled_index = sorted(sampled_index)
        sampled_index = torch.tensor(sampled_index, dtype=torch.int64, device=device)
        return sampled_index

    @staticmethod
    def _make_key_padding_mask(t_length: torch.Tensor, temporal_dim):
        """
        @param t_length: [B]
        @param temporal_dim: int, the max length of the temporal dimension
        """
        B = t_length.size(dim=0)
        mask = torch.range(0, temporal_dim - 1, device=t_length.device)
        mask = rearrange(mask, "t -> 1 t")
        t_length = rearrange(t_length, "b -> b 1")

        mask = mask >= t_length
        return mask

    def forward(self, q, k, v, key_length=None):
        # [t n c]
        Lk = k.shape[0]
        sampled_index = self.sample_index(Lk, q.device)
        modified_key_length = self.modify_key_length(key_length, sampled_index)

        k = k[sampled_index, :, :]
        v = v[sampled_index, :, :]

        if key_length is not None:
            mask = self._make_key_padding_mask(modified_key_length, k.size(dim=0))
        else:
            mask = None

        return self.attn(q, k, v, key_padding_mask=mask)


if __name__ == "__main__":
    attn = BucketRandomAttention(
        d_model=256,
        num_heads=8,
        bucket_size=4,
    )
    qkv = torch.rand(20, 2, 256)
    result, _ = attn(qkv, qkv, qkv, key_length=torch.tensor([10, 20]))
    print(result.shape)
