from torch import nn
import torch
from typing import Optional, Union, Tuple

from xformers.components.attention.core import (
    scaled_query_key_softmax,
    AttentionMask,
    _has_cpp_library,
    _apply_dropout,
    bmm,
)

if _has_cpp_library:
    from xformers.components.attention._sputnik_sparse import SparseCS
from xformers.components.attention import Attention
from xformers.components.multi_head_dispatch import (
    InputProjection,
    _split_heads,
    _fold_heads,
    InputProjectionConfig,
    RotaryEmbedding,
    constant_,
)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    att_mask: Optional[Union[AttentionMask, "SparseCS", torch.Tensor]],
    dropout: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    if isinstance(att_mask, SparseCS) or (att_mask is not None and att_mask.is_sparse):
        raise NotImplementedError(
            "Sparse attention is not implemented for scaled dot product attention"
        )

    att = scaled_query_key_softmax(q, k, att_mask=att_mask)
    #  Optional dropout, could be part of the masking in the future

    att = _apply_dropout(att, dropout)

    # Get to the predicted values, for all heads
    # y = att @ v  # (N, S, S) x (N, S, hs) -> (N, S, hs)
    y = bmm(att, v)
    return y, att


class ScaledDotProduct(Attention):
    r"""
    Implementing the Scaled Dot-Product attention proposed in
    `Attention is all you need`_, Vaswani et al.

    .. _`Attention is all you need`: https://arxiv.org/abs/1706.03762v5
    """

    mask: Optional[AttentionMask]

    def __init__(
        self,
        dropout: float = 0.0,
        causal: bool = False,
        seq_len: Optional[int] = None,
        to_seq_len: Optional[int] = None,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.attn_drop = nn.Dropout(dropout, inplace=False)
        self.causal = causal
        self.seq_len = seq_len

        if causal and seq_len is not None:
            self.mask = AttentionMask.make_causal(seq_len, to_seq_len)
        else:
            self.mask = None

        # Properties specific to this attention mechanism
        self.supports_attention_mask = True
        self.supports_key_padding_mask = False

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[Union[AttentionMask, torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        att_mask    A 2D or 3D mask which ignores attention at certain positions.

                    - If the mask is boolean, a value of True will keep the value,
                        while a value of False will mask the value.

                        Key padding masks (dimension: batch x sequence length) and attention masks
                        (dimension: sequence length x sequence length OR batch x sequence length x sequence length)
                        can be combined and passed in here. Method maybe_merge_masks provided in the utils can be
                        used for that merging.

                    - If the mask has the float type, then an additive mask is expected (masked values are -inf)

        """

        # Convenience, create an attention mask if a tensor was passed
        if att_mask is not None and isinstance(att_mask, torch.Tensor):
            # By default we don't know of the causality, and a check would be expensive
            att_mask = (
                AttentionMask.from_bool(att_mask)
                if att_mask.dtype == torch.bool
                else AttentionMask(att_mask, is_causal=False)
            )

        # Handle a possibly deferred causal mask handling
        mask = self.mask
        if self.causal and self.mask is None:
            mask = AttentionMask.make_causal(
                seq_len=q.shape[-2],
                to_seq_len=q.shape[-2],
                device=q.device,
                dtype=q.dtype,
            )

        # Merge the optional causal mask and the user-provided mask
        if mask is not None:
            mask = mask.to(dtype=q.dtype, device=q.device)

            att_mask = att_mask + mask if att_mask is not None else mask

        # Try to handle a case where the sequence is smaller than the mask
        if (
            att_mask is not None
            and q.shape[-2] == k.shape[-2]
            and q.shape[-2] < att_mask.shape[1]
        ):
            if isinstance(att_mask, AttentionMask):
                att_mask = att_mask.make_crop(seq_len=q.shape[-2])
            else:
                logger.error(
                    "Mismatching sparse attention mask and sequence length."
                    + " Please pad the inputs or adjust the attention mask"
                )
                raise NotImplementedError

        # Attend: (B x nh, S, hs) x (B x nh, hs, S) -> (B x nh, S, S)
        y, attn_weight = scaled_dot_product_attention(
            q=q, k=k, v=v, att_mask=att_mask, dropout=self.attn_drop
        )
        return y, attn_weight


class MultiHeadDispatch(nn.Module):
    """
    A multi-head masked self-attention dispatch mechanism, with a projection at the end,
    following the architecture proposed in `Attention is all you need`_, Vaswani et al.

    The actual attention mechanism can vary, as well as the projections.
    This can be used to wrap the proposed attention mechanisms and make them multi-head aware,
    but it is optional.

    Args:
        dim_model: The model/embedding dimension
        num_heads: The number of heads being used
        attention: The attention mechanism (needs to be registered to the xformers library)
        bias: Whether to use bias for the projections : (Q, K, V, Output)
        residual_dropout: Amount of dropout on the residual path
        use_separate_proj_weight: Use different weights for the Q, K, V projections
        dim_key: Optionally use a different dimension for the key
        dim_value:  Optionally use a different dimension for the value
        in_proj_container: Optionally provide the input projection module
        use_rotary_embeddings: Use rotary embeddings
        out_proj: Optionally provide the output projection module


    .. _`Attention is all you need`: https://arxiv.org/abs/1706.03762v5
    """

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        attention: Attention,
        bias: Tuple[bool, bool, bool, bool] = (True, True, True, True),
        residual_dropout: float = 0.0,
        use_separate_proj_weight: bool = True,
        dim_key: Optional[int] = None,
        dim_value: Optional[int] = None,
        in_proj_container: Optional[InputProjection] = None,
        use_rotary_embeddings: Optional[bool] = False,
        out_proj: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        super().__init__()

        if isinstance(bias, bool):
            logger.warning(
                "Single bias value provided for the MHA projections."
                + f" Assuming the same parameter ({bias}) is to be used everywhere"
            )
            bias = (bias, bias, bias, bias)

        assert (
            dim_model % num_heads == 0
        )  # static preset for now, each head works on 1/d the embeddings, could be relaxed
        assert num_heads > 0
        assert isinstance(
            attention, ScaledDotProduct
        ), "Only my ScaledDotProduct is supported for this class"

        # Popular default is that all latent dimensions are the same
        dim_key, dim_value = map(lambda x: x if x else dim_model, (dim_key, dim_value))

        self.num_heads = num_heads
        self.dim_key_head = dim_key // num_heads
        self.dim_value_head = dim_value // num_heads
        self.dim_model = dim_model
        self.attention = attention

        # key, query, value projections for all heads
        # critical options are
        # - are we sharing weights ?
        # - are we adding biases ?
        if attention.requires_input_projection:
            self.in_proj_container = (
                in_proj_container
                if in_proj_container is not None
                else InputProjection(
                    query_proj_params=InputProjectionConfig(
                        dim_model, dim_key, bias=bias[0]
                    ),
                    key_proj_params=InputProjectionConfig(
                        dim_model, dim_key, bias=bias[1]
                    ),
                    value_proj_params=InputProjectionConfig(
                        dim_model, dim_value, bias=bias[2]
                    ),
                    use_separate_proj_weight=use_separate_proj_weight,
                )
            )

        # Optional rotary embeddings
        self.rotary_embeddings = (
            RotaryEmbedding(self.dim_key_head) if use_rotary_embeddings else None
        )

        # Regularization
        self.resid_drop = nn.Dropout(residual_dropout, inplace=False)

        # Output projection
        self.proj = (
            out_proj if out_proj else nn.Linear(dim_model, dim_model, bias=bias[3])
        )
        if isinstance(self.proj, nn.Linear) and self.proj.bias is not None:
            constant_(self.proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        att_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Expected input dimensions are [batch size, sequence length, embed dim]
        Output dimensions are [batch size, sequence length, embed dim]
        """

        if key is None:
            key = query
        if value is None:
            value = query

        if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
            max_batch = max((query.shape[0], key.shape[0], value.shape[0]))
            query, key, value = map(
                lambda x: x.expand(max_batch, -1, -1), [query, key, value]
            )

        B, S_Q, _ = query.size()  # Batch x Sequence x Embedding (latent)
        _, S_K, _ = key.size()  # K, Q's sequence length could differ

        # Catch different query and key length but a causal attention
        if S_Q != S_K:
            assert not self.attention.requires_same_k_q_dimensions, "This attention mechanism requires query and key to have the same sequence (context) lengths"

            if hasattr(self.attention, "causal"):
                assert not self.attention.causal, (
                    "Causal attention is not supported when key and query have different sequence lengths.\n"
                    + "In that case causality is ill-determined. Please pad your sequences accordingly"
                )

        kw_mask_args = {}
        if att_mask is not None:
            assert (
                self.attention.supports_attention_mask
            ), "This attention does not support attention masks"
            kw_mask_args["att_mask"] = att_mask

        if key_padding_mask is not None:
            assert (
                self.attention.supports_key_padding_mask
            ), "This attention does not support key padding masks"
            kw_mask_args["key_padding_mask"] = key_padding_mask

        if self.attention.requires_skip_multi_head:
            return self.attention(query, key, value, **kw_mask_args)

        # Calculate query, key, values for all heads in batch
        if self.attention.requires_input_projection:
            q, k, v = self.in_proj_container(query=query, key=key, value=value)
        else:
            k, q, v = key, query, value

        # Check the dimensions properly
        def check(t, name):
            assert (
                t.shape[2] % self.num_heads == 0
            ), f"the {name} embeddings need to be divisible by the number of heads"

        check(q, "projected query")
        check(v, "projected value")
        check(k, "projected key")

        # Optional: rotary embedding, add relative positioning information
        if self.rotary_embeddings:
            # rotary requires the head dimension
            q = _split_heads(q, B, S_Q, self.num_heads, self.dim_key_head)
            k = _split_heads(k, B, S_K, self.num_heads, self.dim_key_head)
            v = _split_heads(v, B, S_K, self.num_heads, self.dim_value_head)

            q, k = self.rotary_embeddings(q=q, k=k)

            if not self.attention.requires_head_dimension:
                q, k, v = q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1)

        else:
            # Reshape k/q/v to either expose the heads, or fold the head dimension into the batch
            reshape_fn = (
                _split_heads if self.attention.requires_head_dimension else _fold_heads
            )

            q = reshape_fn(q, B, S_Q, self.num_heads, self.dim_key_head)
            k = reshape_fn(k, B, S_K, self.num_heads, self.dim_key_head)
            v = reshape_fn(v, B, S_K, self.num_heads, self.dim_value_head)

        # Self-attend
        y, attn_weight = self.attention(q, k, v, **kw_mask_args)

        # Re-assemble all head outputs side by side
        y = (
            y.view(B, self.num_heads, S_Q, self.dim_value_head)
            .transpose(1, 2)
            .flatten(start_dim=2, end_dim=3)
        )
        attn_weight = attn_weight.view(B, self.num_heads, S_Q, S_K)

        # Output projection, dropout and good to go
        y = self.resid_drop(self.proj(y))

        # Return the same sequence size as the input
        return y, attn_weight


if __name__ == "__main__":
    attn = MultiHeadDispatch(512, 8, ScaledDotProduct()).to("cuda:1")
    keys = torch.rand(2, 7, 512).to("cuda:1")
    querys = torch.rand(2, 16, 512).to("cuda:1")
    output = attn(querys, keys, keys)
    print(output.shape)

