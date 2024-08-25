from .clip import clip
from torch import nn
import torch
from einops import rearrange, repeat
from .clip.model import VisionTransformer
from .clip.model import CLIP, LayerNorm
from dataclasses import dataclass
from collections import namedtuple


@dataclass
class ViFiClipViTArch:

    input_resolution: int = 224
    patch_size: int = 16
    width: int = 768
    layers: int = 12
    heads: int = 12
    output_dim: int = 512

    @property
    def design_detials(self):
        return dict(
            trainer='ViFi_CLIP',
            vision_depth=0,
            vision_ctx=0,
        )


def vit_b_16():
    return ViFiClipViTArch()
    # ViFiClipViTArch(224, 16, 768, 12, 12, 512)


class PromptLearner(nn.Module):

    def __init__(
            self,
            embed_dim,
            num_prompts,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_prompts = num_prompts
        scale = embed_dim ** -0.5
        self.prompt_embed = nn.Parameter(
            scale * torch.randn(num_prompts, embed_dim, dtype=torch.float32), requires_grad=True)
        self.post_ln = LayerNorm(embed_dim)

    def forward(self, x):
        # [c+hw n c]
        N = x.shape[1]
        prompt = repeat(self.prompt_embed, 'p c -> p n c', n=N)
        return torch.cat([prompt, x], dim=0)

    def extract_prompt(self, x):
        # [p+c+hw, n, c]
        return self.post_ln(x[:self.num_prompts, :, :])


class FeedForwardFusion(nn.Module):

    def __init__(self, d_model, neck_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.proj1 = nn.Linear(d_model, neck_dim)
        self.proj2 = nn.Linear(neck_dim, d_model)

    def forward(self, x):
        # [n, s, c]
        x = self.proj1(x)
        x = torch.mean(x, dim=1, keepdim=False)
        return self.proj2(x)


class ClipEncoder(nn.Module):

    def __init__(
            self,
            vifi_clip_ckpt: str,
            num_prompts,
            output_dim,
            num_local_prompts=0,
            local_prompts_depth=0,
            fusion_expand_factor=2,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_prompts = num_prompts
        arch = vit_b_16()
        design_details = arch.design_detials
        design_details['vision_depth'] = num_local_prompts
        design_details['vision_ctx'] = local_prompts_depth
        self.visual: VisionTransformer = VisionTransformer(
            arch.input_resolution, arch.patch_size, arch.width, arch.layers, arch.heads, arch.output_dim, design_details
        )
        self._load_vit(vifi_clip_ckpt)
        self._freeze_vit()

        # global prompt
        d_model = self.visual.ln_post.normalized_shape[0]
        self.prompt = PromptLearner(d_model, num_prompts)

        # fusiong layer
        self.fusion = FeedForwardFusion(
            d_model, d_model * fusion_expand_factor)

        # fineal proj
        self.out_proj = nn.Linear(d_model, output_dim, bias=False)

    def _load_vit(self, ckpt):
        import re
        from collections import OrderedDict

        ckpt = torch.load(ckpt, map_location='cpu')
        p: OrderedDict = ckpt['model']

        pattern = re.compile(r'^module.image_encoder.')
        replacement = ''
        new_dict = OrderedDict()

        for key, value in p.items():
            if pattern.match(key):
                new_key = pattern.sub(replacement, key)
                new_dict[new_key] = value
        result = self.visual.load_state_dict(new_dict, strict=False)
        for key in result.missing_keys:
            print("Warning: The following keys are missing in the state_dict:")
            print(key)
        del ckpt

    def _freeze_vit(self):
        for p in self.visual.parameters():
            p.requires_grad = False

    ClipEncoderOutput = namedtuple('ClipEncoderOutput', ['out', 't_length'])

    def forward(self, x, t_length):
        # x: [n c s h w]
        N = x.shape[0]
        x = rearrange(x, 'n c s h w -> (n s) c h w')

        x = self._vit_stem_forward(x)
        # after positional encoding add prompt
        x = rearrange(x, 'n t c -> t n c')
        x = self.prompt(x)
        x = rearrange(x, 't n c -> n t c')
        x = self._vit_block_forward(x)
        cls, prompts, local_prompt = self._post_forward(x)

        if local_prompt is None:
            x = self.fusion(torch.cat([cls.unsqueeze(1), prompts], dim=1))
        else:
            x = self.fusion(
                torch.cat([cls.unsqueeze(1), prompts, local_prompt], dim=1))

        x = self.out_proj(x)

        return self.ClipEncoderOutput(
            out=rearrange(x, '(n t) c -> n c t', n=N),
            t_length=t_length
        )

    def _vit_stem_forward(self, x):
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        return x

    def _vit_block_forward(self, x):
        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.visual.VPT_shallow:
            visual_ctx = self.visual.VPT.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.visual.prompt_till_layer_visual == 0
        # Normal code as before
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x

    def _post_forward(self, x):
        x = rearrange(x, 'n t c -> t n c')

        # now cls is behind the prompts
        cls = self.visual.ln_post(x[self.num_prompts, :, :])
        # if self.visual.proj is not None:
        #     cls = cls @ self.visual.proj

        # get prompts
        prompts = self.prompt.extract_prompt(x)
        prompts = rearrange(prompts, 't n c -> n t c')

        local_prompt = None
        if self.visual.VPT_shallow:
            local_prompt = x[-self.visual.VPT.shape[0]:, :, :]
            local_prompt = rearrange(local_prompt, 't n c -> n t c')

        return cls, prompts, local_prompt

    def __forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x
