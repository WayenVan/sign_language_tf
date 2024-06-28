import torch
from PIL import Image
import open_clip
from open_clip.transformer import VisionTransformer, _expand_token, Transformer
from open_clip import CLIP

# print(open_clip.list_pretrained())
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

image = preprocess(Image.open("resources/flow.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])
# image = torch.nn.functional.interpolate(image, (192, 192))
    



def vit_stem_forward(vit: VisionTransformer, x):
        x = vit.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(vit.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + vit.positional_embedding.to(x.dtype)

        x = vit.patch_dropout(x)
        x = vit.ln_pre(x)

        return x

def transformer_block_forward(vit: VisionTransformer, x):
    x = x.permute(1, 0, 2)  # NLD -> LND
    t: Transformer = vit.transformer
    for i in range(len(t.resblocks)+1):
        if t.grad_checkpointing and not torch.jit.is_scripting():
            raise NotImplementedError()
        else:

            if i < len(t.resblocks):
                x = t.resblocks[i](x, attn_mask=None)
    x = x.permute(1, 0, 2)  # LND -> NLD
    return x 

def vit_post_forward(vit: VisionTransformer, x):
    if vit.attn_pool is not None:
        if vit.attn_pool_contrastive is not None:
            # This is untested, WIP pooling that should match paper
            x = vit.ln_post(x)  # TBD LN first or separate one after each pool?
            tokens = vit.attn_pool(x)
            if vit.attn_pool_type == 'parallel':
                pooled = vit.attn_pool_contrastive(x)
            else:
                assert vit.attn_pool_type == 'cascade'
                pooled = vit.attn_pool_contrastive(tokens)
        else:
            # this is the original OpenCLIP CoCa setup, does not match paper
            x = vit.attn_pool(x)
            x = vit.ln_post(x)
            pooled, tokens = vit._global_pool(x)
    elif vit.final_ln_after_pool:
        pooled, tokens = vit._global_pool(x)
        pooled = vit.ln_post(pooled)
    else:
        x = vit.ln_post(x)
        pooled, tokens = vit._global_pool(x)

    if vit.proj is not None:
        pooled = pooled @ vit.proj

    return pooled, tokens

    
with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    
    vit = model.visual
    x = vit_stem_forward(vit, image)
    x = transformer_block_forward(vit, x)
    image_features2, _ = vit_post_forward(vit, x)
    
    a = (image_features == image_features2)
    print(a)