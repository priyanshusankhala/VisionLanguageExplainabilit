import torch
import torch.nn.functional as F
from PIL import Image

from vision_language_models import ALBEF
from vision_language_models.ALBEF.dataset import clean_caption, test_transform


def load_model(device='cpu'):
    config = dict(
        checkpoint_path='/home/ts/Downloads/mscoco.pth',
        # download:
        #  * Pretrain: https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF.pth
        #  * COCO: https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/mscoco.pth
        #  * Flickr: https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/flickr30k.pth
        bert_config='../vision_language_models/ALBEF/configs/config_bert.json',
        cache_dir=None,  # for hugging-face, default is .cache/

        image_res=384,
        queue_size=65536,
        momentum=0.995,
        vision_width=768,
        embed_dim=256,
        temp=0.07,
        distill=False,
        checkpoint_vit=False,
        checkpoint_bert=False,
    )
    model = ALBEF.from_cktp(config)
    model.to(device)
    model.eval()

    return model


def cross_attention_gradcam(model, image_input, text_input, blks, average_heads=False):
    image_embeds = model.visual_encoder(image_input)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
    output = model.text_encoder(text_input.input_ids,
                                attention_mask=text_input.attention_mask,
                                encoder_hidden_states=image_embeds,
                                encoder_attention_mask=image_atts,
                                return_dict=True)
    vl_embeddings = output.last_hidden_state[:, 0, :]
    vl_output = model.itm_head(vl_embeddings)
    loss = vl_output[:, 1].sum()

    model.zero_grad()
    loss.backward()

    mask = text_input.attention_mask.view(text_input.attention_mask.size(0), 1, -1, 1, 1)

    block_gradcam = []

    for blk in blks:
        grads = model.text_encoder.base_model.base_model.encoder.layer[blk].crossattention.self.get_attn_gradients()
        grads = grads[:, :, :, 1:].clamp(0).reshape(image_input.size(0), 12, -1, 24, 24) * mask
        grads = grads.mean(dim=2)  # mean over all text tokens

        cams = model.text_encoder.base_model.base_model.encoder.layer[blk].crossattention.self.get_attention_map()
        cams = cams[:, :, :, 1:].reshape(image_input.size(0), 12, -1, 24, 24) * mask
        cams = cams.mean(dim=2)  # mean over all text tokens

        gradcam = cams * grads
        gradcam = gradcam[0]

        if average_heads:
            gradcam = gradcam.mean(0)  # mean over all heads
        block_gradcam.append(gradcam)

    block_gradcam = torch.stack(block_gradcam)
    return block_gradcam.cpu().detach().numpy()


def visual_attention_gradcam(model, image_input, text_input, block_id=11, average_heads=True):
    model.visual_encoder.blocks[block_id].attn.register_hook = True

    image_embeds = model.visual_encoder(image_input)
    image_feat = F.normalize(model.vision_proj(image_embeds[:, 0, :]), dim=-1)
    text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
                                     return_dict=True, mode='text')
    text_embeds = text_output.last_hidden_state
    text_feat = F.normalize(model.text_proj(text_embeds[:, 0, :]), dim=-1)
    sim = image_feat @ text_feat.t() / model.temp
    loss = sim.diag().sum()

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        grad = model.visual_encoder.blocks[block_id].attn.get_attn_gradients().detach()
        cam = model.visual_encoder.blocks[block_id].attn.get_attention_map().detach()
        cam = cam[:, :, 0, 1:].reshape(image_input.size(0), -1, 24, 24)
        grad = grad[:, :, 0, 1:].reshape(image_input.size(0), -1, 24, 24).clamp(0)
        gradcam = (cam * grad)
        gradcam = gradcam[0]
        if average_heads:
            gradcam = gradcam.mean(0)  # mean over all heads

    return gradcam.cpu().detach().numpy()
