from itertools import chain

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from ..dataset import clean_caption


class ValidationRetrievalCallback(pl.Callback):
    def __init__(self,
                 k_test: int,
                 text_bs: int = 256,
                 verbose: bool = True):
        super().__init__()
        self.k_test = k_test
        self.text_bs = text_bs
        self.verbose = verbose

    def itm_eval(self, scores_i2t, scores_t2i, txt2img, img2txt):
        print(scores_i2t, scores_t2i)
        print(scores_i2t.shape, scores_t2i.shape)

        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        eval_result = {'txt_r1': tr1,
                       'txt_r5': tr5,
                       'txt_r10': tr10,
                       'txt_r_mean': tr_mean,
                       'img_r1': ir1,
                       'img_r5': ir5,
                       'img_r10': ir10,
                       'img_r_mean': ir_mean,
                       'r_mean': r_mean}
        return eval_result

    @torch.no_grad()
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("Starting evaluation")

        dataloader = trainer.val_dataloaders[0]
        print(dataloader.dataset)
        print(len(dataloader), dataloader.sampler)

        pl_module.model.eval()

        texts = list(chain.from_iterable(dataloader.dataset.labels))
        num_text = len(texts)
        text_feats = []
        text_embeds = []
        text_atts = []
        it = range(0, num_text, self.text_bs)
        it = tqdm(it, desc='Encoding Text') if self.verbose else it
        for i in it:
            text = texts[i: min(num_text, i + self.text_bs)]
            text_input = pl_module.model.tokenizer(clean_caption(text, max_words=30), padding='max_length',
                                                   truncation=True, max_length=30, return_tensors="pt").to(pl_module.device)
            text_output = pl_module.model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
                                                       mode='text')
            text_feat = text_output.last_hidden_state
            text_embed = F.normalize(pl_module.model.text_proj(text_feat[:, 0, :]))
            text_embeds.append(text_embed)
            text_feats.append(text_feat)
            text_atts.append(text_input.attention_mask)
        text_embeds = torch.cat(text_embeds, dim=0)
        text_feats = torch.cat(text_feats, dim=0)
        text_atts = torch.cat(text_atts, dim=0)
        print(text_embeds.shape, text_feats.shape, text_atts.shape)

        image_feats = []
        image_embeds = []
        it = tqdm(dataloader, desc='Encoding Images', total=len(dataloader)) if self.verbose else dataloader
        for image, _, _ in it:
            image = image.to(pl_module.device)
            image_feat = pl_module.model.visual_encoder(image)
            image_embed = pl_module.model.vision_proj(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1)

            image_feats.append(image_feat)
            image_embeds.append(image_embed)

        image_feats = torch.cat(image_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)
        print(f"{image_feats.shape=}, {image_embeds.shape=}")

        image_feats = pl_module.all_gather(image_feats)
        image_embeds = pl_module.all_gather(image_embeds)
        print(f"{image_feats.shape=}, {image_embeds.shape=}")
        image_feats = image_feats.reshape(-1, *image_feats.shape[2:])
        image_embeds = image_embeds.reshape(-1, image_embeds.shape[2])
        print(f"{image_feats.shape=}, {image_embeds.shape=}")

        sims_matrix = image_embeds @ text_embeds.t()
        score_matrix_i2t = torch.full((len(dataloader.dataset), len(texts)), -100.0).to(pl_module.device)
        print(f"{sims_matrix.shape=}, {sims_matrix.device=}")

        num_tasks = torch.distributed.get_world_size()
        rank = pl_module.global_rank

        print(f"{rank=} {num_tasks=}")
        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        it = enumerate(sims_matrix[start:end])
        it = tqdm(it, desc='Fusion Encoder i->t', total=sims_matrix.size(0)) if self.verbose else it
        for i, sims in it:
            topk_sim, topk_idx = sims.topk(k=self.k_test, dim=0)

            encoder_output = image_feats[start + i].repeat(self.k_test, 1, 1)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(pl_module.device)
            output = pl_module.model.text_encoder(encoder_embeds=text_feats[topk_idx],
                                                  attention_mask=text_atts[topk_idx],
                                                  encoder_hidden_states=encoder_output,
                                                  encoder_attention_mask=encoder_att,
                                                  return_dict=True,
                                                  mode='fusion'
                                                  )
            score = pl_module.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_i2t[start + i, topk_idx] = score

        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full((len(texts), len(dataloader.dataset)), -100.0).to(pl_module.device)

        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        it = enumerate(sims_matrix[start:end])
        it = tqdm(it, desc='Fusion Encoder t->i', total=sims_matrix.size(0) // num_tasks) if self.verbose else it
        for i, sims in it:
            topk_sim, topk_idx = sims.topk(k=self.k_test, dim=0)
            encoder_output = image_feats[topk_idx]
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(pl_module.device)
            output = pl_module.model.text_encoder(encoder_embeds=text_feats[start + i].repeat(self.k_test, 1, 1),
                                                  attention_mask=text_atts[start + i].repeat(self.k_test, 1),
                                                  encoder_hidden_states=encoder_output,
                                                  encoder_attention_mask=encoder_att,
                                                  return_dict=True,
                                                  mode='fusion'
                                                  )
            score = pl_module.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2i[start + i, topk_idx] = score

        print("Score matrix for reduce", score_matrix_i2t.shape, (score_matrix_i2t == -100).sum())
        score_matrix_i2t = trainer.strategy.reduce(score_matrix_i2t, reduce_op='sum')
        score_matrix_t2i = trainer.strategy.reduce(score_matrix_t2i, reduce_op='sum')
        print("Score matrix after reduce", score_matrix_i2t.shape, (score_matrix_i2t == -100).sum())

        if trainer.is_global_zero:
            results = self.itm_eval(scores_i2t=score_matrix_i2t.cpu().numpy(),
                                    scores_t2i=score_matrix_t2i.cpu().numpy(),
                                    img2txt=dataloader.dataset.img2txt,
                                    txt2img=dataloader.dataset.txt2img)
            print(results)
            pl_module.log_dict(results)

        pl_module.model.train()
        torch.cuda.empty_cache()
        print("Evaluation finished")
