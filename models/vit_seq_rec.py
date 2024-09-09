# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import numpy as np

from os.path import join as pjoin

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

from timm.models._registry import register_model
from timm.models import VisionTransformer
from timm.models._helpers import load_state_dict
from timm.models.helpers import load_pretrained

from pytorch_metric_learning.losses import SupConLoss

# __all__ = ['VITSEQREC']

# def _cfg(url='', **kwargs):
#     return {
#         'url': url,
#         'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
#         'crop_pct': .9, 'interpolation': 'bicubic',
#         'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
#         'classifier': 'head',
#         **kwargs
#     }
#
# default_cfgs = {
#     'vit_base_seq_rec_patch16': _cfg(),
# }


class DecoderConfig:
    def __init__(self,
                 num_labels,
                 hidden_size,
                 label_level,
                 dropout_rate,
                 decoder_num_layers,
                 mlp_dim,
                 num_heads,
                 attention_dropout_rate,
                 ):
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.label_level = label_level
        self.dropout_rate = dropout_rate
        self.decoder_num_layers = decoder_num_layers
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.attention_dropout_rate = attention_dropout_rate

def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}

def get_max_attention_index(attention_scores):
    """
    Gets the index of the class with the maximum attention score for each token.

    Args:
      attention_scores: A tensor of attention scores with size [bs, num_head, num_class, num_token].

    Returns:
      max_attention_index: A tensor of size [bs, num_class] containing the index of the class
                           with the maximum attention score for each sample in the batch.
    """
    # Compute the mean attention scores across the num_head dimension
    mean_attention_scores = attention_scores.mean(dim=1)  # Shape: [bs, num_class, num_token]

    return mean_attention_scores

class AttentionProbs(nn.Module):
    def __init__(self, config):
        super(AttentionProbs, self).__init__()
        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, k_v=None, attn_mask=None):

        is_cross_attention = k_v is not None

        mixed_query_layer = self.query(hidden_states)
        if is_cross_attention:
            mixed_key_layer = self.key(k_v.cuda())
            mixed_value_layer = self.value(k_v)
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attn_mask is not None:
            # print('before attn_mask:', attn_mask.size())
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1)
            # print('attn_mask:', attn_mask.size())
            attention_scores = attention_scores.masked_fill_(attn_mask.bool(), -np.inf)

        attention_probs = self.softmax(attention_scores)

        return attention_probs

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.attention_dropout_rate)
        self.proj_dropout = Dropout(config.attention_dropout_rate)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, k_v=None, attn_mask=None):

        is_cross_attention = k_v is not None

        mixed_query_layer = self.query(hidden_states)
        if is_cross_attention:
            mixed_key_layer = self.key(k_v.cuda())
            # mixed_key_layer = k_v.cuda()    # FIXME
            mixed_value_layer = self.value(k_v)
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attn_mask is not None:
            # print('before attn_mask:', attn_mask.size())
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1)
            # print('attn_mask:', attn_mask.size())
            attention_scores = attention_scores.masked_fill_(attn_mask.bool(), -np.inf)

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)      # FIXME !!! value_layer !!!
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.mlp_dim)
        self.fc2 = Linear(config.mlp_dim, config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super(DecoderBlock, self).__init__()
        # self.self_attn = Attention(config, vis)
        # self.self_attn_layer_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.cross_attn = Attention(config)
        self.cross_attn_layer_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, x, encoder_y, self_attn_mask):
        # residual = x
        # x = self.self_attn_layer_norm(x)
        # x = self.self_attn(x, attn_mask=self_attn_mask)
        # x = x + residual

        residual = x
        x = self.cross_attn_layer_norm(x)
        x = self.cross_attn(x, encoder_y)
        x = x + residual

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + residual
        return x

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.layer = nn.ModuleList()
        # self.decoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.decoder_num_layers):
            layer = DecoderBlock(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x, encoder_y, seq_mask=None):
        for layer_block in self.layer:
            x = layer_block(x, encoder_y, seq_mask)
        # x = self.decoder_norm(x)
        return x

class LabelEmbeddings(nn.Module):
    """Construct the embeddings from label, position embeddings.
    """
    def __init__(self, config):
        super(LabelEmbeddings, self).__init__()
        self.tgt_vocab_size = config.num_labels
        self.emb_size = config.hidden_size
        self.embedding = nn.Embedding(self.tgt_vocab_size, self.emb_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, config.label_level, config.hidden_size))
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.dropout = Dropout(config.dropout_rate)

    def forward(self, x):
        # print('x size:',x.size())
        # print('self.position_embeddings size',self.position_embeddings.size())
        embeddings = self.embedding(x)
        # print('after embed x size:',x.size())
        embeddings += self.position_embeddings[:,0:embeddings.size(1),:]
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class ViTSEQREC(nn.Module):
    def __init__(self,  img_size=(224, 224),
                        patch_size=16,
                        in_chans=3,
                        num_classes=283,
                        embed_dim=768,
                        depth=12,
                        num_heads=12,
                        mlp_ratio=4.,
                        drop_rate=0.,
                        drop_path_rate=0.2,
                        pretrained_path=None,):
        super().__init__()

        self.encoder = VisionTransformer(img_size=img_size,
                                        patch_size=patch_size,
                                        in_chans=in_chans,
                                        num_classes=num_classes,
                                        embed_dim=embed_dim,
                                        depth=depth,
                                        num_heads=num_heads,
                                        mlp_ratio=mlp_ratio,
                                        drop_rate=drop_rate,
                                        drop_path_rate=drop_path_rate
                                        )

        # SEQ modules
        self.decoder_config = DecoderConfig(
            num_labels=num_classes,
            hidden_size=embed_dim,
            label_level=4,
            dropout_rate=drop_rate,
            decoder_num_layers=1,
            mlp_dim=3072,
            num_heads=num_heads,
            attention_dropout_rate=drop_rate,
        )
        self.species_kv = nn.Parameter(torch.randn(num_classes, embed_dim))
        # TODO: positional encoding for species kv
        self.pos_embed_kv = nn.Parameter(torch.randn(num_classes, embed_dim) * .02) # same as vision_transformer

        self.decoder = Decoder(self.decoder_config)

        # Loss
        self.criterion = SupConLoss(temperature=0.10)

        # Do not comment !!!
        if pretrained_path is not None:
            # Load pretrained model
            state_dict = load_state_dict(pretrained_path)
            incompatible_keys = self.encoder.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained model: {incompatible_keys}")


    def forward(self, x, targets=None):
        encoded = self.encoder.forward_features(x)
        # Repeat the parameter tensor along the batch dimension
        species_kv_original = self.species_kv + self.pos_embed_kv
        species_kv = species_kv_original.repeat(x.shape[0], 1, 1)

        # Stage 0: Family prediction
        vis_tokens_f = self.decoder(encoded, species_kv)    # visual tokens contextualized with family labels
        # Force latent keys to encode distinctive information about each class by predicting the correct family
        # selected_lat = species_kv_original[targets[0,:]].unsqueeze(1)
        # out_lat_f = self.encoder.forward_head(selected_lat)
        scores_f = self.encoder.forward_head(vis_tokens_f)

        # Stage 1: Genus prediction
        vis_tokens_g = self.decoder(vis_tokens_f, species_kv)    # visual tokens contextualized with genus labels
        # selected_lat = species_kv_original[targets[1,:]].unsqueeze(1)
        # out_lat_g = self.encoder.forward_head(selected_lat)
        scores_g = self.encoder.forward_head(vis_tokens_g)

        # Stage 2: Species prediction
        vis_tokens_s = self.decoder(vis_tokens_g, species_kv)    # visual tokens contextualized with species labels
        # selected_lat = species_kv_original[targets[2,:]].unsqueeze(1)
        # out_lat_s = self.encoder.forward_head(selected_lat)
        scores_s = self.encoder.forward_head(vis_tokens_s)

        scores = torch.cat((scores_f.unsqueeze(1), scores_g.unsqueeze(1), scores_s.unsqueeze(1)), dim=1) # , out_lat_f.unsqueeze(1), out_lat_g.unsqueeze(1), out_lat_s.unsqueeze(1)

        # Loss
        # contrast_loss = con_loss(vis_tokens_f[:, 0], targets[0,:].view(-1))
        # contrast_loss_latent = self.criterion(F.normalize(species_kv_original[targets[0,:]], p=2, dim=1), targets[0,:].view(-1))      \
        #                     + self.criterion(F.normalize(species_kv_original[targets[1,:]], p=2, dim=1), targets[1,:].view(-1))    \
        #                     + self.criterion(F.normalize(species_kv_original[targets[2,:]], p=2, dim=1), targets[2,:].view(-1))
        #
        # contrast_loss_cls = self.criterion(F.normalize(vis_tokens_f[:, 0], p=2, dim=1), targets[0,:].view(-1))      \
        #                 + self.criterion(F.normalize(vis_tokens_g[:, 0], p=2, dim=1), targets[1,:].view(-1))    \
        #                 + self.criterion(F.normalize(vis_tokens_s[:, 0], p=2, dim=1), targets[2,:].view(-1))
        #
        # contrast_loss = contrast_loss_latent + contrast_loss_cls

        return scores #, contrast_loss

# @register_model
# def vit_base_seq_rec_patch16(pretrained=False, **kwargs):
#     model = ViTSEQREC(img_size=kwargs.get('input_size', [224, 224]),
#                         patch_size=16,
#                         in_chans=3,
#                         embed_dim=768,
#                         depth=12,
#                         num_heads=12,
#                         mlp_ratio=4.,
#                         drop_rate=0.,
#                         drop_path_rate=0.2, **kwargs)
#     model.default_cfg = default_cfgs['MetaFG_0']
#     if pretrained:
#         load_pretrained(
#             model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
#     return model
