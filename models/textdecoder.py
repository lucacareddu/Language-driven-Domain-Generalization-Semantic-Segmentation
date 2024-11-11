import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

from typing import Dict, List, Optional, Tuple, Union

from models.denseclip import DenseCLIPContextDecoder


class TokenDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))

        self.transform = nn.Linear(self.embed_dims, self.embed_dims)
        nn.init.kaiming_uniform_(self.transform.weight, a=math.sqrt(5))

    def forward(
        self, feats: Tensor, tokens: Tensor, batch_first=True
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2)
        delta_feat = self.forward_delta_feat(
            feats,
            tokens
        )
        feats = feats + delta_feat
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor) -> Tensor:
        feats = self.transform(feats)
        attn = torch.einsum("kbc,mc->kbm", feats, tokens)
        if True:
            attn = attn * (self.embed_dims**-0.5)
            attn = F.softmax(attn, dim=-1)
        delta_f = torch.einsum(
            "kbm,mc->kbc",
            attn,
            self.mlp_token2feat(tokens),
        )
        delta_f = self.mlp_delta_f(delta_f)
        return delta_f


class TokenDecoder(nn.Module):
    def __init__(
            self,
            layers: int = 6,
            token_length: int = 19,
            embed_dims: int = 256,
            query_dims: int = 256,
            gamma_init: float = 1e-4,
        ) -> None:
            super().__init__()

            self.learnable_tokens = nn.Parameter(torch.empty([token_length, embed_dims]))        
            nn.init.trunc_normal_(self.learnable_tokens, std=.02)

            self.layers = nn.ModuleList([TokenDecoderLayer() for _ in range(layers)])

            self.out_proj = nn.Linear(embed_dims, query_dims)
            self.gamma = nn.Parameter(torch.ones(embed_dims) * gamma_init)
    
    def forward(self, feats: Tensor) -> Tuple[Tensor, Tensor]:
        x = feats
        for layer in self.layers:
            x = layer(x, self.learnable_tokens)
        x = self.out_proj(x)
        feats = feats + self.gamma * x   
        if torch.rand(1) < 1e-2:
            print(self.gamma)
        return feats, self.learnable_tokens


class TextDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.context_decoder = DenseCLIPContextDecoder(transformer_layers=6, visual_dim=512)
        # self.token_decoder = TokenDecoder(layers=3)

        # self.layers = nn.ModuleList([nn.MultiheadAttention(embed_dim=256, num_heads=4) for _ in range(3)])

    def forward(self, text: Tensor, visual: Tensor) -> Tuple[Tensor, Tensor]:
        contextualized_text = self.context_decoder(text=text, visual=visual)
        # text_emb, text_queries = self.token_decoder(feats=contextualized_text)

        # for layer in self.layers:
        #     text_queries, _ = layer(text_queries, queries, queries)

        return contextualized_text#text_emb, text_queries
    