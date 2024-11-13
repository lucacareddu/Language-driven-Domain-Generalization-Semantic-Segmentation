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
    def __init__(self, visual_dim, text_dim, return_keys, out_dim=256):
        super().__init__()   
        
        self.text_proj = nn.Parameter(torch.randn(text_dim, text_dim)) 
        
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.visual_norm.apply(self._init_weights)

        scale = visual_dim ** -0.5
        self.visual_proj = nn.Parameter(torch.randn(visual_dim, text_dim) * scale)        

        self.context_decoder = DenseCLIPContextDecoder(transformer_width=256,
                                                        transformer_heads=4,
                                                        transformer_layers=3,
                                                        visual_dim=text_dim,
                                                        dropout=0.1)

        if return_keys:
            self.keys_proj = nn.Linear(text_dim, out_dim)
            self.keys_proj.apply(self._init_weights)
        
        self.return_keys = return_keys

        self.queries_proj = nn.Linear(text_dim, out_dim)
        self.queries_proj.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.LayerNorm):            
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, text: Tensor, visual: Tensor):
        text_emb = text @ self.text_proj
        text_emb = text_emb.expand(visual.shape[0],-1,-1)

        visual_emb = self.visual_norm(visual)
        visual_emb = visual @ self.visual_proj

        contextualized_text = self.context_decoder(text=text_emb, visual=visual_emb)
        
        keys = self.keys_proj(contextualized_text) if self.return_keys else None          
        queries = self.queries_proj(contextualized_text.mean(0))

        return keys, queries
    