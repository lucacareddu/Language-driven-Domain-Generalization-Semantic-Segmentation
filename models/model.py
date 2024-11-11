import torch
from torch import nn

from transformers import CLIPModel, CLIPProcessor
from models.reins import *
from models.textdecoder import TextDecoder
from models.neck import MultiLevelNeck
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DGSSModel(nn.Module):
    def __init__(self, clip_name, ignore_index, text_prompts=None, reins=True, text_decoder=True, nqueries=100, nclasses=19):
        super().__init__()

        self.encoders = CLIPModel.from_pretrained(clip_name)
        
        if text_prompts is not None:
            tokenizer = CLIPProcessor.from_pretrained(clip_name)
            text_tokenized = tokenizer(text=text_prompts, return_tensors="pt", padding=True)
            self.text_ids = text_tokenized["input_ids"].cuda()
            self.text_att = text_tokenized["attention_mask"].cuda()

            self.visclip_proj = nn.Linear(768,512)
            self.visclip_proj.apply(self.__init_weights)

            self.textclip_proj = nn.Linear(512,512)
            self.textclip_proj.apply(self.__init_weights)

            text_decoder = True
        else:
            text_decoder = False
        
        self.neck = MultiLevelNeck(in_channels=[768] * 3, out_channels=768)

        configuration = Mask2FormerConfig(num_labels=nclasses, ignore_value=ignore_index, num_queries=(nclasses if text_decoder else nqueries))
        self.vision_decoder = Mask2FormerForUniversalSegmentation(configuration)
        self.vision_decoder_processor = Mask2FormerImageProcessor()
        
        # self.vision_decoder = SegformerDecoder(input_dim=256, hidden_size=256, num_hidden_states=10)     

        if reins:     
            reins_config=dict(
                token_length=nqueries,
                embed_dims=256,
                num_layers=10,
                patch_size=16,
                link_token_to_query=True,
                lora_dim=nclasses,
            )
            self.encoders.vision_model.encoder.reins = LoRAReins(**reins_config) 
            
            if hasattr(self.vision_decoder.model.transformer_module, 'queries_features'):
                del self.vision_decoder.model.transformer_module.queries_features    

        if text_decoder:
            self.text_decoder = TextDecoder()

            self.text_proj = nn.Linear(512,256)
            self.text_proj.apply(self.__init_weights)

            self.queries_proj = nn.Linear(512,256)
            self.queries_proj.apply(self.__init_weights)

            self.vision_decoder.model.pixel_level_module.decoder.encoder.crss_att = nn.ModuleList([nn.MultiheadAttention(embed_dim=256, num_heads=configuration.num_attention_heads, batch_first=True) for _ in range(configuration.encoder_layers)])
            self.vision_decoder.model.pixel_level_module.decoder.encoder.crss_att.apply(self.__init_weights)

            self.vision_decoder.model.pixel_level_module.decoder.encoder.context_text_pos = nn.Embedding(nclasses, 256)
            self.vision_decoder.model.pixel_level_module.decoder.encoder.context_text_pos.apply(self.__init_weights)

            if hasattr(self.vision_decoder.model.transformer_module, 'queries_features'):
                del self.vision_decoder.model.transformer_module.queries_features
        
        self.is_reins = reins
        self.is_text_decoder = text_decoder


    def __init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    
    def train(self, mode: bool = True):
        super().train(mode)

        if mode and self.is_reins:
            set_requires_grad(self.encoders, ["reins"])
            set_train(self.encoders, ["reins"])
        elif mode and not self.is_reins:
            set_requires_grad(self.encoders, ["vision_model"])
            set_train(self.encoders, ["vision_model"])


    def forward(self, pixel_values, bin_masks, classes, return_logits=False):      
        vision_outputs = self.encoders.get_image_features(pixel_values=pixel_values, output_hidden_states=True, interpolate_pos_encoding=True) 
        vision_hidden_states = vision_outputs["outputs"]["hidden_states"]
        vision_hidden_states = (vision_hidden_states[4], vision_hidden_states[7], vision_hidden_states[10])

        keys = None
        queries = None

        if self.is_text_decoder:
            text_outputs = self.encoders.get_text_features(input_ids=self.text_ids, attention_mask=self.text_att)
            text_cls_token = text_outputs["outputs"]["pooler_output"]#["pooled_cls"]
            text_emb = self.textclip_proj(text_cls_token)

            context_text = self.text_decoder(text=text_emb.expand(vision_hidden_states[-1].shape[0],-1,-1), visual=self.visclip_proj(vision_hidden_states[-1]))
            keys = self.text_proj(context_text)            
            queries = self.queries_proj(context_text.mean(0))
        elif self.is_reins:
            _, queries = self.encoders.vision_model.encoder.reins.return_auto(None)

        if keys is not None:
            # To cross-attention layers in pixel decoder (mask2former encoder) as key-value
            self.vision_decoder.model.pixel_level_module.decoder.encoder.context_text = keys
        
        if queries is not None:
            # To cross-attention layers in transformer decoder layers (mask2former decoder) as queries
            self.vision_decoder.model.transformer_module.queries_tensor = queries

        multi_scale_feats = self.neck(vision_hidden_states)

        decoder_outputs = self.vision_decoder(pixel_values=multi_scale_feats, mask_labels=bin_masks, class_labels=classes)

        loss = decoder_outputs.loss
        
        if return_logits:
            upsampled_logits = self.vision_decoder_processor.post_process_semantic_segmentation(decoder_outputs, target_sizes=[pixel_values.shape[-2:]]*pixel_values.shape[0])
            upsampled_logits = torch.cat(upsampled_logits)
            return loss, upsampled_logits
        
        return loss
    

    def print_trainable_params(self, round_to_millions=True, decimals=2):
        self.train()

        trainable_params = {"TOTAL": sum(p.numel() for p in self.parameters() if p.requires_grad),
                            "CLIP": sum(p.numel() for n,p in self.encoders.named_parameters() if "reins" not in n and p.requires_grad),
                            "CLIP_VISION": sum(p.numel() for n,p in self.encoders.vision_model.named_parameters() if "reins" not in n and p.requires_grad),
                            "CLIP_TEXT": sum(p.numel() for n,p in self.encoders.text_model.named_parameters() if "reins" not in n and p.requires_grad),
                            "NECK": sum(p.numel() for p in self.neck.parameters() if p.requires_grad),
                            "MASK2FORMER": sum(p.numel() for p in self.vision_decoder.parameters() if p.requires_grad)}
        
        if self.is_reins:
            trainable_params.update({"REINS (VISION)": sum(p.numel() for p in self.encoders.vision_model.encoder.reins.parameters() if p.requires_grad)})
        
        if self.is_text_decoder:
            trainable_params.update({"TEXT_DECODER": sum(p.numel() for p in self.text_decoder.parameters() if p.requires_grad)})
        
        if round_to_millions:
            trainable_params = {k:round((v/1e6), decimals) for k,v in trainable_params.items()}

        print("TRAINABLE PARAMS (M):")
        [print(f"   {k}: {v}") for k,v in trainable_params.items()]
        print()


    def print_frozen_modules(self):
        self.train()

        trainable_modules = {"MODEL": self.training,
                            "CLIP": self.encoders.training,
                            "CLIP_VISION": self.encoders.vision_model.training,
                            "CLIP_TEXT": self.encoders.text_model.training,
                            "NECK": self.neck.training,
                            "MASK2FORMER": self.vision_decoder.training}
        if self.is_reins:
            trainable_modules.update({"REINS (VISION)": self.encoders.vision_model.encoder.reins.training})
        
        if self.is_text_decoder:
            trainable_modules.update({"TEXT_DECODER": self.text_decoder.training})

        print("IS FROZEN:")
        [print(f"   {k}: {not v}") for k,v in trainable_modules.items()]
        print()
