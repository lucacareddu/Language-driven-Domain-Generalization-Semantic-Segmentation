import torch
from torch import nn

from transformers import ViTModel
from transformers import CLIPModel, CLIPProcessor
from models.neck import ViTNeck, DenseCLIPNeck
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from models.textdecoder import TextDecoder

from models.reins import set_train, set_requires_grad

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DGSSModel(nn.Module):
    def __init__(self, encoder_name, ignore_value, text_prompts=None, nclasses=19, freeze_text_encoder=True, use_text_keys=False, nqueries=100):
        super().__init__()

        self.has_text_decoder = "clip" in encoder_name and text_prompts is not None
        self.freeze_text_encoder = freeze_text_encoder if self.has_text_decoder else True

        self.encoder_name = encoder_name

        encoder_config = {"vit":"google/vit-base-patch32-224-in21k",
                          "tiny_clip":"wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
                          "clip":"openai/clip-vit-base-patch32"}[encoder_name]
        
        encoder_visual_dim = {"vit":768, "tiny_clip":256, "clip":768}[encoder_name]
        encoder_text_dim = {"vit":None, "tiny_clip":256, "clip":512}[encoder_name]    

        self.encoder = {"vit":ViTModel, "tiny_clip":CLIPModel, "clip":CLIPModel}[encoder_name].from_pretrained(encoder_config)

        self.out_indices = {"vit":[3, 5, 7, 11], "tiny_clip":[4, 7, 10], "clip":[3, 5, 7, 11]}[encoder_name][-3:]

        self.neck = ViTNeck(in_channels=[encoder_visual_dim] * 3, out_channels=encoder_visual_dim)
        # self.neck = DenseCLIPNeck(width=encoder_visual_dim)

        if encoder_name == "tiny_clip":
            vision_decoder_config = Mask2FormerConfig(num_labels=nclasses, ignore_value=ignore_value, feature_channels=[encoder_visual_dim] * 3, encoder_layers=1, decoder_layers=3, num_queries=(nclasses if self.has_text_decoder else nqueries))
        else:
            vision_decoder_config = Mask2FormerConfig(num_labels=nclasses, ignore_value=ignore_value, feature_channels=[encoder_visual_dim] * 3, num_queries=(nclasses if self.has_text_decoder else nqueries))
        
        self.vision_decoder = Mask2FormerForUniversalSegmentation(vision_decoder_config)
        self.vision_decoder_processor = Mask2FormerImageProcessor() 

        if self.has_text_decoder:
            tokenizer = CLIPProcessor.from_pretrained(encoder_config)
            text_tokenized = tokenizer(text=text_prompts, return_tensors="pt", padding=True)
            self.text_ids = text_tokenized["input_ids"].cuda()
            self.text_att = text_tokenized["attention_mask"].cuda()

            self.text_decoder = TextDecoder(visual_dim=encoder_visual_dim, text_dim=encoder_text_dim, return_keys=use_text_keys)
            
            del self.vision_decoder.model.transformer_module.queries_features

            if use_text_keys:
                self.vision_decoder.model.pixel_level_module.decoder.encoder.crss_att = nn.ModuleList([nn.MultiheadAttention(embed_dim=vision_decoder_config.hidden_dim, 
                                                                                                                             num_heads=vision_decoder_config.num_attention_heads, 
                                                                                                                             batch_first=True) for _ in range(vision_decoder_config.encoder_layers)])
                self.vision_decoder.model.pixel_level_module.decoder.encoder.crss_att.apply(self._init_weights)

                self.vision_decoder.model.pixel_level_module.decoder.encoder.text_keys_pos = nn.Embedding(nclasses, vision_decoder_config.hidden_dim)
                self.vision_decoder.model.pixel_level_module.decoder.encoder.text_keys_pos.apply(self._init_weights)                


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
    
    
    def train(self, mode: bool = True):
        super().train(mode)

        if mode and "clip" in self.encoder_name and self.freeze_text_encoder:
            set_requires_grad(self.encoder, ["vision_model"])
            set_train(self.encoder, ["vision_model"])


    def forward(self, pixel_values, bin_masks, classes, return_logits=False):
        if self.encoder_name == "vit":      
            vision_outputs = self.encoder(pixel_values=pixel_values, output_hidden_states=True, interpolate_pos_encoding=True)
        else:
            vision_outputs = self.encoder.get_image_features(pixel_values=pixel_values, output_hidden_states=True, interpolate_pos_encoding=True)
        
        vision_hidden_states = vision_outputs["hidden_states"]
        vision_hidden_states = [h for i,h in enumerate(vision_hidden_states) if i in self.out_indices]

        if self.has_text_decoder:
            text_outputs = self.encoder.get_text_features(input_ids=self.text_ids, attention_mask=self.text_att)

            keys, queries = self.text_decoder(text=text_outputs, visual=vision_hidden_states[-1])

            if keys is not None:
                # To cross-attention layers in pixel decoder (mask2former encoder) as key-value
                self.vision_decoder.model.pixel_level_module.decoder.encoder.text_keys = keys
            
            if queries is not None:
                # To cross-attention layers in transformer decoder layers (mask2former decoder) as queries
                self.vision_decoder.model.transformer_module.text_queries = queries

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

        if self.encoder_name == "vit":
            trainable_params = {"TOTAL": sum(p.numel() for p in self.parameters() if p.requires_grad),
                                "VIT": sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
                                "NECK": sum(p.numel() for p in self.neck.parameters() if p.requires_grad),
                                "MASK2FORMER": sum(p.numel() for p in self.vision_decoder.parameters() if p.requires_grad)}
        else:
            trainable_params = {"TOTAL": sum(p.numel() for p in self.parameters() if p.requires_grad),
                                "CLIP": sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
                                "CLIP_VISION": sum(p.numel() for p in self.encoder.vision_model.parameters() if p.requires_grad),
                                "CLIP_TEXT": sum(p.numel() for p in self.encoder.text_model.parameters() if p.requires_grad),
                                "NECK": sum(p.numel() for p in self.neck.parameters() if p.requires_grad),
                                "MASK2FORMER": sum(p.numel() for p in self.vision_decoder.parameters() if p.requires_grad)}
            
            if self.has_text_decoder:
                trainable_params.update({"TEXT_DECODER": sum(p.numel() for p in self.text_decoder.parameters() if p.requires_grad)})
        
        if round_to_millions:
            trainable_params = {k:round((v/1e6), decimals) for k,v in trainable_params.items()}

        print("TRAINABLE PARAMS (M):")
        [print(f"   {k}: {v}") for k,v in trainable_params.items()]
        print()


    def print_frozen_modules(self):
        self.train()

        if self.encoder_name == "vit":
            trainable_modules = {"MODEL": self.training,
                                "VIT": self.encoder.training,
                                "NECK": self.neck.training,
                                "MASK2FORMER": self.vision_decoder.training}
        else:
            trainable_modules = {"MODEL": self.training,
                                "CLIP": self.encoder.training,
                                "CLIP_VISION": self.encoder.vision_model.training,
                                "CLIP_TEXT": self.encoder.text_model.training,
                                "NECK": self.neck.training,
                                "MASK2FORMER": self.vision_decoder.training}
            
            if self.has_text_decoder:
                trainable_modules.update({"TEXT_DECODER": self.text_decoder.training})

        print("IS FROZEN:")
        [print(f"   {k}: {not v}") for k,v in trainable_modules.items()]
        print()
