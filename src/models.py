# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/nanoCLIP
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ImageEncoder(nn.Module):
    SUPPORTED_MODELS = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        # 'dinov2_vitg14' # let's not use huge models for this simple task
    ]
    def __init__(self, output_dim=64,  img_model='dinov2_vits14'):
        super().__init__()
        if img_model not in self.SUPPORTED_MODELS:
            raise ValueError(f'Invalid model name. Choose between {self.SUPPORTED_MODELS}')
        self.encoder = torch.hub.load('facebookresearch/dinov2', img_model)
        
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # unfreeze the last few blocks
        self.num_unfrozen_blocks = 4
        for block in self.encoder.blocks[-self.num_unfrozen_blocks : ]:
            for param in block.parameters():
                param.requires_grad = True
        
        # unfreeze the norm layer
        for param in self.encoder.norm.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(self.encoder.embed_dim, output_dim)
        
    def forward(self, x):
        dino_output = self.encoder.forward_features(x)
        x = dino_output['x_norm_clstoken'] # x_norm_clstoken, x_norm_patchtokens, x_prenorm
        x = self.fc(x)
        return x
    


class TextEncoder(nn.Module):
    def __init__(self, output_dim=64, lang_model="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.lang_model = lang_model
        self.encoder = AutoModel.from_pretrained(lang_model)
        
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # unfreeze the last few encoder layers
        self.num_unfrozen_layers = 4
        for layer in self.encoder.encoder.layer[-self.num_unfrozen_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # unfreeze the pooler layer
        for param in self.encoder.pooler.parameters():
            param.requires_grad = True
        
        
        self.fc = nn.Linear(self.encoder.config.hidden_size, output_dim)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        x = self.fc(x)
        return x