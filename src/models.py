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
    def __init__(self, output_dim=64,  dino_model='dinov2_vits14'):
        super().__init__()
        self.encoder = torch.hub.load('facebookresearch/dinov2', dino_model)
        
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
    def __init__(self, output_dim=64, lang_model="microsoft/MiniLM-L12-H384-uncased"):
        super().__init__()
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