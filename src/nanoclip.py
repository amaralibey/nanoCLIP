# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/nanoCLIP
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import faiss
import numpy as np
import lightning as L
from src.loss import ContrastiveLoss
from src.models import ImageEncoder, TextEncoder


class NanoCLIP(L.LightningModule):
    """ 
    This class defines the pipeline for the nanoCLIP model.
    
    """
    def __init__(
        self,
        txt_model="microsoft/MiniLM-L12-H384-uncased",
        img_model='dinov2_vits14',
        embed_size=64, # output dimension of the encoder
        lr=0.0001,
        warmup_epochs=0,
        weight_decay=0.0001,
        milestones=[5, 10, 15],
        lr_mult=0.1,
    ):
        super().__init__()
        
        self.txt_model = txt_model
        self.img_model = img_model
        self.embed_size = embed_size
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.lr_mult = lr_mult
        # Let's save all hyperparameters to hparams file (for reproducibility)
        self.save_hyperparameters()
        
        self.img_encoder = ImageEncoder(self.embed_size, self.img_model)
        self.txt_encoder = TextEncoder(self.embed_size, self.txt_model)
        self.loss_fn = ContrastiveLoss()

    
    def configure_optimizers(self):
        """
        Define the optimizer and the learning rate scheduler.
        """
        optimizer_params = [
            {"params": self.img_encoder.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
            {"params": self.txt_encoder.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
        ]
        optimizer = torch.optim.AdamW(optimizer_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_mult
        )    
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Define how a single optimization step is executed.
        """
        if self.trainer.current_epoch < self.warmup_epochs:
            total_warmup_steps = self.warmup_epochs * self.trainer.num_training_batches
            lr_scale = min(1.0, (self.trainer.global_step + 1) / total_warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        optimizer.step(closure=optimizer_closure)
        self.log('_LR', optimizer.param_groups[-1]['lr'], prog_bar=False, logger=True)
    
    def forward(self, image, captions, masks):
        """ 
        Define the forward pass of the pipeline.
        """
        # compute image embeddings
        image_embedding = self.img_encoder(image) # (batch_size, out_dim)
        image_embedding = F.normalize(image_embedding, p=2, dim=-1) # normalize embeddings
        
        # compute text embeddings
        text_embedding = self.txt_encoder(captions, masks) # (batch_size, nb_captions, out_dim)
        text_embedding = F.normalize(text_embedding, p=2, dim=-1) # normalize embeddings
        
        return image_embedding, text_embedding
    
    def training_step(self, batch, batch_idx):
        """ 
        Define a single training step (one batch pass).
        
        ImageEncoder ──┐
                       ├──► ContrastiveLoss   
        TextEncoder  ──┘
        """
        images, captions, masks = batch
        
        if len(captions.shape) == 3: # flatten captions to (batch_size*nb_caps, cap_len) cuz we have multiple captions per image
            B, nb_captions, cap_len = captions.shape
            B, nb_masks, mask_len = masks.shape
            captions = captions.view(B*nb_captions, cap_len) 
            masks = masks.view(B*nb_masks, mask_len)
        else:
            nb_captions = 1
            
        img_descriptors, txt_descriptors = self(images, captions, masks)
        
        if nb_captions > 1: # reshape back to (B, nb_captions, out_dim)
            txt_descriptors = txt_descriptors.view(B, nb_captions, -1)
        
        
        loss, batch_accuracy = self.loss_fn(img_descriptors, txt_descriptors)
        
        self.log("loss", loss, prog_bar=True, logger=True)
        self.log("batch_acc", batch_accuracy, prog_bar=True, logger=True)
        return loss
    
    def on_validation_epoch_start(self):
        self.validation_descriptors = {"img": [], "txt": []}
        
    def validation_step(self, batch, batch_idx):
        """ 
        Define a single validation step (one batch pass).
        """
        images, captions, masks = batch
        
        img_descriptors, txt_descriptors = self(images, captions, masks)
        img_descriptors = img_descriptors.detach().cpu().numpy()
        txt_descriptors = txt_descriptors.detach().cpu().numpy()
        
        self.validation_descriptors["img"].append(img_descriptors)
        self.validation_descriptors["txt"].append(txt_descriptors)
    
    def on_validation_epoch_end(self):
        """ 
        Calculate the recall at 1, 5, and 10 for the validation set.
        """
        img_descriptors = np.concatenate(self.validation_descriptors["img"], axis=0) # (N, out_dim)
        txt_descriptors = np.concatenate(self.validation_descriptors["txt"], axis=0) # (N, nb_cap, out_dim)
        
        # create dummy labels
        B = img_descriptors.shape[0]    
        labels = np.arange(B)

        # use faiss to calculate recall, images are gallery and texts are queries
        recall_1, recall_5, recall_10 = self._calculate_recall(img_descriptors, txt_descriptors, labels, k_values=[1, 5, 10])
        self.log("recall@1", recall_1, prog_bar=True, logger=True)
        self.log("recall@5", recall_5, prog_bar=True, logger=True)
        self.log("recall@10", recall_10, prog_bar=False, logger=True)

        # clear the validation descriptors for the next epoch
        self.validation_descriptors.clear()
    
    @staticmethod
    def _calculate_recall(img_descriptors, txt_descriptors, labels, k_values=[1, 5, 10]):
        """ 
        Calculate the recall at k for the given img_descriptors as gallery
        and txt_descriptors as queries.
        """
        embed_size = img_descriptors.shape[1]
        faiss_index = faiss.IndexFlatL2(embed_size) 
        
        faiss_index.add(img_descriptors) # add images to the index
        _, predictions = faiss_index.search(txt_descriptors, max(k_values)) # search for the top k images for each text query
        
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], labels[q_idx])):
                    correct_at_k[i:] += 1
                    break
        
        correct_at_k /= len(labels)
                
        return correct_at_k