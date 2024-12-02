import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """ 
    symmetric cross-entropy contrastive loss. 
    It aims to maximize the cosine similarity 
    between matched image-text pairs while 
    minimizing it for unmatched pairs within a batch
    
    This version supports multiple captions per image.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embedding, text_embedding):
        batch_size = image_embedding.shape[0]
        
        # we create dummy labels for the batch.
        labels = torch.arange(batch_size, device=image_embedding.device)
        
        # in case we have multiple captions per image
        if len(text_embedding.shape) == 3:
            # TODO: vectorize this loop
            for i in range(text_embedding.shape[1]):
                logits = torch.matmul(image_embedding, text_embedding[:, i].T) / self.temperature
                loss_i2t = F.cross_entropy(logits, labels)
                loss_t2i = F.cross_entropy(logits.T, labels)
                if i == 0:
                    total_loss = (loss_i2t + loss_t2i) / 2
                else:
                    total_loss += (loss_i2t + loss_t2i) / 2
            total_loss /= text_embedding.shape[1]
            
        # in case we have a single caption per image
        else:
            logits = torch.matmul(image_embedding, text_embedding.T) / self.temperature
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            total_loss = (loss_i2t + loss_t2i) / 2          
        
        # compute text2image (t2i)  accuracy (only for logging purposes)
        pred_t2i = torch.argmax(logits, dim=0)
        acc_t2i = (pred_t2i == labels).float().mean()
        
        return total_loss, acc_t2i