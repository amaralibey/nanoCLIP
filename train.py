# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/nanoCLIP
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import argparse

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger

from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer

from src.nanoclip import NanoCLIP
from src.dataset import Flickr30k, CollateFlickr


seed = 42


def train(batch_size, lr, dim, dev):
    
    seed_everything(seed, workers=True)
    
    # txt_model = "albert-base-v2"
    # txt_model = "distilbert-base-uncased"
    # txt_model = "bert-base-uncased"
    txt_model = "sentence-transformers/all-MiniLM-L6-v2" # (~20M params)
    # txt_model = "sentence-transformers/all-MiniLM-L12-v2"
    # txt_model = "huawei-noah/TinyBERT_General_4L_312D"
    # txt_model = "sentence-transformers/all-mpnet-base-v2"
    # txt_model = "microsoft/MiniLM-L12-H384-uncased"
    
    
    # let's define the model.
    model = NanoCLIP(
        txt_model=txt_model,
        img_model="dinov2_vits14", # 'dinov2_vitb14' (60M params) or 'dinov2_vits14' (20M params)
        embed_size=dim,
        lr=lr,
        weight_decay=0.001,
        warmup_epochs=10,
        milestones=[15, 30, 40],
        lr_mult=0.1,
    )
    
    train_transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.RandomPerspective(0.4, p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.), # no hue because it distorts the colors
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    
    train_dataset = Flickr30k('./datasets/flickr30k', split='train', img_transform=train_transform)
    val_dataset = Flickr30k('./datasets/flickr30k', split='val', img_transform=valid_transform)
    
    # use the same tokenizer as the one used in the text model.
    tokenizer = AutoTokenizer.from_pretrained(txt_model)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=CollateFlickr(tokenizer, max_length=80, captions_to_use='all') # captions_to_use='random' or 'first' or 'all'
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        collate_fn=CollateFlickr(tokenizer, captions_to_use='first') # in eval we use the first caption only
    )
    
    
    
    tensorboard_logger = TensorBoardLogger(
        save_dir=f"./logs",
        name=f"nano_clip",
        default_hp_metric=False
    )
    
    checkpoint_cb = ModelCheckpoint(
        monitor="recall@5",
        filename="epoch:[{epoch:02d}]_recall@5:[{recall@5:.4f}]]",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=1,
        mode="max",
    )
    
    trainer = Trainer(
        accelerator="gpu",
        devices=[0],
        logger=tensorboard_logger,
        precision="16-mixed",
        max_epochs=50,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_cb, RichProgressBar()],
        log_every_n_steps=10,
        fast_dev_run=dev,
        enable_model_summary=True,
    )
    
    trainer.fit(model, train_dataloader, val_dataloader)







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train parameters")
    
    parser.add_argument("--dev", action="store_true", help="Enable fast dev run (one train and validation iteration).")
    parser.add_argument("--bs", type=int, default=64, help="Batch size.")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimensionality.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate.")
    args = parser.parse_args()
    
    train(batch_size=args.bs, lr=args.lr, dim=args.dim,  dev=args.dev)
