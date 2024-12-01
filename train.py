import argparse
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.loggers import TensorBoardLogger

from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer


from rich.traceback import install
install() # this is for better traceback formatting

from src.framework import Framework
from dataset import Flickr30k, CollateFlickr


seed = 42


def train(batch_size, dev=False):
    
    seed_everything(seed, workers=True)
    
    # lang_model = "albert-base-v2"
    # lang_model = "distilbert-base-uncased"
    # lang_model = "bert-base-uncased"
    lang_model = "sentence-transformers/all-MiniLM-L6-v2"
    # lang_model = "sentence-transformers/all-MiniLM-L12-v2"
    # lang_model = "huawei-noah/TinyBERT_General_4L_312D"
    # lang_model = "sentence-transformers/all-mpnet-base-v2"
    # lang_model = "microsoft/MiniLM-L12-H384-uncased"
    
    tokenizer = AutoTokenizer.from_pretrained(lang_model)
    
    model = Framework(
        lang_model=lang_model,
        lr=0.0001,
        warmup_epochs=10,
        weight_decay=0.001,
        milestones=[15, 30, 40],
        lr_mult=0.3,
    )
    
    
    train_transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomPerspective(0.4, p=0.5),
        transforms.RandomVerticalFlip(0.1),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.),
        # transforms.RandAugment(3, 15),
        # transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    
    train_dataset = Flickr30k('./dataset', split='train', img_transform=train_transform)
    val_dataset = Flickr30k('./dataset', split='val', img_transform=valid_transform)
    


    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=12, 
        pin_memory=True, 
        collate_fn=CollateFlickr(tokenizer, max_length=64, captions_to_use='all') # captions_to_use='random' or 'first' or 'all'
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=12, 
        pin_memory=True,
        collate_fn=CollateFlickr(tokenizer, captions_to_use='first') # in eval we use the first caption only
    )
    
    
    
    tensorboard_logger = TensorBoardLogger(
        save_dir=f"./logs",
        name=f"im_txt_matching",
        default_hp_metric=False
    )
    
    checkpoint_cb = ModelCheckpoint(
        monitor="recall",
        filename="epoch({epoch:02d})_step({step:04d})_recall[{recall:.4f}]]",
        auto_insert_metric_name=False,
        save_weights_only=False,
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
        callbacks=[
            checkpoint_cb,
            RichProgressBar(theme=RichProgressBarTheme()),
        ],
        reload_dataloaders_every_n_epochs=1,
        # accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        fast_dev_run=dev,
        enable_model_summary=True,
    )
    
    trainer.fit(model, train_dataloader, val_dataloader)

# add argparse to pass the config file


if __name__ == "__main__":
    # add argparse to pass the dev
    parser = argparse.ArgumentParser(description="Train parameters")
    
    parser.add_argument("--dev", action="store_true", help="Enable fast dev run (one train and validation iteration).")
    parser.add_argument("--bs", type=int, default=128, help="Batch size.")
    args = parser.parse_args()
    
    train(batch_size=args.bs, dev=args.dev)
