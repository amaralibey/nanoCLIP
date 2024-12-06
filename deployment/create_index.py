# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/nanoCLIP
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

from pathlib import Path
import sys
import os

# Get the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch
import torch.nn.functional as F
from torchvision import transforms as T

import faiss
from tqdm import tqdm

from src.nanoclip import NanoCLIP
from deployment.load_album import AlbumDataset

img_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class NanoCLIPDeployer:
    def __init__(self, 
                model_checkpoint='../logs/nano_clip/version_12/checkpoints/epoch:[17]_recall@5:[0.6517]].ckpt', 
                txt_model = "sentence-transformers/all-MiniLM-L6-v2",
                img_model = 'dinov2_vits14',
                embed_size = 64,
                device = 'cuda',
                gallery_path = './gallery/photos',
                img_transform = img_transform,
                batch_size = 32,
                num_workers = 4
                 ):
        
        # check if device is available
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA is not available. Using CPU instead.")
            device = 'cpu'
        
        self.device = torch.device(device)
        self.gallery_path = Path(gallery_path).resolve()
        
        # Ensure paths exist
        if not Path(model_checkpoint).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")
        if not self.gallery_path.exists():
            raise FileNotFoundError(f"Gallery path not found: {self.gallery_path}")

        # Initialize the model
        self.model = NanoCLIP(txt_model=txt_model, img_model=img_model, embed_size=embed_size)
        state_dict = torch.load(model_checkpoint, weights_only=True)
        self.model.load_state_dict(state_dict['state_dict'])
        self.model.eval()
        print("Model loaded successfully.")
        
        # Separate the encoders
        self.img_encoder = self.model.img_encoder.eval()
        self.txt_encoder = self.model.txt_encoder.eval()
        
        # Save encoders for future use. Save them in the same directory as this script (/deploy)
        torch.save(self.img_encoder.state_dict(), Path(__file__).parent.resolve() / "img_encoder_state_dict.pth") 
        torch.save(self.txt_encoder.state_dict(), Path(__file__).parent.resolve() / "txt_encoder_state_dict.pth")
        print(f"Encoders saved in: {Path(__file__).parent.resolve()}")
        
        
        dataset = AlbumDataset(root_dir=self.gallery_path, transform=img_transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        img_embeddings = self.compute_img_embeddings(dataloader, self.img_encoder, normalize=True, device=self.device)

        # create and index and save it to disk in the same directory as the gallery parrent directory
        index_path = self.gallery_path.parent / f"{self.gallery_path.name}.faiss"
        self.create_and_save_index(img_embeddings, index_path.as_posix())
        print(f"Index created and saved in: {index_path}")
        
        
        
    @staticmethod
    def compute_img_embeddings(dataloader, img_encoder, normalize=True, device='cuda'):
        img_encoder.eval()
        img_encoder.to(device)
        img_embeddings = []
        for imgs, _ in tqdm(dataloader, desc="Embedding gallery images"):
            imgs = imgs.to(device)
            with torch.no_grad():
                img_img_embeddings = img_encoder(imgs)
                if normalize:
                    img_img_embeddings = F.normalize(img_img_embeddings, p=2, dim=1)
            img_embeddings.append(img_img_embeddings.cpu())
        return torch.cat(img_embeddings, dim=0)
    
    @staticmethod
    def create_and_save_index(img_embeddings, save_path):
        index = faiss.IndexFlatL2(img_embeddings.size(1)) # create the index
        index.add(img_embeddings) # add the image embeddings to the index
        faiss.write_index(index, save_path) # save the index to disk


#------------------------------------------------------------


def main():
    deployer = NanoCLIPDeployer(
        model_checkpoint='logs/nano_clip/version_12/checkpoints/epoch:[17]_recall@5:[0.6517]].ckpt',
        txt_model = "sentence-transformers/all-MiniLM-L6-v2",
        img_model = 'dinov2_vits14',
        embed_size = 64,
        device = 'cuda',
        gallery_path = './gallery/photos',
    )
    
if __name__ == '__main__':
    main()  # Run the main function
    