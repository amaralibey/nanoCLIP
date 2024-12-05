# ----------------------------------------------------------------------------
# Copyright (c) 2024 Amar Ali-bey
#
# OpenVPRLab: https://github.com/amaralibey/nanoCLIP
#
# Licensed under the MIT License. See LICENSE file in the project root.
# ----------------------------------------------------------------------------

import os
import sys
# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import faiss
from transformers import AutoTokenizer
import gradio as gr

from src.models import TextEncoder
from deployment.load_album import AlbumDataset

class ImageSearchEngine:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: int = 64,
        gallery_folder: str = "photos",
        device: str = 'cpu'
    ):
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA is not available. Using CPU instead.")
            device = 'cpu'
        self.device = torch.device(device)
        self.setup_model(model_name, output_dim)
        self.setup_gallery(gallery_folder)
        
    def setup_model(self, model_name: str, output_dim: int) -> None:
        """Initialize and load the text encoder model."""
        self.txt_encoder = TextEncoder(
            output_dim=output_dim,
            lang_model=model_name
        ).to(self.device)
        
        # Load the pre-trained weights for the text encoder
        # 
        weights_path = Path(__file__).parent.resolve() / 'txt_encoder_state_dict.pth'
        # check if the weights file exists
        if not weights_path.exists():
            raise FileNotFoundError(f"Text encoder weights not found: {weights_path}, make sure to run the create_index.py script.")
        weights = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.txt_encoder.load_state_dict(weights)
        self.txt_encoder.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def setup_gallery(self, gallery_folder: str) -> None:
        """Setup the image gallery and FAISS index."""
        gallery_path = Path(__file__).parent.parent.resolve() / f'gallery/{gallery_folder}'
        # check if the gallery folder exists
        if not gallery_path.exists():
            raise FileNotFoundError(f"Album folder {gallery_path} not found")
        # we use the AlbumDataset class to load the image paths (we won't load the images themselves)
        # this is more efficient than loading the images directly, because Gradio will load them 
        # given the paths returned by the search method.
        self.dataset = AlbumDataset(gallery_path, transform=None)
        
        # Load the FAISS index
        # the index file should be in the same folder as the gallery 
        # and has the same name as the folder being indexed 
        index_path = gallery_path.parent / f"{gallery_folder}.faiss"
        self.index = faiss.read_index(index_path.as_posix())
        
    @torch.no_grad()
    def encode_query(self, query_text: str) -> torch.Tensor:
        """Encode the text query into embeddings."""
        inputs = self.tokenizer(query_text, truncation=True, return_tensors="pt")
        inputs = inputs['input_ids'].to(self.device)
        
        embedding = self.txt_encoder(inputs)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu()
    
    def search(self, query_text: str, k: int = 10) -> List[Tuple[str, Optional[str]]]:
        """Search for images matching the query text."""
        if len(query_text) < 3: # avoid searching for very short queries
            return []
            
        query_embedding = self.encode_query(query_text)
        dist, indices = self.index.search(query_embedding, k)
        # you can filter results according to a threshold on the distance
        return [(self.dataset.imgs[idx], None) for idx in indices[0]]

class GalleryUI:
    def __init__(self, search_engine: ImageSearchEngine):
        self.search_engine = search_engine
        self.css_path = Path(__file__).parent / 'style.css'
        
    def load_css(self) -> str:
        """Load CSS styles from file."""
        with open(self.css_path) as f:
            return f.read()
            
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(css=self.load_css(), theme=gr.themes.Soft(text_size='lg')) as demo:
            with gr.Column(elem_classes="container"):
                self._create_header()
                self._create_search_section()
                self._create_footer()
                
            self._setup_callbacks(demo)
            return demo
    
    def _create_header(self) -> None:
        """Create the header section."""
        with gr.Column(elem_classes="header"):
            gr.Markdown("# Gallery Search")
            gr.Markdown("Search through your collection of photos with AI")
    
    def _create_search_section(self) -> None:
        """Create the search interface section."""
        with gr.Column():
            self.query_text = gr.Textbox(
                placeholder="Example: Riding my horse",
                label="Search Query",
                elem_classes="search-input",
                autofocus=True,
                container=False
            )
            
            with gr.Row(visible=False): # this is hidden for now, but you can show it if you want
                self.number_of_results = gr.Dropdown(
                    choices=[4,8,12,16,24,30],
                    value=30,
                    label="Results per page",
                    elem_classes="dropdown"
                )
                
            self.gallery = gr.Gallery(
                label="Search Results",
                columns=3,
                object_fit="cover",
                elem_classes="gallery",
                container=False,
            )
    
    def _create_footer(self) -> None:
        """Create the footer section."""
        with gr.Column(elem_classes="footer"):
            gr.Markdown(
                """Created by [Amar Ali-bey](https://amaralibey.github.io) | 
                [View on GitHub](https://github.com/amaralibey/nanoCLIP)"""
            )
    
    def _setup_callbacks(self, demo: gr.Blocks) -> None:
        """Setup the interface callbacks."""
        self.query_text.change(
            self.search_engine.search,
            inputs=[self.query_text, self.number_of_results],
            outputs=self.gallery,
            show_progress='hidden',
        )
        
        self.number_of_results.change(
            self.search_engine.search,
            inputs=[self.query_text, self.number_of_results],
            outputs=self.gallery
        )


if __name__ == "__main__":
    search_engine = ImageSearchEngine(
        model_name = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim = 64,
        gallery_folder = "photos",
    )
    ui = GalleryUI(search_engine)
    demo = ui.create_interface()
    
    # Launch the interface on port 7860
    # 0.0.0.0 makes the interface available on all network interfaces (through wifi or LAN for example)
    demo.launch(server_name="0.0.0.0", server_port=7860)