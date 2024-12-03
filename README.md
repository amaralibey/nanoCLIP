# nanoCLIP

A lightweight Text-to-Image retrieval model.

What you can learn from this repo:

- Train an efficient text-to-image retrieval model on Flickr30k dataset
- Learn how to use PyTorch Lightning to train, validate, track and iterate over the entire pipline
- Create an efficient index for your gallery and deploy your web app using Gradio
- Have fun

## Getting started

1. This project structure is as follows:

   ```
   nanoCLIP
   ├── datasets          		# The training dataset resides here
   ├── deployment
   │   ├── create_index.py    	# To load a trained model and create an index for an Album.
   │   ├── load_album.py       	# Create a dataset containing all images in an album (inc. subfolder)
   │   └── gradio_app.py   	# Gradio APP. contains also inference code (query -> images)
   ├── gallery           		# Your gallery of photos and albums that will be deployed
   ├── logs              		# To track the experiments and save weights and Tensorboard logs
   ├── src
   │   ├── dataset.py    		# PyTorch dataset to read the train/val images and their captions
   │   ├── loss.py       		# Implements a contrastive loss to supervise the training
   │   ├── models.py     		# Contains the implementation of the Image Encoder and Text Encoder
   │   └── nanoclip.py   		# Lightning module to train, validate, and track the training
   └── train.py          		# Script to launch the training via the command line
   ```
