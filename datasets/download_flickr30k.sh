#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get the project root directory
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if Kaggle is installed
if ! command_exists kaggle; then
    echo "Kaggle is not installed. Please install Kaggle using: pip install kaggle"
    exit 1
fi

# Set the dataset name and download path
dataset_name="eeshawn/flickr30k" # dataset name on Kaggle
download_path="$PROJECT_ROOT/datasets/flickr30k" # path to download the dataset

# Create the download directory if it doesn't exist
mkdir -p "$download_path"


# make sur the folder flickr30k_images does not exist, otherwise the dataset has already been downloaded
if [ -d "$download_path/flickr30k_images" ]; then
    echo "Flick3 k dataset already downloaded to $download_path."
    exit 0
fi


# Download the dataset
echo "Downloading Flickr30k dataset from Kaggle..."
echo "To path: $download_path"
kaggle datasets download "eeshawn/flickr30k" -p "$download_path" --unzip

if [ $? -eq 0 ]; then
    echo "Flickr30k downloaded successfully to $download_path."
else
    echo "Error downloading dataset."
fi


echo "Download completed. Dataset available at $download_path."