import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
import os
import shutil
import tifffile
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import rasterio
from rasterio.plot import show
from PIL import Image
from sklearn.model_selection import train_test_split


'''
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(os.path.join(data_dir, "images"))
        self.labels = {}  # Dictionary to store labels
        self.artifacts = {}  # Dictionary to store artifact annotations
        
        # Load labels and artifact annotations
        for image_file in self.image_files:
            # Assuming labels are stored in a text file with image file names and corresponding labels
            with open(os.path.join(data_dir, "labels.txt"), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    image_name, label = line.strip().split(',')
                    self.labels[image_name] = int(label)
            
            # Assuming artifact annotations are stored as images with the same file names as the original images
            artifact_path = os.path.join(data_dir, "artifacts", image_file)
            if os.path.exists(artifact_path):
                self.artifacts[image_file] = artifact_path

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(os.path.join(self.data_dir, "images", img_name))
        label = self.labels[img_name]
        
        # If there are artifacts, load the annotation image
        if img_name in self.artifacts:
            artifact_image = Image.open(self.artifacts[img_name])
        else:
            artifact_image = None

        if self.transform:
            image = self.transform(image)
            if artifact_image:
                artifact_image = self.transform(artifact_image)

        return image, label, artifact_image

# Define transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])'''
def link_label1_to_image(train_images_dir, train_labels_dir):
    labels={}
    artifacts={}
# Get the list of filenames in the images directory
    train_image_filenames = os.listdir(train_images_dir)

# Assuming label filenames correspond to image filenames with "_label" appended
    train_label_filenames = os.listdir(train_labels_dir)

# Create full paths for images and labels
    train_image_paths = [os.path.join(train_images_dir, filename) for filename in train_image_filenames]
    train_label_paths = [os.path.join(train_labels_dir, filename) for filename in train_label_filenames]

    for idx in range(len(train_label_paths)):
        labels[train_image_paths[idx]]=1
        artifacts[train_image_paths[idx]]=train_label_paths[idx]
    return labels, artifacts
def link_label0_to_image(train_images_dir, save_label_dir):
    if not os.path.exists(save_label_dir):
        os.makedirs(save_label_dir)
    labels={}
    artifacts={}
# Get the list of filenames in the images directory
    train_image_filenames = os.listdir(train_images_dir)

# Assuming label filenames correspond to image filenames with "_label" appended
    

# Create full paths for images and labels
    train_image_paths = [os.path.join(train_images_dir, filename) for filename in train_image_filenames]
    train_label_paths=[]

    for idx in range(len(train_image_paths)):
        image = Image.open(train_image_paths[idx])
        artifact_image = Image.new('RGB', image.size, (0, 0, 0))
        train_label_paths.append(os.path.join(save_label_dir, train_image_filenames[idx]))
        if not os.path.exists(train_label_paths[idx]):
            artifact_image.save(os.path.join(save_label_dir, train_image_filenames[idx]))  
        labels[train_image_paths[idx]]=0
        artifacts[train_image_paths[idx]]=train_label_paths[idx]
    return labels, artifacts


