#PROGRAMMER: GLORIUS MUSYAWA

# importing libraries that will help in load the model from the checkpoint
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision import datasets
from torchvision import models

# Importing important libraries for data pre-processing and visualizing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Other necessary libraries
import time
import json
import argparse
from PIL import Image
import requests
import tarfile
import os
import shutil

# Argument parser for CLI options
parser = argparse.ArgumentParser(description="Image prediction script")
parser.add_argument('--gpu', action='store_true', help='Use GPU if available for training')
parser.add_argument('--image-path', type=str, required=True, help="Path to the image to predict")
parser.add_argument('--cat_json-path', type=str, required=True, help="Path to the cat_to_name json file")
parser.add_argument('--checkpoint-path',type=str,required = True, help="Path to model's saved checkpoint")
parser.add_argument('--topk',type = int, default = 5, help = 'Top k predictions of the model')
args = parser.parse_args()

# Detect if a GPU is available
if args.gpu and torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
    print("Using GPU for training")
else:
    device = torch.device("cpu")  # Fall back to CPU
    print("Using CPU for training")

#rebuilding the structure of my classifier
class FlowerClassifier(nn.Module):
    def __init__(self,base_model,output_size,in_features,hidden_units):
        super(FlowerClassifier,self).__init__()
        self.base_model = base_model
        self.in_features = in_features
        for param in base_model.features.parameters():
            param.requires_grad = False
                   
        self.classifier = nn.Sequential(
            nn.Linear(in_features,hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units,1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024,102)
        )
    
    def forward(self,x):
        x = self.base_model.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x   

    
# Load model checkpoint
def load_checkpoint(filepath):
    try:
        checkpoint = torch.load(filepath)
        
        # Extract necessary information from the checkpoint
        architecture = checkpoint['architecture']
        in_features = checkpoint['in_features']
        hidden_units = checkpoint['hidden_units']
        output_size = checkpoint['output_size']


        # Initialize the base model based on architecture (e.g., vgg16, resnet, etc.)
        base_model = getattr(models, architecture)(pretrained=True)
        # Load the model
        model = FlowerClassifier(base_model,output_size,in_features,hidden_units)
        
        # Freeze feature extraction layers
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace classifier with the one from the checkpoint
        model.classifier = checkpoint['classifier']
        
        # Load the model state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Attach the class-to-index mapping
        model.class_to_idx = checkpoint['class_to_idx']
        
        # Move model to device
        model.to(device)

        print("Checkpoint loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

model = load_checkpoint(args.checkpoint_path)

if model is not None:
    print("Model successfully loaded!.")



# Defining the process_image function to process a PIL image for use in a PyTorch model
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Load the image
    pil_image = Image.open(image_path)

    # Resize: Ensure the shortest side is 256 pixels, maintaining aspect ratio
    pil_image = pil_image.resize((256, 256))

    # Center crop to 224x224
    width, height = pil_image.size
    new_width, new_height = 224, 224
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    pil_image = pil_image.crop((left, top, right, bottom))

    # Convert to a numpy array and scale values
    np_image = np.array(pil_image) / 255.0

    # Normalize each color channel
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions to [color, height, width]
    np_image = np_image.transpose((2, 0, 1))

    return np_image



# Optional : Defining the image visualization function
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes the color channel is the last dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing (denormalize the image)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Clip the image to ensure pixel values are between 0 and 1
    image = np.clip(image, 0, 1)
    
    # Display the image
    ax.imshow(image)
    if title:
        ax.set_title(title)
    
    ax.axis('off')  # Remove axes for better visualization
    
    return ax


# Defining the predict function 
def predict(image_path, model, topk):
    print(f"Predicting for image at: {image_path}")
    """Predict the class (or classes) of an image using a trained deep learning model."""
    # Process the image
    processed_image = process_image(image_path)

    # Convert to a PyTorch tensor and move to the appropriate device
    image_tensor = torch.from_numpy(processed_image).type(torch.FloatTensor).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Forward pass
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)  # Apply softmax to convert logits to probabilities

    # Get the top K probabilities and their corresponding class indices
    top_probs, top_classes = probs.topk(topk, dim=1)

    # Convert to lists for easier handling
    top_probs = top_probs.cpu().numpy().flatten().tolist()
    top_classes = top_classes.cpu().numpy().flatten().tolist()

    # Map class indices to actual class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[c] for c in top_classes]

    return top_probs, top_classes


# Defining function to load the category to name mapping json file
def load_category_names(cat_json):
    with open(cat_json, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


# Optional:defining the sanity Check function
def sanity_check(image_path, model, cat_to_name, topk):
    """Perform a sanity check by displaying the image and prediction probabilities."""
    # Get predictions
    probs, classes = predict(image_path, model, topk)
    
    # Map class indices to flower names
    flower_names = [cat_to_name[cls] for cls in classes]
    
    # Process the image for display
    processed_image = process_image(image_path)
    image_tensor = torch.from_numpy(processed_image).type(torch.FloatTensor)
    
    # Plot the image and bar chart
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), nrows=2)
    imshow(image_tensor, ax=ax1)
    ax1.set_title(flower_names[0])  # Display the top prediction as the title
    
    # Bar chart for probabilities
    ax2.barh(flower_names, probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(range(len(flower_names)))
    ax2.set_yticklabels(flower_names)
    ax2.set_xlim(0, 1.1)
    ax2.invert_yaxis()  # Highest probability at the top
    ax2.set_xlabel('Probability')
    
    plt.tight_layout()
    plt.show()



# Getting the image path
image_path = args.image_path

# Pre-processing the image
process_image(image_path)

#loading the category to name mapping json file
cat_to_name = load_category_names(args.cat_json_path)

# Getting the topk user value
topk = args.topk

# Predicting the results
probs,classes = predict(args.image_path,model,topk = topk)
proba = [round(p,3) for p in probs]
flower_names = [cat_to_name[cls] for cls in classes]

#Printing out the results for the user
print(f'The predicted result is :\n {flower_names[0]} flower\n {("*" * 50)}')
print(f'Here are the top {topk} predictions of the flower in the image:\n {("*" * 50)}\n {proba}\n {flower_names}')

# Performing sanity check
# feel free to perform the sanity check too below by calling it!

#sanity_check(args.image_path, model, cat_to_name, topk)
