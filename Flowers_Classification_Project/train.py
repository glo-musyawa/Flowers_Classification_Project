# PROGRAMMER: GLORIUS MUSYAWA

# Import necessary PyTorch libraries for building and training models
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from torchvision import datasets
from torchvision import models

# Import libraries for visualization and data handling
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Other utility libraries
import time
import json
from PIL import Image
import requests
import tarfile
import os
import shutil
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a Flower Classifier")
# gpu availability for use argument
parser.add_argument('--gpu', action='store_true', help='Use GPU if available for training')

# Model architecture argument (with choices)
parser.add_argument('--arch', default='vgg16', choices=['vgg16', 'alexnet'], 
                    help='Model architecture (default: vgg16)')

# Directory to save the model checkpoint
parser.add_argument('--save_dir', default='./', help='Directory to save checkpoints (default: current directory)')

# Learning rate argument
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate (default: 0.0001)')

# Number of hidden units for the classifier
parser.add_argument('--hidden_units', type=int, default= 4096, help='Number of hidden units (default: 4096)')

# Number of training epochs
parser.add_argument('--epochs', type=int, default=23, help='Number of training epochs (default: 23)')
# Parse the command-line arguments
args = parser.parse_args()

# Set device based on GPU availability
if args.gpu and torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
    print("Using GPU for training")
else:
    device = torch.device("cpu")  # Fall back to CPU
    print("Using CPU for training")

# Defining Input_features for the model chosen
if args.arch == 'vgg16':
    model = models.vgg16(pretrained = True)
    in_features = 25088
elif args.arch == "alexnet":
    model = models.alexnet(pretrained = True)
    in_features = 9216
# Move the model to the appropriate device
model.to(device)



# Define the URL and folder paths
url = "https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz"
folder_name = "flowers"
file_name = "flower_data.tar.gz"
file_path = os.path.join(folder_name, file_name)

# Remove the folder or symbolic link if it already exists (equivalent to `rm -rf flowers`)
try:
    if os.path.islink(folder_name) or os.path.isfile(folder_name):
        os.remove(folder_name)  # Remove the symbolic link or file
    elif os.path.isdir(folder_name):
        shutil.rmtree(folder_name)  # Remove the directory
    print(f"Removed existing {folder_name} folder/file/soft link, if any.")
except FileNotFoundError:
    pass  # If the file or directory does not exist, do nothing

# Create the folder
os.makedirs(folder_name)
print(f"Created folder: {folder_name}")

# Download the file
response = requests.get(url, stream=True)

# Save the file in the 'flowers' folder
with open(file_path, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            file.write(chunk)

print(f"Downloaded {file_name} to {folder_name}")

# Extract the file in the 'flowers' folder
if file_path.endswith("tar.gz"):
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=folder_name)
        print(f"Extracted {file_name} to {folder_name}")

# Clean up by removing the tar.gz file after extraction
os.remove(file_path)
print(f"Removed the downloaded tar.gz file: {file_path}")

# Paths for data directories

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# Defining transforms for the training, validation, and testing sets

train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

validation_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#Loading the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir,transform = train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform = validation_test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform = validation_test_transforms)

# Defining dataloaders
train_loader =DataLoader(train_dataset,batch_size = 32, shuffle= True,num_workers = 4)
valid_loader = DataLoader(valid_dataset, batch_size = 32, shuffle = False, num_workers = 4)
test_loader = DataLoader(test_dataset, shuffle = False )

# Load class-to-name mapping

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


#Defining the classifier network
class FlowerClassifier(nn.Module):
    def __init__(self,base_model,output_size,in_features,hidden_units):
        super(FlowerClassifier,self).__init__()
        self.base_model = base_model
        self.in_features = in_features

        # Freeze base model parameters
        for param in base_model.features.parameters():
            param.requires_grad = False
            
        #Creating a new classifier        
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


#Initializing the model

output_size = 102
# Use specified hidden units 
hidden_units = args.hidden_units

model = FlowerClassifier(model,output_size = output_size,in_features = in_features,hidden_units = hidden_units) # The dataset has 102 classes

model.to(device)

#Defining the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

#Defining the training function
def train_model(model,criterion,optimizer,num_epochs = 23):
    since = time.time()
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 35)
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs,labels in train_loader:
            inputs,labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            #forward pass
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            #backward pass optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs,1)
            running_corrects += torch.sum(preds == labels.data)
            
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Train Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}')
        
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(valid_loader.dataset)
        val_acc = val_corrects.double() / len(valid_loader.dataset)
        
        print(f'Val Loss: {val_loss:.4f} Accuracy: {val_acc:.4f}')
        
        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    print(f'Best val Acc: {best_acc:.4f}')

# Train the model
train_model(model, criterion, optimizer, num_epochs=args.epochs)
        


# Evaluating the model
def test_model(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    test_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)

    test_loss = test_loss / len(dataloader.dataset)
    test_acc = test_corrects.double() / len(dataloader.dataset)

    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')

# Evaluate the trained model on the test dataset
print("Evaluating the model on the test dataset:")
test_model(model, test_loader)



# Saving the model checkpoint
# Attach class-to-index mapping
model.class_to_idx = train_dataset.class_to_idx

# Create the checkpoint dictionary
checkpoint = {
    "model_state_dict": model.state_dict(),
    "class_to_idx": model.class_to_idx,
    "optimizer_state_dict": optimizer.state_dict(),
    "epochs": 23,
    "learning_rate": optimizer.param_groups[0]['lr'],
    "architecture": args.arch,
    "classifier": model.classifier,
    "in_features": in_features,
    "hidden_units": hidden_units,
    "output_size": 102
}

# Save the checkpoint to a file
checkpoint_name = f"{args.save_dir}/flower_classifier_checkpoint_{args.arch}.pth"
torch.save(checkpoint, checkpoint_name)

print(f"The model's checkpoint is saved as '{checkpoint_name}'.")
