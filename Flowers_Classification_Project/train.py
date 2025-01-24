# Glorious Musyawa Project
# Check torch version and CUDA status if GPU is enabled.
import torch
print(torch.__version__)
print(torch.cuda.is_available())# Should return True when GPU is enabled. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Should return cuda if GPU is enabled 
print(device)


# Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

# Importing important pytorch libraries for computer vision
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

from PIL import Image
import requests
import tarfile
import os
import shutil


# ## Load the data

# Download the dataset 

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

# ## Data Description
# The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Defining transforms for the training, validation, and testing sets
#Both Validation and test sets will undergo the same transforms 
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


# Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# Building and training the classifier

# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# ## Note for Workspace users: 
# If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# Building and training the network
#Using Vgg16 as my backbone(pre-trained) network
#Loading the pre-trained model
vgg16 = models.vgg16(pretrained = True)

#Freezing the pre-trained model's parameters
#Why?- So that only the feedforward layers will be updated during training

for param in vgg16.features.parameters():
    param.requires_grad = False




#Defining the classifier network
class FlowerClassifier(nn.Module):
    def __init__(self,output_size):
        super(FlowerClassifier,self).__init__()
        self.vgg16 = models.vgg16(pretrained = True)
        self.vgg16.features.requires_grad = False
        
#Creating a new classifier        
        
        self.classifier = nn.Sequential(
            nn.Linear(25088,4096), #Starting from vgg16's output feature layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096,1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024,102)
        )
    
    def forward(self,x):
        x = self.vgg16.features(x)
        x = x.view(-1,25088)
        x = self.classifier(x)
        return x   

#instantiating the model
model = FlowerClassifier(102) # The dataset has 102 classes

#Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


#Defining the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(),lr=0.0001)

#Defining the training loop
#I intent to time the training

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
train_model(model, criterion, optimizer, num_epochs=23)
        


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

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


# Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# Saving the model checkpoint
 # Attach class-to-index mapping
model.class_to_idx = train_dataset.class_to_idx

# Create the checkpoint dictionary
checkpoint = {
    "model_state_dict": model.state_dict(),
    "class_to_idx": model.class_to_idx,
    "optimizer_state_dict": optimizer.state_dict(),
    "epochs": 23,
    "learning_rate": 0.0001,
    "architecture": "vgg16",
    "classifier": model.classifier,
    "output_size": 102
}

# Save the checkpoint to a file
torch.save(checkpoint, "flower_classifier_checkpoint.pth")

print("The model's checkpoint is saved as 'flower_classifier_checkpoint.pth'.")
