import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import cv2 
import numpy as np
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Define U-Net architecture
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            DoubleConv(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(128, 64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        x1 = self.encoder[0](x)
        x2 = self.encoder[2](x1)
        x3 = self.encoder[4](x2)
        x4 = self.encoder[6](x3)
        x5 = self.encoder[8](x4)
        
        d1 = self.decoder[0](x5)
        d1 = torch.cat([d1, x4], dim=1)
        d2 = self.decoder[1](d1)
        d2 = self.decoder[2](d2)
        d2 = torch.cat([d2, x3], dim=1)
        d3 = self.decoder[3](d2)
        d3 = self.decoder[4](d3)
        d3 = torch.cat([d3, x2], dim=1)
        d4 = self.decoder[5](d3)
        d4 = self.decoder[6](d4)
        d4 = torch.cat([d4, x1], dim=1)
        out = self.decoder[7](d4)
        
        return out

# Define training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, model_save_path='best_train_model.pth'):
    best_val_loss = float('inf')  # Initialize best validation loss
    last_improvement = 0
    model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
        avg_train_loss = running_loss / len(train_loader)
        # Validate the model
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images.to(device))
                loss = criterion(outputs, labels.to(device))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

        # Save the model if validation loss has decreased
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            last_improvement = epoch
            torch.save(model.state_dict(), model_save_path)
        elif epoch - last_improvement > 5:  # Patience of 5 epochs
            print(f'No improvement in validation loss for 5 epochs. Stopping training.')
            break

    print('Training completed.')



def save(self, path):
        torch.save(self.model.state_dict(), path)
    
def load(self, path):
        self.model.load_state_dict(torch.load(path))

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path) 

# Define test function
def test_model(model_path, test_loader, criterion, output_folder):
    if model_path is not None:
            model=load(model_path)
    create_folder_if_not_exists(output_folder)   
    model.eval()
    total_correct = 0
    total_samples = 0
    test_loss=0.0
    idx=0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
          
            loss = criterion(outputs, labels.to(device))
            test_loss += loss.item()
            # Assuming predicted_mask is the tensor containing the predicted mask
            predicted_mask = predicted.squeeze(0).cpu().numpy()  # Assuming batch size is 1

# Convert the predicted mask to a binary mask (0s and 255s)
            predicted_mask_binary = (predicted_mask * 255).astype(np.uint8)

# Save predicted mask image using OpenCV
            cv2.imwrite(f'{output_folder}/mask_{idx}.png', predicted_mask_binary)
            idx+=1
    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy}", f"Test Loss:{test_loss}")

