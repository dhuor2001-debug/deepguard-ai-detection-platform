import torch
import timm
import os
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from huggingface_hub import login
from PIL import Image

# Configuration
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "deepguard_efficientnet.pth"
NUM_CLASSES = 100  # CIFAR-100 has 100 classes

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def train():
    print(f"Using device: {DEVICE}")
    
    # Load CIFAR-100 dataset from torchvision
    print("Loading CIFAR-100 dataset...")
    
    try:
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    except Exception as e:
        print(f"Error loading CIFAR-100: {e}")
        print("Could not load CIFAR-100 dataset. Please ensure:")
        print("1. You have internet connection")
        print("2. You have enough disk space (~500MB)")
        return
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    print("Creating EfficientNet B0 model...")
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    model.to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Train"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Val"):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_loss /= len(val_loader)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model saved with validation accuracy: {val_acc:.2f}%")
    
    print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
