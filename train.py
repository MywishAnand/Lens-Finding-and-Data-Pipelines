import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import LensClassifier
import torchvision.transforms.v2 as transforms

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    base_dir = '/Users/mywishanand/Documents/5-Lens-Finding-Data-Pipelines/lens-finding-test'
    train_lenses = os.path.join(base_dir, 'train_lenses')
    train_nonlenses = os.path.join(base_dir, 'train_nonlenses')
    test_lenses = os.path.join(base_dir, 'test_lenses')
    test_nonlenses = os.path.join(base_dir, 'test_nonlenses')
    
    # Data augmentation for training dataset
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    
    test_transform = None
    
    train_loader, test_loader, pos_weight = get_dataloaders(
        train_lenses, train_nonlenses, test_lenses, test_nonlenses, 
        batch_size=64, train_transform=train_transform, test_transform=test_transform
    )
    
    print(f"Positive weight (num_neg/num_pos) for loss balancing: {pos_weight:.2f}")
    
    model = LensClassifier(pretrained=True).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    num_epochs = 15
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
        val_loss /= len(test_loader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val/Test Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            # Save absolute path to make sure it loads correctly from any CWD
            save_path = '/Users/mywishanand/Documents/5-Lens-Finding-Data-Pipelines/best_model.pth'
            torch.save(model.state_dict(), save_path)
            print("  --> Saved Best Model")

if __name__ == '__main__':
    main()
