import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from dataset import get_dataloaders
from model import LensClassifier

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    base_dir = '/Users/mywishanand/Documents/5-Lens-Finding-Data-Pipelines/lens-finding-test'
    test_lenses = os.path.join(base_dir, 'test_lenses')
    test_nonlenses = os.path.join(base_dir, 'test_nonlenses')
    
    # We only need the test loader, so we can mock train paths
    _, test_loader, _ = get_dataloaders(
        test_lenses, test_nonlenses, test_lenses, test_nonlenses, 
        batch_size=64, train_transform=None, test_transform=None
    )
    
    model = LensClassifier(pretrained=False)
    
    model_path = '/Users/mywishanand/Documents/5-Lens-Finding-Data-Pipelines/best_model.pth'
    if not os.path.exists(model_path):
        print(f"Model weights not found at {model_path}. Please train the model first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_labels = []
    all_probs = []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            # The model outputs raw logits
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f"ROC Curve computed. AUC Score: {roc_auc:.4f}")
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    plot_path = '/Users/mywishanand/Documents/5-Lens-Finding-Data-Pipelines/roc_curve.png'
    plt.savefig(plot_path)
    print(f"Saved ROC curve plot to {plot_path}")

if __name__ == '__main__':
    evaluate()
