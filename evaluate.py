import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def evaluate_task(model, test_loader, device):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.float().to(device)
            
            preds, _, _ = model(images)
            
            # Using np.atleast_1d prevents errors if batch size happens to be 1
            all_preds.extend(np.atleast_1d(preds.cpu().numpy()))
            all_labels.extend(np.atleast_1d(labels.cpu().numpy()))
            
    preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, preds_binary) * 100.0
    
    try:
        auc = roc_auc_score(all_labels, all_preds) * 100.0
    except ValueError:
        auc = 0.0 
        
    return acc, auc
