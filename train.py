import torch
import torch.nn as nn
import numpy as np
from dataset import get_dataloaders
from model import DevFDViT
from evaluate import evaluate_task

def calculate_orthogonal_loss(model, moe_inputs, lambda_1, lambda_2):
    t = len(model.moe_layers[0].experts) - 1      # index of current Fake-LoRA
    if t < 2:                                     # no previous Fake-LoRAs yet
        return 0.0
    L_ort = 0.0
    for l_idx, moe in enumerate(model.moe_layers):
        Bt = moe.experts[t].B
        r = Bt.size(1)

        Ht = moe_inputs[l_idx].detach().reshape(-1, moe_inputs[l_idx].size(-1))
        Ht = Ht[torch.randperm(Ht.size(0), device=Ht.device)[:2048]]   # subsample tokens
        _, _, Vh = torch.linalg.svd(Ht.float(), full_matrices=False)
        Vt_r = Vh[:r]                                                   # [r, d_in]

        for i in range(1, t):
            Bi = moe.experts[i].B.detach()
            O_it = Bt.t() @ Bi
            G_it = Vt_r @ Bi
            L_ort = L_ort + lambda_1 * (O_it ** 2).sum() + lambda_2 * (G_it ** 2).sum()
    return L_ort / (t - 1)

def train_continual():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DevFDViT().to(device)
    
    tasks = [
        "processed_data/FF++",
        "processed_data/DFDC-P",
        "processed_data/DFD",
        "processed_data/CDF2_from_list"
    ]
    
    criterion_cls = nn.BCELoss()
    lambda_3 = 0.2
    acc_matrix = np.zeros((len(tasks), len(tasks)))
    
    # Store the 20% test splits for historical evaluation
    historical_test_loaders = []
    
    for task_idx, task_path in enumerate(tasks):
        print(f"\n{'='*40}\nStarting Task {task_idx+1}: {task_path}\n{'='*40}")
        if task_idx > 0:
            model.add_task()
            model.to(device)
            
        # Get both the 80% train and 20% test splits
        train_loader, test_loader = get_dataloaders(task_path, batch_size=32, train_ratio=0.8)
        historical_test_loaders.append(test_loader)
        
        # Penalize the classifier head's learning rate to prevent drift
        head_params = list(model.head.parameters())
        base_params = [p for n, p in model.named_parameters() if not n.startswith('head.') and p.requires_grad]
        
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': 1e-4},
            {'params': head_params, 'lr': 1e-5}
        ], betas=(0.9, 0.999))
        
        for epoch in range(20):
            model.train()
            # Massively increased orthogonal penalties to protect old spaces
            if epoch < 5:    l1, l2 = 0.5, 0.5
            elif epoch < 10: l1, l2 = 1.0, 0.1
            else: l1, l2 = 1.0, 0.01
                
            epoch_loss = 0
            cls_sum = ort_sum = llb_sum = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.float().to(device)
                
                optimizer.zero_grad()
                preds, L_llb, moe_inputs = model(images, labels)
                
                L_cls = criterion_cls(preds, labels)
                L_ort = calculate_orthogonal_loss(model, moe_inputs, l1, l2)
                
                total_loss = L_cls + L_ort + (lambda_3 * L_llb)
                total_loss.backward()
                model.freeze_old_router_rows()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                cls_sum += L_cls.item(); ort_sum += float(L_ort); llb_sum += float(L_llb)
                
            n = len(train_loader)
            print(f"Epoch {epoch+1}/20 - Loss: {epoch_loss/n:.4f} "
      f"(cls {cls_sum/n:.4f}, ort {ort_sum/n:.4f}, llb {llb_sum/n:.4f})")

        print(f"\n--- Evaluating after completing Task {task_idx+1} ---")
        # Evaluate against the 20% test splits of all tasks seen so far
        for eval_idx in range(task_idx + 1):
            acc, auc = evaluate_task(model, historical_test_loaders[eval_idx], device)
            acc_matrix[task_idx, eval_idx] = acc
            print(f"Test on Task {eval_idx+1}: Acc = {acc:.2f}%, AUC = {auc:.2f}%")
            
        current_aa = np.mean(acc_matrix[task_idx, :task_idx+1])
        print(f"Average Accuracy (AA): {current_aa:.2f}%")
        
        if task_idx > 0:
            forgetting = [acc_matrix[j, j] - acc_matrix[task_idx, j] for j in range(task_idx)]
            current_af = np.mean(forgetting)
            print(f"Average Forgetting (AF): {current_af:.2f}%")

if __name__ == "__main__":
    train_continual()
