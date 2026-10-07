import random
import torch
import torch.nn as nn
import numpy as np
from dataset import get_dataloaders
from model import DevFDViT
from evaluate import evaluate_task


def gradient_basis(H, r):
    """Top-r right singular vectors of the input matrix; eigh runs on CPU for stability."""
    H = H.reshape(-1, H.size(-1)).float()
    C = (H.T @ H).double().cpu()
    _, V = torch.linalg.eigh(C)                # eigenvalues ascending
    return V[:, -r:].T.float()                 # [r, d_in] on CPU


def calculate_orthogonal_loss(model, moe_inputs, lambda_1, lambda_2, use_grad_term=False):
    t = len(model.moe_layers[0].experts) - 1   # index of current Fake-LoRA
    if t < 2:                                  # Task 1: no previous Fake-LoRA yet
        return 0.0
    L_ort = 0.0
    for l_idx, moe in enumerate(model.moe_layers):
        Bt = moe.experts[t].B
        Vt_r = None
        if use_grad_term:                      # constant w.r.t. parameters: monitoring only
            Vt_r = gradient_basis(moe_inputs[l_idx].detach(), Bt.size(1)).to(Bt.device)
        for i in range(1, t):
            Bi = moe.experts[i].B.detach()
            L_ort = L_ort + lambda_1 * ((Bt.t() @ Bi) ** 2).sum()
            if use_grad_term:
                L_ort = L_ort + lambda_2 * ((Vt_r @ Bi) ** 2).sum()
    return L_ort / (t - 1)

def train_continual(rank=32, epochs=20, n_tasks=4, holdout=False,
                    batch_size=32, freeze_head=False, freeze_real=False, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DevFDViT(rank=rank).to(device)

    tasks = [
        "processed_data/FF++",
        "processed_data/DFDC-P",
        "processed_data/DFD",
        "processed_data/CDF2_from_list",
    ][:n_tasks]

    criterion_cls = nn.BCELoss()
    lambda_3 = 0.2
    acc_matrix = np.zeros((len(tasks), len(tasks)))
    historical_test_loaders = []

    for task_idx, task_path in enumerate(tasks):
        print(f"\n{'='*40}\nStarting Task {task_idx+1}: {task_path}\n{'='*40}")
        if task_idx > 0:
            model.add_task(freeze_real=freeze_real)
            model.to(device)
            if freeze_head:                              # experiment A
                for p in model.head.parameters():
                    p.requires_grad = False

        train_loader, test_loader = get_dataloaders(
            task_path, batch_size=batch_size, train_ratio=0.8, holdout=holdout)
        historical_test_loaders.append(test_loader)

        head_params = [p for p in model.head.parameters() if p.requires_grad]
        base_params = [p for n, p in model.named_parameters()
                       if not n.startswith('head.') and p.requires_grad]
        groups = [{'params': base_params, 'lr': 1e-4}]
        if head_params:
            groups.append({'params': head_params, 'lr': 1e-5})
        optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999))

        for epoch in range(epochs):
            frac = epoch / epochs                        # paper: 25% / 25% / 50% of epochs
            if frac < 0.25:   l1, l2 = 0.5, 0.5
            elif frac < 0.5:  l1, l2 = 1.0, 0.1
            else:             l1, l2 = 1.0, 0.01

            model.train()
            epoch_loss = cls_sum = ort_sum = llb_sum = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.float().to(device)

                optimizer.zero_grad()
                preds, L_llb, moe_inputs = model(images, labels)

                L_cls = criterion_cls(preds, labels)
                L_ort = calculate_orthogonal_loss(model, moe_inputs, l1, l2)

                total_loss = L_cls + L_ort + lambda_3 * L_llb
                total_loss.backward()
                model.freeze_old_router_rows()
                optimizer.step()

                epoch_loss += total_loss.item()
                cls_sum += L_cls.item(); ort_sum += float(L_ort); llb_sum += float(L_llb)

            n = len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/n:.4f} "
                  f"(cls {cls_sum/n:.4f}, ort {ort_sum/n:.4f}, llb {llb_sum/n:.4f})")

        print(f"\n--- Evaluating after completing Task {task_idx+1} ---")
        for eval_idx in range(task_idx + 1):
            acc, auc = evaluate_task(model, historical_test_loaders[eval_idx], device)
            acc_matrix[task_idx, eval_idx] = acc
            print(f"Test on Task {eval_idx+1}: Acc = {acc:.2f}%, AUC = {auc:.2f}%")

        print(f"Average Accuracy (AA): {np.mean(acc_matrix[task_idx, :task_idx+1]):.2f}%")
        torch.save(acc_matrix, f"acc_matrix_task{task_idx+1}.pt")
        if task_idx > 0:
            forgetting = [acc_matrix[j, j] - acc_matrix[task_idx, j] for j in range(task_idx)]
            print(f"Average Forgetting (AF): {np.mean(forgetting):.2f}%")

    return acc_matrix


if __name__ == "__main__":
    train_continual(rank=32, freeze_real=True)
