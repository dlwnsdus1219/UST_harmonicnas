import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchsummary import summary

from sklearn.metrics import f1_score

import wandb_vgg

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Training Loop
def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()     # Back Propagation
        optimizer.step()
    return loss.item()

# Validation 과정(with 테스트 데이터셋)
def validate_one_epoch(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    total_metrics = {'accuracy': 0, 'f1_score': 0}
    num_batches = 0

    with torch.no_grad():
        for images, labels in test_loader:
             images, labels = images.to(device), labels.to(device)
             outputs = model(images)
             loss = criterion(outputs, labels)
             total_loss += loss.item()

             _, predicted = torch.max(outputs.data, 1)
             total = labels.size(0)
             correct = (predicted == labels).sum().item()
             accuracy = correct / total
             predicted_cpu = predicted.cpu()
             labels_cpu = labels.cpu()
             f1 = f1_score(labels_cpu, predicted_cpu, average='macro')

             total_metrics['accuracy'] += accuracy
             total_metrics['f1_score'] += f1
             num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}

    accr = avg_metrics['accuracy']
    f1s = avg_metrics['f1_score']
    
    # 한 에포크마다 validation 하고 기록!!
    wandb_vgg.valid_wandb(accr, f1s)

    return avg_loss