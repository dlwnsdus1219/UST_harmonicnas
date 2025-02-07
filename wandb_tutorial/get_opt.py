import torch
import torch.nn as nn
import torch.optim as optim

def get_optimizer(optimizer_name, model_parameters, learning_rate, momentum=0.0, weight_decay=0.0):
    if optimizer_name == 'adam':
        return torch.optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(model_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported Optimizer: {optimizer_name}")