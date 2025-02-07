import wandb

def valid_wandb(accuracy, f1_score):
    wandb.log({
        'val_accuracy': accuracy,
        'val_f1_score': f1_score
    })


def write_all_epoch(train_loss, val_loss, epoch):
    wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch': epoch
            })