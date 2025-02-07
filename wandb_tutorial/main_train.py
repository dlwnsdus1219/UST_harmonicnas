import vgg16 as VG
import get_opt as go
import test_and_valid as tv
import dtl

import wandb
import wandb_vgg as wv

import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

wandb.login()

def train_model(config=None):
    with wandb.init(config=config):
        config = wandb.config

        model = VG.VGG16(num_classes=10)
        model = model.to(device)

        train_loader = dtl.create_train_loader(config.batch_size)
        test_loader = dtl.create_test_loader(config.batch_size)

        optimizer = go.get_optimizer(
            optimizer_name=config.optimizer,
            model_parameters=model.parameters(),
            learning_rate=config.learning_rate,
            momentum=config.get('momentum', 0.0),
            weight_decay=config.get('weight_decay', 0.0)
        )

        criterion = nn.CrossEntropyLoss()
        wandb.watch(model)

        for epoch in range(config.n_epochs):
            train_loss = tv.train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss = tv.validate_one_epoch(model, test_loader, criterion)
            wv.write_all_epoch(train_loss, val_loss, epoch)
            print(f"Epoch {epoch+1}/{config.n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


## 학습을 어케 진행할 것인가??
sweep_configuration = {
     'method': 'bayes',
     'name': 'sweep-bayes',         # 그냥 실험 이름 기입 ㄱㄱ
     'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
     ## 어떤 하이퍼 파라미터를 최적화??
     'parameters': {
         'batch_size': {'values': [16, 32, 64]},
         'n_epochs': {'values': [3, 5, 10]},
         'learning_rate': {'max': 0.1, 'min': 0.0001},
         'optimizer': {
             'values': ['adam', 'sgd', 'rmsprop', 'adamw']
         },
         'momentum': {'values': [0.0, 0.9]},
         'weight_decay': {'values': [0.0, 0.001, 0.0001]}
    }
}

sweep_id = wandb.sweep(
    sweep = sweep_configuration,
    entity = 'junylee00-chonnam-national-university',        # 팀 이름
    project = 'VGG_WandB_advanced'
)

wandb.agent(sweep_id, function=train_model, count=30)       # 에포크 돌릴 main 함수(train model) => 결과적으로 총 30개 모델!!

wandb.finish()