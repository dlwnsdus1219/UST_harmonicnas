import wandb

# # Sweep 설정 정의하기
# sweep_config = {
#     'method': 'bayes',
#     'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
#     'parameters': {
#         ## 진화 알고리즘 관련 파라미터
#         'evo_parent_popu_size': {'values': [6, 8, 10]},
#         "evo_survival_ratio": {"values": [0.3, 0.5, 0.7]},
#         "evo_mutate_size": {"values": [32, 64, 128]},
#         "evo_crossover_size": {"values": [32, 64, 128]},
#         "evo_mutate_prob": {"values": [0.2, 0.3, 0.4]},
#         "evo_crossover_prob": {"values": [0.7, 0.8, 0.9]},
#         "evo_iter": {"values": [30, 40, 50]},

#         "inner_survival_ratio": {"values": [0.2, 0.25, 0.3]},
#         ## 모델 학습 관련 파라미터
#         'arch_learning_rate': {'values': [0.001, 0.003, 0.005]},
#         'arch_weight_decay': {'values': [0.001, 0.005, 0.0001]}, 
#         'weight_decay': {'values': [0.0001, 0.0005, 0.001]},
#         'dropout': {'min': 0.2, 'max': 0.5},
#         #### 파라미터 더 이어서 써 주세요 ####
#     }
# }

# Sweep 생성하기
def create_sweep(sweep_config, entity, proj_name):
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity=entity,
        project=proj_name
    )
    return sweep_id

## 그리고 sweep의 실행 함수
def run_sweep(sweep_id, train_fn, count):
    wandb.agent(sweep_id, function=train_fn, count=count)