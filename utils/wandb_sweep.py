import wandb

# Sweep 생성하기
def create_sweep(sweep_config, entity, proj_name):
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity=entity,
        project=proj_name
    )
    return sweep_id

## 그리고 sweep의 실행 함수
def run_sweep(sweep_id, function, count):
    wandb.agent(sweep_id, function=function, count=count)