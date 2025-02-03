import wandb
import random
from datetime import datetime
import yaml
from utils.config import setup

# args = setup('./configs/search_config_avmnist.yml')

## sweep_config.yml을 로드하여, 설정값을 WandB에 적용!!

def init_wandb(project_name='harmonicnas_avmnist', config_path='./configs/sweep_config.yml'):
    """WandB 프로젝트 초기화 & YaML 파일 로드하기"""
    wandb.init(project=project_name, 
               name=f"exp_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}",
            )
    
    # yaml 설정 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # WandB에 설정값을 업데이트!!
    wandb.config.update(config)

    return wandb.config



def log_wandb(evo_outer, backbones1, backbones2, parent_ooe_popu, args):
    """진화 과정 중 성능 기록"""
    wandb.log({
        "evolution_iteration": evo_outer,
        "num_backbones_selected": len(backbones1),
        "num_fusion_networks": len(parent_ooe_popu),
        "best_accuracy": max([b['Acc@1'] for b in backbones1] + [b['Acc@1'] for b in backbones2]),
        "avg_accuracy": sum([b['Acc@1'] for b in backbones1] + [b['Acc@1'] for b in backbones2]) / (len(backbones1) + len(backbones2)),
        "crossover_prob": args.evo_search_outer.crossover_prob,
        "mutate_prob": args.evo_search_outer.mutate_prob
    })


def finalize_wandb(best_model):
    """최종 결과 저장"""
    wandb.summary["best_model_accuracy"] = best_model["Acc@1"]
    wandb.summary["best_model_params"] = best_model
    wandb.finish()