import wandb
import random
from datetime import datetime
import yaml
import math
from utils.config import setup

# args = setup('./configs/search_config_avmnist.yml')

## sweep_config.yml을 로드하여, 설정값을 WandB에 적용!!

def init_wandb(project_name, config_path):
    wandb.login()

    """WandB 프로젝트 초기화 & YaML 파일 로드하기"""
    wandb.init(project=project_name, 
               name=f"exp_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}",      # 개별 model 실험 이름 따로 저장 ㄱㄱ
            )
    
    # yaml 설정 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # WandB에 설정값을 업데이트!!
    wandb.config.update(config)

    return wandb.config



def log_wandb(evo_outer, backbones1, backbones2, parent_ooe_popu, args):
    """Fusion Network 학습 과정 로깅하기"""
    accu_b1 = [b['Acc@1'] for b in backbones1]
    accu_b2 = [b['Acc@1'] for b in backbones2]

    fusion_acc = [b['Acc@1'] for b in parent_ooe_popu if 'Acc@1' in b]
    """진화 과정 중 성능 기록"""
    wandb.log({
        "evolution_iteration": evo_outer,
        "num_backbones_selected": len(backbones1),
        "num_fusion_networks": len(parent_ooe_popu),

        # 개별 백본 정확도
        "best_accuracy": max(accu_b1 + accu_b2),
        "avg_accuracy": sum(accu_b1 + accu_b2) / (len(accu_b1) + len(accu_b2)),

        # Fusion 백본 정확도
        "best_fusion_accuracy": max(fusion_acc) if fusion_acc else None,
        "avg_fusion_accuracy": sum(fusion_acc) / len(fusion_acc) if fusion_acc else None,

        # 부모 개체 & 새로운 자식 개체 수
        "num_parents_survived": math.ceil(args.evo_search_outer.parent_popu_size * args.evo_search_inner.survival_ratio),
        "num_children_generated": len(parent_ooe_popu) - math.ceil(args.evo_search_outer.parent_popu_size * args.evo_search_inner.survival_ratio),
        
        # 교차 & 변이 확률
        "crossover_prob": args.evo_search_outer.crossover_prob,
        "mutate_prob": args.evo_search_outer.mutate_prob
    })

# def log_wandb_fusion(epoch, train_loss, train_f1, test_loss, test_f1, fusion_arch, args):
#     """Fusion Network 학습 과정 중 WandB에 기록"""
#     wandb.log({
#         "epoch": epoch,
#         "train_loss": train_loss,
#         "train_f1": train_f1,
#         "test_loss": test_loss,
#         "test_f1": test_f1,
#         "fusion_epochs": args.fusion_epochs,
#         "num_fusion_steps": args.num_input_nodes,  # Fusion Cell 개수
#         "fusion_architecture": str(fusion_arch)  # 아키텍처 구조 저장
#     })

def log_wandb_fusion(epoch, phase, loss, metric, args):
    """
    WandB에 Fusion Network 학습 로그 기록

    Args:
        epoch (int): 현재 학습 epoch
        phase (str): 학습 단계 ('train', 'dev', 'test')
        loss (float): 현재 단계의 손실 값
        metric (float): 현재 단계의 F1-score 또는 Accuracy 값
        args (Namespace): 실험 설정 값 (args.task를 사용하여 로그 키 설정)
    """
    # F1-score or Accuracy를 선택하여 기록
    metric_name = f"{phase}_f1" if args.task == 'multilabel' else f"{phase}_accuracy"

    # WandB 로그 기록
    wandb.log({
        "epoch": epoch,
        f"{phase}_loss": loss,
        metric_name: metric
    })

def finalize_wandb(best_model):
    if best_model is None:
        print("⚠ Warning: No best model found. Skipping wandb summary update.")
        wandb.finish()
        return
    """최종 결과 저장"""
    wandb.summary["best_model_accuracy"] = best_model["Acc@1"]
    wandb.summary["best_model_params"] = best_model
    wandb.finish()