## 분산 학습 환경에서 모델 동기화, 데이터 수집 처리 등 지원
import functools
import logging
import pickle

import torch
import torch.nn as nn
import torch.distributed as dist

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_my_model(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model

def is_master_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def get_parallel_model(model, device):

    if get_world_size() >= 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True
        )
    else:
        raise NotImplementedError
    return model


def reduce_eval_results(summary, gpu):
    if get_world_size() == 1:  # Single GPU mode
        print("Single GPU detected: skipping reduce operation.")
        return [eval(summary)]  # 현재 summary를 평가 결과로 반환
    
    summary = summary + "".join([" "] * (2000-len(summary)))
    #send summary to rank 0
    summary = torch.tensor([ord(c) for c in summary]).cuda(gpu)
    summary_list = [torch.zeros_like(summary) for _ in range(dist.get_world_size())]

    dist.all_gather(summary_list, summary)
    group = []
    for _i in range(dist.get_world_size()):
        s = "".join([chr(c) for c in summary_list[_i]])
        group.append(eval(s))
    return  group



# def reduce_final_fusion_results()