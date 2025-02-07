## AVMNIST 데이터셋 처리 모듈

import torch
import numpy as np
from torch.utils.data import DataLoader
import sys

# import data_transform
from .avmnist.sound import Sound
from .avmnist.mnist import MNIST
from .avmnist.soundmnist import SoundMNIST      #본격적으로 Sound, MNIST 합친 멀티모달 데이터 다룸!!

def build_avmnist_image_loader(args):
    
    dataset_training = MNIST(root=args.dataset_dir+'mnist/', per_class_num=105, train=True)     # 학습 데이터 내, 클래스 당 105개의 sample만 사용!!
    dataset_test = MNIST(root=args.dataset_dir+'mnist/', train=False)               # 훈련 데이터 읽어들이기
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_training)       # 분산 학습 환경에서 데이터를 균등하게 분배한다(GPU) 
    else:
        train_sampler = None
    
    if args.distributed and getattr(args, 'distributed_val', True):
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = None
    
    eval_batch_size = min(args.batch_size, 16) \
        if not getattr(args, 'eval_only', False) else args.batch_size
    
    # train 데이터 불러오기
    train_loader = DataLoader(
            dataset_training, 
            batch_size=args.batch_size, 
            shuffle=(train_sampler is None), 
            sampler = train_sampler,
            drop_last = getattr(args, 'drop_last', True),
            num_workers=args.data_loader_workers_per_gpu,
            pin_memory=True
        ) 
    
    # test 데이터 불러오기
    test_loader = DataLoader(
            dataset_test, 
            batch_size=eval_batch_size, 
            shuffle=False, 
            num_workers=args.data_loader_workers_per_gpu,
            drop_last=False,
            pin_memory=True,
            sampler=test_sampler
        ) 
    print("in avmnist dataloader ",len(train_loader))
    return train_loader, test_loader, train_sampler


def build_avmnist_sound_loader(args):
    # 클래스 당 100개의 샘플 사용!!
    dataset_training = Sound(sound_root=args.dataset_dir+'sound_450/', per_class_num=100, train=True)
    dataset_test = Sound(sound_root=args.dataset_dir+'sound_450/', per_class_num=100, train=False)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_training)
    else:
        train_sampler = None
    
    if args.distributed and getattr(args, 'distributed_val', True):
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = None
    
    eval_batch_size = min(args.batch_size, 16) \
        if not getattr(args, 'eval_only', False) else args.batch_size
    
    
    train_loader = DataLoader(
            dataset_training, 
            batch_size=args.batch_size, 
            shuffle=(train_sampler is None), 
            sampler = train_sampler,
            drop_last = getattr(args, 'drop_last', True),
            num_workers=args.data_loader_workers_per_gpu,
            pin_memory=True
        ) 
        
    test_loader = DataLoader(
            dataset_test, 
            batch_size=eval_batch_size, 
            shuffle=False, 
            num_workers=args.data_loader_workers_per_gpu,
            drop_last=False,
            pin_memory=True,
            sampler=test_sampler
        ) 
    # print("in avmnist dataloader ",len(train_loader))
    return train_loader, test_loader, train_sampler


## 멀티모달 데이터 로드하기
def build_avmnist_loader(args, train_shuffle=True, flatten_audio=False, flatten_image=False, \
                         unsqueeze_channel=True, generate_sample=False, normalize_image=True, normalize_audio=True):   
    img_root = args.dataset_dir+'mnist/'
    sound_root=args.dataset_dir+'sound_450/'

    ## 한 클래스의 개수를 적게?(60) or 많게?(105)
    if(args.small_dataset):
        dataset_training = SoundMNIST(img_root, sound_root, 
                                      per_class_num=60, train=True, aug="auto_augment_tf")     # 새로 추가 ㄱㄱ
        dataset_test = SoundMNIST(img_root, sound_root, 
                                  per_class_num=60, train=False, aug="default")
    else:     
        dataset_training = SoundMNIST(img_root, sound_root, 
                                      per_class_num=105, train=True, aug="auto_augment_tf")
        dataset_test = SoundMNIST(img_root, sound_root, 
                                  per_class_num=150, train=False, aug="default")
    
    ## 분산 학습 처리
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_training)       # 멀티 GPU에 데이터 분배해서 처리 ㄱㄱ
    else: 
        train_sampler = None
    
    if args.distributed and getattr(args, 'distributed_val', True):
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = None
        
    eval_batch_size = min(args.batch_size, 16) \
        if not getattr(args, 'eval_only', False) else args.batch_size     
    
    ## 데이터로더 학습 및 생성
    train_loader = DataLoader(
            dataset_training, 
            batch_size=args.batch_size,         # 한 번에 불러올 batch size == 256  (얘도 WandB Sweep 통해서 변경 가능)
            shuffle=(train_sampler is None),    # 분산 학습 아닐 경우, 데이터 셔플링 ㄱㄱ
            sampler = train_sampler,            
            drop_last = getattr(args, 'drop_last', True),
            num_workers=args.data_loader_workers_per_gpu,       # 데이터 로드 시 사용할 CPU 스레드 수
            pin_memory=True     # 데이터를 CPU에서 GPU로 보내기
        )
    
    test_loader = DataLoader(
            dataset_test, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.data_loader_workers_per_gpu,
            drop_last=False,
            pin_memory=True,
            sampler=test_sampler
        )
    
    return train_loader, test_loader, train_sampler



