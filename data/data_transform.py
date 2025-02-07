# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
## 데이터 변환 관련 코드!!

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from .auto_augment_tf import (
    auto_augment_policy,
    AutoAugment,
)

IMAGENET_PIXEL_MEAN = [123.675, 116.280, 103.530] 
IMAGENET_PIXEL_STD = [58.395, 57.12, 57.375]

## 이미지 데이터 증강 ㄱㄱ
def get_data_transform(is_training, augment,
                       train_crop_size=28, test_scale=28, test_crop_size=28, interpolation=Image.BICUBIC):

    # train_crop_size = getattr(args, 'train_crop_size', 28)
    # test_scale = getattr(args, 'test_scale', 28)
    # test_crop_size = getattr(args, 'test_crop_size', 28)

    # interpolation = Image.BICUBIC
    # if getattr(args, 'interpolation', None) and  args.interpolation == 'bilinear':
    #     interpolation = Image.BILINEAR 
    
    if interpolation == 'bilinear':
        interpolation = Image.BILINEAR

    da_args = {
        'train_crop_size': train_crop_size,
        'test_scale': test_scale,
        'test_crop_size': test_crop_size,
        'interpolation': interpolation
    }

    policy='v0'


    if augment == 'default':
        return build_default_transform(is_training, **da_args)
    elif augment == 'auto_augment_tf':
        # policy = getattr(args,  'auto_augment_policy', 'v0')
        return build_imagenet_auto_augment_tf_transform(is_training, policy=policy, **da_args)
    else:
        raise ValueError(augment)


def get_normalize():
    # normalize = transforms.Normalize(
    #     mean=torch.Tensor(IMAGENET_PIXEL_MEAN) / 255.0,
    #     std=torch.Tensor(IMAGENET_PIXEL_STD) / 255.0,
    # )
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    return normalize

## 기본 변환
def build_default_transform(
    is_training, train_crop_size=28, test_scale=28, test_crop_size=28, interpolation=Image.BICUBIC
):
    normalize = get_normalize()     # 먼저 이미지 데이터 정규화!!
    if is_training:     # 학습 데이터
        ret = transforms.Compose(
            [
                transforms.RandomResizedCrop(train_crop_size, interpolation=interpolation), # 랜덤 크롭
                transforms.RandomHorizontalFlip(),  # 좌우 뒤집기
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),  # 텐서 변환 ㄱㄱ
                normalize,
            ]
        )
    else:               # 테스트 데이터
        ret = transforms.Compose(
            [
                transforms.Resize(test_scale, interpolation=interpolation), # 크기 조정
                transforms.CenterCrop(test_crop_size),  # 중앙 크롭
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return ret

## AutoAugment 적용!! (기본 변환 + 이미지 회전, 밝기 조절, 색조 변환 등 다양한 변화 양상 추가)
def build_imagenet_auto_augment_tf_transform(
    is_training, policy='v0', train_crop_size=28, test_scale=28, test_crop_size=28, interpolation=Image.BICUBIC
):

    normalize = get_normalize()
    img_size = train_crop_size
    aa_params = {
        "translate_const": int(img_size * 0.45),
        "img_mean": tuple(round(x) for x in IMAGENET_PIXEL_MEAN),
    }

    aa_policy = AutoAugment(auto_augment_policy(policy, aa_params))

    if is_training:
        ret = transforms.Compose(
            [
                transforms.RandomResizedCrop(train_crop_size, interpolation=interpolation),
                transforms.RandomHorizontalFlip(),
                aa_policy,      # 여기서 auto augmentation 적용!!
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:       # 기본적인 변형만 적용(resize, centercrop, tensor 변환)
        ret = transforms.Compose(
            [
                transforms.Resize(test_scale, interpolation=interpolation),
                transforms.CenterCrop(test_crop_size),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return ret

