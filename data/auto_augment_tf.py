# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
## 데이터 증강에 관련된 모듈입니다아아아아아아아

""" Auto Augment
Implementation adapted from  timm: https://github.com/rwightman/pytorch-image-models
"""
import random
import math
from PIL import Image, ImageOps, ImageEnhance
import PIL


_PIL_VER = tuple([int(x) for x in PIL.__version__.split('.')[:2]])

_FILL = (128, 128, 128) # Gray

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.

_HPARAMS_DEFAULT = dict(
    translate_const=100,    # 이미지 이동시키는 픽셀 기준값
    img_mean=_FILL,         # 증강 시 사용되는 이미지의 기본적인 평균 색상값 => 즉, 어느 색상으로 채울 건가??
)

_RANDOM_INTERPOLATION = (Image.NEAREST, Image.BILINEAR, Image.BICUBIC)

## 1. 다양한 이미지 증강 연산 정의!!
def _interpolation(kwargs):
    """랜덤 보간법 설정"""
    interpolation = kwargs.pop('resample', Image.NEAREST)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation

## PIL 버전이 낮으면, fillcolor 매개변수를 제거한다
def _check_args_tf(kwargs):
    if 'fillcolor' in kwargs and _PIL_VER < (5, 0):
        kwargs.pop('fillcolor')
    kwargs['resample'] = _interpolation(kwargs)


def shear_x(img, factor, **kwargs):
    """x축 방향으로 이미지 찌그러트리기"""
    _check_args_tf(kwargs)
    # fillcolor를 kwargs에서 제거 (중복 방지)
    kwargs.pop('fillcolor', None)
    # fillcolor 값 설정 (흑백 → (0,), 컬러 → 0)
    fillcolor = (0,) if img.mode == 'L' else 0
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(img, factor, **kwargs):
    """y축 방향으로 이미지 찌그러트리기"""
    _check_args_tf(kwargs)
    # fillcolor를 kwargs에서 제거
    kwargs.pop('fillcolor', None)
    # fillcolor 값 설정 (흑백(L) → (0,), 컬러(RGB) → 0)
    fillcolor = (0,) if img.mode == 'L' else 0
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), fillcolor=fillcolor, **kwargs)


def translate_x_rel(img, pct, **kwargs):
    """x축 방향 가로 이동(pct만큼)"""
    _check_args_tf(kwargs)
    # fillcolor를 kwargs에서 제거 (중복 방지)
    kwargs.pop('fillcolor', None)
    # fillcolor 값 설정 (흑백 → (0,), 컬러 → 0)
    fillcolor = (0,) if img.mode == 'L' else 0
    pixels = pct * img.size[0]      # 이미지 높이 비율에 따라 이동할 픽셀 계산!!
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(img, pct, **kwargs):
    """y축 방향 세로 이동(pct만큼)"""
    _check_args_tf(kwargs)
    # fillcolor를 kwargs에서 제거 (중복 방지)
    kwargs.pop('fillcolor', None)
    # fillcolor 값 설정 (흑백 → (0,), 컬러 → 0)
    fillcolor = (0,) if img.mode == 'L' else 0
    pixels = pct * img.size[1]
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def translate_x_abs(img, pixels, **kwargs):
    """가로 방향, 픽셀 개수만큼 이동"""
    _check_args_tf(kwargs)
    # fillcolor를 kwargs에서 제거 (중복 방지)
    kwargs.pop('fillcolor', None)
    # fillcolor 값 설정 (흑백 → (0,), 컬러 → 0)
    fillcolor = (0,) if img.mode == 'L' else 0
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_abs(img, pixels, **kwargs):
    """세로 방향, 픽셀 개수만큼 이동"""
    _check_args_tf(kwargs)
    # fillcolor를 kwargs에서 제거 (중복 방지)
    kwargs.pop('fillcolor', None)
    # fillcolor 값 설정 (흑백 → (0,), 컬러 → 0)
    fillcolor = (0,) if img.mode == 'L' else 0
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def rotate(img, degrees, **kwargs):
    """이미지 회전(시계 or 반시계)"""
    _check_args_tf(kwargs)
    # fillcolor 값을 kwargs에서 제거
    kwargs.pop('fillcolor', None)
    # fillcolor을 올바른 형식으로 변환
    fillcolor = (0,) if img.mode == 'L' else 0

    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, fillcolor=fillcolor, **kwargs)
    elif _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
        )
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, fillcolor=fillcolor, **kwargs)
    else:
        return img.rotate(degrees, resample=kwargs.get('resample', Image.BICUBIC), fillcolor=fillcolor)


def auto_contrast(img, **__):
    """자동 대비 조절"""
    return ImageOps.autocontrast(img)


def invert(img, **__):
    """색상 반전"""
    return ImageOps.invert(img)


def equalize(img, **__):
    """히스토그램 균등화"""
    return ImageOps.equalize(img)


def solarize(img, thresh, **__):
    """thresh 값보다 밝은 픽셀만 색상 반전"""
    return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    """밝은 영역 강조(thresh 보다 밝으 픽셀에 add만큼 밝기 증가)"""
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def posterize(img, bits_to_keep, **__):
    """색상 단순화"""
    if bits_to_keep >= 8:
        return img
    bits_to_keep = max(1, bits_to_keep)  # prevent all 0 images
    return ImageOps.posterize(img, bits_to_keep)


def contrast(img, factor, **__):
    """대비 조절"""
    return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__):
    """색조 조절"""
    return ImageEnhance.Color(img).enhance(factor)


def brightness(img, factor, **__):
    """밝기 조절"""
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    """선명도 조절"""
    return ImageEnhance.Sharpness(img).enhance(factor)


def _randomly_negate(v):
    """50% 확률로 값을 음수로 변환!!"""
    return -v if random.random() > 0.5 else v


def _rotate_level_to_arg(level):
    """이미지 회전"""
    # range [-30, 30]
    level = (level / _MAX_LEVEL) * 15.
    level = _randomly_negate(level)
    return (level,)


def _enhance_level_to_arg(level):
    """대비, 색상, 선명도 등등"""
    # range [0.1, 1.9]
    return ((level / _MAX_LEVEL) * 1.4 + 0.1,)


def _shear_level_to_arg(level):
    """이미지 기울이기"""
    # range [-0.3, 0.3]
    level = (level / _MAX_LEVEL) * 0.2      # 주의) 기울임이 심하면 이미지가 찌그러짐
    level = _randomly_negate(level)
    return (level,)


def _translate_abs_level_to_arg(level, translate_const):
    """이미지 절대 이동"""
    level = (level / _MAX_LEVEL) * float(translate_const)
    level = _randomly_negate(level)
    return (level,)

def _translate_abs_level_to_arg2(level):
    """이미지 절대 이동"""
    # level = (level / _MAX_LEVEL) * float(_HPARAMS_DEFAULT['translate_const'])
    level = (level / _MAX_LEVEL) * 50  # 이동 거리 감소 (기존 250 → 50)
    level = _randomly_negate(level)
    return (level,)

def _translate_rel_level_to_arg(level):
    # range [-0.45, 0.45]
    level = (level / _MAX_LEVEL) * 0.45
    level = _randomly_negate(level)
    return (level,)


# def level_to_arg(hparams):
#     return {
#         'AutoContrast': lambda level: (),
#         'Equalize': lambda level: (),
#         'Invert': lambda level: (),
#         'Rotate': _rotate_level_to_arg,
#         # FIXME these are both different from original impl as I believe there is a bug,
#         # not sure what is the correct alternative, hence 2 options that look better
#         'Posterize': lambda level: (int((level / _MAX_LEVEL) * 4) + 4,),  # range [4, 8]
#         'Posterize2': lambda level: (4 - int((level / _MAX_LEVEL) * 4),),  # range [4, 0]
#         'Solarize': lambda level: (int((level / _MAX_LEVEL) * 256),),  # range [0, 256]
#         'SolarizeAdd': lambda level: (int((level / _MAX_LEVEL) * 110),),  # range [0, 110]
#         'Color': _enhance_level_to_arg,
#         'Contrast': _enhance_level_to_arg,
#         'Brightness': _enhance_level_to_arg,
#         'Sharpness': _enhance_level_to_arg,
#         'ShearX': _shear_level_to_arg,
#         'ShearY': _shear_level_to_arg,
#         'TranslateX': lambda level: _translate_abs_level_to_arg(level, hparams['translate_const']),
#         'TranslateY': lambda level: _translate_abs_level_to_arg(level, hparams['translate_const']),
#         'TranslateXRel': lambda level: _translate_rel_level_to_arg(level),
#         'TranslateYRel': lambda level: _translate_rel_level_to_arg(level),
#     }

## 자동 변환 이름 딕셔너리 지정!!
NAME_TO_OP = {
    'AutoContrast': auto_contrast,
    'Equalize': equalize,
    'Invert': invert,
    'Rotate': rotate,
    'Posterize': posterize,
    'Posterize2': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x_abs,
    'TranslateY': translate_y_abs,
    'TranslateXRel': translate_x_rel,
    'TranslateYRel': translate_y_rel,
}


def pass_fn(input):
    return ()


def _conversion0(input):
    return (int((input / _MAX_LEVEL) * 4) + 4,)


def _conversion1(input):
    return (4 - int((input / _MAX_LEVEL) * 4),)


def _conversion2(input):
    return (int((input / _MAX_LEVEL) * 256),)


def _conversion3(input):
    return (int((input / _MAX_LEVEL) * 110),)

##2. 랜덤하게 증강 연산 적용!!
class AutoAugmentOp:
    def __init__(self, name, prob, magnitude, hparams={}):
        self.aug_fn = NAME_TO_OP[name]      # 변환 연산 여기서 찾아!!
        # self.level_fn = level_to_arg(hparams)[name]
        if name == 'AutoContrast' or name == 'Equalize' or name == 'Invert':    # 단순 변환
            self.level_fn = pass_fn
        elif name == 'Rotate':      # 회전 각도 정하기
            self.level_fn = _rotate_level_to_arg
        elif name == 'Posterize':
            self.level_fn = _conversion0
        elif name == 'Posterize2':
            self.level_fn = _conversion1
        elif name == 'Solarize':
            self.level_fn = _conversion2
        elif name == 'SolarizeAdd':
            self.level_fn = _conversion3
        elif name == 'Color' or name == 'Contrast' or name == 'Brightness' or name == 'Sharpness':
            self.level_fn = _enhance_level_to_arg
        elif name == 'ShearX' or name == 'ShearY':
            self.level_fn = _shear_level_to_arg
        elif name == 'TranslateX' or name == 'TranslateY':      # 이미지 이동 변환 거리 정하기!!
            self.level_fn = _translate_abs_level_to_arg2
        elif name == 'TranslateXRel' or name == 'TranslateYRel':
            self.level_fn = _translate_rel_level_to_arg
        else:
            print("{} not recognized".format({}))
        self.prob = prob        # 변환 적용 확률
        self.magnitude = magnitude      # 변환 강도 설정 ㄱㄱ
        # If std deviation of magnitude is > 0, we introduce some randomness
        # in the usually fixed policy and sample magnitude from normal dist
        # with mean magnitude and std-dev of magnitude_std.
        # NOTE This is being tested as it's not in paper or reference impl.
        self.magnitude_std = 0.5  # FIXME add arg/hparam
        self.kwargs = {
            'fillcolor': hparams['img_mean'] if 'img_mean' in hparams else _FILL,
            'resample': hparams['interpolation'] if 'interpolation' in hparams else _RANDOM_INTERPOLATION
        }
    ## 실제 변환은 여기서
    def __call__(self, img):
        if self.prob < random.random():
            ## 만약 prob=0.8 이면, 80% 확률로 변환 적용, 20%는 그대로!!
            return img
        magnitude = self.magnitude
        if self.magnitude_std and self.magnitude_std > 0:
            ## 가우시안 분포 이용하여 magnitude 값을 랜덤 샘플링 ㄱㄱ
            magnitude = random.gauss(magnitude, self.magnitude_std)
        magnitude = min(_MAX_LEVEL, max(0, magnitude))
        level_args = self.level_fn(magnitude)
        return self.aug_fn(img, *level_args, **self.kwargs)     # 변환 강도 조정 후 변환 적용!!


# def auto_augment_policy_v0(hparams=_HPARAMS_DEFAULT):
#     """AutoAugment 정책 로그 함수(v0)"""
#     # ImageNet policy from TPU EfficientNet impl, cannot find
#     # a paper reference.
#     policy = [
#         [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
#         [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
#         [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
#         [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
#         [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
#         [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
#         [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
#         [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
#         [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
#         [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
#         [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
#         [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
#         [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
#         [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
#         [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
#         [('Rotate', 1.0, 7), ('TranslateYRel', 0.8, 9)],
#         [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
#         [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
#         [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
#         [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
#         [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
#         [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
#         [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
#         [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
#         [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
#     ]
#     pc = [[AutoAugmentOp(*a, hparams) for a in sp] for sp in policy]
#     return pc

def auto_augment_policy_v0(hparams=_HPARAMS_DEFAULT):
    """AutoAugment 정책 수정 버전 (이동 변환 감소 & 색 변환 조정)"""
    policy = [
        [('Equalize', 0.8, 1), ('ShearY', 0.5, 3)],  # Shear 강도 감소
        [('Color', 0.3, 6), ('Equalize', 0.7, 3)],  # Color 강도 감소
        [('Rotate', 0.6, 7), ('Equalize', 0.6, 2)],  # Rotate만 유지
        [('Solarize', 0.5, 2), ('Equalize', 0.3, 5)],  # Solarize 확률 감소
        [('Equalize', 0.8, 7), ('AutoContrast', 0.5, 3)],  
        [('ShearX', 0.2, 5), ('Rotate', 0.5, 6)],  # Shear 강도 낮춤
        [('Color', 0.5, 4), ('Equalize', 0.6, 1)],  
        [('Invert', 0.3, 5), ('Rotate', 0.6, 3)],  
        [('Equalize', 1.0, 6), ('ShearY', 0.4, 3)],  
        [('Posterize', 0.3, 4), ('AutoContrast', 0.3, 4)],  
        [('Solarize', 0.3, 6), ('Color', 0.5, 5)],  # Solarize 강도 감소
        [('Rotate', 1.0, 6), ('TranslateYRel', 0.4, 5)],  # Translate 확률 감소
    ]
    pc = [[AutoAugmentOp(*a, hparams) for a in sp] for sp in policy]
    return pc

def auto_augment_policy_original(hparams=_HPARAMS_DEFAULT):
    """AutoAugment 정책 로그 함수(original)"""
    # ImageNet policy from https://arxiv.org/abs/1805.09501
    policy = [
        [('Posterize', 0.4, 8), ('Rotate', 0.6, 9)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
        [('Posterize', 0.6, 7), ('Posterize', 0.6, 6)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
        [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
        [('Posterize', 0.8, 5), ('Equalize', 1.0, 2)],
        [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
        [('Equalize', 0.6, 8), ('Posterize', 0.4, 6)],
        [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
        [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
        [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Rotate', 0.8, 8), ('Color', 1.0, 2)],
        [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
        [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
        [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
        [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
        [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
        [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
        [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
        [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
    ]
    pc = [[AutoAugmentOp(*a, hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy(name='v0', hparams=_HPARAMS_DEFAULT):
    """자동 증강 정책 소환(hparams 값에 따라)"""
    if name == 'original':
        return auto_augment_policy_original(hparams)
    elif name == 'v0':
        return auto_augment_policy_v0(hparams)
    else:
        print("Unknown auto_augmentation policy {}".format(name))
        raise AssertionError()


class AutoAugment:

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, img):
        sub_policy = random.choice(self.policy)
        for op in sub_policy:
            img = op(img)
        return img
