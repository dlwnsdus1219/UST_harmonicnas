# Wav 파일을 MFCC로 바꿔라!!(ML 모델에 학습시키기 위해)

import numpy as np
import librosa      # for 오디오 신호 처리
import os
from PIL import Image
from ..sound_augment import apply_augmentation
# from keras.utils import to_categorical

# 오디오 파일의 MFCC 변환
def wav2mfcc(file_path, max_pad_len=20, apply_aug=False):
    wave, sr = librosa.load(file_path, mono=True, sr=None)      # 파형, 샘플링 레이트 반환하기(mono는 1채널, sr=None은 원래 샘플링 레이트 유지)
    
    wave = np.asfortranarray(wave[::3])     # 3개 샘플 중 하나만 추출 -> sr 줄이기!! -> 계산량 줄이고 처리량 높이기
    
    # Train 데이터만 Audio 증강 적용!!(Test 데이터는 ㄴㄴ)
    if apply_aug:
        wave = apply_augmentation(wave, sr)

    
    mfcc = librosa.feature.mfcc(y=wave, sr=8000, n_mfcc=20, n_fft=256)   # 본격적으로 MFCC 추출(샘플링 레이트 8000Hz, 20개의 MFCC 계수)
    # n_fft: 윈도우 크기 설정(입력 신호의 길이에 맞게끔)
    ## n_fft를 줄이면, 시간 해상도 up -> 시간 프레임 개수 늘어남, 주파수 해상도 down -> 주파수별 표현이 덜 세밀

    ## 패딩(MFCC 특징 길이 고정)
    pad_width = max_pad_len - mfcc.shape[1]
  
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')    # 2번째 차원(시간 축) 방향으로 0을 패딩해 준다
    return mfcc

def get_data(root):

    labels = []
    mfccs = []

    for f in os.listdir(root):
        if f.endswith('.wav'):
            # MFCC
            mfccs.append(wav2mfcc(root + f))

            # List of labels
            label = f.split('_')[0]
            labels.append(label)

    return np.asarray(mfccs), labels

if __name__ == '__main__':
    dir = '/home/etri01/jy/harmonicnas/HNAS_AVMNIST/soundmnist/'
    root = dir+'sound/0/'
    mfccs, labels = get_data(root)
    print(mfccs[0])
    print(len(labels))      # 150
    print(mfccs[0].shape)   # (20, 20)


# MFCC 구조: 2차원 행렬(행 - MFCC 계수(n_mfcc=20), 열 - 시간 축 방향으로 패딩(max_pad_len=20))