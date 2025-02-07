import random
import numpy as np
import librosa
import pydub

from scipy.signal import butter, filtfilt

def add_white_noise(audio, noise_level):
    """오디오에 화이트 노이즈를 추가하여 일반화 성능 Up!!"""
    noise = np.random.randn(len(audio))
    augmented_radio = audio + noise_level * noise
    return np.clip(augmented_radio, -1.0, 1.0)

def pitch_shift(audio, sr, n_steps=2):
    """오디오의 피치를 높이거나 낮추는 변환"""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def time_stretch(audio, sr, rate):
    """ 오디오 속도를 빠르게 or 느리게 변환 (길이 보정 포함) """
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    if len(stretched) > len(audio):
        stretched = stretched[:len(audio)]
    else:
        stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
    return stretched

# def change_volume(audio_path, db_change=5):
#     """ 오디오 볼륨을 증가시키거나 감소 """
#     sound = pydub.AudioSegment.from_file(audio_path)
#     sound = sound + db_change  # 볼륨 증가 (dB)
#     samples = np.array(sound.get_array_of_samples()).astype(np.float32) / 32768.0
#     return samples, sound.frame_rate  # numpy 배열과 샘플링 레이트 반환

# def reverse_audio(audio_path):
#     """ 오디오를 뒤집어서 numpy array로 변환 """
#     sound = pydub.AudioSegment.from_file(audio_path)
#     sound = sound.reverse()
#     samples = np.array(sound.get_array_of_samples()).astype(np.float32) / 32768.0
#     return samples, sound.frame_rate

def lowpass_filter(audio, cutoff=1000, sr=16000, order=5):
    """저주파 필터링(낮은 주파수만 남겨요, 위상 유지)"""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, audio)

def apply_augmentation(audio, sr, 
                    #    audio_path = None
                       ):
    """랜덤 오디오 증강 함수"""
    augmentations = [
        lambda x: add_white_noise(x, noise_level=0.005),
        lambda x: pitch_shift(x, sr, n_steps=random.choice([-2, 2])),
        lambda x: time_stretch(x, sr, rate=random.uniform(0.8, 1.2)),
        lambda x: lowpass_filter(x, cutoff=random.choice([500, 1000, 1500]), sr=sr),
    ]
    # # `change_volume`, `reverse_audio`는 파일 경로가 필요하므로 따로 처리
    # if audio_path:
    #     augmentations.extend([
    #         lambda x: change_volume(audio_path, db_change=random.choice([-5, 5]))[0],
    #         lambda x: reverse_audio(audio_path)[0]
    #     ])
    aug_func = random.choice(augmentations)
    return aug_func(audio)