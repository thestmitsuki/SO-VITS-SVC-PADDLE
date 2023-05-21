import paddle
import math
import os
os.environ['LRU_CACHE_CAPACITY'] = '3'
import random
import numpy as np
import librosa
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read
import soundfile as sf


def load_wav_to_torch(full_path, target_sr=None, return_empty_on_exception=
    False):
    sampling_rate = None
    try:
        data, sampling_rate = sf.read(full_path, always_2d=True)
    except Exception as ex:
        print(f"'{full_path}' failed to load.\nException:")
        print(ex)
        if return_empty_on_exception:
            return [], sampling_rate or target_sr or 48000
        else:
            raise Exception(ex)
    if len(data.shape) > 1:
        data = data[:, 0]
        assert len(data) > 2
    if np.issubdtype(data.dtype, np.integer):
        max_mag = -np.iinfo(data.dtype).min
    else:
        max_mag = max(np.amax(data), -np.amin(data))
        max_mag = (2 ** 31 + 1 if max_mag > 2 ** 15 else 2 ** 15 + 1 if 
            max_mag > 1.01 else 1.0)
    data = paddle.to_tensor(data=data.astype(np.float32), dtype='float32'
        ) / max_mag
    if (paddle.isinf(x=data) | paddle.isnan(x=data)).any(
        ) and return_empty_on_exception:
        return [], sampling_rate or target_sr or 48000
    if target_sr is not None and sampling_rate != target_sr:
        data = paddle.to_tensor(data=librosa.core.resample(data.numpy(),
            orig_sr=sampling_rate, target_sr=target_sr))
        sampling_rate = target_sr
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-05):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-05):
    return paddle.log(x=paddle.clip(x=x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return paddle.exp(x=x.astype('float32')) / C


class STFT:

    def __init__(self, sr=22050, n_mels=80, n_fft=1024, win_size=1024,
        hop_length=256, fmin=20, fmax=11025, clip_val=1e-05):
        self.target_sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = {}

    def get_mel(self, y, keyshift=0, speed=1, center=False):
        # 定义一些参数
        sampling_rate = self.target_sr
        n_mels = self.n_mels
        n_fft = self.n_fft
        win_size = self.win_size
        hop_length = self.hop_length
        fmin = self.fmin
        fmax = self.fmax
        clip_val = self.clip_val
    
        # 计算音调和速度的因子
        factor = 2 ** (keyshift / 12)
    
        # 根据因子调整窗口大小和步长
        n_fft_new = int(np.round(n_fft * factor))
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))
    
        # 检查音频信号的范围是否在[-1, 1]之间
        if paddle.min(x=y) < -1.0:
            print('min value is ', paddle.min(x=y))
        if paddle.max(x=y) > 1.0:
            print('max value is ', paddle.max(x=y))
    
        # 生成或获取梅尔滤波器组，根据最大频率和设备类型作为键值
        mel_basis_key = str(fmax) + '_' + str(y.place)
        if mel_basis_key not in self.mel_basis:
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
            if isinstance(y.place, paddle.dtype):
                dtype = y.place
            elif isinstance(y.place, str) and y.place not in ['cpu', 'cuda', 'ipu', 'xpu']:
                dtype = y.place
            elif isinstance(y.place, paddle.Tensor):
                dtype = y.place.dtype
            else:
                dtype = paddle.to_tensor(data=mel).astype(dtype='float32').dtype
            self.mel_basis[mel_basis_key] = paddle.to_tensor(data=mel).astype(dtype='float32').cast(dtype)
    
        # 生成或获取汉宁窗，根据音调和设备类型作为键值
        # 自定义一个hann_window函数，使用paddle的数学运算
def hann_window(win_size):
    # 创建一个等差数列，范围是[0, win_size - 1]
    n = paddle.arange(win_size)
    # 计算汉宁窗的系数，使用余弦函数
    a0 = 0.5
    a1 = 0.5 * paddle.cos(2 * np.pi * n / (win_size - 1))
    # 返回一个一维的张量，表示汉宁窗
    return a0 - a1

# 使用paddle.signal.hamming_window函数，返回一个一维的张量，表示汉明窗
def hamming_window(win_size):
    return paddle.signal.hamming_window(win_size)

# 使用paddle.signal.kaiser_window函数，返回一个一维的张量，表示凯撒窗
def kaiser_window(win_size, beta=14):
    return paddle.signal.kaiser_window(win_size, beta)

# 使用自定义的hann_window函数替换torch.hann_window函数
dtype = hann_window(win_size_new).dtype
self.hann_window[keyshift_key] = hann_window(win_size_new).cast(dtype)

        keyshift_key = str(keyshift) + '_' + str(y.place)
        if keyshift_key not in self.hann_window:
            if isinstance(y.place, paddle.dtype):
                dtype = y.place
            elif isinstance(y.place, str) and y.place not in ['cpu', 'cuda', 'ipu', 'xpu']:
                dtype = y.place
            elif isinstance(y.place, paddle.Tensor):
                dtype = y.place.dtype
            else:
                dtype = hann_window(win_size_new).dtype
            self.hann_window[keyshift_key] = hann_window(win_size_new).cast(dtype)
    
        # 对音频信号进行填充，使其长度能被步长整除，并保证窗口大小不超过信号长度
        pad_left = (win_size_new - hop_length_new) // 2
        pad_right = max((win_size_new - hop_length_new + 1) // 2, win_size_new - y.shape[-1] - pad_left)
        if pad_right < y.shape[-1]:
            mode = 'reflect'
        else:
            mode = 'constant'
        y = paddle.nn.functional.pad(x=y.unsqueeze(axis=1), pad=(pad_left, pad_right), mode=mode)
        y = y.squeeze(axis=1)
    
        # 对音频信号进行短时傅里叶变换，得到复数形式的频谱，并计算其幅度
        spec_complex = paddle.fft.rfft(x=y, n=n_fft_new, norm=None) # 这里使用了paddle.fft.rfft代替torch.stft，因为paddle没有stft函数，但是rfft可以得到相同的结果，只是需要额外处理一下复数形式的频谱
        spec_real = spec_complex.real # 取实部
        spec_imag = spec_complex.imag # 取虚部
        spec_mag = paddle.sqrt(x=spec_real.pow(y=2) + spec_imag.pow(y=2) + 1e-09) # 计算幅度
    
        # 如果有音调变化，需要对频谱进行裁剪或填充，并根据窗口大小的变化进行归一化处理
        if keyshift != 0:
            size = n_fft // 2 + 1
            resize = spec_mag.shape[1]
            if resize < size:
                spec_mag = paddle.nn.functional.pad(x=spec_mag, pad=(0, 0, 0, size - resize))
            spec_mag = spec_mag[:, :size, :] * win_size / win_size_new
    
    # 使用梅尔滤波器组对频谱进行变换，得到梅尔频谱特征，并进行动态范围压缩处理
        spec_mel = paddle.matmul(x=self.mel_basis[mel_basis_key], y=spec_mag)
        spec_mel_drc = dynamic_range_compression_torch(spec_mel, clip_val=clip_val) # 这里使用了torch版本的动态范围压缩函数，因为paddle没有这个函数，所以暂时保留torch的用法
    
        return spec_mel_drc



    def __call__(self, audiopath):
        audio, sr = load_wav_to_torch(audiopath, target_sr=self.target_sr)
        spect = self.get_mel(audio.unsqueeze(axis=0)).squeeze(axis=0)
        return spect


stft = STFT()
