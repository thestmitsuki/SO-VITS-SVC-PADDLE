import paddle
import math
import os
import random
import numpy as np
import librosa
import librosa.util as librosa_util
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-05):
    """
    PARAMS
    ------
    C: compression factor
    """
    return paddle.log(x=paddle.clip(x=x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return paddle.exp(x=x.astype('float32')) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def spectrogram_paddle(y, n_fft, sampling_rate, hop_size, win_size, center=False
 ):
    if paddle.min(x=y) < -1.0:
        print('min value is ', paddle.min(x=y))
    if paddle.max(x=y) > 1.0:
        print('max value is ', paddle.max(x=y))
    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.place)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        # 使用 paddle.to_tensor 函数代替 torch.hann_window 函数，并转换数据类型
        hann_window[wnsize_dtype_device] = paddle.to_tensor(paddle.signal.get_window("hanning", win_size)).astype(y.dtype)
    if center:
        # 使用 paddle.nn.functional.pad 函数代替 torch.nn.functional.pad 函数，并指定 pad_mode 参数
        y = paddle.nn.functional.pad(x=y.unsqueeze(axis=1), pad=(int((n_fft -
        hop_size) / 2), int((n_fft - hop_size) / 2)), mode='reflect')
    y = y.squeeze(axis=1)
    # 使用 paddle.signal.stft 函数代替 torch.stft 函数，并指定参数名
    spec = paddle.signal.stft(y, n_fft=n_fft, hop_length=hop_size, win_length=win_size,
                              window=hann_window[wnsize_dtype_device], center=center, pad_mode=
                              'reflect', normalized=False, onesided=True)
    # 使用 paddle.pow 函数代替 torch.pow 函数，并指定参数名
    spec = paddle.sqrt(x=paddle.pow(spec, y=2).sum(axis=-1) + 1e-06)
    return spec



def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.place)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels,
            fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = paddle.to_tensor(data=mel).cast(spec
            .dtype)
    spec = paddle.matmul(x=mel_basis[fmax_dtype_device], y=spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_paddle(y, n_fft, num_mels, sampling_rate, hop_size,
 win_size, fmin, fmax, center=False):
    if paddle.min(x=y) < -1.0:
        print('min value is ', paddle.min(x=y))
    if paddle.max(x=y) > 1.0:
        print('max value is ', paddle.max(x=y))
    global mel_basis, hann_window
    dtype_device = str(y.dtype) + '_' + str(y.place)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels,
                             fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = paddle.to_tensor(data=mel).astype(y.dtype)
    if wnsize_dtype_device not in hann_window:
        # 使用 paddle.to_tensor 函数代替 torch.hann_window 函数，并转换数据类型
        hann_window[wnsize_dtype_device] = paddle.to_tensor(paddle.signal.get_window("hanning", win_size)).astype(y.dtype)
    if center:
        # 使用 paddle.nn.functional.pad 函数代替 torch.nn.functional.pad 函数，并指定 pad_mode 参数
        y = paddle.nn.functional.pad(x=y.unsqueeze(axis=1), pad=(int((n_fft -
        hop_size) / 2), int((n_fft - hop_size) / 2)), mode='reflect')
    y = y.squeeze(axis=1)
    # 使用 paddle.signal.stft 函数代替 torch.stft 函数，并指定参数名
    spec = paddle.signal.stft(y, n_fft=n_fft, hop_length=hop_size, win_length=win_size,
                              window=hann_window[wnsize_dtype_device], center=center, pad_mode=
                              'reflect', normalized=False, onesided=True)
    # 使用 paddle.pow 函数代替 torch.pow 函数，并指定参数名
    spec = paddle.sqrt(x=paddle.pow(spec, y=2).sum(axis=-1) + 1e-06)
    spec = paddle.matmul(x=mel_basis[fmax_dtype_device], y=spec) 
    spec = spectral_normalize_paddle(spec)
    return spec