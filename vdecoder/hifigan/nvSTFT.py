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
            return [], sampling_rate or target_sr or 32000
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
        return [], sampling_rate or target_sr or 32000
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

    def get_mel(self, y, center=False):
        sampling_rate = self.target_sr
        n_mels = self.n_mels
        n_fft = self.n_fft
        win_size = self.win_size
        hop_length = self.hop_length
        fmin = self.fmin
        fmax = self.fmax
        clip_val = self.clip_val
        if paddle.min(x=y) < -1.0:
            print('min value is ', paddle.min(x=y))
        if paddle.max(x=y) > 1.0:
            print('max value is ', paddle.max(x=y))
        if fmax not in self.mel_basis:
            mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=
                                 n_mels, fmin=fmin, fmax=fmax)
            if isinstance(y.place, paddle.dtype):
                dtype = y.place
            elif isinstance(y.place, str) and y.place not in ['cpu', 'cuda',
                                                              'ipu', 'xpu']:
                dtype = y.place
            elif isinstance(y.place, paddle.Tensor):
                dtype = y.place.dtype
            else:
                dtype = paddle.to_tensor(data=mel).astype(dtype='float32'
                                                          ).dtype
            self.mel_basis[str(fmax) + '_' + str(y.place)] = paddle.to_tensor(
                data=mel).astype(dtype='float32').cast(dtype)
            if isinstance(y.place, paddle.dtype):
                dtype = y.place
            elif isinstance(y.place, str) and y.place not in ['cpu', 'cuda',
                                                              'ipu', 'xpu']:
                dtype = y.place
            elif isinstance(y.place, paddle.Tensor):
                dtype = y.place.dtype
            else:
                # 修改了这一行
                dtype = paddleaudio.functional.window.hann_window(self.win_size).dtype
                # 修改了这一行
                self.hann_window[str(y.place)] = paddleaudio.functional.window.hann_window(self.win_size).cast(dtype)
        y = paddle.nn.functional.pad(x=y.unsqueeze(axis=1), pad=(int((n_fft -
                                                                      hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        y = y.squeeze(axis=1)
        # 修改了这一行
        spec = paddleaudio.functional.stft(y, n_fft=n_fft,
                                           hop_length=hop_length,
                                           win_length=win_size,
                                           window=self.hann_window[str(y.place)],
                                           center=center,
                                           normalized=False,
                                           onesided=True)
        spec = paddle.sqrt(x=spec.pow(y=2).sum(axis=-1) + 1e-09)
        spec = paddle.matmul(x=self.mel_basis[str(fmax) + '_' + str(y.place)],
                             y=spec)
        spec = dynamic_range_compression_torch(spec, clip_val=clip_val)
        return spec

    def __call__(self, audiopath):
        audio, sr = load_wav_to_torch(audiopath, target_sr=self.target_sr)
        spect = self.get_mel(audio.unsqueeze(axis=0)).squeeze(axis=0)
        return spect




stft = STFT()
