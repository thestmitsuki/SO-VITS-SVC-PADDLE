import paddle
import numpy as np
from vdecoder.nsf_hifigan.nvSTFT import STFT
from vdecoder.nsf_hifigan.models import load_model
from torchaudio.transforms import Resample


class Enhancer:

    def __init__(self, enhancer_type, enhancer_ckpt, device=None):
        if device is None:
            device = 'cuda' if paddle.device.cuda.device_count(
                ) >= 1 else 'cpu'
        self.place = device
        if enhancer_type == 'nsf-hifigan':
            self.enhancer = NsfHifiGAN(enhancer_ckpt, device=self.place)
        else:
            raise ValueError(f' [x] Unknown enhancer: {enhancer_type}')
        self.resample_kernel = {}
        self.enhancer_sample_rate = self.enhancer.sample_rate()
        self.enhancer_hop_size = self.enhancer.hop_size()

    def enhance(self, audio, sample_rate, f0, hop_size, adaptive_key=0,
        silence_front=0):
        start_frame = int(silence_front * sample_rate / hop_size)
        real_silence_front = start_frame * hop_size / sample_rate
        audio = audio[:, int(np.round(real_silence_front * sample_rate)):]
        f0 = f0[:, start_frame:, :]
        adaptive_factor = 2 ** (-adaptive_key / 12)
        adaptive_sample_rate = 100 * int(np.round(self.enhancer_sample_rate /
            adaptive_factor / 100))
        real_factor = self.enhancer_sample_rate / adaptive_sample_rate
        if sample_rate == adaptive_sample_rate:
            audio_res = audio
        else:
            key_str = str(sample_rate) + str(adaptive_sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate,
                    adaptive_sample_rate, lowpass_filter_width=128).to(self
                    .place)
            audio_res = self.resample_kernel[key_str](audio)
        n_frames = int(audio_res.shape[-1] // self.enhancer_hop_size + 1)
        f0_np = f0.squeeze(axis=0).squeeze(axis=-1).cpu().numpy()
        f0_np *= real_factor
        time_org = hop_size / sample_rate * np.arange(len(f0_np)) / real_factor
        time_frame = (self.enhancer_hop_size / self.enhancer_sample_rate *
            np.arange(n_frames))
        f0_res = np.interp(time_frame, time_org, f0_np, left=f0_np[0],
            right=f0_np[-1])
        if isinstance(self.place, paddle.dtype):
            dtype = self.place
        elif isinstance(self.place, str) and self.place not in ['cpu',
            'cuda', 'ipu', 'xpu']:
            dtype = self.place
        elif isinstance(self.place, paddle.Tensor):
            dtype = self.place.dtype
        else:
            dtype = paddle.to_tensor(data=f0_res).unsqueeze(axis=0).astype(
                dtype='float32').dtype
        f0_res = paddle.to_tensor(data=f0_res).unsqueeze(axis=0).astype(dtype
            ='float32').cast(dtype)
        enhanced_audio, enhancer_sample_rate = self.enhancer(audio_res, f0_res)
        if adaptive_factor != 0:
            key_str = str(adaptive_sample_rate) + str(enhancer_sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(adaptive_sample_rate,
                    enhancer_sample_rate, lowpass_filter_width=128).to(self
                    .place)
            enhanced_audio = self.resample_kernel[key_str](enhanced_audio)
        if start_frame > 0:
            enhanced_audio = paddle.nn.functional.pad(x=enhanced_audio, pad
                =(int(np.round(enhancer_sample_rate * real_silence_front)), 0))
        return enhanced_audio, enhancer_sample_rate


class NsfHifiGAN(paddle.nn.Layer):

    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = 'cuda' if paddle.device.cuda.device_count(
            ) >= 1 else 'cpu'
        self.place = device
        print('| Load HifiGAN: ', model_path)
        self.model, self.h = load_model(model_path, device=self.place)

    def sample_rate(self):
        return self.h.sampling_rate

    def hop_size(self):
        return self.h.hop_size

    def forward(self, audio, f0):
        stft = STFT(self.h.sampling_rate, self.h.num_mels, self.h.n_fft,
                    self.h.win_size, self.h.hop_size, self.h.fmin, self.h.fmax)
        with paddle.no_grad():
            mel = stft.get_mel(audio)
            # 使用 reshape 函数代替 view 函数，注意保持内存布局不变
            enhanced_audio = self.model(mel, f0[:, :mel.shape[-1]]).reshape([-1])
        return enhanced_audio, self.h.sampling_rate

