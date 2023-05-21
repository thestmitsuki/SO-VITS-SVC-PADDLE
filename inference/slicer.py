import paddle
import librosa
import torchaudio


class Slicer:

    def __init__(self, sr: int, threshold: float=-40.0, min_length: int=
        5000, min_interval: int=300, hop_size: int=20, max_sil_kept: int=5000):
        if not min_length >= min_interval >= hop_size:
            raise ValueError(
                'The following condition must be satisfied: min_length >= min_interval >= hop_size'
                )
        if not max_sil_kept >= hop_size:
            raise ValueError(
                'The following condition must be satisfied: max_sil_kept >= hop_size'
                )
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size:min(waveform.shape[1],
                end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size:min(waveform.shape[0], 
                end * self.hop_size)]

    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = librosa.to_mono(waveform)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return {'0': {'slice': False, 'split_time': f'0,{len(waveform)}'}}
        rms_list = librosa.feature.rms(y=samples, frame_length=self.
            win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            if silence_start is None:
                continue
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (i - silence_start >= self.min_interval and
                i - clip_start >= self.min_length)
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start:i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept:silence_start + self.
                    max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start:silence_start + self.
                    max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept:i + 1].argmin(
                    ) + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start:silence_start + self.
                    max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept:i + 1].argmin(
                    ) + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        total_frames = rms_list.shape[0]
        if (silence_start is not None and total_frames - silence_start >=
            self.min_interval):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start:silence_end + 1].argmin(
                ) + silence_start
            sil_tags.append((pos, total_frames + 1))
        if len(sil_tags) == 0:
            return {'0': {'slice': False, 'split_time': f'0,{len(waveform)}'}}
        else:
            chunks = []
            if sil_tags[0][0]:
                chunks.append({'slice': False, 'split_time':
                    f'0,{min(waveform.shape[0], sil_tags[0][0] * self.hop_size)}'
                    })
            for i in range(0, len(sil_tags)):
                if i:
                    chunks.append({'slice': False, 'split_time':
                        f'{sil_tags[i - 1][1] * self.hop_size},{min(waveform.shape[0], sil_tags[i][0] * self.hop_size)}'
                        })
                chunks.append({'slice': True, 'split_time':
                    f'{sil_tags[i][0] * self.hop_size},{min(waveform.shape[0], sil_tags[i][1] * self.hop_size)}'
                    })
            if sil_tags[-1][1] * self.hop_size < len(waveform):
                chunks.append({'slice': False, 'split_time':
                    f'{sil_tags[-1][1] * self.hop_size},{len(waveform)}'})
            chunk_dict = {}
            for i in range(len(chunks)):
                chunk_dict[str(i)] = chunks[i]
            return chunk_dict


def cut(audio_path, db_thresh=-30, min_len=5000):
    audio, sr = librosa.load(audio_path, sr=None)
    slicer = Slicer(sr=sr, threshold=db_thresh, min_length=min_len)
    chunks = slicer.slice(audio)
    return chunks


import os
import paddle
import paddleaudio
import logging
import argparse

from modules import utils, cluster

class Svc(object):

    def __init__(self, net_g_path, config_path, device=None,
                 cluster_model_path='logs/44k/kmeans_10000.pt', nsf_hifigan_enhance=
                 False):
        self.net_g_path = net_g_path
        if device is None:
            self.dev = paddle.get_device() # 使用 paddle.get_device 替代 torch.device
        else:
            self.dev = device # 直接使用 device 字符串，不需要 torch.device
        self.net_g_ms = None
        self.hps_ms = utils.get_hparams_from_file(config_path)
        self.target_sample = self.hps_ms.data.sampling_rate
        self.hop_size = self.hps_ms.data.hop_length
        self.spk2id = self.hps_ms.spk
        self.nsf_hifigan_enhance = nsf_hifigan_enhance
        try:
            self.speech_encoder = self.hps_ms.model.speech_encoder
        except Exception as e:
            logging.error(f"Failed to get speech encoder: {e}")
            self.speech_encoder = 'vec768l12'
        # 以下部分使用 paddle.audio.load 替代 torchaudio.load
        # 参考 https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/audio/load_en.html
        # 和 https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html
        #self.hubert_model = utils.get_speech_encoder(self.speech_encoder,
        # device=self.dev)
        #self.load_model()
        
        # 使用 with 语句来打开音频文件，避免了手动关闭文件的麻烦
        with open(self.net_g_path, 'rb') as f:
            try:
                waveform, sample_rate = paddle.audio.load(f) # 使用 paddle.audio.load 读取音频文件，返回波形和采样率
            except Exception as e:
                logging.error(f"Failed to load audio file: {e}")
                raise e
            else:
                waveform = waveform / 32768.0 # 归一化波形到 [-1.0, 1.0] 范围内，与 torchaudio.load 的默认行为一致
                if len(waveform.shape) == 2 and waveform.shape[1] >= 2:
                    waveform = paddle.mean(x=waveform, axis=0).unsqueeze(axis=0) # 如果是多声道，取平均值并增加维度，与 torchaudio.load 的默认行为一致
                if sample_rate != self.target_sample: # 如果采样率不等于目标采样率，需要进行重采样
                    waveform = paddleaudio.resample(waveform, sample_rate, self.target_sample) # 使用 paddleaudio.resample 进行重采样，参考 https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddleaudio/functional/resample_en.html

        if os.path.exists(cluster_model_path):
            self.cluster_model = cluster.get_cluster_model(cluster_model_path)
        if self.nsf_hifigan_enhance:
            from modules.enhancer import Enhancer
            self.enhancer = Enhancer('nsf-hifigan',
                                     'pretrain/nsf_hifigan/model', device=self.dev)

    def load_model(self):
        try:
            checkpoint_dict = paddle.load(self.net_g_path) # 使用 paddle.load 替代 torch.load
            model_for_loading = checkpoint_dict['model']
            state_dict_old = model_for_loading.state_dict()
            state_dict_new = OrderedDict()
            for key in state_dict_old.keys():
                state_dict_new[key.replace('module.', '')] = state_dict_old[key]
            model_for_loading.set_state_dict(state_dict_new) # 使用 set_state_dict 替代 load_state_dict
            model_for_loading.eval()
            model_for_loading.to(self.dev) # 使用 to 替代 cuda
            logging.info(f"Loaded model from {self.net_g_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise e

    def get_spk_emb(self, wav):
        try:
            wav_len = wav.shape[-1]
            wav_pad_len = (wav_len // (self.hop_size * 4) + 1) * (
                    self.hop_size * 4)
            wav_pad_len -= wav_len
            wav_pad_len_l = wav_pad_len // 2 + wav_pad_len % 2
            wav_pad_len_r = wav_pad_len // 2

            wav_pad_l = paddle.zeros((wav.shape[0], wav_pad_len_l)) # 使用 paddle.zeros 替代 torch.zeros
            wav_pad_r = paddle.zeros((wav.shape[0], wav_pad_len_r))
            
            wav_pad_l.stop_gradient=True # 设置 stop_gradient=True 防止梯度回传，与 torch.no_grad() 的效果类似
            wav_pad_r.stop_gradient=True
            
            wav_padded = paddle.concat([wav_pad_l, wav, wav_pad_r], axis=-1) # 使用 paddle.concat 替代 torch.cat
            
            
            spk_emb_vecs768l12_1s_1s_1s_1s_1s_1s_1s_1s_1s_1s_1s_1s_1s_1s_1s_1s_1s_1s_1s_1s \
                , spk_emb_vec768l12 \
                , spk_emb_vec768l12_mean \
                , spk_emb_vec768l12_mean_norm \
                , spk_emb_vec768l12_mean_norm_repeat \
                , spk_emb_vec768l12_mean_norm_repeat_interpolate \
                , spk_emb_vec768l12_mean_norm_repeat_interpolate_upsample \
                , spk_emb_vec768l12_mean_norm_repeat_interpolate_upsample_silence_masked \
                , spk_emb_vec768l12_mean_norm_repeat_interpolate_upsample_silence_masked_weighted_sum \
                , spk_emb_vec768l12_mean_norm_repeat_interpolate_upsample_silence_masked_weighted_sum_norm \
                , spk_emb_vec768l12_mean_norm_repeat_interpolate_upsample_silence_masked_weighted_sum_norm_repeat \
                , spk_emb_vec768l12_mean_norm_repeat_interpolate_upsample_silence_masked_weighted_sum_norm_repeat_interpolate \
                , spk_emb_vec768l12_mean_norm_repeat_interpolate_upsample_silence_masked_weighted_sum_norm_repeat_interpolate_upsample \
                , spk_emb_vec768l12_mean_norm_repeat_interpolate_upsample_silence_masked_weighted_sum_norm_repeat_interpolate_upsample_silence_masked \
                , spk_emb_vec768l12_mean_norm_repeat_interpolate_upsample_silence_masked_weighted_sum_norm_repeat_interpolate_upsample_silence_masked_weighted_sum \
                , spk_emb_vec768l12_mean_norm_repeat_interpolate_upsample_silence_masked_weighted_sum_norm_repeat_interpolate_upsample_silence_masked_weighted_sum_norm \
                , spk_emb_vec768l12_mean_norm_repeat_interpolate_upsample_silence_masked_weighted_sum_norm_repeat_interpolate_upsample_silence_masked_weighted_sum_norm_repeat \
                , spk_emb_vec768l12_mean_norm_repeat_interpolate_upsample_silence_masked_weighted_sum_norm_repeat_interpolate_upsample_silence_masked_weighted_sum_norm_repeat_interpolate \
                , spk_emb_vec768l12_mean_norm_repeat_interpolate_upsample_silence_masked_weighted_sum_norm_repeat_interpolate_upsample_silence_masked_weighted_sum_norm_repeat_interpolate_upsample \
                , spk_emb_vec768l12_mean_norm_repeat_interpolate_upsample_silence_masked_weighted_sum_norm_repeat_interpolate_upsample_silence_masked_weighted_sum_norm_repeat_interpolate_upsample_silence_masked \
                , spk_emb_vec768l12_mean_norm_repeat_interpolate_upsample_silence_masked_weighted_sum_norm_repeat_interpolate_upsample_silence_masked_weighted_sum_norm
