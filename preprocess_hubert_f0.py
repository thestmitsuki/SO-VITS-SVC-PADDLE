import paddle
import math
import multiprocessing
import os
import argparse
from random import shuffle
from glob import glob
from tqdm import tqdm
from modules.mel_processing import spectrogram_torch
import json
import utils
import logging
logging.getLogger('numba').setLevel(logging.WARNING)
import librosa
import numpy as np
hps = utils.get_hparams_from_file('configs/config.json')
sampling_rate = hps.data.sampling_rate
hop_length = hps.data.hop_length
speech_encoder = hps['model']['speech_encoder']


def process_one(filename, hmodel, f0p):
    wav, sr = librosa.load(filename, sr=sampling_rate)
    soft_path = filename + '.soft.pt'
    if not os.path.exists(soft_path):
        device = paddle.set_device('gpu' if paddle.device.cuda.device_count() >=
            1 else 'cpu')
        wav16k = librosa.resample(wav, orig_sr=sampling_rate, target_sr=16000)
        if isinstance(device, paddle.dtype):
            dtype = device
        elif isinstance(device, str) and device not in ['cpu', 'cuda',
            'ipu', 'xpu']:
            dtype = device
        elif isinstance(device, paddle.Tensor):
            dtype = device.dtype
        else:
            dtype = paddle.to_tensor(data=wav16k).dtype
        wav16k = paddle.to_tensor(data=wav16k).cast(dtype)
        c = hmodel.encoder(wav16k)
        paddle.save(obj=c.cpu(), path=soft_path, protocol=4)
    f0_path = filename + '.f0.npy'
    if not os.path.exists(f0_path):
        f0_predictor = utils.get_f0_predictor(f0p, sampling_rate=
            sampling_rate, hop_length=hop_length, device=None, threshold=0.05)
        f0, uv = f0_predictor.compute_f0_uv(wav)
        np.save(f0_path, np.asanyarray((f0, uv), dtype=object))
    spec_path = filename.replace('.wav', '.spec.pt')
    if not os.path.exists(spec_path):
        audio, sr = utils.load_wav_to_torch(filename)
        if sr != hps.data.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(sr,
                hps.data.sampling_rate))
        audio_norm = audio / hps.data.max_wav_value
        audio_norm = audio_norm.unsqueeze(axis=0)
        spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.
            data.sampling_rate, hps.data.hop_length, hps.data.win_length,
            center=False)
        spec = paddle.squeeze(x=spec, axis=0)
        paddle.save(obj=spec, path=spec_path, protocol=4)


def process_batch(filenames, f0p):
    print('Loading hubert for content...')
    device = 'cuda' if paddle.device.cuda.device_count() >= 1 else 'cpu'
    hmodel = utils.get_speech_encoder(speech_encoder, device=device)
    print('Loaded hubert.')
    for filename in tqdm(filenames):
        process_one(filename, hmodel, f0p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='dataset/44k', help=
        'path to input dir')
    parser.add_argument('--f0_predictor', type=str, default='dio', help=
        'Select F0 predictor, can select crepe,pm,dio,harvest, default pm(note: crepe is original F0 using mean filter)'
        )
    args = parser.parse_args()
    f0p = args.f0_predictor
    print(speech_encoder)
    print(f0p)
    filenames = glob(f'{args.in_dir}/*/*.wav', recursive=True)
    shuffle(filenames)
    multiprocessing.set_start_method('spawn', force=True)
    num_processes = 1
    chunk_size = int(math.ceil(len(filenames) / num_processes))
    chunks = [filenames[i:i + chunk_size] for i in range(0, len(filenames),
        chunk_size)]
    print([len(c) for c in chunks])
    processes = [multiprocessing.Process(target=process_batch, args=(chunk,
        f0p)) for chunk in chunks]
    for p in processes:
        p.start()
