import paddle
import os
import glob
import re
import sys
import argparse
import logging
import json
import subprocess
import warnings
import random
import functools
import librosa
import numpy as np
from scipy.io.wavfile import read
from modules.commons import sequence_mask
MATPLOTLIB_FLAG = False
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging
f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def normalize_f0(f0, x_mask, uv, random_scale=True):
    uv_sum = paddle.sum(x=uv, axis=1, keepdim=True)
    uv_sum[uv_sum == 0] = 9999
    means = paddle.sum(x=f0[:, 0, :] * uv, axis=1, keepdim=True) / uv_sum
    if random_scale:
        if isinstance(f0.place, paddle.dtype):
            dtype = f0.place
        elif isinstance(f0.place, str) and f0.place not in ['cpu', 'cuda',
            'ipu', 'xpu']:
            dtype = f0.place
        elif isinstance(f0.place, paddle.Tensor):
            dtype = f0.place.dtype
        else:
            dtype = paddle.empty(shape=[f0.shape[0], 1]).uniform_(min=0.8,
                max=1.2).dtype
        factor = paddle.empty(shape=[f0.shape[0], 1]).uniform_(min=0.8, max=1.2
            ).cast(dtype)
    else:
        if isinstance(f0.place, paddle.dtype):
            dtype = f0.place
        elif isinstance(f0.place, str) and f0.place not in ['cpu', 'cuda',
            'ipu', 'xpu']:
            dtype = f0.place
        elif isinstance(f0.place, paddle.Tensor):
            dtype = f0.place.dtype
        else:
            dtype = paddle.ones(shape=[f0.shape[0], 1]).dtype
        factor = paddle.ones(shape=[f0.shape[0], 1]).cast(dtype)
    f0_norm = (f0 - means.unsqueeze(axis=-1)) * factor.unsqueeze(axis=-1)
    if paddle.isnan(x=f0_norm).any():
        exit(0)
    return f0_norm * x_mask


def plot_data_to_numpy(x, y):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use('Agg')
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 2))
    plt.plot(x)
    plt.plot(y)
    plt.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def f0_to_coarse(f0):
    is_torch = isinstance(f0, paddle.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 +
        f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (
        f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).astype(dtype='int32') if is_torch else np.rint(
        f0_mel).astype(np.int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(
        ), f0_coarse.min())
    return f0_coarse


def get_content(cmodel, y):
    with paddle.no_grad():
        c = cmodel.extract_features(y.squeeze(axis=1))[0]
    x = c
    perm_18 = list(range(x.ndim))
    perm_18[1] = 2
    perm_18[2] = 1
    c = x.transpose(perm=perm_18)
    return c


def get_f0_predictor(f0_predictor, hop_length, sampling_rate, **kargs):
    if f0_predictor == 'pm':
        from modules.F0Predictor.PMF0Predictor import PMF0Predictor
        f0_predictor_object = PMF0Predictor(hop_length=hop_length,
            sampling_rate=sampling_rate)
    elif f0_predictor == 'crepe':
        from modules.F0Predictor.CrepeF0Predictor import CrepeF0Predictor
        f0_predictor_object = CrepeF0Predictor(hop_length=hop_length,
            sampling_rate=sampling_rate, device=kargs['device'], threshold=
            kargs['threshold'])
    elif f0_predictor == 'harvest':
        from modules.F0Predictor.HarvestF0Predictor import HarvestF0Predictor
        f0_predictor_object = HarvestF0Predictor(hop_length=hop_length,
            sampling_rate=sampling_rate)
    elif f0_predictor == 'dio':
        from modules.F0Predictor.DioF0Predictor import DioF0Predictor
        f0_predictor_object = DioF0Predictor(hop_length=hop_length,
            sampling_rate=sampling_rate)
    else:
        raise Exception('Unknown f0 predictor')
    return f0_predictor_object


import paddle
import paddle.nn as nn
import paddle.optimizer as optim

def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=
 False):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = paddle.load(checkpoint_path)
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None and not skip_optimizer and checkpoint_dict[
        'optimizer'] is not None:
        optimizer.set_state_dict(state_dict=checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (saved_state_dict[
                k].shape, v.shape)
        except:
            print('error, %s is not in the checkpoint' % k)
            logger.info('%s is not in the checkpoint' % k)
        new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.set_state_dict(state_dict=new_state_dict)
    print('load ')
    logger.info("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration




def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path
    ):
    logger.info('Saving model and optimizer state at iteration {} to {}'.
        format(iteration, checkpoint_path))
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    paddle.save(obj={'model': state_dict, 'iteration': iteration,
        'optimizer': optimizer.state_dict(), 'learning_rate': learning_rate
        }, path=checkpoint_path, protocol=4)


def clean_checkpoints(path_to_models='logs/44k/', n_ckpts_to_keep=2,
    sort_by_time=True):
    """Freeing up space by deleting saved ckpts

  Arguments:
  path_to_models    --  Path to the model directory
  n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
  sort_by_time      --  True -> chronologically delete ckpts
                        False -> lexicographically delete ckpts
  """
    ckpts_files = [f for f in os.listdir(path_to_models) if os.path.isfile(
        os.path.join(path_to_models, f))]
    name_key = lambda _f: int(re.compile('._(\\d+)\\.pth').match(_f).group(1))
    time_key = lambda _f: os.path.getmtime(os.path.join(path_to_models, _f))
    sort_key = time_key if sort_by_time else name_key
    x_sorted = lambda _x: sorted([f for f in ckpts_files if f.startswith(_x
        ) and not f.endswith('_0.pth')], key=sort_key)
    to_del = [os.path.join(path_to_models, fn) for fn in x_sorted('G')[:-
        n_ckpts_to_keep] + x_sorted('D')[:-n_ckpts_to_keep]]
    del_info = lambda fn: logger.info(f'.. Free up space by deleting ckpt {fn}'
        )
    del_routine = lambda x: [os.remove(x), del_info(x)]
    rs = [del_routine(fn) for fn in to_del]


def summarize(writer, global_step, scalars={}, histograms={}, images={},
    audios={}, audio_sampling_rate=22050):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats='HWC')
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex='G_*.pth'):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x


import paddle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect='auto', origin='lower',
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel('Frames')
    plt.ylabel('Channels')
    plt.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    """Class Method: *.reshape, not convert, please check whether it is paddle.Tensor.*/Optimizer.*/nn.Layer.*, and convert manually"""
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


import paddle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    """Class Method: *.reshape, not convert, please check whether it is paddle.Tensor.*/Optimizer.*/nn.Layer.*, and convert manually"""
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data



def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return paddle.to_tensor(data=data.astype(np.float32), dtype='float32'
        ), sampling_rate


def load_filepaths_and_text(filename, split='|'):
    with open(filename, encoding='utf-8') as f:
        """Class Method: *.split, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=
        './configs/config.json', help='JSON file for configuration')
    parser.add_argument('-m', '--model', type=str, required=True, help=
        'Model name')
    args = parser.parse_args()
    model_dir = os.path.join('./logs', args.model)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    config_path = args.config
    config_save_path = os.path.join(model_dir, 'config.json')
    if init:
        with open(config_path, 'r') as f:
            data = f.read()
        with open(config_save_path, 'w') as f:
            f.write(data)
    else:
        with open(config_save_path, 'r') as f:
            data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, 'config.json')
    with open(config_save_path, 'r') as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path):
    with open(config_path, 'r') as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    return hparams


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, '.git')):
        logger.warn(
            '{} is not a git repository, therefore hash value comparison will be ignored.'
            .format(source_dir))
        return
    cur_hash = subprocess.getoutput('git rev-parse HEAD')
    path = os.path.join(model_dir, 'githash')
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn(
                'git hash values are different. {}(saved) != {}(current)'.
                format(saved_hash[:8], cur_hash[:8]))
    else:
        open(path, 'w').write(cur_hash)


def get_logger(model_dir, filename='train.log'):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


def repeat_expand_2d(content, target_len):
    src_len = content.shape[-1]
    if isinstance(content.place, paddle.dtype):
        dtype = content.place
    elif isinstance(content.place, str) and content.place not in ['cpu',
        'cuda', 'ipu', 'xpu']:
        dtype = content.place
    elif isinstance(content.place, paddle.Tensor):
        dtype = content.place.dtype
    else:
        dtype = paddle.zeros(shape=[content.shape[0], target_len], dtype=
            'float32').dtype
    target = paddle.zeros(shape=[content.shape[0], target_len], dtype='float32'
        ).cast(dtype)
    temp = paddle.arange(end=src_len + 1) * target_len / src_len
    current_pos = 0
    for i in range(target_len):
        if i < temp[current_pos + 1]:
            target[:, i] = content[:, current_pos]
        else:
            current_pos += 1
            target[:, i] = content[:, current_pos]
    return target


def mix_model(model_paths, mix_rate, mode):
    mix_rate = paddle.to_tensor(data=mix_rate, dtype='float32') / 100
    model_tem = paddle.load(path=model_paths[0])
    models = [paddle.load(path=path)['model'] for path in model_paths]
    if mode == 0:
        mix_rate = paddle.nn.functional.softmax(x=mix_rate, axis=0)
    for k in model_tem['model'].keys():
        model_tem['model'][k] = paddle.zeros_like(x=model_tem['model'][k])
        for i, model in enumerate(models):
            model_tem['model'][k] += model[k] * mix_rate[i]
    paddle.save(obj=model_tem, path=os.path.join(os.path.curdir,
        'output.pth'), protocol=4)
    return os.path.join(os.path.curdir, 'output.pth')


class HParams:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
