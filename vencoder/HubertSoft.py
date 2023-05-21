import paddle
from vencoder.encoder import SpeechEncoder
from vencoder.hubert import hubert_model


class HubertSoft(SpeechEncoder):

    def __init__(self, vec_path='pretrain/hubert-soft-0d54a1f4.pt', device=None
        ):
        print('load model(s) from {}'.format(vec_path))
        hubert_soft = hubert_model.hubert_soft(vec_path)
        if device is None:
            self.dev = paddle.device('cuda' if paddle.device.cuda.device_count() >= 1 else 'cpu')
        else:
            self.dev = paddle.device(device)
        self.hidden_dim = 256
        if isinstance(self.dev, paddle.dtype):
            dtype = self.dev
        elif isinstance(self.dev, str) and self.dev not in ['cpu', 'cuda',
            'ipu', 'xpu']:
            dtype = self.dev
        elif isinstance(self.dev, paddle.Tensor):
            dtype = self.dev.dtype
        else:
            dtype = hubert_soft.dtype
        self.model = hubert_soft.cast(dtype)

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:
            feats = feats.mean(axis=-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats[None, None, :]
        with paddle.no_grad():
    units = self.model.units(feats)
    x = units
    perm_21 = list(range(x.ndim))
    perm_21[1] = 2
    perm_21[2] = 1
    return x.transpose(perm=perm_21)
