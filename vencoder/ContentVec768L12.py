import paddle
from vencoder.encoder import SpeechEncoder
from fairseq import checkpoint_utils


class ContentVec768L12(SpeechEncoder):

    def __init__(self, vec_path='pretrain/checkpoint_best_legacy_500.pt',
        device=None):
        print('load model(s) from {}'.format(vec_path))
        self.hidden_dim = 768
        models, saved_cfg, task = (checkpoint_utils.
            load_model_ensemble_and_task([vec_path], suffix=''))
        if device is None:
            self.dev = paddle.device('cuda' if paddle.device.cuda.device_count() >= 1 else 'cpu')
        else:
            self.dev = paddle.device(device)
        if isinstance(self.dev, paddle.dtype):
            dtype = self.dev
        elif isinstance(self.dev, str) and self.dev not in ['cpu', 'cuda',
            'ipu', 'xpu']:
            dtype = self.dev
        elif isinstance(self.dev, paddle.Tensor):
            dtype = self.dev.dtype
        else:
            dtype = models[0].dtype
        self.model = models[0].cast(dtype)
        self.model.eval()

    def encoder(self, wav):
        feats = wav
        if feats.dim() == 2:
            feats = feats.mean(axis=-1)
        assert feats.dim() == 1, feats.dim()
        """Class Method: *.view, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        feats = paddle.reshape(feats, [1, -1])
        # 第一处：torch.BoolTensor改为paddle.Tensor，并指定dtype为paddle.bool
        padding_mask = paddle.Tensor(feats.shape, dtype=paddle.bool).fill_(value=False)
        if isinstance(wav.place, paddle.dtype):
            dtype = wav.place
        elif isinstance(wav.place, str) and wav.place not in ['cpu', 'cuda',
            'ipu', 'xpu']:
            dtype = wav.place
        elif isinstance(wav.place, paddle.Tensor):
            dtype = wav.place.dtype
        else:
            dtype = feats.dtype
        if isinstance(wav.place, paddle.dtype):
            dtype = wav.place
        elif isinstance(wav.place, str) and wav.place not in ['cpu', 'cuda',
            'ipu', 'xpu']:
            dtype = wav.place
        elif isinstance(wav.place, paddle.Tensor):
            dtype = wav.place.dtype
        else:
            dtype = padding_mask.dtype
        inputs = {'source': feats.cast(dtype), 'padding_mask': padding_mask
            .cast(dtype), 'output_layer': 12}
        with paddle.no_grad():
            logits = self.model.extract_features(**inputs)
        x = logits[0]
        perm_19 = list(range(x.ndim))
        perm_19[1] = 2
        perm_19[2] = 1
        return x.transpose(perm=perm_19)
