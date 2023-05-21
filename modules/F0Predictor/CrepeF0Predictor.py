import paddle
from modules.F0Predictor.F0Predictor import F0Predictor
from modules.F0Predictor.crepe import CrepePitchExtractor


class CrepeF0Predictor(F0Predictor):

    def __init__(self, hop_length=512, f0_min=50, f0_max=1100, device=None,
        sampling_rate=44100, threshold=0.05, model='full'):
        self.F0Creper = CrepePitchExtractor(hop_length=hop_length, f0_min=
            f0_min, f0_max=f0_max, device=device, threshold=threshold,
            model=model)
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.place = device
        self.threshold = threshold
        self.sampling_rate = sampling_rate

    def compute_f0(self, wav, p_len=None):
        if isinstance(self.place, paddle.dtype):
            dtype = self.place
        elif isinstance(self.place, str) and self.place not in ['cpu',
            'cuda', 'ipu', 'xpu']:
            dtype = self.place
        elif isinstance(self.place, paddle.Tensor):
            dtype = self.place.dtype
        else:
            dtype = paddle.to_tensor(data=wav, dtype='float32').dtype
        x = paddle.to_tensor(data=wav, dtype='float32').cast(dtype)
        if p_len is None:
            p_len = x.shape[0] // self.hop_length
        else:
            assert abs(p_len - x.shape[0] // self.hop_length
                ) < 4, 'pad length error'
        f0, uv = self.F0Creper(x[None, :].astype(dtype='float32'), self.
            sampling_rate, pad_to=p_len)
        return f0

    def compute_f0_uv(self, wav, p_len=None):
        if isinstance(self.place, paddle.dtype):
            dtype = self.place
        elif isinstance(self.place, str) and self.place not in ['cpu',
            'cuda', 'ipu', 'xpu']:
            dtype = self.place
        elif isinstance(self.place, paddle.Tensor):
            dtype = self.place.dtype
        else:
            dtype = paddle.to_tensor(data=wav, dtype='float32').dtype
        x = paddle.to_tensor(data=wav, dtype='float32').cast(dtype)
        if p_len is None:
            p_len = x.shape[0] // self.hop_length
        else:
            assert abs(p_len - x.shape[0] // self.hop_length
                ) < 4, 'pad length error'
        f0, uv = self.F0Creper(x[None, :].astype(dtype='float32'), self.
            sampling_rate, pad_to=p_len)
        return f0, uv
