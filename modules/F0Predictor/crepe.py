import paddle
from typing import Optional, Union
try:
    from typing import Literal
except Exception as e:
    from typing_extensions import Literal
import numpy as np
import torchcrepe
import scipy


def repeat_expand(content: Union[paddle.Tensor, np.ndarray], target_len:
    int, mode: str='nearest'):
    """Repeat content to target length.
    This is a wrapper of torch.nn.functional.interpolate.

    Args:
        content (torch.Tensor): tensor
        target_len (int): target length
        mode (str, optional): interpolation mode. Defaults to "nearest".

    Returns:
        torch.Tensor: tensor
    """
    ndim = content.ndim
    if content.ndim == 1:
        content = content[None, None]
    elif content.ndim == 2:
        content = content[None]
    assert content.ndim == 3
    is_np = isinstance(content, np.ndarray)
    if is_np:
        content = paddle.to_tensor(data=content)
    results = paddle.nn.functional.interpolate(x=content, size=target_len,
        mode=mode)
    if is_np:
        results = results.numpy()
    if ndim == 1:
        return results[0, 0]
    elif ndim == 2:
        return results[0]


class BasePitchExtractor:

    def __init__(self, hop_length: int=512, f0_min: float=50.0, f0_max:
        float=1100.0, keep_zeros: bool=True):
        """Base pitch extractor.

        Args:
            hop_length (int, optional): Hop length. Defaults to 512.
            f0_min (float, optional): Minimum f0. Defaults to 50.0.
            f0_max (float, optional): Maximum f0. Defaults to 1100.0.
            keep_zeros (bool, optional): Whether keep zeros in pitch. Defaults to True.
        """
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.keep_zeros = keep_zeros

    def __call__(self, x, sampling_rate=44100, pad_to=None):
        raise NotImplementedError('BasePitchExtractor is not callable.')

    def post_process(self, x, sampling_rate, f0, pad_to):
        if isinstance(f0, np.ndarray):
            if isinstance(x.place, paddle.dtype):
                dtype = x.place
            elif isinstance(x.place, str) and x.place not in ['cpu', 'cuda',
                'ipu', 'xpu']:
                dtype = x.place
            elif isinstance(x.place, paddle.Tensor):
                dtype = x.place.dtype
            else:
                dtype = paddle.to_tensor(data=f0).astype(dtype='float32').dtype
            f0 = paddle.to_tensor(data=f0).astype(dtype='float32').cast(dtype)
        if pad_to is None:
            return f0
        f0 = repeat_expand(f0, pad_to)
        if self.keep_zeros:
            return f0
        vuv_vector = paddle.zeros_like(x=f0)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0
        nzindex = paddle.nonzero(x=f0).squeeze()
        f0 = paddle.index_select(x=f0, axis=0, index=nzindex).cpu().numpy()
        time_org = self.hop_length / sampling_rate * nzindex.cpu().numpy()
        time_frame = np.arange(pad_to) * self.hop_length / sampling_rate
        if f0.shape[0] <= 0:
            return paddle.zeros(shape=pad_to, dtype='float32'), paddle.zeros(
                shape=pad_to, dtype='float32')
        if f0.shape[0] == 1:
            return paddle.ones(shape=pad_to, dtype='float32') * f0[0
                ], paddle.ones(shape=pad_to, dtype='float32')
        f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
        vuv_vector = vuv_vector.cpu().numpy()
        vuv_vector = np.ceil(scipy.ndimage.zoom(vuv_vector, pad_to / len(
            vuv_vector), order=0))
        return f0, vuv_vector


class MaskedAvgPool1d(paddle.nn.Layer):

    def __init__(self, kernel_size: int, stride: Optional[int]=None,
        padding: Optional[int]=0):
        """An implementation of mean pooling that supports masked values.

        Args:
            kernel_size (int): The size of the median pooling window.
            stride (int, optional): The stride of the median pooling window. Defaults to None.
            padding (int, optional): The padding of the median pooling window. Defaults to 0.
        """
        super(MaskedAvgPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x, mask=None):
        ndim = x.dim()
        if ndim == 2:
            x = x.unsqueeze(axis=1)
        assert x.dim(
            ) == 3, 'Input tensor must have 2 or 3 dimensions (batch_size, channels, width)'
        if mask is None:
            mask = ~paddle.isnan(x=x)
        assert x.shape == mask.shape, 'Input tensor and mask must have the same shape'
        masked_x = paddle.where(condition=mask, x=x, y=paddle.zeros_like(x=x))
        ones_kernel = paddle.ones(shape=[x.shape[1], 1, self.kernel_size])
        sum_pooled = paddle.nn.functional.conv1d(x=masked_x, weight=
            ones_kernel, stride=self.stride, padding=self.padding, groups=x
            .shape[1])
        valid_count = paddle.nn.functional.conv1d(x=mask.astype(dtype=
            'float32'), weight=ones_kernel, stride=self.stride, padding=
            self.padding, groups=x.shape[1])
        valid_count = valid_count.clip(min=1)
        avg_pooled = sum_pooled / valid_count
        avg_pooled[avg_pooled == 0] = float('nan')
        if ndim == 2:
            return avg_pooled.squeeze(axis=1)
        return avg_pooled


class MaskedMedianPool1d(paddle.nn.Layer):

    def __init__(self, kernel_size: int, stride: Optional[int]=None,
        padding: Optional[int]=0):
        """An implementation of median pooling that supports masked values.

        This implementation is inspired by the median pooling implementation in
        https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598

        Args:
            kernel_size (int): The size of the median pooling window.
            stride (int, optional): The stride of the median pooling window. Defaults to None.
            padding (int, optional): The padding of the median pooling window. Defaults to 0.
        """
        super(MaskedMedianPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x, mask=None):
        ndim = x.dim()
        if ndim == 2:
            x = x.unsqueeze(axis=1)
        assert x.dim(
        ) == 3, 'Input tensor must have 2 or 3 dimensions (batch_size, channels, width)'
        if mask is None:
            mask = ~paddle.isnan(x=x)
        assert x.shape == mask.shape, 'Input tensor and mask must have the same shape'
        masked_x = paddle.where(condition=mask, x=x, y=paddle.zeros_like(x=x))
        x = paddle.nn.functional.pad(x=masked_x, pad=(self.padding, self.
        padding), mode='reflect')
        mask = paddle.nn.functional.pad(x=mask.astype(dtype='float32'), pad
        =(self.padding, self.padding), mode='constant', value=0)
        # 使用 paddle.nn.functional.unfold 函数代替 torch.Tensor.unfold 方法，并指定参数名
        x = paddle.nn.functional.unfold(x=x, kernel_sizes=[1, self.kernel_size], strides=[1, self.stride])
        # 使用 paddle.nn.functional.unfold 函数代替 torch.Tensor.unfold 方法，并指定参数名
        mask = paddle.nn.functional.unfold(x=mask.astype(dtype='float32'), kernel_sizes=[1, self.kernel_size], strides=[1, self.stride])
        # 使用 paddle.reshape 函数代替 torch.Tensor.view 方法，并指定参数名
        x = paddle.reshape(x=x, shape=x.shape[:3] + (-1,))
        # 使用 paddle.reshape 函数代替 torch.Tensor.view 方法，并指定参数名
        mask = paddle.reshape(x=mask.astype(dtype='float32'), shape=mask.shape[:3] + (-1,))
        x_masked = paddle.where(condition=mask.astype(dtype='bool'), x=x, y
        =paddle.to_tensor(data=[float('inf')], dtype='float32').cast(dtype=x.dtype)
        )
        x_sorted, _ = paddle.sort(x=x_masked, axis=-1)
        valid_count = mask.sum(axis=-1)
        median_idx = paddle.trunc(paddle.divide(valid_count - 1, 2)).clip(min=0
        )
        median_pooled = x_sorted.take_along_axis(axis=-1, indices=
        median_idx.unsqueeze(axis=-1).astype(dtype='int64')).squeeze(axis
        =-1)
        median_pooled[paddle.isinf(x=median_pooled)] = float('nan')
        if ndim == 2:
            return median_pooled.squeeze(axis=1)
        return median_pooled



class CrepePitchExtractor(BasePitchExtractor):

    def __init__(self, hop_length: int=512, f0_min: float=50.0, f0_max:
    float=1100.0, threshold: float=0.05, keep_zeros: bool=False, device
    =None, model: Literal['full', 'tiny']='full', use_fast_filters:
    bool=True, decoder='viterbi'):
        super().__init__(hop_length, f0_min, f0_max, keep_zeros)
        if decoder == 'viterbi':
            self.decoder = torchcrepe.decode.viterbi
        elif decoder == 'argmax':
            self.decoder = torchcrepe.decode.argmax
        elif decoder == 'weighted_argmax':
            self.decoder = torchcrepe.decode.weighted_argmax
        else:
            raise 'Unknown decoder'
        self.threshold = threshold
        self.model = model
        self.use_fast_filters = use_fast_filters
        self.hop_length = hop_length
        if device is None:
            # 使用 paddle.device.get_device() 函数代替 torch.device.cuda.device_count() 函数，并指定参数名
            self.dev = paddle.device.get_device(device='gpu' if paddle.device.get_device() >= 1 else 'cpu')
        else:
            # 使用 paddle.device.get_device() 函数代替 torch.device 函数，并指定参数名
            self.dev = paddle.device.get_device(device=device)
        if self.use_fast_filters:
            # 使用 paddle.dtype 获取数据类型，并指定参数名
            dtype = paddle.dtype(x=MaskedMedianPool1d(3, 1, 1))
            self.median_filter = MaskedMedianPool1d(3, 1, 1).cast(dtype=dtype)
            # 使用 paddle.dtype 获取数据类型，并指定参数名
            dtype = paddle.dtype(x=MaskedAvgPool1d(3, 1, 1))
            self.mean_filter = MaskedAvgPool1d(3, 1, 1).cast(dtype=dtype)

    def __call__(self, x, sampling_rate=44100, pad_to=None):
        """Extract pitch using crepe.



        Args:
            x (torch.Tensor): Audio signal, shape (1, T).
            sampling_rate (int, optional): Sampling rate. Defaults to 44100.
            pad_to (int, optional): Pad to length. Defaults to None.

        Returns:
            torch.Tensor: Pitch, shape (T // hop_length,).
        """
        assert x.ndim == 2, f'Expected 2D tensor, got {x.ndim}D tensor.'
        assert x.shape[0
            ] == 1, f'Expected 1 channel, got {x.shape[0]} channels.'
        if isinstance(self.dev, paddle.dtype):
            dtype = self.dev
        elif isinstance(self.dev, str) and self.dev not in ['cpu', 'cuda',
            'ipu', 'xpu']:
            dtype = self.dev
        elif isinstance(self.dev, paddle.Tensor):
            dtype = self.dev.dtype
        else:
            dtype = x.dtype
        x = x.cast(dtype)
        f0, pd = torchcrepe.predict(x, sampling_rate, self.hop_length, self
            .f0_min, self.f0_max, pad=True, model=self.model, batch_size=
            1024, device=x.place, return_periodicity=True, decoder=self.decoder
            )
        if self.use_fast_filters:
            pd = self.median_filter(pd)
        else:
            pd = torchcrepe.filter.median(pd, 3)
        pd = torchcrepe.threshold.Silence(-60.0)(pd, x, sampling_rate, 512)
        f0 = torchcrepe.threshold.At(self.threshold)(f0, pd)
        if self.use_fast_filters:
            f0 = self.mean_filter(f0)
        else:
            f0 = torchcrepe.filter.mean(f0, 3)
        f0 = paddle.where(condition=paddle.isnan(x=f0), x=paddle.full_like(
            x=f0, fill_value=0), y=f0)[0]
        if paddle.all(x=(f0 == 0).astype(dtype='bool')):
            rtn = f0.cpu().numpy() if pad_to == None else np.zeros(pad_to)
            return rtn, rtn
        return self.post_process(x, sampling_rate, f0, pad_to)
