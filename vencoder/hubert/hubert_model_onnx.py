import paddle
import copy
import random
from typing import Optional, Tuple


class Hubert(paddle.nn.Layer):

    def __init__(self, num_label_embeddings: int=100, mask: bool=True):
        super().__init__()
        self._mask = mask
        self.feature_extractor = FeatureExtractor()
        self.feature_projection = FeatureProjection()
        self.positional_embedding = PositionalConvEmbedding()
        self.norm = paddle.nn.LayerNorm(normalized_shape=768, epsilon=1e-05,
            weight_attr=None, bias_attr=None)
        self.dropout = paddle.nn.Dropout(p=0.1)
>>>        self.encoder = TransformerEncoder(torch.nn.TransformerEncoderLayer(
            768, 12, 3072, activation='gelu', batch_first=True), 12)
        self.proj = paddle.nn.Linear(in_features=768, out_features=256)
>>>        self.masked_spec_embed = torch.nn.Parameter(paddle.empty(shape=[768
            ], dtype='float32').uniform_())
>>>        self.label_embedding = torch.nn.Embedding(num_label_embeddings, 256)

    def mask(self, x: paddle.Tensor) ->Tuple[paddle.Tensor, paddle.Tensor]:
        mask = None
        if self.training and self._mask:
            mask = _compute_mask((x.shape[0], x.shape[1]), 0.8, 10, x.place, 2)
            if isinstance(x.dtype, paddle.dtype):
                dtype = x.dtype
            elif isinstance(x.dtype, str) and x.dtype not in ['cpu', 'cuda',
                'ipu', 'xpu']:
                dtype = x.dtype
            elif isinstance(x.dtype, paddle.Tensor):
                dtype = x.dtype.dtype
            else:
                dtype = self.masked_spec_embed.dtype
            x[mask] = self.masked_spec_embed.cast(dtype)
        return x, mask

    def encode(self, x: paddle.Tensor, layer: Optional[int]=None) ->Tuple[
        paddle.Tensor, paddle.Tensor]:
        x = self.feature_extractor(x)
        x = x
        perm_22 = list(range(x.ndim))
        perm_22[1] = 2
        perm_22[2] = 1
        x = self.feature_projection(x.transpose(perm=perm_22))
        x, mask = self.mask(x)
        x = x + self.positional_embedding(x)
        x = self.dropout(self.norm(x))
        x = self.encoder(x, output_layer=layer)
        return x, mask

    def logits(self, x: paddle.Tensor) ->paddle.Tensor:
>>>        logits = torch.cosine_similarity(x.unsqueeze(axis=2), self.
            label_embedding.weight.unsqueeze(axis=0).unsqueeze(axis=0), dim=-1)
        return logits / 0.1


class HubertSoft(Hubert):

    def __init__(self):
        super().__init__()

    def units(self, wav: paddle.Tensor) ->paddle.Tensor:
        wav = paddle.nn.functional.pad(x=wav, pad=((400 - 320) // 2, (400 -
            320) // 2))
        x, _ = self.encode(wav)
        return self.proj(x)

    def forward(self, x):
        return self.units(x)


class FeatureExtractor(paddle.nn.Layer):

    def __init__(self):
        super().__init__()
        self.conv0 = paddle.nn.Conv1D(in_channels=1, out_channels=512,
            kernel_size=10, stride=5, bias_attr=False)
        self.norm0 = paddle.nn.GroupNorm(num_groups=512, num_channels=512,
            epsilon=1e-05, weight_attr=None, bias_attr=None)
        self.conv1 = paddle.nn.Conv1D(in_channels=512, out_channels=512,
            kernel_size=3, stride=2, bias_attr=False)
        self.conv2 = paddle.nn.Conv1D(in_channels=512, out_channels=512,
            kernel_size=3, stride=2, bias_attr=False)
        self.conv3 = paddle.nn.Conv1D(in_channels=512, out_channels=512,
            kernel_size=3, stride=2, bias_attr=False)
        self.conv4 = paddle.nn.Conv1D(in_channels=512, out_channels=512,
            kernel_size=3, stride=2, bias_attr=False)
        self.conv5 = paddle.nn.Conv1D(in_channels=512, out_channels=512,
            kernel_size=2, stride=2, bias_attr=False)
        self.conv6 = paddle.nn.Conv1D(in_channels=512, out_channels=512,
            kernel_size=2, stride=2, bias_attr=False)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = paddle.nn.functional.gelu(x=self.norm0(self.conv0(x)))
        x = paddle.nn.functional.gelu(x=self.conv1(x))
        x = paddle.nn.functional.gelu(x=self.conv2(x))
        x = paddle.nn.functional.gelu(x=self.conv3(x))
        x = paddle.nn.functional.gelu(x=self.conv4(x))
        x = paddle.nn.functional.gelu(x=self.conv5(x))
        x = paddle.nn.functional.gelu(x=self.conv6(x))
        return x


class FeatureProjection(paddle.nn.Layer):

    def __init__(self):
        super().__init__()
        self.norm = paddle.nn.LayerNorm(normalized_shape=512, epsilon=1e-05,
            weight_attr=None, bias_attr=None)
        self.projection = paddle.nn.Linear(in_features=512, out_features=768)
        self.dropout = paddle.nn.Dropout(p=0.1)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class PositionalConvEmbedding(paddle.nn.Layer):

    def __init__(self):
        super().__init__()
        self.conv = paddle.nn.Conv1D(in_channels=768, out_channels=768,
            kernel_size=128, padding=128 // 2, groups=16)
        self.conv = paddle.nn.utils.weight_norm(Layer=self.conv, name=
            'weight', dim=2)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = x
        perm_23 = list(range(x.ndim))
        perm_23[1] = 2
        perm_23[2] = 1
        x = self.conv(x.transpose(perm=perm_23))
        x = paddle.nn.functional.gelu(x=x[:, :, :-1])
        x = x
        perm_24 = list(range(x.ndim))
        perm_24[1] = 2
        perm_24[2] = 1
        return x.transpose(perm=perm_24)


class TransformerEncoder(paddle.nn.Layer):

>>>    def __init__(self, encoder_layer: torch.nn.TransformerEncoderLayer,
        num_layers: int) ->None:
        super(TransformerEncoder, self).__init__()
        self.layers = paddle.nn.LayerList(sublayers=[copy.deepcopy(
            encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src: paddle.Tensor, mask: paddle.Tensor=None,
        src_key_padding_mask: paddle.Tensor=None, output_layer: Optional[
        int]=None) ->paddle.Tensor:
        output = src
        for layer in self.layers[:output_layer]:
            output = layer(output, src_mask=mask, src_key_padding_mask=
                src_key_padding_mask)
        return output


def _compute_mask(shape: Tuple[int, int], mask_prob: float, mask_length:
>>>    int, device: torch.device, min_masks: int=0) ->paddle.Tensor:
    batch_size, sequence_length = shape
    if mask_length < 1:
        raise ValueError('`mask_length` has to be bigger than 0.')
    if mask_length > sequence_length:
        raise ValueError(
            f'`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length} and `sequence_length`: {sequence_length}`'
            )
    num_masked_spans = int(mask_prob * sequence_length / mask_length +
        random.random())
    num_masked_spans = max(num_masked_spans, min_masks)
    if num_masked_spans * mask_length > sequence_length:
        num_masked_spans = sequence_length // mask_length
    mask = paddle.zeros(shape=(batch_size, sequence_length), dtype='bool')
    uniform_dist = paddle.ones(shape=(batch_size, sequence_length - (
        mask_length - 1)))
    mask_indices = paddle.multinomial(x=uniform_dist, num_samples=
        num_masked_spans)
    """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    mask_indices = mask_indices.unsqueeze(axis=-1).expand(shape=(batch_size,
        num_masked_spans, mask_length)).reshape(batch_size, 
        num_masked_spans * mask_length)
    """Class Method: *.reshape, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    offsets = paddle.arange(end=mask_length)[None, None, :].expand(shape=(
        batch_size, num_masked_spans, mask_length)).reshape(batch_size, 
        num_masked_spans * mask_length)
    mask_idxs = mask_indices + offsets
    """Class Method: *.scatter, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
>>>    mask = mask.scatter(1, mask_idxs, True)
    return mask


def hubert_soft(path: str) ->HubertSoft:
    """HuBERT-Soft from `"A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion"`.
    Args:
        path (str): path of a pretrained model
    """
    hubert = HubertSoft()
    checkpoint = paddle.load(path=path)
>>>    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint,
        'module.')
    hubert.set_state_dict(state_dict=checkpoint)
    hubert.eval()
    return hubert
