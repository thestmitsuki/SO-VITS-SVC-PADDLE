import paddle
import modules.attentions as attentions
import modules.commons as commons
import modules.modules as modules
import utils
from modules.commons import init_weights, get_padding
from vdecoder.hifigan.models import Generator
from utils import f0_to_coarse


class ResidualCouplingBlock(paddle.nn.Layer):

    def __init__(self, channels, hidden_channels, kernel_size,
        dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels
        self.flows = paddle.nn.LayerList()
        for i in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer(channels,
                hidden_channels, kernel_size, dilation_rate, n_layers,
                gin_channels=gin_channels, mean_only=True))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class Encoder(paddle.nn.Layer):

    def __init__(self, in_channels, out_channels, hidden_channels,
        kernel_size, dilation_rate, n_layers, gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.pre = paddle.nn.Conv1D(in_channels=in_channels, out_channels=
            hidden_channels, kernel_size=1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate,
            n_layers, gin_channels=gin_channels)
        self.proj = paddle.nn.Conv1D(in_channels=hidden_channels,
            out_channels=out_channels * 2, kernel_size=1)

    def forward(self, x, x_lengths, g=None):
        if isinstance(x.dtype, paddle.dtype):
            dtype = x.dtype
        elif isinstance(x.dtype, str) and x.dtype not in ['cpu', 'cuda',
            'ipu', 'xpu']:
            dtype = x.dtype
        elif isinstance(x.dtype, paddle.Tensor):
            dtype = x.dtype.dtype
        else:
            dtype = paddle.unsqueeze(x=commons.sequence_mask(x_lengths, x.
                shape[2]), axis=1).dtype
        x_mask = paddle.unsqueeze(x=commons.sequence_mask(x_lengths, x.
            shape[2]), axis=1).cast(dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = paddle.split(x=stats, num_or_sections=stats.shape[1] //
            self.out_channels, axis=1)
        z = (m + paddle.randn(shape=m.shape, dtype=m.dtype) * paddle.exp(x=
            logs.astype('float32'))) * x_mask
        return z, m, logs, x_mask


class TextEncoder(paddle.nn.Layer):

    def __init__(self, out_channels, hidden_channels, kernel_size, n_layers,
                 gin_channels=0, filter_channels=None, n_heads=None, p_dropout=None):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.proj = paddle.nn.Conv1D(in_channels=hidden_channels,
                                     out_channels=out_channels * 2, kernel_size=1)
        # 修改了这一行
        self.f0_emb = paddle.nn.Embedding(256, hidden_channels)
        self.enc_ = attentions.Encoder(hidden_channels, filter_channels,
                                       n_heads, n_layers, kernel_size, p_dropout)

    def forward(self, x, x_mask, f0=None, z=None):
        x = self.f0_emb(f0)
        perm_36 = list(range(x.ndim))
        perm_36[1] = 2
        perm_36[2] = 1
        x = x + x.transpose(perm=perm_36)
        x = self.enc_(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = paddle.split(x=stats, num_or_sections=stats.shape[1] //
                               self.out_channels, axis=1)
        z = (m + z * paddle.exp(x=logs.astype('float32'))) * x_mask
        return z, m, logs, x_mask



class DiscriminatorP(paddle.nn.Layer):

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False
        ):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = (paddle.nn.utils.weight_norm if use_spectral_norm == False
             else paddle.nn.utils.spectral_norm)
        self.convs = paddle.nn.LayerList(sublayers=[norm_f(paddle.nn.Conv2D
            (in_channels=1, out_channels=32, kernel_size=(kernel_size, 1),
            stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(paddle.nn.Conv2D(in_channels=32, out_channels=128,
            kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(
            get_padding(kernel_size, 1), 0))), norm_f(paddle.nn.Conv2D(
            in_channels=128, out_channels=512, kernel_size=(kernel_size, 1),
            stride=(stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(paddle.nn.Conv2D(in_channels=512, out_channels=1024,
            kernel_size=(kernel_size, 1), stride=(stride, 1), padding=(
            get_padding(kernel_size, 1), 0))), norm_f(paddle.nn.Conv2D(
            in_channels=1024, out_channels=1024, kernel_size=(kernel_size, 
            1), stride=1, padding=(get_padding(kernel_size, 1), 0)))])
        self.conv_post = norm_f(paddle.nn.Conv2D(in_channels=1024,
            out_channels=1, kernel_size=(3, 1), stride=1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - t % self.period
            x = paddle.nn.functional.pad(x=x, pad=(0, n_pad), mode='reflect')
            t = t + n_pad
        # 修改了这一行
        x = x.reshape([b, c, t // self.period, self.period])
        for l in self.convs:
            x = l(x)
            x = paddle.nn.functional.leaky_relu(x=x, negative_slope=modules
                                                .LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = paddle.flatten(x=x, start_axis=1, stop_axis=-1)
        return x, fmap



class DiscriminatorS(paddle.nn.Layer):

    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = (paddle.nn.utils.weight_norm if use_spectral_norm == False
             else paddle.nn.utils.spectral_norm)
        self.convs = paddle.nn.LayerList(sublayers=[norm_f(paddle.nn.Conv1D
            (in_channels=1, out_channels=16, kernel_size=15, stride=1,
            padding=7)), norm_f(paddle.nn.Conv1D(in_channels=16,
            out_channels=64, kernel_size=41, stride=4, groups=4, padding=20
            )), norm_f(paddle.nn.Conv1D(in_channels=64, out_channels=256,
            kernel_size=41, stride=4, groups=16, padding=20)), norm_f(
            paddle.nn.Conv1D(in_channels=256, out_channels=1024,
            kernel_size=41, stride=4, groups=64, padding=20)), norm_f(
            paddle.nn.Conv1D(in_channels=1024, out_channels=1024,
            kernel_size=41, stride=4, groups=256, padding=20)), norm_f(
            paddle.nn.Conv1D(in_channels=1024, out_channels=1024,
            kernel_size=5, stride=1, padding=2))])
        self.conv_post = norm_f(paddle.nn.Conv1D(in_channels=1024,
            out_channels=1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = paddle.nn.functional.leaky_relu(x=x, negative_slope=modules
                .LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = paddle.flatten(x=x, start_axis=1, stop_axis=-1)
        return x, fmap


class F0Decoder(paddle.nn.Layer):

    def __init__(self, out_channels, hidden_channels, filter_channels,
                 n_heads, n_layers, kernel_size, p_dropout, spk_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.spk_channels = spk_channels
        self.prenet = paddle.nn.Conv1D(in_channels=hidden_channels,
                                       out_channels=hidden_channels, kernel_size=3, padding=1)
        self.decoder = attentions.FFT(hidden_channels, filter_channels,
                                      n_heads, n_layers, kernel_size, p_dropout)
        self.proj = paddle.nn.Conv1D(in_channels=hidden_channels,
                                     out_channels=out_channels, kernel_size=1)
        self.f0_prenet = paddle.nn.Conv1D(in_channels=1, out_channels=
                                          hidden_channels, kernel_size=3, padding=1)
        self.cond = paddle.nn.Conv1D(in_channels=spk_channels, out_channels
                                     =hidden_channels, kernel_size=1)

    def forward(self, x, norm_f0, x_mask, spk_emb=None):
        # 修改了这一行
        x = x.detach()
        if spk_emb is not None:
            x = x + self.cond(spk_emb)
        x += self.f0_prenet(norm_f0)
        x = self.prenet(x) * x_mask
        x = self.decoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask
        return x



class SynthesizerTrn(paddle.nn.Layer):
    """
    Synthesizer for Training
    """

    def __init__(self, spec_channels, segment_size, inter_channels,
                 hidden_channels, filter_channels, n_heads, n_layers, kernel_size,
                 p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                 gin_channels, ssl_dim, n_speakers, sampling_rate=44100, **kwargs):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.ssl_dim = ssl_dim
        # 修改了这一行
        self.emb_g = paddle.nn.Embedding(n_speakers, gin_channels)
        self.pre = paddle.nn.Conv1D(in_channels=ssl_dim, out_channels=
                                    hidden_channels, kernel_size=5, padding=2)
        self.enc_p = TextEncoder(inter_channels, hidden_channels,
                                 filter_channels=filter_channels, n_heads=n_heads, n_layers=
                                 n_layers, kernel_size=kernel_size, p_dropout=p_dropout)
        hps = {'sampling_rate': sampling_rate, 'inter_channels':
               inter_channels, 'resblock': resblock, 'resblock_kernel_sizes':
               resblock_kernel_sizes, 'resblock_dilation_sizes':
               resblock_dilation_sizes, 'upsample_rates': upsample_rates,
               'upsample_initial_channel': upsample_initial_channel,
               'upsample_kernel_sizes': upsample_kernel_sizes, 'gin_channels':
               gin_channels}
        self.dec = Generator(h=hps)
        self.enc_q = Encoder(spec_channels, inter_channels, hidden_channels,
                             5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels,
                                          5, 1, 4, gin_channels=gin_channels)
        self.f0_decoder = F0Decoder(1, hidden_channels, filter_channels,
                                    n_heads, n_layers, kernel_size, p_dropout, spk_channels=
                                    gin_channels)
        # 修改了这一行
        self.emb_uv = paddle.nn.Embedding(2, hidden_channels)
        self.predict_f0 = False


    def forward(self, c, f0, mel2ph, uv, noise=None, g=None):
        decoder_inp = paddle.nn.functional.pad(x=c, pad=[0, 0, 1, 0])
        mel2ph_ = mel2ph.unsqueeze(axis=2).tile(repeat_times=[1, 1, c.shape
            [-1]])
        x = paddle.take_along_axis(arr=decoder_inp, axis=1, indices=mel2ph_)
        perm_37 = list(range(x.ndim))
        perm_37[1] = 2
        perm_37[2] = 1
        c = x.transpose(perm=perm_37)
        if isinstance(c.place, paddle.dtype):
            dtype = c.place
        elif isinstance(c.place, str) and c.place not in ['cpu', 'cuda',
            'ipu', 'xpu']:
            dtype = c.place
        elif isinstance(c.place, paddle.Tensor):
            dtype = c.place.dtype
        else:
            dtype = (paddle.ones(shape=c.shape[0]) * c.shape[-1]).dtype
        c_lengths = (paddle.ones(shape=c.shape[0]) * c.shape[-1]).cast(dtype)
        g = g.unsqueeze(axis=0)
        x = self.emb_g(g)
        perm_38 = list(range(x.ndim))
        perm_38[1] = 2
        perm_38[2] = 1
        g = x.transpose(perm=perm_38)
        if isinstance(c.dtype, paddle.dtype):
            dtype = c.dtype
        elif isinstance(c.dtype, str) and c.dtype not in ['cpu', 'cuda',
            'ipu', 'xpu']:
            dtype = c.dtype
        elif isinstance(c.dtype, paddle.Tensor):
            dtype = c.dtype.dtype
        else:
            dtype = paddle.unsqueeze(x=commons.sequence_mask(c_lengths, c.
                shape[2]), axis=1).dtype
        x_mask = paddle.unsqueeze(x=commons.sequence_mask(c_lengths, c.
            shape[2]), axis=1).cast(dtype)
        x = self.emb_uv(uv.astype(dtype='int64'))
        perm_39 = list(range(x.ndim))
        perm_39[1] = 2
        perm_39[2] = 1
        x = self.pre(c) * x_mask + x.transpose(perm=perm_39)
        if self.predict_f0:
            lf0 = 2595.0 * paddle.log10(x=1.0 + f0.unsqueeze(axis=1) / 700.0
                ) / 500
            norm_lf0 = utils.normalize_f0(lf0, x_mask, uv, random_scale=False)
            pred_lf0 = self.f0_decoder(x, norm_lf0, x_mask, spk_emb=g)
            f0 = (700 * (paddle.pow(x=10, y=pred_lf0 * 500 / 2595) - 1)
                ).squeeze(axis=1)
        z_p, m_p, logs_p, c_mask = self.enc_p(x, x_mask, f0=f0_to_coarse(f0
            ), z=noise)
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g, f0=f0)
        return o
