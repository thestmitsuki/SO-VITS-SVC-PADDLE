import paddle
import os
import json
from .env import AttrDict
import numpy as np
from .utils import init_weights, get_padding
LRELU_SLOPE = 0.1


def load_model(model_path, device='cuda'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    if isinstance(device, paddle.dtype):
        dtype = device
    elif isinstance(device, str) and device not in ['cpu', 'cuda', 'ipu', 'xpu'
        ]:
        dtype = device
    elif isinstance(device, paddle.Tensor):
        dtype = device.dtype
    else:
        dtype = Generator(h).dtype
    generator = Generator(h).cast(dtype)
    cp_dict = paddle.load(path=model_path)
    generator.set_state_dict(state_dict=cp_dict['generator'])
    generator.eval()
    generator.remove_weight_norm()
    del cp_dict
    return generator, h


class ResBlock1(paddle.nn.Layer):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = paddle.nn.LayerList(sublayers=[paddle.nn.utils.
            weight_norm(Layer=paddle.nn.Conv1D(in_channels=channels,
            out_channels=channels, kernel_size=kernel_size, stride=1,
            dilation=dilation[0], padding=get_padding(kernel_size, dilation
            [0]))), paddle.nn.utils.weight_norm(Layer=paddle.nn.Conv1D(
            in_channels=channels, out_channels=channels, kernel_size=
            kernel_size, stride=1, dilation=dilation[1], padding=
            get_padding(kernel_size, dilation[1]))), paddle.nn.utils.
            weight_norm(Layer=paddle.nn.Conv1D(in_channels=channels,
            out_channels=channels, kernel_size=kernel_size, stride=1,
            dilation=dilation[2], padding=get_padding(kernel_size, dilation
            [2])))])
        self.convs1.apply(fn=init_weights)
        self.convs2 = paddle.nn.LayerList(sublayers=[paddle.nn.utils.
            weight_norm(Layer=paddle.nn.Conv1D(in_channels=channels,
            out_channels=channels, kernel_size=kernel_size, stride=1,
            dilation=1, padding=get_padding(kernel_size, 1))), paddle.nn.
            utils.weight_norm(Layer=paddle.nn.Conv1D(in_channels=channels,
            out_channels=channels, kernel_size=kernel_size, stride=1,
            dilation=1, padding=get_padding(kernel_size, 1))), paddle.nn.
            utils.weight_norm(Layer=paddle.nn.Conv1D(in_channels=channels,
            out_channels=channels, kernel_size=kernel_size, stride=1,
            dilation=1, padding=get_padding(kernel_size, 1)))])
        self.convs2.apply(fn=init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = paddle.nn.functional.leaky_relu(x=x, negative_slope=
                LRELU_SLOPE)
            xt = c1(xt)
            xt = paddle.nn.functional.leaky_relu(x=xt, negative_slope=
                LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            paddle.nn.utils.remove_weight_norm(Layer=l)
        for l in self.convs2:
            paddle.nn.utils.remove_weight_norm(Layer=l)


class ResBlock2(paddle.nn.Layer):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = paddle.nn.LayerList(sublayers=[paddle.nn.utils.
            weight_norm(Layer=paddle.nn.Conv1D(in_channels=channels,
            out_channels=channels, kernel_size=kernel_size, stride=1,
            dilation=dilation[0], padding=get_padding(kernel_size, dilation
            [0]))), paddle.nn.utils.weight_norm(Layer=paddle.nn.Conv1D(
            in_channels=channels, out_channels=channels, kernel_size=
            kernel_size, stride=1, dilation=dilation[1], padding=
            get_padding(kernel_size, dilation[1])))])
        self.convs.apply(fn=init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = paddle.nn.functional.leaky_relu(x=x, negative_slope=
                LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            paddle.nn.utils.remove_weight_norm(Layer=l)


def padDiff(x):
    return paddle.nn.functional.pad(x=paddle.nn.functional.pad(x=x, pad=(0,
        0, -1, 1), mode='constant', value=0) - x, pad=(0, 0, 0, -1), mode=
        'constant', value=0)


class SineGen(paddle.nn.Layer):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
           sine_amp = 0.1, noise_std = 0.003,
           voiced_threshold = 0,
           flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
          segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=
                 0.003, voiced_threshold=0, flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse

    def _f02uv(self, f0):
        # 修改了这一行
        uv = (f0 > self.voiced_threshold).astype('float32')
        return uv


    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        rad_values = f0_values / self.sampling_rate % 1
        rand_ini = paddle.rand(shape=[f0_values.shape[0], f0_values.shape[2]])
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        if not self.flag_for_pulse:
            tmp_over_one = paddle.cumsum(x=rad_values, dim=1) % 1
            tmp_over_one_idx = padDiff(tmp_over_one) < 0
            cumsum_shift = paddle.zeros_like(x=rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
            sines = paddle.sin(x=paddle.cumsum(x=rad_values + cumsum_shift,
                dim=1) * 2 * np.pi)
        else:
            uv = self._f02uv(f0_values)
            uv_1 = paddle.roll(x=uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            tmp_cumsum = paddle.cumsum(x=rad_values, dim=1)
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            i_phase = paddle.cumsum(x=rad_values - tmp_cumsum, dim=1)
            sines = paddle.cos(x=i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with paddle.no_grad():
            f0_buf = paddle.zeros(shape=[f0.shape[0], f0.shape[1], self.dim])
            if isinstance(f0.place, paddle.dtype):
                dtype = f0.place
            elif isinstance(f0.place, str) and f0.place not in ['cpu',
                'cuda', 'ipu', 'xpu']:
                dtype = f0.place
            elif isinstance(f0.place, paddle.Tensor):
                dtype = f0.place.dtype
            else:
                dtype = paddle.to_tensor(data=[[range(1, self.harmonic_num +
                    2)]], dtype='float32').dtype
            fn = paddle.multiply(x=f0, y=paddle.to_tensor(data=[[range(1, 
                self.harmonic_num + 2)]], dtype='float32').cast(dtype))
            sine_waves = self._f02sine(fn) * self.sine_amp
            uv = self._f02uv(f0)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * paddle.randn(shape=sine_waves.shape, dtype=
                sine_waves.dtype)
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(paddle.nn.Layer):
    """ SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1,
        add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp,
            add_noise_std, voiced_threshod)
        self.l_linear = paddle.nn.Linear(in_features=harmonic_num + 1,
            out_features=1)
        self.l_tanh = paddle.nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        noise = paddle.randn(shape=uv.shape, dtype=uv.dtype
            ) * self.sine_amp / 3
        return sine_merge, noise, uv


class Generator(paddle.nn.Layer):

    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h['resblock_kernel_sizes'])
        self.num_upsamples = len(h['upsample_rates'])
        # 修改了这一行
        self.f0_upsamp = paddle.nn.Upsample(scale_factor=np.prod(h[
            'upsample_rates']))
        self.m_source = SourceModuleHnNSF(sampling_rate=h['sampling_rate'],
                                          harmonic_num=8)
        self.noise_convs = paddle.nn.LayerList()
        self.conv_pre = paddle.nn.utils.weight_norm(Layer=paddle.nn.Conv1D(
            in_channels=h['inter_channels'], out_channels=h[
                'upsample_initial_channel'], kernel_size=7, stride=1, padding=3))
        resblock = ResBlock1 if h['resblock'] == '1' else ResBlock2
        self.ups = paddle.nn.LayerList()
        for i, (u, k) in enumerate(zip(h['upsample_rates'], h[
                'upsample_kernel_sizes'])):
            c_cur = h['upsample_initial_channel'] // 2 ** (i + 1)
            self.ups.append(paddle.nn.utils.weight_norm(Layer=paddle.nn.
                                                        Conv1DTranspose(in_channels=h['upsample_initial_channel'] //
                                                                        2 ** i, out_channels=h['upsample_initial_channel'] // 2 **
                                                                        (i + 1), kernel_size=k, stride=u, padding=(k - u) // 2)))
            if i + 1 < len(h['upsample_rates']):
                stride_f0 = np.prod(h['upsample_rates'][i + 1:])
                self.noise_convs.append(paddle.nn.Conv1D(in_channels=1,
                                                         out_channels=c_cur, kernel_size=stride_f0 * 2, stride=
                                                         stride_f0, padding=stride_f0 // 2))
            else:
                self.noise_convs.append(paddle.nn.Conv1D(in_channels=1,
                                                         out_channels=c_cur, kernel_size=1))
        self.resblocks = paddle.nn.LayerList()
        for i in range(len(self.ups)):
            ch = h['upsample_initial_channel'] // 2 ** (i + 1)
            for j, (k, d) in enumerate(zip(h['resblock_kernel_sizes'], h[
                    'resblock_dilation_sizes'])):
                self.resblocks.append(resblock(h, ch, k, d))
        self.conv_post = paddle.nn.utils.weight_norm(Layer=paddle.nn.Conv1D
                                                     (in_channels=ch, out_channels=1, kernel_size=7, stride=1,
                                                      padding=3))
        self.ups.apply(fn=init_weights)
        self.conv_post.apply(fn=init_weights)
        self.cond = paddle.nn.Conv1D(in_channels=h['gin_channels'],
                                     out_channels=h['upsample_initial_channel'], kernel_size=1)


    def forward(self, x, f0, g=None):
        x = self.f0_upsamp(f0[:, None])
        perm_9 = list(range(x.ndim))
        perm_9[1] = 2
        perm_9[2] = 1
        f0 = x.transpose(perm=perm_9)
        har_source, noi_source, uv = self.m_source(f0)
        x = har_source
        perm_10 = list(range(x.ndim))
        perm_10[1] = 2
        perm_10[2] = 1
        har_source = x.transpose(perm=perm_10)
        x = self.conv_pre(x)
        x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = paddle.nn.functional.leaky_relu(x=x, negative_slope=LRELU_SLOPE
                )
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = paddle.nn.functional.leaky_relu(x=x)
        x = self.conv_post(x)
        x = paddle.tanh(x=x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            paddle.nn.utils.remove_weight_norm(Layer=l)
        for l in self.resblocks:
            l.remove_weight_norm()
        paddle.nn.utils.remove_weight_norm(Layer=self.conv_pre)
        paddle.nn.utils.remove_weight_norm(Layer=self.conv_post)


class DiscriminatorP(paddle.nn.Layer):

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False
        ):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = (paddle.nn.utils.weight_norm if use_spectral_norm == False
             else paddle.nn.utils.spectral_norm)
        self.convs = paddle.nn.LayerList(sublayers=[norm_f(paddle.nn.Conv2D
            (in_channels=1, out_channels=32, kernel_size=(kernel_size, 1),
            stride=(stride, 1), padding=(get_padding(5, 1), 0))), norm_f(
            paddle.nn.Conv2D(in_channels=32, out_channels=128, kernel_size=
            (kernel_size, 1), stride=(stride, 1), padding=(get_padding(5, 1
            ), 0))), norm_f(paddle.nn.Conv2D(in_channels=128, out_channels=
            512, kernel_size=(kernel_size, 1), stride=(stride, 1), padding=
            (get_padding(5, 1), 0))), norm_f(paddle.nn.Conv2D(in_channels=
            512, out_channels=1024, kernel_size=(kernel_size, 1), stride=(
            stride, 1), padding=(get_padding(5, 1), 0))), norm_f(paddle.nn.
            Conv2D(in_channels=1024, out_channels=1024, kernel_size=(
            kernel_size, 1), stride=1, padding=(2, 0)))])
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
            x = paddle.nn.functional.leaky_relu(x=x, negative_slope=LRELU_SLOPE
                                                )
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = paddle.flatten(x=x, start_axis=1, stop_axis=-1)
        return x, fmap



class MultiPeriodDiscriminator(paddle.nn.Layer):

    def __init__(self, periods=None):
        super(MultiPeriodDiscriminator, self).__init__()
        self.periods = periods if periods is not None else [2, 3, 5, 7, 11]
        self.discriminators = paddle.nn.LayerList()
        for period in self.periods:
            self.discriminators.append(DiscriminatorP(period))

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(paddle.nn.Layer):

    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = (paddle.nn.utils.weight_norm if use_spectral_norm == False
             else paddle.nn.utils.spectral_norm)
        self.convs = paddle.nn.LayerList(sublayers=[norm_f(paddle.nn.Conv1D
            (in_channels=1, out_channels=128, kernel_size=15, stride=1,
            padding=7)), norm_f(paddle.nn.Conv1D(in_channels=128,
            out_channels=128, kernel_size=41, stride=2, groups=4, padding=
            20)), norm_f(paddle.nn.Conv1D(in_channels=128, out_channels=256,
            kernel_size=41, stride=2, groups=16, padding=20)), norm_f(
            paddle.nn.Conv1D(in_channels=256, out_channels=512, kernel_size
            =41, stride=4, groups=16, padding=20)), norm_f(paddle.nn.Conv1D
            (in_channels=512, out_channels=1024, kernel_size=41, stride=4,
            groups=16, padding=20)), norm_f(paddle.nn.Conv1D(in_channels=
            1024, out_channels=1024, kernel_size=41, stride=1, groups=16,
            padding=20)), norm_f(paddle.nn.Conv1D(in_channels=1024,
            out_channels=1024, kernel_size=5, stride=1, padding=2))])
        self.conv_post = norm_f(paddle.nn.Conv1D(in_channels=1024,
            out_channels=1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = paddle.nn.functional.leaky_relu(x=x, negative_slope=LRELU_SLOPE
                )
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = paddle.flatten(x=x, start_axis=1, stop_axis=-1)
        return x, fmap


class MultiScaleDiscriminator(paddle.nn.Layer):

    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = paddle.nn.LayerList(sublayers=[DiscriminatorS
            (use_spectral_norm=True), DiscriminatorS(), DiscriminatorS()])
        self.meanpools = paddle.nn.LayerList(sublayers=[paddle.nn.AvgPool1D
            (kernel_size=4, stride=2, padding=2, exclusive=False), paddle.
            nn.AvgPool1D(kernel_size=4, stride=2, padding=2, exclusive=False)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += paddle.mean(x=paddle.abs(x=rl - gl))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = paddle.mean(x=(1 - dr) ** 2)
        g_loss = paddle.mean(x=dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = paddle.mean(x=(1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses
