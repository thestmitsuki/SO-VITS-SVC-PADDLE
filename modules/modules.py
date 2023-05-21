import paddle
import copy
import math
import numpy as np
import scipy
import modules.commons as commons
from modules.commons import init_weights, get_padding
LRELU_SLOPE = 0.1


class LayerNorm(paddle.nn.Layer):

    def __init__(self, channels, eps=1e-05):
        super().__init__()
        self.channels = channels
        self.eps = eps
        # 使用 paddle.create_parameter 函数代替 torch.nn.Parameter 函数，并指定参数名
        self.gamma = paddle.create_parameter(shape=channels, dtype='float32', name='gamma')
        self.beta = paddle.create_parameter(shape=channels, dtype='float32', name='beta')

    def forward(self, x):
        x = x
        perm_7 = list(range(x.ndim))
        perm_7[1] = -1
        perm_7[-1] = 1
        x = x.transpose(perm=perm_7)
        # 调用 paddle.nn.functional.layer_norm 函数进行归一化，并指定参数名
        x = paddle.nn.functional.layer_norm(x=x, normalized_shape=(self.channels,), weight=self.gamma, bias=self.beta, epsilon=self.eps)
        x = x
        perm_8 = list(range(x.ndim))
        perm_8[1] = -1
        perm_8[-1] = 1
        return x.transpose(perm=perm_8)



class ConvReluNorm(paddle.nn.Layer):

    def __init__(self, in_channels, hidden_channels, out_channels,
        kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, 'Number of layers should be larger than 0.'
        self.conv_layers = paddle.nn.LayerList()
        self.norm_layers = paddle.nn.LayerList()
        self.conv_layers.append(paddle.nn.Conv1D(in_channels=in_channels,
            out_channels=hidden_channels, kernel_size=kernel_size, padding=
            kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = paddle.nn.Sequential(paddle.nn.ReLU(), paddle.nn.
            Dropout(p=p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(paddle.nn.Conv1D(in_channels=
                hidden_channels, out_channels=hidden_channels, kernel_size=
                kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = paddle.nn.Conv1D(in_channels=hidden_channels,
            out_channels=out_channels, kernel_size=1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DDSConv(paddle.nn.Layer):
    """
  Dialted and Depth-Separable Convolution
  """

    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.drop = paddle.nn.Dropout(p=p_dropout)
        self.convs_sep = paddle.nn.LayerList()
        self.convs_1x1 = paddle.nn.LayerList()
        self.norms_1 = paddle.nn.LayerList()
        self.norms_2 = paddle.nn.LayerList()
        for i in range(n_layers):
            dilation = kernel_size ** i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(paddle.nn.Conv1D(in_channels=channels,
                out_channels=channels, kernel_size=kernel_size, groups=
                channels, dilation=dilation, padding=padding))
            self.convs_1x1.append(paddle.nn.Conv1D(in_channels=channels,
                out_channels=channels, kernel_size=1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = paddle.nn.functional.gelu(x=y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = paddle.nn.functional.gelu(x=y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class WN(paddle.nn.Layer):

    def __init__(self, hidden_channels, kernel_size, dilation_rate,
        n_layers, gin_channels=0, p_dropout=0):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.in_layers = paddle.nn.LayerList()
        self.res_skip_layers = paddle.nn.LayerList()
        self.drop = paddle.nn.Dropout(p=p_dropout)
        if gin_channels != 0:
            cond_layer = paddle.nn.Conv1D(in_channels=gin_channels,
                out_channels=2 * hidden_channels * n_layers, kernel_size=1)
            self.cond_layer = paddle.nn.utils.weight_norm(Layer=cond_layer,
                name='weight')
        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = paddle.nn.Conv1D(in_channels=hidden_channels,
                out_channels=2 * hidden_channels, kernel_size=kernel_size,
                dilation=dilation, padding=padding)
            in_layer = paddle.nn.utils.weight_norm(Layer=in_layer, name=
                'weight')
            self.in_layers.append(in_layer)
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            res_skip_layer = paddle.nn.Conv1D(in_channels=hidden_channels,
                out_channels=res_skip_channels, kernel_size=1)
            res_skip_layer = paddle.nn.utils.weight_norm(Layer=
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = paddle.zeros_like(x=x)
        n_channels_tensor = paddle.to_tensor(data=[self.hidden_channels],
            dtype='int32')
        if g is not None:
            g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.
                    hidden_channels, :]
            else:
                g_l = paddle.zeros_like(x=x_in)
            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l,
                n_channels_tensor)
            acts = self.drop(acts)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            paddle.nn.utils.remove_weight_norm(Layer=self.cond_layer)
        for l in self.in_layers:
            paddle.nn.utils.remove_weight_norm(Layer=l)
        for l in self.res_skip_layers:
            paddle.nn.utils.remove_weight_norm(Layer=l)


class ResBlock1(paddle.nn.Layer):

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
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

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = paddle.nn.functional.leaky_relu(x=x, negative_slope=
                LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = paddle.nn.functional.leaky_relu(x=xt, negative_slope=
                LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            paddle.nn.utils.remove_weight_norm(Layer=l)
        for l in self.convs2:
            paddle.nn.utils.remove_weight_norm(Layer=l)


class ResBlock2(paddle.nn.Layer):

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = paddle.nn.LayerList(sublayers=[paddle.nn.utils.
            weight_norm(Layer=paddle.nn.Conv1D(in_channels=channels,
            out_channels=channels, kernel_size=kernel_size, stride=1,
            dilation=dilation[0], padding=get_padding(kernel_size, dilation
            [0]))), paddle.nn.utils.weight_norm(Layer=paddle.nn.Conv1D(
            in_channels=channels, out_channels=channels, kernel_size=
            kernel_size, stride=1, dilation=dilation[1], padding=
            get_padding(kernel_size, dilation[1])))])
        self.convs.apply(fn=init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = paddle.nn.functional.leaky_relu(x=x, negative_slope=
                LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            paddle.nn.utils.remove_weight_norm(Layer=l)


class Log(paddle.nn.Layer):

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            # 使用 paddle.clip 函数代替 torch.clamp_min 函数，并指定参数名
            y = paddle.log(x=paddle.clip(x, min=1e-05)) * x_mask
            logdet = paddle.sum(x=-y, axis=[1, 2])
            return y, logdet
        else:
            x = paddle.exp(x=x.astype('float32')) * x_mask
            return x



class Flip(paddle.nn.Layer):

    def forward(self, x, *args, reverse=False, **kwargs):
        x = paddle.flip(x=x, axis=[1])
        if not reverse:
            logdet = paddle.zeros(shape=x.shape[0]).cast(x.dtype)
            return x, logdet
        else:
            return x


class ElementwiseAffine(paddle.nn.Layer):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # 使用 paddle.create_parameter 函数代替 torch.nn.Parameter 函数，并指定参数名
        self.m = paddle.create_parameter(shape=[channels, 1], dtype='float32', name='m')
        self.logs = paddle.create_parameter(shape=[channels, 1], dtype='float32', name='logs')

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + paddle.exp(x=self.logs.astype('float32')) * x
            y = y * x_mask
            logdet = paddle.sum(x=self.logs * x_mask, axis=[1, 2])
            return y, logdet
        else:
            x = (x - self.m) * paddle.exp(x=(-self.logs).astype('float32')
            ) * x_mask
            return x



class ResidualCouplingLayer(paddle.nn.Layer):

    def __init__(self, channels, hidden_channels, kernel_size,
        dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False):
        assert channels % 2 == 0, 'channels should be divisible by 2'
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = paddle.nn.Conv1D(in_channels=self.half_channels,
            out_channels=hidden_channels, kernel_size=1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers,
            p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = paddle.nn.Conv1D(in_channels=hidden_channels,
            out_channels=self.half_channels * (2 - mean_only), kernel_size=1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = paddle.split(x=x, num_or_sections=[self.half_channels] * 2,
            axis=1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = paddle.split(x=stats, num_or_sections=[self.
                half_channels] * 2, axis=1)
        else:
            m = stats
            logs = paddle.zeros_like(x=m)
        if not reverse:
            x1 = m + x1 * paddle.exp(x=logs.astype('float32')) * x_mask
            x = paddle.concat(x=[x0, x1], axis=1)
            logdet = paddle.sum(x=logs, axis=[1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * paddle.exp(x=(-logs).astype('float32')) * x_mask
            x = paddle.concat(x=[x0, x1], axis=1)
            return x
