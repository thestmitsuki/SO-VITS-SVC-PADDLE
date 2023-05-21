import paddle
from onnxexport.model_onnx import SynthesizerTrn
import utils


def main(NetExport):
    path = 'SoVits4.0'
    if NetExport:
        device = 'cpu'
        hps = utils.get_hparams_from_file(f'checkpoints/{path}/config.json')
        SVCVITS = SynthesizerTrn(hps.data.filter_length // 2 + 1, hps.train
            .segment_size // hps.data.hop_length, **hps.model)
        _ = utils.load_checkpoint(f'checkpoints/{path}/model.pth', SVCVITS,
            None)
        if isinstance(device, paddle.dtype):
            dtype = device
        elif isinstance(device, str) and device not in ['cpu', 'cuda',
            'ipu', 'xpu']:
            dtype = device
        elif isinstance(device, paddle.Tensor):
            dtype = device.dtype
        else:
            dtype = SVCVITS.eval().dtype
        _ = SVCVITS.eval().cast(dtype)
        for i in SVCVITS.parameters():
            i.stop_gradient = not False
        n_frame = 10
        test_hidden_unit = paddle.rand(shape=[1, n_frame, 256])
        test_pitch = paddle.rand(shape=[1, n_frame])
        test_mel2ph = paddle.arange(start=0, end=n_frame).astype('int64')[None]
        test_uv = paddle.ones(shape=[1, n_frame], dtype='float32')
        test_noise = paddle.randn(shape=[1, 192, n_frame])
        test_sid = paddle.to_tensor(data=[0], dtype='int64')
        input_names = ['c', 'f0', 'mel2ph', 'uv', 'noise', 'sid']
        output_names = ['audio']
        if isinstance(device, paddle.dtype):
            dtype = device
        elif isinstance(device, str) and device not in ['cpu', 'cuda',
            'ipu', 'xpu']:
            dtype = device
        elif isinstance(device, paddle.Tensor):
            dtype = device.dtype
        else:
            dtype = test_hidden_unit.dtype
        if isinstance(device, paddle.dtype):
            dtype = device
        elif isinstance(device, str) and device not in ['cpu', 'cuda',
            'ipu', 'xpu']:
            dtype = device
        elif isinstance(device, paddle.Tensor):
            dtype = device.dtype
        else:
            dtype = test_pitch.dtype
        if isinstance(device, paddle.dtype):
            dtype = device
        elif isinstance(device, str) and device not in ['cpu', 'cuda',
            'ipu', 'xpu']:
            dtype = device
        elif isinstance(device, paddle.Tensor):
            dtype = device.dtype
        else:
            dtype = test_mel2ph.dtype
        if isinstance(device, paddle.dtype):
            dtype = device
        elif isinstance(device, str) and device not in ['cpu', 'cuda',
            'ipu', 'xpu']:
            dtype = device
        elif isinstance(device, paddle.Tensor):
            dtype = device.dtype
        else:
            dtype = test_uv.dtype
        if isinstance(device, paddle.dtype):
            dtype = device
        elif isinstance(device, str) and device not in ['cpu', 'cuda',
            'ipu', 'xpu']:
            dtype = device
        elif isinstance(device, paddle.Tensor):
            dtype = device.dtype
        else:
            dtype = test_noise.dtype
        if isinstance(device, paddle.dtype):
            dtype = device
        elif isinstance(device, str) and device not in ['cpu', 'cuda',
            'ipu', 'xpu']:
            dtype = device
        elif isinstance(device, paddle.Tensor):
            dtype = device.dtype
        else:
            dtype = test_sid.dtype
        torch.onnx.export(SVCVITS, (test_hidden_unit.cast(dtype),
            test_pitch.cast(dtype), test_mel2ph.cast(dtype), test_uv.cast(
            dtype), test_noise.cast(dtype), test_sid.cast(dtype)),
            f'checkpoints/{path}/model.onnx', dynamic_axes={'c': [0, 1],
            'f0': [1], 'mel2ph': [1], 'uv': [1], 'noise': [2]},
            do_constant_folding=False, opset_version=16, verbose=False,
            input_names=input_names, output_names=output_names)


if __name__ == '__main__':
    main(True)
