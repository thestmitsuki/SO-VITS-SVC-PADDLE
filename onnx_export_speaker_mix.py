import paddle
from torchaudio.models.wav2vec2.utils import import_fairseq_model
from fairseq import checkpoint_utils
from onnxexport.model_onnx_speaker_mix import SynthesizerTrn
import utils


def get_hubert_model():
    vec_path = 'hubert/checkpoint_best_legacy_500.pt'
    print('load model(s) from {}'.format(vec_path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([
        vec_path], suffix='')
    model = models[0]
    model.eval()
    return model


def main(HubertExport, NetExport):
    path = 'yuuka'
    """if HubertExport:
        device = torch.device("cpu")
        vec_path = "hubert/checkpoint_best_legacy_500.pt"
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [vec_path],
            suffix="",
        )
        original = models[0]
        original.eval()
        model = original
        test_input = torch.rand(1, 1, 16000)
        model(test_input)
        torch.onnx.export(model,
                          test_input,
                          "hubert4.0.onnx",
                          export_params=True,
                          opset_version=16,
                          do_constant_folding=True,
                          input_names=['source'],
                          output_names=['embed'],
                          dynamic_axes={
                              'source':
                                  {
                                      2: "sample_length"
                                  },
                          }
                          )"""
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
        test_hidden_unit = paddle.rand(shape=[1, 10, SVCVITS.gin_channels])
        test_pitch = paddle.rand(shape=[1, 10])
        test_mel2ph = paddle.to_tensor(data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            dtype='int64').unsqueeze(axis=0)
        test_uv = paddle.ones(shape=[1, 10], dtype='float32')
        test_noise = paddle.randn(shape=[1, 192, 10])
        export_mix = False
        test_sid = paddle.to_tensor(data=[0], dtype='int64')
        spk_mix = []
        if export_mix:
            n_spk = len(hps.spk)
            for i in range(n_spk):
                spk_mix.append(1.0 / float(n_spk))
            test_sid = paddle.to_tensor(data=spk_mix)
            SVCVITS.export_chara_mix(n_spk)
        input_names = ['c', 'f0', 'mel2ph', 'uv', 'noise', 'sid']
        output_names = ['audio']
        SVCVITS.eval()
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
    main(False, True)
