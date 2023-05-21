import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.nn.parallel import DataParallel as DDP
import logging
import multiprocessing
import time
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
import os
import json
import argparse
import itertools
import math
import modules.commons as commons
import utils
from data_utils import TextAudioSpeakerLoader, TextAudioCollate
from models import SynthesizerTrn, MultiPeriodDiscriminator
from modules.losses import kl_loss, generator_loss, discriminator_loss, feature_loss
from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
paddle.set_flags(**{'FLAGS_cudnn_exhaustive_search': 1}
global_step = 0
start_time = time.time()


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert paddle.device.cuda.device_count(
        ) >= 1, 'CPU training is not allowed.'
    hps = utils.get_hparams()
    n_gpus = paddle.device.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = hps.train.port
    paddle.distributed.spawn(run, args=(n_gpus, hps), nprocs=n_gpus)




    # 初始化分布式环境
    paddle.distributed.init_parallel_env(backend='gloo' if os.name == 'nt' else 'nccl', init_method='env://', world_size=n_gpus, rank=rank)
    # 获取当前进程的rank
    rank = paddle.distributed.get_rank()
    # 获取总设备数
    device_num = paddle.distributed.get_world_size()

    # 设置随机种子和设备
    paddle.seed(seed=hps.train.seed)
    paddle.device.set_device(device=rank)

    # 创建数据集和数据加载器
    collate_fn = TextAudioCollate()
    all_in_mem = hps.train.all_in_mem
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps,
    all_in_mem=all_in_mem)
    num_workers = 5 if multiprocessing.cpu_count(
    ) > 4 else multiprocessing.cpu_count()
    if all_in_mem:
        num_workers = 0
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=hps.train. batch_size, shuffle=False, use_shared_memory=True, num_workers= num_workers,             collate_fn=collate_fn)  

    # 只在rank为0的进程中创建logger，writer和writer_eval
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = paddle.callbacks.VisualDL(log_dir=hps.model_dir)
        writer_eval = paddle.callbacks.VisualDL(log_dir=os.path .join(hps.model_dir, 'eval'))
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files,
        hps, all_in_mem=all_in_mem)
        eval_loader = paddle.io.DataLoader(eval_dataset, batch_size=1, shuffle=False, use_shared_memory=False, num_workers =1, drop_last= False, collate_fn=collate_fn)

    # 创建模型
    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1, hps.train.
    segment_size // hps.data.hop_length, **hps.model)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)

    # 使用paddle.DataParallel包装模型
    net_g = DDP(net_g)
    net_d = DDP(net_d)

    # 创建优化器
    optim_g = paddle.optimizer.AdamW(learning_rate=hps.train.learning_rate, betas=hps.train.betas, epsilon=hps.train.eps, parameters=net_g.parameters())
    optim_d = paddle.optimizer.AdamW(learning_rate=hps.train.learning_rate, betas=hps.train.betas, epsilon=hps.train.eps, parameters=net_d.parameters())

    # 保存和加载模型参数
    paddle.save(net_g.state_dict(), "net_g.pdparams")
    paddle.save(net_d.state_dict(), "net_d.pdparams")
    net_g.set_state_dict(paddle.load("net_g.pdparams"))
    net_d.set_state_dict(paddle.load("net_d.pdparams"))

    skip_optimizer = False

    # 创建模型
    net_g = ToyModel()
    net_d = ToyModel()

    # 创建指数衰减的学习率调度器
    lr_g = paddle.optimizer.lr.ExponentialDecay(learning_rate=hps.train.learning_rate, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    lr_d = paddle.optimizer.lr.ExponentialDecay(learning_rate=hps.train.learning_rate, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    # 创建优化器，并传入学习率调度器
    optim_g = paddle.optimizer.AdamW(learning_rate=lr_g, betas=hps.train.betas, epsilon=hps.train.eps, parameters=net_g.parameters())
    optim_d = paddle.optimizer.AdamW(learning_rate=lr_d, betas=hps.train.betas, epsilon=hps.train.eps, parameters=net_d.parameters())

    # 创建梯度缩放器
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    for epoch in epochs:
        for input, target in data:
            optim_g.clear_grad()
            optim_d.clear_grad()
            # 运行前向传播，并使用autocast自动选择合适的精度
            with paddle.amp.auto_cast():
                output = net_g(input)
                loss = loss_fn(output, target)
            # 使用梯度缩放器放大损失，并进行反向传播
            scaled = scaler.scale(loss)
            scaled.backward()
            # 使用梯度缩放器更新优化器，并缩小梯度
            scaler.step(optim_g)
            scaler.step(optim_d)
            scaler.update()

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if epoch > 1:
            scheduler_g.step()
            scheduler_d.step()
        if epoch <= warmup_epoch:
            for param_group in optim_g.param_groups:
                param_group['lr'
                    ] = hps.train.learning_rate / warmup_epoch * epoch
            for param_group in optim_d.param_groups:
                param_group['lr'
                    ] = hps.train.learning_rate / warmup_epoch * epoch
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g,
                optim_d], [scheduler_g, scheduler_d], scaler, [train_loader,
                eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g,
                optim_d], [scheduler_g, scheduler_d], scaler, [train_loader,
                None], None, None)


    # 创建模型
    net_g = ToyModel()
    net_d = ToyModel()

    # 创建指数衰减的学习率调度器
    lr_g = paddle.optimizer.lr.ExponentialDecay(learning_rate=hps.train.learning_rate, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    lr_d = paddle.optimizer.lr.ExponentialDecay(learning_rate=hps.train.learning_rate, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    # 创建优化器，并传入学习率调度器
    optim_g = paddle.optimizer.AdamW(learning_rate=lr_g, betas=hps.train.betas, epsilon=hps.train.eps, parameters=net_g.parameters())
    optim_d = paddle.optimizer.AdamW(learning_rate=lr_d, betas=hps.train.betas, epsilon=hps.train.eps, parameters=net_d.parameters())

    # 创建梯度缩放器
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    for epoch in epochs:
        for input, target in data:
            optim_g.clear_grad()
            optim_d.clear_grad()
            # 运行前向传播，并使用auto_cast自动选择合适的精度
            with paddle.amp.auto_cast():
                y_hat, ids_slice, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q
                ), pred_lf0, norm_lf0, lf0 = net_g(c, f0, uv, spec, g=g,
                c_lengths=lengths, spec_lengths=lengths)
                y_mel = commons.slice_segments(mel, ids_slice, hps.train.
                segment_size // hps.data.hop_length)
                y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(axis=1), hps.
                data.filter_length, hps.data.n_mel_channels, hps.data.
                sampling_rate, hps.data.hop_length, hps.data.win_length,
                hps.data.mel_fmin, hps.data.mel_fmax)
                y = commons.slice_segments(y, ids_slice * hps.data.hop_length,
                hps.train.segment_size)
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            # 使用梯度缩放器放大损失，并进行反向传播
            loss_mel = paddle.nn.functional.l1_loss(input=y_mel, label=
            y_hat_mel) * hps.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask
            ) * hps.train.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_lf0 = paddle.nn.functional.mse_loss(input=pred_lf0,
            label=lf0)
            loss_gen_all = (loss_gen + loss_fm + loss_mel + loss_kl +
            loss_lf0)
            scaled = scaler.scale(loss_gen_all)
            scaled.backward()
            # 使用梯度缩放器更新优化器，并缩小梯度
            scaler.step(optim_g)
            scaler.step(optim_d)
            scaler.update()
        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                reference_loss = 0
                for i in losses:
                    reference_loss += i
                logger.info('Train Epoch: {} [{:.0f}%]'.format(epoch, 100.0 *
                    batch_idx / len(train_loader)))
                logger.info(
                    f'Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}, reference_loss: {reference_loss}'
                    )
                scalar_dict = {'loss/g/total': loss_gen_all, 'loss/d/total':
                    loss_disc_all, 'learning_rate': lr, 'grad_norm_d':
                    grad_norm_d, 'grad_norm_g': grad_norm_g}
                scalar_dict.update({'loss/g/fm': loss_fm, 'loss/g/mel':
                    loss_mel, 'loss/g/kl': loss_kl, 'loss/g/lf0': loss_lf0})
                image_dict = {'slice/mel_org': utils.
                    plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    'slice/mel_gen': utils.plot_spectrogram_to_numpy(
                    y_hat_mel[0].data.cpu().numpy()), 'all/mel': utils.
                    plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    'all/lf0': utils.plot_data_to_numpy(lf0[0, 0, :].cpu().
                    numpy(), pred_lf0[0, 0, :].detach().cpu().numpy()),
                    'all/norm_lf0': utils.plot_data_to_numpy(lf0[0, 0, :].
                    cpu().numpy(), norm_lf0[0, 0, :].detach().cpu().numpy())}
                utils.summarize(writer=writer, global_step=global_step,
                    images=image_dict, scalars=scalar_dict)
            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)
                utils.save_checkpoint(net_g, optim_g, hps.train.
                    learning_rate, epoch, os.path.join(hps.model_dir,
                    'G_{}.pth'.format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.
                    learning_rate, epoch, os.path.join(hps.model_dir,
                    'D_{}.pth'.format(global_step)))
                keep_ckpts = getattr(hps.train, 'keep_ckpts', 0)
                if keep_ckpts > 0:
                    utils.clean_checkpoints(path_to_models=hps.model_dir,
                        n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
        global_step += 1
    if rank == 0:
        global start_time
        now = time.time()
        durtaion = format(now - start_time, '.2f')
        logger.info(f'====> Epoch: {epoch}, cost {durtaion} s')
        start_time = now


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with paddle.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            c, f0, spec, y, spk, _, uv = items
            g = spk[:1]
            spec, y = spec[:1], y[:1]
            c = c[:1]
            f0 = f0[:1]
            uv = uv[:1]
            mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.
                n_mel_channels, hps.data.sampling_rate, hps.data.mel_fmin,
                hps.data.mel_fmax)
            y_hat = generator.module.infer(c, f0, uv, g=g)
            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(axis=1).astype(
                dtype='float32'), hps.data.filter_length, hps.data.
                n_mel_channels, hps.data.sampling_rate, hps.data.hop_length,
                hps.data.win_length, hps.data.mel_fmin, hps.data.mel_fmax)
            audio_dict.update({f'gen/audio_{batch_idx}': y_hat[0],
                f'gt/audio_{batch_idx}': y[0]})
        image_dict.update({f'gen/mel': utils.plot_spectrogram_to_numpy(
            y_hat_mel[0].cpu().numpy()), 'gt/mel': utils.
            plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
    utils.summarize(writer=writer_eval, global_step=global_step, images=
        image_dict, audios=audio_dict, audio_sampling_rate=hps.data.
        sampling_rate)
    generator.train()


if __name__ == '__main__':
    main()
