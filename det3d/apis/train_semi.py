import warnings

import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmcv.utils import build_from_cfg
from torch import distributed as dist

from mmdet3d.datasets import build_dataset
from mmdet3d.utils import find_latest_checkpoint
from mmdet.core import DistEvalHook as MMDET_DistEvalHook
from mmdet.core import EvalHook as MMDET_EvalHook
from mmdet.datasets import build_dataloader as build_mmdet_dataloader
from mmdet.datasets import replace_ImageToTensor
from mmdet.utils import get_root_logger as get_mmdet_root_logger
from mmseg.core import DistEvalHook as MMSEG_DistEvalHook
from mmseg.core import EvalHook as MMSEG_EvalHook
from mmseg.datasets import build_dataloader as build_mmseg_dataloader
from mmseg.utils import get_root_logger as get_mmseg_root_logger


def train_semi_model(model,
                    dataset,
                    semi_dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """A function wrapper for launching model semi training according to cfg.
    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """

    if cfg.model.type in ['EncoderDecoder3D']:
        raise NotImplementedError

    logger = get_mmdet_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    semi_dataset = semi_dataset if isinstance(semi_dataset, (list, tuple)) else [semi_dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']
    data_loaders = [
        build_mmdet_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            runner_type=runner_type,
            persistent_workers=cfg.data.get('persistent_workers', False))
        for ds in dataset
    ]

    semi_data_loaders = [
        build_mmdet_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            runner_type=runner_type,
            persistent_workers=cfg.data.get('persistent_workers', False))
        for ds in semi_dataset
    ]


    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_mmdet_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = MMDET_DistEvalHook if distributed else MMDET_EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)

    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.semi_run(data_loaders, semi_data_loaders, cfg.workflow)
