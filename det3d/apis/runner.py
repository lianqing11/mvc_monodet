from mmcv.runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS
import mmcv
from mmcv.runner.utils import get_host_info
import time
@RUNNERS.register_module()
class SemiEpochBasedRunner(EpochBasedRunner):
    def semi_train(self, data_loader, semi_data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self.semi_data_loader = semi_data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        self.semi_data_loader_iter = iter(self.semi_data_loader)
        for i, data_batch in enumerate(self.data_loader):
            try:
                semi_data_batch = self.semi_data_loader_iter.__next__()
            except:
                self.semi_data_loader_iter = iter(self.semi_data_loader)
                semi_data_batch = self.semi_data_loader_iter.__next__()
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.semi_run_iter(data_batch,
                               semi_data_batch,
                               train_mode=True,
                               **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1


    def semi_run_iter(self,
                      data_batch,
                      semi_data_batch,
                      train_mode,
                      **kwargs):
        if self.batch_processor is not None:
            raise NotImplementedError
            outputs = self.batch_processor(
                self.model, data_batch, semi_data_batch,
                train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(
                data_batch, self.optimizer, semi_data_batch,
                **kwargs)
        else:
            raise NotImplementedError

        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')

        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs






    def semi_run(self,
                data_loaders,
                semi_data_loaders,
                workflow,
                max_epochs=None,
                **kwargs):
        """Start running.
        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            semi_data_loaders (list[:obj:'DataLoader']): Dataloaders for training
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    self.semi_train(data_loaders[i], semi_data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
