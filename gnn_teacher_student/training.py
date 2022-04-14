from typing import Callable
from collections import OrderedDict

from gnn_teacher_student.students import AbstractStudent


class SegmentedFitProcess:

    def __init__(self,
                 model: AbstractStudent,
                 fit_kwargs: dict):
        self.model = model
        self.fit_kwargs = fit_kwargs

        self.total_epochs = fit_kwargs['epochs']
        self.callbacks = OrderedDict()
        self.callbacks[self.total_epochs] = lambda m, h: None
        self.current_epoch = 0

    def __call__(self):

        hists = []
        for epoch, cb in self.callbacks.items():
            epoch_diff = epoch - self.current_epoch

            fit_kwargs = self.fit_kwargs.copy()
            fit_kwargs['epochs'] = epoch_diff
            
            hist = self.model.fit(**fit_kwargs)
            cb(self.model, hist)

            self.current_epoch += epoch_diff

        # TODO: Merge histories
        return hists

    def add_callback(self,
                     epochs: int,
                     callback: Callable) -> None:
        self.callbacks[epochs] = callback


class FitManager:

    def __init__(self):
        pass

    def __call__(self,
                 model: AbstractStudent,
                 fit_kwargs: dict) -> SegmentedFitProcess:
        fit_process = SegmentedFitProcess(model, fit_kwargs)
        self.register_callbacks(fit_process, fit_kwargs, model)

        return fit_process

    def register_callbacks(self,
                           fit_process: SegmentedFitProcess,
                           fit_kwargs: dict,
                           model: AbstractStudent) -> None:
        return
