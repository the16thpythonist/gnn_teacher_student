from typing import Callable, List
from collections import OrderedDict, defaultdict
from pprint import pprint

import tensorflow.keras as ks

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
            hists.append(hist)
            cb(self.model, hist)

            self.current_epoch += epoch_diff

        merged_hist = self.merge_histories(hists)
        return merged_hist

    def merge_histories(self, hists: List[ks.callbacks.History]):
        merged_hist = ks.callbacks.History()

        merged_history = defaultdict(list)
        for hist in hists:
            for key, value in hist.history.items():
                if isinstance(value, list):
                    merged_history[key] += value

        merged_hist.history = merged_history
        return merged_hist

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
