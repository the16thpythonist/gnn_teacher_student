import logging
import time
from typing import Callable, List
from collections import OrderedDict, defaultdict
from pprint import pprint

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

from gnn_teacher_student.students import AbstractStudent


# == TRAINING SEGMENTATION ==================================================================================
# ===========================================================================================================

class SegmentedFitProcess:

    def __init__(self,
                 model: AbstractStudent,
                 fit_kwargs: dict):
        self.model = model
        self.fit_kwargs = fit_kwargs

        self.total_epochs = fit_kwargs['epochs']
        self.callbacks = dict()
        self.callbacks[self.total_epochs] = lambda m, h: None
        self.current_epoch = 0

    def __call__(self):

        hists = []
        for epoch, cb in sorted(self.callbacks.items(), key=lambda i: i[0]):
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


class LockExplanationManager(FitManager):

    def __init__(self,
                 epoch_threshold: int):
        FitManager.__init__(self)
        self.epoch_threshold = epoch_threshold

    def register_callbacks(self, fit_process, fit_kwargs, model):
        fit_process.add_callback(self.epoch_threshold, self.lock_model_explanation_parameters)

    def lock_model_explanation_parameters(self, model, history):
        model.lock_explanation_parameters()
        model.recompile()


# == CUSTOM LOSSES ==========================================================================================
# ===========================================================================================================

class NoLoss(ks.losses.Loss):

    def __init__(self):
        ks.losses.Loss.__init__(self)
        self.name = 'no_loss'

    def call(self, y_true, y_pred):
        return 0.0


class ExplanationLoss(ks.losses.Loss):

    def __init__(self,
                 loss_function: ks.losses.Loss = ks.losses.binary_crossentropy,
                 mask_empty_explanations: bool = True):
        ks.losses.Loss.__init__(self)

        self.loss_function = loss_function
        self.mask_empty_explanations = mask_empty_explanations
        self.name = 'explanation_loss'

    def call(self, y_true, y_pred):
        loss = self.loss_function(y_true, y_pred)
        mask = tf.cast(tf.reduce_max(y_true, axis=-1) > 0, dtype=tf.float32)

        if self.mask_empty_explanations:
            loss *= mask

        loss = tf.reduce_mean(loss, axis=-1)
        return loss


# == CUSTOM CALLBACKS =======================================================================================
# ===========================================================================================================

class EpochCounterCallback(ks.callbacks.Callback):

    def __init__(self):
        super(EpochCounterCallback, self).__init__()
        self.epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch += 1


class LogProgressCallback(EpochCounterCallback):

    def __init__(self,
                 logger: logging.Logger,
                 identifier: str,
                 epoch_step: int):
        super(LogProgressCallback, self).__init__()

        self.logger = logger
        self.identifier = identifier
        self.epoch_step = epoch_step

        self.start_time = time.time()
        self.elapsed_time = 0

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch % self.epoch_step == 0:
            self.elapsed_time = time.time() - self.start_time
            value = logs[self.identifier]

            self.logger.info(
                f'   epoch {str(self.epoch):<5}: '
                f'{self.identifier}={value:.1f} '
                f'elapsed_time={self.elapsed_time:.1f}s'
            )


class LossWeightSwitchCallback(EpochCounterCallback):

    def __init__(self,
                 loss_weights: List[tf.Variable],
                 weights1: List[float],
                 weights2: List[float],
                 epoch_threshold: int):
        super(LossWeightSwitchCallback, self).__init__()

        self.loss_weights = loss_weights
        self.weights1 = weights1
        self.weights2 = weights2
        self.epoch_threshold = epoch_threshold

    def on_epoch_begin(self, epoch, logs=None):
        if self.epoch == 0:
            for var, w1, w2 in zip(self.loss_weights, self.weights1, self.weights2):
                K.set_value(var, w1)

        if self.epoch == self.epoch_threshold:
            for var, w1, w2 in zip(self.loss_weights, self.weights1, self.weights2):
                K.set_value(var, w2)

        super(LossWeightSwitchCallback, self).on_epoch_begin(epoch, logs)


# == STUDENT TRAINING STRATEGIES ============================================================================
# ===========================================================================================================


class StudentTrainingStrategy:

    def __init__(self):
        pass

    def create_kwargs(self):
        raise NotImplemented

    def __call__(self):
        return self.create_kwargs()


class ExplanationPreTraining(StudentTrainingStrategy):

    def __init__(self,
                 loss: List[ks.losses.Loss],
                 epochs: int,
                 post_weights: [1, 0, 0],
                 lock_explanation: bool = False):
        StudentTrainingStrategy.__init__(self)

        self.loss = loss
        self.epochs = epochs
        self.lock_explanation = lock_explanation

        self.loss_weights = [
            tf.Variable(1.0, dtype=tf.float32, trainable=False),
            tf.Variable(1.0, dtype=tf.float32, trainable=False),
            tf.Variable(1.0, dtype=tf.float32, trainable=False)
        ]

        self.loss_weight_switch = LossWeightSwitchCallback(
            loss_weights=self.loss_weights,
            weights1=[0, 1, 1],
            weights2=post_weights,
            epoch_threshold=self.epochs
        )
        self.lock_explanation_manager = LockExplanationManager(
            epoch_threshold=self.epochs
        )

    def create_kwargs(self):
        kwargs = {
            'loss': self.loss,
            'loss_weights': self.loss_weights,
            'callbacks': [
                self.loss_weight_switch,
            ]
        }
        if self.lock_explanation:
            kwargs['fit_manager'] = self.lock_explanation_manager

        return kwargs


