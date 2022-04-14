import logging
import time
import random
import itertools
from copy import deepcopy
from typing import List, Sequence, Dict, Optional, Any, Union, Callable

import scipy
import scipy.stats
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from kgcnn.data.moleculenet import MemoryGraphDataset
from kgcnn.utils.data import ragged_tensor_from_nested_numpy

from gnn_teacher_student.students import StudentTemplate, AbstractStudent


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


class LogProgressCallback(ks.callbacks.Callback):

    def __init__(self,
                 logger: logging.Logger,
                 identifier: str,
                 epoch_step: int):
        ks.callbacks.Callback.__init__(self)

        self.logger = logger
        self.identifier = identifier
        self.epoch_step = epoch_step

        self.start_time = time.time()
        self.elapsed_time = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epoch_step == 0:
            self.elapsed_time = time.time() - self.start_time
            value = logs[self.identifier]

            self.logger.info(
                f'   epoch {str(epoch):<5}: value={value:.1f} elapsed_time={self.elapsed_time:.1f}s'
            )


class StudentTrainingStrategy:

    def __init__(self):
        pass

    def create_kwargs(self):
        raise NotImplemented

    def __call__(self):
        return self.create_kwargs()


class LossWeightSwitchCallback(ks.callbacks.Callback):

    def __init__(self,
                 loss_weights: List[tf.Variable],
                 weights1: List[float],
                 weights2: List[float],
                 epoch_threshold: int):
        ks.callbacks.Callback.__init__(self)

        self.loss_weights = loss_weights
        self.weights1 = weights1
        self.weights2 = weights2
        self.epoch_threshold = epoch_threshold

    def on_epoch_begin(self, epoch, logs=None):
        for var, w1, w2 in zip(self.loss_weights, self.weights1, self.weights2):
            if epoch < self.epoch_threshold:
                K.set_value(var, w1)
            else:
                K.set_value(var, w2)


class LockExplanationCallback(ks.callbacks.Callback):

    def __init__(self,
                 epoch_threshold: int):
        ks.callbacks.Callback.__init__(self)
        self.epoch_threshold = epoch_threshold

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.epoch_threshold:
            while len(self.model._trainable_weights) != 0:
                w = self.model._trainable_weights.pop()
                self.model._non_trainable_weights.append(w)


class ExplanationPreTraining(StudentTrainingStrategy):

    def __init__(self,
                 loss: List[ks.losses.Loss],
                 epochs: int,
                 lock_explanation: bool = False):
        StudentTrainingStrategy.__init__(self)

        self.loss = loss
        self.epochs = epochs
        self.lock_explanation = lock_explanation

        self.loss_weights = [
            tf.Variable(1.0),
            tf.Variable(1.0),
            tf.Variable(1.0)
        ]

        self.loss_weight_switch = LossWeightSwitchCallback(
            loss_weights=self.loss_weights,
            weights1=[0, 1, 1],
            weights2=[1, 0, 0],
            epoch_threshold=self.epochs
        )
        self.lock_explanation = LockExplanationCallback(
            epoch_threshold=self.epochs
        )

    def create_kwargs(self):
        kwargs = {
            'loss': self.loss,
            'loss_weights': self.loss_weights,
            'callbacks': [
                self.loss_weight_switch,
                self.lock_explanation
            ]
        }
        return kwargs


# == STUDENT TEACHER ANALYSIS ===============================================================================
# ===========================================================================================================

class StudentTeacherExplanationAnalysis:

    DATASET_REQUIRED_FIELDS = {
        'node_attributes': {
            'type': list,
        },
        'edge_attributes': {
            'type': list,
        },
        'edge_indices': {
            'type': list,
        },
        'graph_labels': {
            'type': list,
        },
        'node_importances': {
            'type': list,
        },
        'edge_importances': {
            'type': list,
        }
    }

    def __init__(self,
                 student_template: StudentTemplate,
                 epochs: int,
                 batch_size: int,
                 optimizer: ks.optimizers.Optimizer = ks.optimizers.Adam(),
                 prediction_metric: ks.metrics.Metric = ks.metrics.Accuracy(),
                 explanation_metric: ks.metrics.Metric = ks.metrics.MeanAbsoluteError(),
                 variants: Sequence[str] = ('exp', 'ref'),
                 random_state: Optional[int] = None):

        self.student_template = student_template

        self.epochs = epochs
        self.batch_size = batch_size

        self.optimizer = optimizer
        self.prediction_metric = prediction_metric
        self.explanation_metric = explanation_metric

        self.random_state: Optional[int] = random_state

        self.student_models = {}
        self.variants: List[str] = list(variants)
        self.students: Dict[str, AbstractStudent] = {variant: student_template(variant)
                                                     for variant in variants}

        self.results = {}
        self.logger = logging.Logger(self.__class__.__name__)

    def fit(self,
            dataset: Union[Dict[str, list], MemoryGraphDataset],
            train_split: float,
            variant_kwargs: Dict[str, Dict[str, Any]],
            random_state: Optional[int] = 0,
            verbose: int = 0,
            log_progress: Optional[int] = None):

        # ~ Processing the dataset
        # The "dataset" argument is supposed to contain all the data for the training process. This includes
        # the input data (= graph representation consisting of node attributes, edge indices and edge
        # weights) and the explanations (= importance weights for nodes and edges)
        dataset = self._process_dataset(dataset)

        xtrain, ytrain, xtest, ytest = self._split_dataset(dataset, train_split, random_state)
        print(xtrain[2].dtype)
        node_attribute_count = dataset['node_attributes'][0].shape[1]
        edge_attribute_count = dataset['edge_attributes'][0].shape[1]

        for student_variant, model in self.students.items():

            # Builds the model
            model([
                xtrain[0][0:1],
                xtrain[1][0:1],
                xtrain[2][0:1]
            ])

            # Each student model has a flag with which it can signal whether or not it supports batching,
            # only if batching is explicitly configured we use the generally !=1 batch size which was passed
            # to the constructor. Otherwise we use 1
            if model.supports_batching:
                batch_size = self.batch_size
            else:
                batch_size = 1

            # ~ compiling the model
            loss = variant_kwargs[student_variant]['loss']
            loss_weights = variant_kwargs[student_variant]['loss_weights']

            model.compile(
                loss=loss,
                loss_weights=loss_weights,
                optimizer=self.optimizer,
                metrics=[
                    self.prediction_metric,
                    self.explanation_metric
                ]
            )

            # ~ training the model
            self.logger.info(f'starting student training "{model.name}:{student_variant}" '
                             f'[LOSS:] prediction="{loss[0].name}*{loss_weights[0]}" '
                             f'node_importance="{loss[1].name}*{loss_weights[1]}" '
                             f'edge_importance="{loss[2].name}*{loss_weights[2]}" '
                             f'[BATCHING:] batch_size={batch_size} '
                             f'supports_batching="{model.supports_batching}" '
                             f'[MODEL:] parameters={model.count_params()} '
                             f'[TRAINING:] epochs={self.epochs} '
                             f'optimizer={self.optimizer.__class__.__name__} '
                             f'dataset_size={len(dataset)}) ')

            # ~ Assembling the callbacks
            callbacks = []

            if log_progress is not None:
                callbacks.append(LogProgressCallback(
                    logger=self.logger,
                    epoch_step=log_progress,
                    identifier=f'val_output_1_{self.prediction_metric.name}'
                ))

            if 'callbacks' in variant_kwargs[student_variant].keys():
                callbacks += variant_kwargs[student_variant]['callbacks']

            start_time = time.process_time()
            hist: ks.callbacks.History = model.fit(
                xtrain,
                ytrain,
                epochs=self.epochs,
                batch_size=batch_size,
                validation_freq=1,
                validation_data=(xtest, ytest),
                callbacks=callbacks,
                verbose=verbose
            )
            stop_time = time.process_time()

            self.results[student_variant] = {
                'model': model,
                'name': model.name,
                # The dataset which was used in the training process
                'dataset': dataset,
                'xtest': xtest,
                'ytest': ytest,
                # Meta information about the training process
                'duration': stop_time - start_time,
                'epochs': self.epochs,
                'ts': np.arange(self.epochs),
                # Training prediction & explanation loss & metric over all epochs
                'train_prediction_metric': np.array(
                    hist.history[f'output_1_{self.prediction_metric.name}']
                ),
                'train_node_importance_metric': np.array(
                    hist.history[f'output_2_{self.explanation_metric.name}']
                ),
                'train_edge_importance_metric': np.array(
                    hist.history[f'output_3_{self.explanation_metric.name}']
                ),
                # Validation prediction & explanation loss & metric over all epochs
                'test_prediction_metric': np.array(
                    hist.history[f'val_output_1_{self.prediction_metric.name}']
                ),
                'test_node_importance_metric': np.array(
                    hist.history[f'val_output_2_{self.explanation_metric.name}']
                ),
                'test_edge_importance_metric': np.array(
                    hist.history[f'val_output_3_{self.explanation_metric.name}']
                )
            }

        duration = sum(d['duration'] for _, d in self.results.items())
        self.logger.info(f'student teacher analysis complete after {duration/60:.1f} minutes')

        return self.results

    def calc_final(self,
                   results: dict,
                   student_variants: List[str] = ('exp', 'ref'),
                   error_func: Callable = lambda y_true, y_pred: y_true - y_pred[0]):

        for variant in student_variants:
            data = results[variant]
            model = data['model']
            predictions = model(data['xtest'])

            errors = [error_func(y_true, y_pred)
                      for y_true, y_pred in zip(data['ytest'][0], predictions[0])]

            results[variant]['test_errors'] = errors

        r = {}
        print(itertools.combinations(student_variants, 2))
        for variant1, variant2 in itertools.combinations(student_variants, 2):
            errors1 = results[variant1]['test_errors']
            errors2 = results[variant2]['test_errors']

            errors1_avg = np.mean(errors1)
            errors2_avg = np.mean(errors2)

            r[(variant1, variant2)] = {
                f'{variant1}_avg_error':    errors1_avg,
                f'{variant2}_avg_error':    errors2_avg,
                'avg_error_div':            errors1_avg - errors2_avg,
                'wilcoxon_result':          scipy.stats.wilcoxon(errors1, errors2)
            }

        return r


    def plot_final(self,
                   ax: plt.Axes,
                   results: dict,
                   metric: str = 'prediction',
                   error_func: Callable = lambda y_true, y_pred: y_true - y_pred[0],
                   student_variants: List[str] = ('exp', 'ref'),
                   student_variant_color_map: Optional[Dict[str, Any]] = None,
                   marker_size: int = 700,
                   marker_style: str = 'o'):

        # If we do not explicitly get a color dict, then we randomly choose one of matplotlib's predefined
        # TABLEAU colors and use that color for all the student variants
        if student_variant_color_map is None:
            color = random.choice(list(mcolors.TABLEAU_COLORS.values()))
            student_variant_color_map = {variant: color for variant in student_variants}

        variant_errors_map = {}
        for student_variant in student_variants:
            data = results[student_variant]
            model = data['model']
            predictions = model(data['xtest'])

            index = {'prediction': 0, 'node_importance': 1, 'edge_importance': 2}[metric]
            errors = [error_func(y_true, y_pred)
                      for y_true, y_pred in zip(data['ytest'][index], predictions[index])]

            variant_errors_map[student_variant] = errors

        # ~ plotting
        xticks = []
        for i, (student_variant, errors) in enumerate(variant_errors_map.items()):
            error_avg = np.mean(errors)
            error_std = np.std(errors)

            ax.scatter(
                i + 1, error_avg,
                c=student_variant_color_map[student_variant],
                s=marker_size,
                marker=marker_style,
            )

            ax.errorbar(
                i + 1, error_avg,
                xerr=None,
                yerr=error_std,
                ecolor=student_variant_color_map[student_variant],
                elinewidth=0.005 * marker_size,
                capsize=10,
                capthick=0.005 * marker_size,
            )

            xticks.append(i + 1)

        ax.set_xticks(xticks)

        return ax

    def plot_metrics(self,
                     ax: plt.Axes,
                     results: dict,
                     metric: str = 'node_importance',
                     version: str = 'test',
                     student_variants: List[str] = ['exp', 'ref'],
                     student_variant_alpha_map: Dict[str, int] = {'exp': 1.0, 'ref': 0.4},
                     student_variant_line_style_map: Dict[str, str] = {'exp': '-', 'ref': '-'},
                     student_variant_color_map: Optional[Dict[str, Any]] = None) -> plt.Axes:
        random.seed(self.random_state)

        # If we do not explicitly get a color dict, then we randomly choose one of matplotlib's predefined
        # TABLEAU colors and use that color for all the student variants
        if student_variant_color_map is None:
            color = random.choice(list(mcolors.TABLEAU_COLORS.values()))
            student_variant_color_map = {variant: color for variant in student_variants}

        for student_variant in student_variants:

            data = results[student_variant]
            ax.plot(
                data[f'ts'],
                data[f'{version}_{metric}_metric'],
                c=student_variant_color_map[student_variant],
                alpha=student_variant_alpha_map[student_variant],
                ls=student_variant_line_style_map[student_variant],
                label=f'{data["name"]}:{student_variant} ({version})'
            )

        return ax

    @staticmethod
    def _from_snake_case(string: str) -> str:
        string_split = string.split('_')
        return ' '.join([s.capitalize() for s in string_split])

    def _split_dataset(self, dataset: dict, train_split: float, random_state: int) -> tuple:
        # Train Test split
        labels_train, labels_test, \
            nodes_train, nodes_test, \
            edges_train, edges_test, \
            edge_indices_train, edge_indices_test, \
            node_importances_train, node_importances_test, \
            edge_importances_train, edge_importances_test \
            = train_test_split(
                dataset['graph_labels'],
                dataset['node_attributes'],
                dataset['edge_attributes'],
                dataset['edge_indices'],
                dataset['node_importances'],
                dataset['edge_importances'],
                train_size=train_split,
                random_state=random_state
            )

        # The train scores
        xtrain = (
            ragged_tensor_from_nested_numpy(nodes_train),
            ragged_tensor_from_nested_numpy(edges_train),
            ragged_tensor_from_nested_numpy(edge_indices_train)
        )

        xtest = (
            ragged_tensor_from_nested_numpy(nodes_test),
            ragged_tensor_from_nested_numpy(edges_test),
            ragged_tensor_from_nested_numpy(edge_indices_test)
        )

        # The importance scores
        ytrain = (
            # ragged_tensor_from_nested_numpy(labels_train),
            np.array(labels_train),
            ragged_tensor_from_nested_numpy(node_importances_train),
            ragged_tensor_from_nested_numpy(edge_importances_train)
        )

        ytest = (
            # ragged_tensor_from_nested_numpy(labels_test),
            np.array(labels_test),
            ragged_tensor_from_nested_numpy(node_importances_test),
            ragged_tensor_from_nested_numpy(edge_importances_test)
        )

        return xtrain, ytrain, xtest, ytest

    def _process_dataset(self, dataset: Union[Dict[str, list], MemoryGraphDataset]) -> Dict[str, list]:
        # dataset my be a dict
        if isinstance(dataset, dict):
            return self._process_dataset_dict(dataset)

        # but it also may be an object and those cases have to be processed differently
        else:
            return self._process_dataset_object(dataset)

    def _process_dataset_dict(self, dataset: Dict[str, list]) -> Dict[str, list]:
        processed_dataset = {}

        for field, info in self.DATASET_REQUIRED_FIELDS.items():
            # Here we check if the dataset dictionary actually contains all the elements which we need!
            assert field in dataset.keys() and isinstance(dataset[field], info['type']), \
                f'dataset needs to contain field "{field}" of type {info["type"]}'

            # If that is the case we potentially need to convert to ragged tensors
            field_value = dataset[field]
            processed_dataset[field] = [np.array(element) for element in field_value]

        return processed_dataset

    def _process_dataset_object(self, dataset: Any) -> Dict[str, list]:
        processed_dataset = {}

        for field, info in self.DATASET_REQUIRED_FIELDS.items():
            # Here we check if the dataset object actually contains all the fields which are required!
            assert hasattr(dataset, field) and isinstance(getattr(dataset, field), info['type']), \
                f'dataset needs to contain field "{field}" of type {info["type"]}'

            # If that is the case we potentially need to convert to ragged tensors
            field_value = getattr(dataset, field)
            processed_dataset[field] = [np.array(element) for element in field_value]

        return processed_dataset

    @staticmethod
    def _list_to_ragged_tensor(self, elements: list) -> tf.RaggedTensor:
        if not isinstance(elements, tf.RaggedTensor):
            return ragged_tensor_from_nested_numpy(elements)

        else:
            return elements
