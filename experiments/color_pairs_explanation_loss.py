import os
import sys
import json
import pathlib
import random
import itertools
import logging
from typing import Dict, List

import scipy
import scipy.stats
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import matplotlib.pyplot as plt

from gnn_teacher_student.experiment import Experiment
from gnn_teacher_student.data import (generate_color_pairs_dataset,
                                      COLORS_DESCRIPTION,
                                      COLOR_PAIRS_DESCRIPTION)
from gnn_teacher_student.students import StudentTemplate, SimpleAttentionStudent
from gnn_teacher_student.main import StudentTeacherExplanationAnalysis
from gnn_teacher_student.training import (ExplanationPreTraining,
                                          ExplanationLoss,
                                          NoLoss)
from gnn_teacher_student.visualization import (plot_average_with_uncertainty)

import warnings
warnings.filterwarnings('ignore')
warnings.warn = lambda *args, **kwargs: 0
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'font.size': 18,
    'font.family': 'sans-serif',
    'legend.fontsize': 16,
})

PATH = os.path.dirname(pathlib.Path(__file__).parent.absolute())
BASE_PATH = os.getenv('EXPERIMENT_BASE_PATH', os.path.join(PATH, 'results'))

LENGTH = 1000
RANDOMIZE_DATASET = True
REPETITIONS = 10
EPOCHS = 10000
LEARNING_RATE = 0.001
BATCH_SIZE = 128
LOSSES = {
    'Binary Crossentropy': ks.losses.binary_crossentropy,
    'Mean Absolute Error': ks.losses.mean_absolute_error,
    'Mean Squared Error': ks.losses.mean_squared_error
}
DEVICE = '/cpu:0'

NAME = os.path.basename(__file__).replace('.py', '')
DESCRIPTION = """
MOTIVATION
==========
During preliminary experiments I had observed that the quality of the explanation supervision procedure
for the explanation-aware student of the student teacher analysis strongly depended on the choice of
explanation loss function. Although these were mainly heuristic observations. Now it would be interesting
to have some qualitative evidence which loss function works.

DESCRIPTION
===========
This experiment will use one base dataset configuration of the "COLORS" dataset generation method and the
"COUNT COLOR PAIRS" task. For different explanation loss functions and otherwise same parameters, a training
process will be repeated a number of times to get a statistical result for the final validation metric
difference between the explanation and the reference students.
""" + COLORS_DESCRIPTION + COLOR_PAIRS_DESCRIPTION

# This data structure will later on contain all the information about the final prediction performance
# metrics for each of the variations of the experiment variable as well as all the individual repetitions
# (on the level below)
# Specifically, the keys of this dict will be the (string) identifiers for the experiment parameters and
# the values will be dicts again, where the string keys of those dicts will be "exp" and "ref" and the
# corresponding values will be lists contain the final validation error metrics for that student.
final_prediction_metrics_map: Dict[str, Dict[str, List[float]]] = {}

results_map: Dict[str, List[dict]] = {}


with Experiment(base_path=BASE_PATH, name=NAME, description=DESCRIPTION, override=True) as e:
    e.logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    e.copy_code_file(pathlib.Path(__file__).absolute())
    e.total_work = REPETITIONS * len(LOSSES)

    # ~ Creating the dataset
    dataset_kwargs = {
        'length': LENGTH,
        'node_count_cb': lambda: random.randint(5, 50),
        'additional_edge_count_cb': lambda: random.randint(1, 5),
        'colors': [
            (1, 0, 0),  # red
            (0, 1, 0),  # green
            (0, 0, 1),  # blue
            (1, 1, 0),  # yellow
            (0, 1, 1),  # magenta
        ],
        'exclude_empty': True
    }
    dataset = generate_color_pairs_dataset(**dataset_kwargs)

    # ~ Setting up the student template
    student_template = StudentTemplate(
        student_class=SimpleAttentionStudent,
        student_name='attention-student',
        units=2,
        attention_units=1,
        activation='kgcnn>leaky_relu',
        attention_activation='tanh'
    )

    # In this dictionary we save the sample sizes as the keys and the values will be lists consisting of the
    # differences between the prediction metrics of the two student variants at the end of the training
    sample_final_prediction_metric_diffs: Dict[int, List[float]] = {}

    for loss_name, loss_function in LOSSES.items():

        e.log(f'LOSS FUNCTION: "{loss_name}"')

        result_list = []
        for i in range(REPETITIONS):
            e.log(f'   Repetition ({i+1}/{REPETITIONS})')

            # This flag controls whether or not the dataset should be re-generated during each repetition.
            # Setting this flag to True will probably require more repetitions to get a statistically solid
            # result as the dataset randomization does induce a fair bit of noise, but the obtained results
            # should be more solid in that there is a lesser chance that they are influenced by a
            # specifically (un-)favorable dataset.
            if RANDOMIZE_DATASET:
                dataset = generate_color_pairs_dataset(**dataset_kwargs)

            _dataset = {field: [g[field] for g in dataset] for field in dataset[0].keys()}

            student_teacher_analysis = StudentTeacherExplanationAnalysis(
                student_template=student_template,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                optimizer=ks.optimizers.Adam(learning_rate=LEARNING_RATE),
                prediction_metric=ks.metrics.MeanSquaredError(),
                explanation_metric=ks.metrics.MeanAbsoluteError()
            )
            # We also want to be able to follow the progress of the experiment in real time, which is why
            # we also print the log messages to the console here.
            student_teacher_analysis.logger.addHandler(logging.StreamHandler(stream=sys.stdout))

            explanation_pre_training = ExplanationPreTraining(
                loss=[
                    ks.losses.MeanSquaredError(),
                    ExplanationLoss(loss_function=loss_function),
                    ExplanationLoss(loss_function=loss_function)
                ],
                epochs=int(0.25 * EPOCHS),
                post_weights=[1, 0.2, 0.2],
                lock_explanation=False
            )

            with tf.device(DEVICE):
                results = student_teacher_analysis.fit(
                    dataset=_dataset,
                    train_split=0.8,
                    variant_kwargs={
                        'exp': explanation_pre_training(),
                        'ref': {
                            'loss': [
                                ks.losses.MeanSquaredError(),
                                NoLoss(),
                                NoLoss()
                            ],
                            'loss_weights': [1, 0, 0],
                            'callbacks': []
                        },
                    },
                    log_progress=int(0.2 * EPOCHS)
                )
                result_list.append(results)

            e.update()

        # After all the individual repetitions have finished we want to create plots about the results
        colors = {'exp': 'green', 'ref': 'blue'}
        fig, rows = plt.subplots(ncols=3, nrows=REPETITIONS, figsize=(24, 8 * REPETITIONS), squeeze=False)
        for (ax_pred, ax_node, ax_edge), results in zip(rows, result_list):
            student_teacher_analysis.plot_metrics(
                ax=ax_pred,
                results=results,
                metric='prediction',
                version='test',
                student_variant_color_map=colors
            )
            student_teacher_analysis.plot_metrics(
                ax=ax_pred,
                results=results,
                metric='prediction',
                version='train',
                student_variant_color_map=colors,
                student_variant_alpha_map={'exp': 0.2, 'ref': 0.2}
            )
            ax_pred.set_title('Prediction Validation MSE')
            ax_pred.set_xlabel('Epochs')
            ax_pred.set_ylabel('Mean Squared Error')
            ax_pred.set_ylim([0, 10])
            ax_pred.legend()
            student_teacher_analysis.plot_metrics(
                ax=ax_node,
                results=results,
                metric='node_importance',
                version='test',
                student_variant_color_map=colors
            )

            ax_node.set_title('Node Importance Validation MAE')
            ax_node.set_xlabel('Epochs')
            ax_node.set_ylabel('Mean Absolute Error')
            student_teacher_analysis.plot_metrics(
                ax=ax_edge,
                results=results,
                metric='edge_importance',
                student_variant_color_map=colors
            )
            ax_edge.set_title('Edge Importance Validation MAE')
            ax_edge.set_xlabel('Epochs')
            ax_edge.set_ylabel('Mean Absolute Error')

        loss_slag = loss_name.lower().replace(' ', '_')
        fig_path = os.path.join(e.path, f'loss_function_{loss_slag}')
        fig.savefig(fig_path + '.pdf',
                    bbox_inches='tight',
                    pad_inches=0.05)

        final_prediction_metrics = {
            'ref': [result['ref']['test_prediction_metric'][-1] for result in result_list],
            'exp': [result['exp']['test_prediction_metric'][-1] for result in result_list]
        }

        # With that we can compute an average and a standard deviations
        final_prediction_metrics_map[loss_name] = final_prediction_metrics

        results_map[loss_name] = [{
            'ref': {
                'test_prediction_metric': list(result['ref']['test_prediction_metric']),
                'test_node_importance_metric': list(result['ref']['test_node_importance_metric']),
                'test_edge_importance_metric': list(result['ref']['test_edge_importance_metric'])
            },
            'exp': {
                'test_prediction_metric': list(result['exp']['test_prediction_metric']),
                'test_node_importance_metric': list(result['exp']['test_node_importance_metric']),
                'test_edge_importance_metric': list(result['exp']['test_edge_importance_metric'])
            }
        } for result in result_list]

    # Now that we have the statistical results for all the different loss functions, we can save this data
    # and process it further into plots
    fig_bar, ax_bar = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    xs = []
    loss_names = []
    for x, (loss_name, final_prediction_metrics) in enumerate(final_prediction_metrics_map.items()):
        diffs = [m_ref - m_exp
                 for m_ref, m_exp in zip(final_prediction_metrics['ref'], final_prediction_metrics['exp'])]
        avg = np.mean(diffs)
        std = np.std(diffs)

        ax_bar.scatter(x, avg)
        xs.append(x)
        loss_names.append(loss_name)

    ax_bar.set_title('Final Prediction Metric Difference')
    ax_bar.set_ylabel(r'$\text{MSE}_{\text{ref}} - \text{MSE}_{\text{exp}}$')
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels([loss_name.replace(' ', '\n') for loss_name in loss_names])

    fig_bar_path = os.path.join(e.path, 'final_prediction_diff_for_losses')
    fig_bar.savefig(fig_bar_path + '.pdf',
                    bbox_inches='tight',
                    pad_inches=0.05)

    # Saving the raw data
    results_path = os.path.join(e.path, 'results.json')
    with open(results_path, mode='w') as file:
        content = json.dumps(results_map, indent=4)
        file.write(content)

    final_metrics_path = os.path.join(e.path, 'final_prediction_metrics.json')
    with open(final_metrics_path, mode='w') as file:
        content = json.dumps(final_prediction_metrics_map, indent=4)
        file.write(content)
