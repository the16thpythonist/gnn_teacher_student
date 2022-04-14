import os
import sys
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
from gnn_teacher_student.main import (StudentTeacherExplanationAnalysis,
                                      ExplanationPreTraining,
                                      ExplanationLoss,
                                      NoLoss)
from gnn_teacher_student.visualization import (plot_average_with_uncertainty)

import warnings
warnings.filterwarnings('ignore')
warnings.warn = lambda *args, **kwargs: 0
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.compat.v1.enable_eager_execution()

PATH = pathlib.Path(__file__).parent.parent.absolute()
BASE_PATH = os.getenv('EXPERIMENT_BASE_PATH', os.path.join(PATH, 'results'))

LENGTH = 2000
#SAMPLE_RATIOS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
SAMPLE_RATIOS = [1.0]
REPETITIONS = 1
EPOCHS = 100
DEVICE = '/cpu:0'

NAME = os.path.basename(__file__).replace('.py', '')
DESCRIPTION = """
MOTIVATION
----------
When doing a student teacher analysis, the results very much depend on a number of things. One factor
definitely is the quality of the explanations, and that is exactly what we want to find out with the
procedure. Unfortunately, the results also depend on other things as well. For example the results which can
be achieved depend on an equilibrium between the difficulty of the problem and the complexity / ability of
the used student architecture. The main problem is that if the student is too powerful for the problem, it
will learn most of the things on it's own, without needing the explanations. In such a case both student
variants will likely converge to a very good result and there will be no significant difference, meaning
we cannot make an assessment of explanation quality.

Now the hypothesis is: If the previously mentioned case can be observed, there should be an increasing
difference between the student variants when the problem is incrementally made harder.
One method of making the problem harder is by decreasing the dataset size.

DESCRIPTION
-----------
This experiment will create one base dataset and then incrementally use smaller subsets of this base
dataset to perform a repeated student teacher analysis with the goal of plotting the final average
difference of the validation metric over the different dataset sizes (=problem difficulty).
The expectation is that for smaller dataset sizes the learning effect from the explanations.
""" + COLORS_DESCRIPTION + COLOR_PAIRS_DESCRIPTION


with Experiment(base_path=BASE_PATH, name=NAME, description=DESCRIPTION, override=True) as e:
    e.logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    e.total_work = REPETITIONS * len(SAMPLE_RATIOS)

    # ~ Creating the base dataset
    base_dataset = generate_color_pairs_dataset(
        length=LENGTH,
        node_count_cb=lambda: random.randint(5, 50),
        additional_edge_count_cb=lambda: random.randint(1, 5),
        colors=[
            (1, 0, 0),  # red
            (0, 1, 0),  # green
            (0, 0, 1),  # blue
            (1, 1, 0),  # yellow
            (0, 1, 1),  # magenta
        ],
        exclude_empty=True
    )

    # ~ Setting up the student template
    student_template = StudentTemplate(
        student_class=SimpleAttentionStudent,
        student_name='attention_student',
        units=2,
        attention_units=3,
        activation='kgcnn>leaky_relu',
        attention_activation='tanh'
    )

    # In this dictionary we save the sample sizes as the keys and the values will be lists consisting of the
    # differences between the prediction metrics of the two student variants at the end of the training
    sample_final_prediction_metric_diffs: Dict[int, List[float]] = {}

    for sample_ratio in SAMPLE_RATIOS:
        sample_size = int(LENGTH * sample_ratio)

        e.log(f'DATASET SIZE: {sample_size}')
        result_list = []
        for i in range(REPETITIONS):
            e.log(f'   Repetition ({i+1}/{REPETITIONS})')
            dataset = random.sample(base_dataset, sample_size)
            _dataset = {field: [g[field] for g in dataset] for field in dataset[0].keys()}

            student_teacher_analysis = StudentTeacherExplanationAnalysis(
                student_template=student_template,
                epochs=EPOCHS,
                batch_size=32,
                optimizer=ks.optimizers.Adam(learning_rate=0.001),
                prediction_metric=ks.metrics.MeanSquaredError(),
                explanation_metric=ks.metrics.MeanAbsoluteError()
            )
            student_teacher_analysis.logger.addHandler(logging.StreamHandler(stream=sys.stdout))

            explanation_pre_training = ExplanationPreTraining(
                loss=[
                    ks.losses.MeanSquaredError(),
                    ExplanationLoss(),
                    ExplanationLoss()
                ],
                epochs=int(0.25 * EPOCHS),
                lock_explanation=True
            )

            #print(_dataset['edge_indices'][0].dtype)
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
                        }
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
                student_variant_color_map=colors
            )
            ax_pred.set_title('Prediction Validation MSE')
            ax_pred.set_xlabel('Epochs')
            ax_pred.set_ylabel('Mean Squared Error')
            ax_pred.legend()
            student_teacher_analysis.plot_metrics(
                ax=ax_node,
                results=results,
                metric='node_importance',
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

        fig_path = os.path.join(e.path, f'dataset_size_{sample_size}')
        fig.savefig(fig_path + '.pdf')

        # Now we want to calculate the difference between the final prediction MSE's for all the
        # training repetitions
        final_prediction_metric_diffs = [result['ref']['test_prediction_metric'][-1] -
                                         result['exp']['test_prediction_metric'][-1]
                                         for result in result_list]

        # With that we can compute an average and a standard deviations
        sample_final_prediction_metric_diffs[sample_size] = final_prediction_metric_diffs

    # Now that we have the statistical results for all the sample sizes, we can process those into a plot
    sample_sizes = []
    diff_avgs = []
    diff_stds = []
    for sample_size, prediction_diffs in sample_final_prediction_metric_diffs.items():
        diff_avg = np.mean(prediction_diffs)
        diff_std = np.mean(prediction_diffs)
        w, p = scipy.stats.wilcoxon(prediction_diffs)
        significant = p < 0.05

        e.log(f'sample size {sample_size}: avg={diff_avg:.3f} std={diff_std:.3f} p={p:.5f}')
        sample_sizes.append(sample_size)
        diff_avgs.append(diff_avg)
        diff_stds.append(diff_std)

    color_sample = 'green'

    fig_sample, ax_sample = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    fig_sample.patch.set_facecolor('white')

    xs = list(range(len(sample_sizes)))
    plot_average_with_uncertainty(
        ax=ax_sample,
        xs=xs,
        yss=[prediction_diffs for prediction_diffs in sample_final_prediction_metric_diffs.values()],
        color=color_sample,
        fill_alpha=0.05,
    )
    ax_sample.set_title('Difference in final Validation MSE')
    ax_sample.set_ylabel(r'$MSE_{ref} - MSE_{exp}$')
    ax_sample.set_xlabel('Dataset Size')
    ax_sample.set_xticks(xs)
    ax_sample.set_xticklabels(sample_sizes)

    fig_sample_path = os.path.join(e.path, 'final_prediction_diff_over_sample_size')
    fig_sample.savefig(fig_sample_path + '.pdf')
    fig_sample.savefig(fig_sample_path + '.png')





