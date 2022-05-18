import os
import sys
import pathlib
import random
from typing import List

from gnn_teacher_student.data import (generate_color_pairs_dataset,
                                      COLORS_DESCRIPTION,
                                      COLOR_PAIRS_DESCRIPTION)
from gnn_teacher_student.experiment import Experiment

PATH = os.path.dirname(pathlib.Path(__file__).parent.absolute())
BASE_PATH = os.getenv('EXPERIMENT_BASE_PATH', os.path.join(PATH, 'results'))

# == EXPERIMENT VARIABLES ===================================================================================

LENGTH = 1000

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


with Experiment(base_path=BASE_PATH, name=NAME, description=DESCRIPTION, override=True) as e:

    dataset: List[dict] = generate_color_pairs_dataset(
        length=LENGTH,
        node_count_cb=lambda: random.randint(5, 100),
        additional_edge_count_cb=lambda: random.randint(2, 10),
        colors=COLORS,
        color1=MAIN_COLOR,
        color2=PAIR_COLOR,
        exclude_empty=INCLUDE_ZERO_LABEL
    )
