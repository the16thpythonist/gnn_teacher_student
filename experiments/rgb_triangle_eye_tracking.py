import os
import sys
import pathlib
import logging

from gnn_teacher_student.experiment import Experiment

PATH = os.path.dirname(pathlib.Path(__file__).parent.absolute())
BASE_PATH = os.getenv('EXPERIMENT_BASE_PATH', os.path.join(PATH, 'results'))

# == VARIABLE DEFINITIONS ===================================================================================
# Define the most important values / parameters of the experiment here as global variables

VARIABLE_1 = 'hello'
VARIABLE_2 = 'world'

# Name and description are special variables which should always be defined
NAME = os.path.basename(__file__).replace('.py', '')
DESCRIPTION = """
This should be a description of the experiment as whole which can later be used to understand what was
initially done with this experiment.
"""


with Experiment(base_path=BASE_PATH, name=NAME, description=DESCRIPTION, override=True) as e:
    e.logger.addHandler(logging.StreamHandler(stream=sys.stdout))
    e.copy_code_file(pathlib.Path(__file__).absolute())
    # "total_work" is a convenience feature of the Experiment class. By defining the total number of work
    # packages (assuming they all take roughly the same amount of time) and using the update method when
    # a work package is over, estimated remaining time will be logged automatically.
    e.total_work = 100
