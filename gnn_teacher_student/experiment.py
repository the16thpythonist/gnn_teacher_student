import os
import time
import shutil
import logging
import textwrap
import datetime
import traceback
from typing import Optional, Any, Callable

import psutil


class Experiment:

    DESCRIPTION_FILE_NAME = 'EXPERIMENT_DESCRIPTION'
    ERROR_FILE_NAME = 'EXPERIMENT_ERROR'
    REPORT_FILE_NAME = 'EXPERIMENT_REPORT'
    LOG_FILE_NAME = 'EXPERIMENT_LOG'

    def __init__(self,
                 base_path: str,
                 name: str,
                 description: str,
                 override: bool = True,
                 total_work: Optional[int] = None,
                 glob: dict = {}):
        self.base_path = base_path
        self.name = name
        self.description = description
        self.override = override
        self.total_work = total_work
        print(glob)
        self.globals = glob

        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.logger = logging.Logger(self.name)
        self.process = psutil.Process(os.getpid())
        self.work = []
        self.experiment_variables = {}

        # The "override" flag determines the behavior of the class if an experiment folder at the given
        # position already exists. If the flag is True we will delete the previous folder and create a new
        # one. If the flag is False we will augment the folder path a little bit so it does not conflict
        # with the existing one
        self.path = os.path.join(self.base_path, self.name)
        if os.path.exists(self.path):

            if override:
                shutil.rmtree(self.path)

            else:
                path = self.path
                index = 1
                while os.path.exists(path):
                    path = f'{self.path}_{index}'
                    index += 1

                self.path = path

        os.mkdir(self.path)

        # ~ Constructing paths for all artifacts
        self.description_path = os.path.join(self.path, self.DESCRIPTION_FILE_NAME)
        self.error_path = os.path.join(self.path, self.ERROR_FILE_NAME)
        self.report_path = os.path.join(self.path, self.REPORT_FILE_NAME)
        self.log_path = os.path.join(self.path, self.LOG_FILE_NAME)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = logging.FileHandler(self.log_path)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def update(self):
        self.work.append(time.time())

        time_finished = self.work[-1] - self.start_time
        work_finished = len(self.work)
        time_per_work = time_finished / work_finished

        estimated_remaining_time = (self.total_work - work_finished) * time_per_work

        memory_usage = self.process.memory_info().rss  # in bytes

        self.logger.debug(f'work finished ({work_finished}/{self.total_work})')
        self.logger.debug(f'estimated remaining time: {estimated_remaining_time/3600:.1f} hours')
        self.logger.debug(f'Memory usage: {memory_usage / 1024 ** 3:.1f} GB')

    def log(self, message: str):
        self.logger.info(message)

    def create_description_file(self):
        with open(self.description_path, mode='w') as file:
            file.write(self.description)

    def create_report_file(self):
        report_string_lines = [
            'EXPERIMENT_REPORT',
            '=================',
            ''
        ]

        for key, value in self.experiment_variables.items():
            report_string_lines.append(f'{key:<20} = {str(value)}')

        with open(self.report_path, mode='w') as file:
            report_string = '\n'.join(report_string_lines)
            file.write(report_string)

    def create_error_file(self, exception_value, exception_traceback):
        with open(self.error_path, mode='w') as file:
            file.write(f'{exception_value.__class__.__name__.upper()}: {exception_value}')
            file.write('\n\n')
            traceback.print_tb(exception_traceback, file=file)

    def detect_experiment_variables(self):
        for key, value in self.globals.items():
            print(key, key.upper())
            if key.upper() == key:
                self.experiment_variables[key] = value

    def copy_code_file(self, file_path: str):
        """
        This function expects the string path to the Python code module which defines the very experiment.
        This code file will then be copied into the experiment folder as a kind of archive for how this
        specific experiment was defined and conducted including all the parameterization etc.
        """
        code_file_name = os.path.basename(file_path)
        code_file_path = os.path.join(self.path, code_file_name)
        with open(file_path, mode='r') as file_read, open(code_file_path, mode='w') as file_write:
            content = file_read.read()
            content = f'"""COPY OF ORIGINAL CODE FROM {datetime.datetime.now()}"""\n' + content
            file_write.write(content)

    @property
    def has_started(self) -> bool:
        return self.start_time is not None

    @property
    def has_ended(self) -> bool:
        return self.end_time is not None

    @property
    def is_running(self):
        return self.has_started and not self.has_ended

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f'START Experiment "{self.name}" at {datetime.datetime.now()}')
        self.create_description_file()
        self.logger.info(f'Created Experiment Description: "{self.description_path}"')

        self.detect_experiment_variables()

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.end_time = time.time()

        if isinstance(exc_value, Exception):
            self.logger.error(f'An exception has occurred: {exc_value}')
            self.create_error_file(exc_value, exc_tb)
            self.logger.info(f'Created Error Report: "{self.error_path}"')

        self.create_report_file()
        self.logger.info(f'Created Experiment Report: "{self.report_path}"')
        self.logger.info(f'END Experiment "{self.name}" at {datetime.datetime.now()}')

        return True

