import unittest
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

from gnn_teacher_student.students import AbstractStudent
from gnn_teacher_student.training import SegmentedFitProcess


class Student(AbstractStudent):

    def __init__(self, name, variant, units):
        AbstractStudent.__init__(self, name, variant)
        self.supports_batching = True
        self.lay_dense = ks.layers.Dense(units=units)
        self.lay_dense_final = ks.layers.Dense(units=1)

    def call(self, inputs):
        x = self.lay_dense(inputs)
        x = self.lay_dense_final(x)
        return x


class TestSegmentedFitProcess(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.size = 10
        cls.xtrain = np.random.normal(size=(cls.size, 3))
        cls.ytrain = np.random.normal(size=(cls.size, 1))

    def test_basically_works(self):
        student = Student('student', 'exp', 2)
        # building model
        student(self.xtrain)
        # compiling model
        student.compile(loss=ks.losses.MeanSquaredError(), optimizer=ks.optimizers.Adam())

        epochs = 10
        fit_process = SegmentedFitProcess(
            model=student,
            fit_kwargs={
                'epochs': epochs,
                'x': self.xtrain,
                'y': self.ytrain,
                'batch_size': self.size,
                'verbose': 0
            }
        )
        self.assertTrue(isinstance(fit_process, SegmentedFitProcess))
        # The
        self.assertEqual(1, len(fit_process.callbacks))

        hist = fit_process()
        self.assertTrue(isinstance(hist, ks.callbacks.History))
        self.assertEqual(epochs, len(hist.history['loss']))

    def test_single_segmentation_of_training_process_works(self):
        student = Student('student', 'exp', 2)
        # building model
        student(self.xtrain)
        # compiling model
        student.compile(loss=ks.losses.MeanSquaredError(), optimizer=ks.optimizers.Adam())

        epochs = 10
        fit_process = SegmentedFitProcess(
            model=student,
            fit_kwargs={
                'epochs': epochs,
                'x': self.xtrain,
                'y': self.ytrain,
                'batch_size': self.size,
                'verbose': 0
            }

        )
        # This callback will now cause an actual segmentation of the training process at half of the total
        # epochs. We dynamically attach a flag to the model object here. By checking for it's presence later
        # we can make sure that the callback actually got executed
        fit_process.add_callback(int(epochs/2), lambda m, h: setattr(m, 'flag', True))
        # Now the internally there should be two callbacks, we also need to check for the order of the
        # keys of the dict (it is an OrderedDict) they should be the epoch numbers in ascending order.
        # That is important for the correct functionality
        self.assertEqual(2, len(fit_process.callbacks))

        hist = fit_process()
        # Now the model object should have the flag
        self.assertTrue(hasattr(student, 'flag'))
        self.assertTrue(student.flag)
        # Also, even though the training process was split, the merged history should still have the total
        # length of the epochs determined by the initial epochs argument
        self.assertEqual(epochs, len(hist.history['loss']))

