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


