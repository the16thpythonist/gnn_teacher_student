import random

import unittest
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from kgcnn.utils.data import ragged_tensor_from_nested_numpy

from gnn_teacher_student.students import AbstractStudent, StudentTemplate, SimpleAttentionStudent
from .util import generate_simple_random_graph, model_weights_similar, x_elem

# Disable the GPU - Not supported by KGCNN
tf.config.set_visible_devices([], 'GPU')


class Student(AbstractStudent):

    def __init__(self,
                 name,
                 variant,
                 units,
                 flag=True,
                 elements_cb=lambda: [1, 2]):
        AbstractStudent.__init__(self, name, variant)
        self.units = units
        self.flag = flag

        self.elements = elements_cb()

    def lock_explanation_parameters(self) -> None:
        pass

    def unlock_explanation_parameters(self) -> None:
        pass

    def lock_prediction_parameters(self) -> None:
        pass

    def unlock_prediction_parameters(self) -> None:
        pass


class TestAbstractStudent(unittest.TestCase):

    class LocalStudent(AbstractStudent):

        def __init__(self, name: str, variant: str, **kwargs):
            AbstractStudent.__init__(self, name, variant, **kwargs)
            self.layer1 = ks.layers.Dense(units=2)
            self.layer2 = ks.layers.Dense(units=1)

            self.setup_sub_models()

        def call(self, inputs):
            x = self.layer1(inputs)
            x = self.layer2(x)
            return x

        def lock_explanation_parameters(self):
            self.layer1.trainable = False

        def lock_prediction_parameters(self):
            self.layer2.trainable = False

    def test_construction_basically_works(self):
        student = self.LocalStudent('test', 'exp')
        self.assertTrue(isinstance(student, AbstractStudent))
        self.assertTrue(isinstance(student.layer1, ks.layers.Layer))
        self.assertTrue(isinstance(student.layer1.name, str))
        # building the model
        student(np.array([[1, 1]]))
        # compiling the model
        student.compile(
            loss=ks.losses.MeanSquaredError(),
            optimizer=ks.optimizers.Adam()
        )

        self.assertTrue(isinstance(student._sub_models, dict))

        self.assertIn('explanation_locked', student._sub_models)
        self.assertTrue(isinstance(student._sub_models['explanation_locked'], AbstractStudent))
        self.assertTrue(isinstance(student._sub_models['explanation_locked'], self.LocalStudent))

        self.assertIn('prediction_locked', student._sub_models)
        self.assertTrue(isinstance(student._sub_models['prediction_locked'], self.LocalStudent))

        self.assertEqual(student.layer1, student._sub_models['explanation_locked'].layer1)
        self.assertEqual(student.layer1, student._sub_models['prediction_locked'].layer1)


class TestStudentTemplate(unittest.TestCase):

    def test_construction_basically_works(self):
        student_template = StudentTemplate(
            AbstractStudent,
            'test_student'
        )
        self.assertTrue(isinstance(student_template, StudentTemplate))

    def test_construction_fails_without_abstract_student_subclass(self):
        with self.assertRaises(TypeError):
            StudentTemplate(
                int,
                'test_student'
            )

    def test_building_basically_works(self):
        student_template = StudentTemplate(
            Student,
            'test_student',
            units=10,
            flag=False
        )
        explanation_student = student_template('exp')
        self.assertTrue(isinstance(explanation_student, AbstractStudent))
        self.assertTrue(isinstance(explanation_student, Student))
        self.assertEqual('exp', explanation_student.variant)
        self.assertEqual(10, explanation_student.units)
        self.assertEqual(False, explanation_student.flag)

        reference_student = student_template('ref')
        self.assertTrue(isinstance(reference_student, AbstractStudent))
        self.assertTrue(isinstance(reference_student, Student))
        self.assertEqual('ref', reference_student.variant)
        self.assertEqual(10, reference_student.units)
        self.assertEqual(False, reference_student.flag)

    def test_building_fails_with_incorrect_variant_identifier(self):
        with self.assertRaises(ValueError):
            student_template = StudentTemplate(
                Student,
                'test_student',
                units=10
            )
            student_template('hello')


class TestSimpleAttentionStudent(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.node_attribute_count = 2
        cls.edge_attribute_count = 1

        # Generate the input graphs themselves
        cls.graphs = [generate_simple_random_graph(
            node_feature_count=cls.node_attribute_count
        ) for _ in range(10)]
        # Add random labels
        for graph_data in cls.graphs:
            graph_data['graph_labels'] = np.array([random.uniform(0, 4)])
            graph_data['edge_importances'] = np.random.randint(0, 1, size=(graph_data['edge_attributes'].shape[0], 1))
            graph_data['node_importances'] = np.random.randint(0, 1, size=(graph_data['node_attributes'].shape[0], 1))

        # Generate the ragged tensors for training
        cls.x_train = [
            ragged_tensor_from_nested_numpy([gd['node_attributes'] for gd in cls.graphs]),
            ragged_tensor_from_nested_numpy([gd['edge_attributes'] for gd in cls.graphs]),
            ragged_tensor_from_nested_numpy([gd['edge_indices'] for gd in cls.graphs])
        ]
        cls.y_train = [
            ragged_tensor_from_nested_numpy([gd['graph_labels'] for gd in cls.graphs]),
            ragged_tensor_from_nested_numpy([gd['node_importances'] for gd in cls.graphs]),
            ragged_tensor_from_nested_numpy([gd['edge_importances'] for gd in cls.graphs])
        ]

    def test_construction_basically_works(self):
        student = SimpleAttentionStudent(
            name='simple_attention_student',
            variant='exp',
            units=2,
            attention_units=3,
        )

        self.assertTrue(isinstance(student, SimpleAttentionStudent))

    def test_training(self):
        student = SimpleAttentionStudent(
            name='simple_attention_student',
            variant='exp',
            units=2,
            attention_units=3,
        )
        student(self.x_train)

        student.compile(
            loss=ks.losses.MeanSquaredError(),
            optimizer=ks.optimizers.Adam()
        )

        weights_before = student.get_weights()
        student.fit(
            self.x_train, self.y_train, batch_size=2, epochs=1, verbose=0
        )
        weights_after = student.get_weights()

        self.assertFalse(model_weights_similar(weights_before, weights_after))

    def test_recompile(self):
        student = SimpleAttentionStudent(
            name='simple_attention_student',
            variant='exp',
            units=2,
            attention_units=3,
        )
        student(self.x_train)

        student.compile(
            loss=ks.losses.MeanSquaredError(),
            optimizer=ks.optimizers.Adam()
        )

        student.fit(self.x_train, self.y_train, batch_size=2, epochs=1, verbose=0)

        # This will make all the parameters which are used for the explanation non-trainable
        # !important: After such a change is applied the student needs to be recompiled
        student.lock_explanation_parameters()
        student.recompile()

        # Now a training should not affect the explanation weights at all!
        weights_before = student.lay_node_importances.get_weights()
        student.fit(self.x_train, self.y_train, batch_size=2, epochs=1, verbose=0)
        weights_after = student.lay_node_importances.get_weights()
        self.assertTrue(model_weights_similar(weights_before, weights_after))
