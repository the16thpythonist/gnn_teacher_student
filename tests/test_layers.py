import unittest
import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from kgcnn.utils.data import ragged_tensor_from_nested_numpy

from gnn_teacher_student.layers import NodeImportanceLayer
from .util import generate_simple_random_graph, model_weights_similar, x_elem

# Disable the GPU - Not supported by KGCNN
tf.config.set_visible_devices([], 'GPU')


class TestNodeImportanceLayer(unittest.TestCase):

    class Model(ks.models.Model):

        def __init__(self):
            ks.models.Model.__init__(self)

            self.lay_node_importance = NodeImportanceLayer(
                units=10
            )

        def call(self, inputs):
            node_importances = self.lay_node_importance(inputs)
            return node_importances

    @classmethod
    def setUpClass(cls) -> None:
        cls.node_attribute_count = 2
        cls.edge_attribute_count = 1

        cls.graphs = [generate_simple_random_graph(
            node_feature_count=cls.node_attribute_count
        ) for _ in range(10)]
        for graph_data in cls.graphs:
            graph_data['graph_labels'] = np.ones(
                shape=(len(graph_data['node_attributes']), ),
                dtype=np.int32
            )

        cls.x_train = [
            ragged_tensor_from_nested_numpy([gd['node_attributes'] for gd in cls.graphs]),
            ragged_tensor_from_nested_numpy([gd['edge_attributes'] for gd in cls.graphs]),
            ragged_tensor_from_nested_numpy([gd['edge_indices'] for gd in cls.graphs])
        ]
        cls.y_train = [
            ragged_tensor_from_nested_numpy([gd['graph_labels'] for gd in cls.graphs])
        ]

    def test_dataset_exists(self):
        self.assertTrue(hasattr(self, 'graphs'))
        self.assertTrue(isinstance(self.graphs, list))
        self.assertNotEqual(0, len(self.graphs))
        self.assertTrue(isinstance(self.graphs[0], dict))

        self.assertTrue(hasattr(self, 'x_train'))
        self.assertTrue(isinstance(self.x_train, list))

    def test_construction_basically_works(self) -> None:
        layer = NodeImportanceLayer(
            units=10
        )
        self.assertTrue(isinstance(layer, NodeImportanceLayer))

    def test_training(self) -> None:

        # Constructing a simple model
        model = self.Model()
        model.compile(
            loss=ks.losses.BinaryCrossentropy(),
            optimizer=ks.optimizers.Adam()
        )
        model(self.x_train)  # build the model

        weights_before = model.get_weights()
        model.fit(self.x_train, self.y_train, batch_size=2, epochs=1, verbose=0)
        weights_after = model.get_weights()

        # The model weights should be a little bit different after training
        self.assertFalse(model_weights_similar(weights_before, weights_after))

    def test_output(self):
        model = self.Model()

        # The output of the layer is the
        x = x_elem(self.x_train, 2)
        node_importances = model(x)
        # batch size 1
        self.assertEqual(1, node_importances.shape[0])
        # The graph size is ragged -> None
        self.assertEqual(None, node_importances.shape[1])
        # Attention coefficients are a single value per node
        self.assertEqual(1, node_importances.shape[2])

        # This layer uses a softmax activation on all the elements in the end, this means that
        # all the individual elements should be normalized such that they add up to 1
        importance_sum = float(tf.reduce_sum(node_importances))
        self.assertAlmostEqual(1, importance_sum)

    def test_lock_parameters(self) -> None:
        # Constructing a simple model
        model = self.Model()
        # ! IMPORTANT: model.compile() has to be called after changing the trainability of parameters or it
        # will not take effect!
        model.lay_node_importance.trainable = False
        model.compile(
            loss=ks.losses.BinaryCrossentropy(),
            optimizer=ks.optimizers.Adam()
        )
        model(self.x_train)

        weights_before = model.get_weights()
        model.fit(self.x_train, self.y_train, batch_size=2, epochs=1, verbose=0)
        weights_after = model.get_weights()

        self.assertTrue(model_weights_similar(weights_before, weights_after))


