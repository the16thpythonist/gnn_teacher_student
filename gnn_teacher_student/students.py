from copy import deepcopy, copy
from typing import Tuple, Callable
from pprint import pprint

import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K
from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.modules import LazyConcatenate, ActivationEmbedding, DenseEmbedding

from gnn_teacher_student.layers import NodeImportanceLayer, EdgeImportanceLayer, AttentiveLayer


class StudentVariants:

    EXPLANATION = 'exp'
    REFERENCE = 'ref'
    TEMPLATE = 'tem'


class AbstractStudent(ks.models.Model):

    def __init__(self,
                 name: str,
                 variant: str,
                 is_original: bool = True,
                 **kwargs):
        ks.models.Model.__init__(self, name=name, **kwargs)

        self.variant: str = variant
        self.supports_batching: bool = False

        self.prediction_locked = None
        self.explanation_locked = None

    def duplicate(self):
        c = self.__class__(name='', variant='', is_original=False)
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                setattr(c, key, value)

        return c

    def lock_explanation_parameters(self):
        raise NotImplemented

    def unlock_explanation_parameters(self):
        raise NotImplemented

    def lock_prediction_parameters(self):
        raise NotImplemented

    def unlock_prediction_parameters(self):
        raise NotImplemented


class StudentTemplate(object):

    def __init__(self,
                 student_class: type,
                 student_name: str,
                 **kwargs):

        if not issubclass(student_class, AbstractStudent):
            raise TypeError(f'The student class "{student_class.__name__}" passed to the StudentTemplate needs to be '
                            f'a subclass of the abstract base class AbstractStudent!')

        self.student_class = student_class
        self.student_name = student_name
        self.kwargs = kwargs

    def build(self, variant: str) -> AbstractStudent:
        if variant not in ['exp', 'ref']:
            raise ValueError(f'The string "{variant}" is not a valid identifier to describe a StudentVariant! '
                             f'Please choose any of the following: (ref, exp)')

        return self.student_class(
            name=self.student_name,
            variant=variant,
            **self.kwargs
        )

    def __call__(self, variant: str) -> AbstractStudent:
        return self.build(variant)


# == CONCRETE STUDENT IMPLEMENTATIONS ==


class SimpleAttentionStudent(AbstractStudent):

    def __init__(self,
                 units: int = 1,
                 attention_units: int = 1,
                 use_bias: bool = True,
                 activation: str = 'kgcnn>leaky_relu',
                 attention_activation: str = 'tanh',
                 pooling_method='sum',
                 lay_pooling_cb: Callable = lambda: PoolingNodes(pooling_method='sum'),
                 lay_out_cb: Callable = lambda: DenseEmbedding(units=1, activation='linear'),
                 **kwargs):
        AbstractStudent.__init__(self, **kwargs)
        self.supports_batching = True

        self.lay_node_importances = NodeImportanceLayer(
            units=attention_units,
            activation=attention_activation,
            use_bias=use_bias,
            pooling_method=pooling_method,
        )

        self.lay_edge_importances = EdgeImportanceLayer(
            units=attention_units,
            activation=attention_activation,
            use_bias=use_bias
        )

        self.lay_attentive = AttentiveLayer(
            units=units,
            activation=activation,
            use_bias=use_bias,
            pooling_method=pooling_method,
        )

        self.lay_pooling = lay_pooling_cb()
        self.lay_out = lay_out_cb()

        self.setup_sub_models()

    def call(self, inputs):
        node_input, edge_input, edge_index_input = inputs

        node_importances = self.lay_node_importances([node_input, edge_input, edge_index_input])
        edge_importances = self.lay_edge_importances([node_input, edge_input, edge_index_input])

        x = self.lay_attentive([
            node_input,
            node_importances,
            edge_input,
            edge_importances,
            edge_index_input
        ])
        x = self.lay_pooling(x)
        x = self.lay_out(x)

        node_importances = tf.reduce_sum(node_importances, axis=-1)
        edge_importances = tf.reduce_sum(edge_importances, axis=-1)

        return x, node_importances, edge_importances

    def lock_explanation_parameters(self) -> None:
        self.lay_node_importances.trainable = False
        self.lay_edge_importances.trainable = False

    def lock_prediction_parameters(self) -> None:
        self.lay_attentive.trainable = False
