import tensorflow.keras as ks

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import DenseEmbedding, ActivationEmbedding, LazyConcatenate
from kgcnn.layers.gather import GatherNodesIngoing, GatherNodesOutgoing
from kgcnn.layers.pooling import PoolingLocalMessages, PoolingWeightedLocalMessages


class NodeImportanceLayer(GraphBaseLayer):

    def __init__(self,
                 units: int,
                 activation: str = 'kgcnn>leaky_relu',
                 use_bias: bool = True,
                 pooling_method: str = 'sum',
                 trainable: bool = True,
                 **kwargs):
        GraphBaseLayer.__init__(self, **kwargs)

        self.lay_gather_in = GatherNodesIngoing()
        self.lay_gather_out = GatherNodesOutgoing()
        self.lay_cat = LazyConcatenate(axis=-1)

        self.lay_pooling = PoolingLocalMessages(pooling_method=pooling_method)
        self.lay_dense_activation = DenseEmbedding(
            units=units,
            activation=activation,
            use_bias=use_bias
        )
        self.lay_dense_final = DenseEmbedding(
            units=1,
            activation='linear',
            use_bias=True
        )

    def call(self, inputs):
        node_input, edge_input, edge_index_input = inputs

        n_in = self.lay_gather_in([node_input, edge_index_input])
        n_out = self.lay_gather_out([node_input, edge_index_input])
        messages = self.lay_cat([n_in, n_out])

        node_embeddings = self.lay_pooling([node_input, messages, edge_index_input])
        node_embeddings = self.lay_dense_activation(node_embeddings)
        node_importances = self.lay_dense_final(node_embeddings)
        node_importances = ks.activations.softmax(node_importances, axis=1)

        return node_importances

    @property
    def trainable(self) -> bool:
        return (self.lay_dense_activation._layer_dense.trainable and
                self.lay_dense_final._layer_dense.trainable)

    @trainable.setter
    def trainable(self, value: bool) -> None:
        self.lay_dense_activation._layer_dense.trainable = value
        self.lay_dense_final._layer_dense.trainable = value


class EdgeImportanceLayer(GraphBaseLayer):

    def __init__(self,
                 units: int,
                 activation: str = 'kgcnn>leaky_relu',
                 use_bias: bool = True,
                 trainable: bool = True,
                 **kwargs):
        GraphBaseLayer.__init__(self, **kwargs)

        self.lay_gather_in = GatherNodesIngoing()
        self.lay_gather_out = GatherNodesOutgoing()
        self.lay_cat = LazyConcatenate(axis=-1)

        self.lay_dense_activation = DenseEmbedding(
            units=units,
            activation=activation,
            use_bias=use_bias
        )

        self.lay_dense_final = DenseEmbedding(
            units=1,
            activation='linear',
            use_bias=False
        )

    def call(self, inputs):
        node_input, edge_input, edge_index_input = inputs

        n_in = self.lay_gather_in([node_input, edge_index_input])
        n_out = self.lay_gather_out([node_input, edge_index_input])
        messages = self.lay_cat([n_in, n_out])

        edge_embeddings = self.lay_dense_activation(messages)
        edge_importances = self.lay_dense_final(edge_embeddings)
        edge_importances = ks.activations.softmax(edge_importances, axis=1)

        return edge_importances

    @property
    def trainable(self) -> bool:
        return (self.lay_dense_activation._layer_dense.trainable and
                self.lay_dense_final._layer_dense.trainable)

    @trainable.setter
    def trainable(self, value: bool) -> None:
        self.lay_dense_activation._layer_dense.trainable = value
        self.lay_dense_final._layer_dense.trainable = value


class AttentiveLayer(GraphBaseLayer):

    def __init__(self,
                 units: int,
                 activation: str = 'kgcnn>leaky_relu',
                 use_bias: bool = True,
                 pooling_method: str = 'sum',
                 **kwargs):
        GraphBaseLayer.__init__(self, **kwargs)

        self.lay_gather_out = GatherNodesOutgoing()
        self.lay_concat = LazyConcatenate(axis=-1)

        self.lay_dense = DenseEmbedding(
            units=units,
            activation=activation,
            use_bias=use_bias
        )

        self.lay_pooling = PoolingWeightedLocalMessages(
            pooling_method=pooling_method
        )

    def call(self, inputs):
        node_input, node_importances, edge_input, edge_importances, edge_index_input = inputs

        n_out = self.lay_gather_out([node_input, edge_index_input])
        messages = self.lay_concat([n_out, edge_input])

        x = self.lay_pooling([node_input, messages, edge_index_input, edge_importances])
        x = x * node_importances
        x = self.lay_dense(x)

        return x

    @property
    def trainable(self) -> bool:
        return self.lay_dense._layer_dense.trainable

    @trainable.setter
    def trainable(self, value: bool) -> None:
        self.lay_dense._layer_dense.trainable = value
