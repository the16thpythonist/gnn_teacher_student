from typing import List, Callable

import tensorflow.keras as ks

from kgcnn.layers.base import GraphBaseLayer
from kgcnn.layers.modules import DenseEmbedding, ActivationEmbedding, LazyConcatenate
from kgcnn.layers.gather import GatherNodesIngoing, GatherNodesOutgoing
from kgcnn.layers.pooling import PoolingLocalMessages, PoolingWeightedLocalMessages, PoolingNodes
from kgcnn.layers.conv.gcn_conv import GCN


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


class NodeImportanceSubNetwork(GraphBaseLayer):

    def __init__(self,
                 unitss: List[int],
                 activation: str = 'kgcnn>leaky_relu',
                 use_bias: bool = True,
                 conv_layer_cb: Callable = GCN,
                 pooling_method: str = 'sum',
                 use_softmax: bool = True,
                 **kwargs):
        GraphBaseLayer.__init__(self, **kwargs)
        self.unitss = unitss
        self.use_softmax = use_softmax

        self.conv_layers = []
        for units in self.unitss:
            lay_conv = conv_layer_cb(
                units=units,
                activation=activation,
                use_bias=use_bias,
                pooling_method=pooling_method
            )
            self.conv_layers.append(lay_conv)

        self.lay_dense_final = DenseEmbedding(
            units=1,
            activation='linear',
            use_bias=False
        )

    def call(self, inputs):
        node_input, edge_input, edge_index_input = inputs

        x = node_input
        for lay_conv in self.conv_layers:
            x = lay_conv([x, edge_input, edge_index_input])

        node_importances = self.lay_dense_final(x)
        if self.use_softmax:
            node_importances = ks.activations.softmax(node_importances, axis=1)

        return node_importances


class EdgeImportanceSubNetwork(GraphBaseLayer):

    def __init__(self,
                 unitss: List[int],
                 activation: str = 'kgcnn>leaky_relu',
                 use_bias: bool = True,
                 conv_layer_cb: Callable = GCN,
                 pooling_method: str = 'sum',
                 use_softmax: bool = True,
                 use_gather_in: bool = True,
                 use_gather_out: bool = False,
                 **kwargs):
        GraphBaseLayer.__init__(self, **kwargs)
        self.unitss = unitss
        self.use_softmax = use_softmax
        self.use_gather_in = use_gather_in
        self.use_gather_out = use_gather_out

        self.conv_layers = []
        for units in self.unitss:
            lay_conv = conv_layer_cb(
                units=units,
                activation=activation,
                use_bias=use_bias,
                pooling_method=pooling_method
            )
            self.conv_layers.append(lay_conv)

        self.lay_gather_in = GatherNodesIngoing()
        self.lay_gather_out = GatherNodesOutgoing()
        self.lay_concat = LazyConcatenate(axis=-1)

        self.lay_dense_final = DenseEmbedding(
            units=1,
            activation='linear',
            use_bias=False
        )

    def call(self, inputs):
        node_input, edge_input, edge_index_input = inputs

        x = node_input
        for lay_conv in self.conv_layers:
            x = lay_conv([x, edge_input, edge_index_input])


        _concat_list = [edge_input]
        if self.use_gather_in:
            n_in = self.lay_gather_in([x, edge_index_input])
            _concat_list.append(n_in)
        if self.use_gather_out:
            n_out = self.lay_gather_out([x, edge_index_input])
            _concat_list.append(n_out)

        messages = self.lay_concat(_concat_list)
        edge_importances = self.lay_dense_final(messages)
        if self.use_softmax:
            edge_importances = ks.activations.softmax(edge_importances, axis=1)

        return edge_importances


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


class ConvolutionalSubNetwork(GraphBaseLayer):

    def __init__(self,
                 unitss: List[int],
                 activation: str = 'kgcnn>leaky_relu',
                 use_bias: bool = False,
                 pooling_method: str = 'sum',
                 lay_conv_cb: Callable = GCN,
                 lay_pooling_cb: Callable = lambda: PoolingNodes(pooling_method='sum'),
                 lay_out_cb: Callable = lambda: DenseEmbedding(units=1, activation='linear'),
                 **kwargs):
        GraphBaseLayer.__init__(self)
        self.unitss = unitss

        self.conv_layers = []
        for units in self.unitss:
            lay_conv = lay_conv_cb(
                units=units,
                use_bias=use_bias,
                activation=activation,
                pooling_method=pooling_method
            )
            self.conv_layers.append(lay_conv)

        self.lay_pooling = lay_pooling_cb()
        self.lay_out = lay_out_cb()

    def call(self, inputs):
        node_input, edge_input, edge_index_input = inputs

        x = node_input
        for lay_conv in self.conv_layers:
            x = lay_conv([x, edge_input, edge_index_input])

        x = self.lay_pooling(x)
        x = self.lay_out(x)

        return x
