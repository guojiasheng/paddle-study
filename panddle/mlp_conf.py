# -*- coding: utf-8 -*-
import paddle.v2 as paddle
from paddle.v2 import data_type as dtype
from paddle.v2 import layer
from utils import ModelType


class MLP():

    def __init__(self, input_dim, model_type=ModelType.create_classification(), is_infer=False):
        self.input_dim = input_dim
        self.model_type = model_type
        self.is_infer = is_infer
        self._declare_input_layers()
        self.subModel = self._build_model_submodel()
        self.model = self._build_classification_model(self.subModel)

    def _declare_input_layers(self):

        self.input = layer.data(
            name='input',
            type=paddle.data_type.sparse_float_vector(self.input_dim))

        if not self.is_infer:
            self.label = paddle.layer.data(
                name='label', type=dtype.dense_vector(1))

    '''
    size (int) – The layer dimension
    '''
    def _build_model_submodel(self):
        fc = layer.fc(input=self.input,
                      size=1,
                      act=paddle.activation.Relu())
        return fc


    def _build_classification_model(self, input):


        '''
        定义多层网络
        :param input:
        :return:
        '''
        def multilayer_perceptron(input):
            dropout_probability = 0.1
            fc1 = paddle.layer.fc(input=input, size=10, act=paddle.activation.Relu())
            drop1 = paddle.layer.dropout(input=fc1, dropout_rate=dropout_probability)
            fc2 = paddle.layer.fc(input=drop1, size=10, act=paddle.activation.Relu())
            drop2 = paddle.layer.dropout(input=fc2, dropout_rate=dropout_probability)
            fc3 = paddle.layer.fc(input=drop2, size=10, act=paddle.activation.Relu())
            return fc3

        self.output = layer.fc(
            input=multilayer_perceptron,
            size=1,
            act=paddle.activation.Sigmoid())

        if not self.is_infer:
            self.train_cost = paddle.layer.multi_binary_label_cross_entropy_cost(
                input=self.output, label=self.label)

        return self.output
