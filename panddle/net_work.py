# -*- coding: utf-8 -*-
import paddle.v2 as paddle
from paddle.v2 import data_type as dtype
from paddle.v2 import layer
from utils import ModelType


class IModel(object):

    def __init__(self, input_dim, work_type , model_type=ModelType.create_classification(), is_infer=False):
        self.input_dim = input_dim
        self.model_type = model_type
        self.is_infer = is_infer
        self.work_type = work_type
        self._declare_input_layers()
        self.model = self._build_model_submodel()

    def _declare_input_layers(self):

        self.input = layer.data(
            name='input',
            type=paddle.data_type.sparse_float_vector(self.input_dim))

        if not self.is_infer:
            self.label = paddle.layer.data(
                name='label', type=dtype.dense_vector(1))

    def _build_model_submodel(self):
        act_function = paddle.activation.Sigmoid()
        self.output = layer.fc(input=self.input,
                      size=1,
                      act=act_function)

        if not self.is_infer:
             #self.train_cost = paddle.layer.classification_cost(input=self.output, label=self.label)
             self.train_cost = paddle.layer.multi_binary_label_cross_entropy_cost(
                 input=self.output, label=self.label)

        return self.output


