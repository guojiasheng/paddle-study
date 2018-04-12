# -*- coding: utf-8 -*-

import numpy as np
import paddle.v2 as paddle

# init paddle
paddle.init(use_gpu=False)

# step 1： 定义网络结构
x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(2))
y_predict = paddle.layer.fc(input=x, size=1, act=paddle.activation.Linear())
y = paddle.layer.data(name='y', type=paddle.data_type.dense_vector(1))

# step 2： 创建cost函数
cost = paddle.layer.square_error_cost(input=y_predict, label=y)
parameters = paddle.parameters.create(cost)

# step 3: 创建优化器

optimizer = paddle.optimizer.Momentum(momentum=0)

# step 4 : 创建一个 trainer
trainer = paddle.trainer.SGD(cost=cost,
                             parameters=parameters,
                             update_equation=optimizer)

t_y = np.array([[-2], [-3], [-7], [-7]])


def event_handler(event):
    """
    event的处理函数，把所有的训练过程都包装成了一个trainer，然后调用这个event_handler来处理比如打印loss信息这样的事情
    :param event:
    :return:
    """
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 1 == 0:
            print "Pass %d, Batch %d, Cost %f" % (event.pass_id, event.batch_id,
                                                  event.cost)
    if isinstance(event, paddle.event.EndPass):
        if event.pass_id % 10 == 0:
            with open('params_pass_%d.tar' % event.pass_id, 'w') as f:
                trainer.save_parameter_to_tar(f)

            result = trainer.test(reader=paddle.batch(
                train_test(), batch_size=2))
            print "Test with Pass %d, Cost %f, %s\n" % (
                event.pass_id, result.cost, result.metrics)


def train_reader():
    train_x = np.array([[1, 1], [1, 2], [3, 4], [5, 2]])
    train_y = np.array([[-2], [-3], [-7], [-7]])

    def reader():
        for i in xrange(train_y.shape[0]):
            yield train_x[i], train_y[i]

    return reader


def train_test():
    train_x = np.array([[1, 1], [1, 2], [3, 4], [5, 2]])
    train_y = np.array([[-2], [-3], [-7], [-7]])

    def reader():
        for i in xrange(train_y.shape[0]):
            yield train_x[i], train_y[i]

    return reader


# define feeding map
feeding = {'x': 0, 'y': 1}

# training ，持续的给数据
trainer.train(
    reader=paddle.batch(
        train_reader(), batch_size=1),
    feeding=feeding,
    event_handler=event_handler,
    num_passes=100)
