# -*- coding: utf-8 -*-
import argparse
import paddle.v2 as paddle
from net_work import IModel
from mlp_conf import MLP
from logistc_conf import LogisticModel
from utils import DataSet
import gzip

def parse_args():
    '''
    读取参数，训练文件，测试文件，模型存储路径
    python train.py --train_data_path train_lr --model model
    '''
    parser = argparse.ArgumentParser(description="PaddlePaddle LR example")
    parser.add_argument(
        '--train_data_path',
        type=str,
        required=True,
        help="path of training dataset")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help="train model lr or mp")

    parser.add_argument(
        '--model_output_prefix',
        type=str,
        default='./model/trainModel',
        help='prefix of path for model to store (default: ./model/trainModel)')
    return parser.parse_args()


def train():
    args = parse_args()
    batch_size = 2
    input_dim = 13
    epoch = 100  ##迭代次数
    paddle.init(use_gpu=False, trainer_count=1)  ##paddle初始化,是否使用 gpu, gpu、cpu的个数

    if args.model == "MLP":
        model = MLP(input_dim,args.model)
    else:
        model = LogisticModel(input_dim, args.model)

    params = paddle.parameters.create(model.train_cost)

    optimizer = paddle.optimizer.RMSProp()

    #optimizer = paddle.optimizer.AdaGrad(learning_rate=1e-3,
             #                                  regularization=paddle.optimizer.L2Regularization(rate=1e-3))


    trainer = paddle.trainer.SGD(cost=model.train_cost,
                                 parameters=params,
                                 update_equation=optimizer)

    dataset = DataSet()
    def __event_handler__(event):
        if isinstance(event,paddle.event.EndIteration):
            num_samples = event.batch_id*batch_size
            if event.batch_id == 0:
                pass
            if event.batch_id % 100 == 0:
                print ("epoch %d, Samples %d, Cost %f" % (
                    event.pass_id, num_samples, event.cost))
            if event.batch_id % 100 == 0:
                # print(event.metrics)
                path = "{}-epoch-{}-batch-{}.tar.gz".format(args.model_output_prefix,event.batch_id,event.cost)
                with gzip.open(path,"w") as f:
                    trainer.save_parameter_to_tar(f)

    feeding_index = {'input': 0, 'label': 1}

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(dataset.mk_data(args.train_data_path,model),buf_size=10),
            batch_size=batch_size
        ),
        feeding=feeding_index,
        event_handler=__event_handler__,
        num_passes=epoch
    )

if __name__ == '__main__':
    train()

