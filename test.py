import paddle.v2 as paddle
import ssl


# Initialize PaddlePaddle.
paddle.init(use_gpu=False, trainer_count=1)

# Configure the neural network.
x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(13))
y_predict = paddle.layer.fc(input=x, size=1, act=paddle.activation.Linear())

input_data = [item for item in paddle.dataset.uci_housing.test()()]

# Infer using provided test data.
probs = paddle.infer(
    output_layer=y_predict,
    parameters=paddle.dataset.uci_housing.model(),
    input=input_data)



for i in xrange(len(probs)):
    print 'Predicted price: ${:,.2f}'.format(probs[i][0] * 1000),input_data[i]