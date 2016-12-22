import six
import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F
import nutszebra_chainer
import functools
from collections import defaultdict


class BN_ReLU_Conv(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(BN_ReLU_Conv, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(in_channel),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x, train=False):
        return self.conv(F.relu(self.bn(x, test=not train)))

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)


class Highway_Conv(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        stride = stride if type(stride) is not int else (stride, stride)
        super(Highway_Conv, self).__init__()
        modules = []
        modules += [('conv', BN_ReLU_Conv(in_channel, out_channel, filter_size, stride, pad))]
        modules += [('conv_highway_3x3', BN_ReLU_Conv(out_channel, out_channel, filter_size, stride, pad))]
        modules += [('conv_highway_1x1_1', BN_ReLU_Conv(out_channel, out_channel, 1, 1, 0))]
        modules += [('conv_highway_1x1_2', BN_ReLU_Conv(out_channel, out_channel, 1, 1, 0))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad

    def weight_initialization(self):
        [link.weight_initialization() for _, link in self.modules]

    def count_parameters(self):
        return int(np.sum([link.count_parameters() for _, link in self.modules]))

    @staticmethod
    def concatenate_zero_pad(x, h_shape, volatile, h_type):
        _, x_channel, _, _ = x.data.shape
        batch, h_channel, h_y, h_x = h_shape
        if x_channel == h_channel:
            return x
        pad = chainer.Variable(np.zeros((batch, h_channel - x_channel, h_y, h_x), dtype=np.float32), volatile=volatile)
        if h_type is not np.ndarray:
            pad.to_gpu()
        return F.concat((x, pad))

    def maybe_pooling(self, x):
        if 2 in self.stride:
            return F.average_pooling_2d(x, 1, 2, 0)
        return x

    def __call__(self, x, train=False):
        h = self.conv(x, train)
        x = self.concatenate_zero_pad(self.maybe_pooling(x), h.data.shape, h.volatile, type(h.data))
        highway = self.conv_highway_3x3(x, train)
        highway = self.conv_highway_1x1_1(highway, train)
        highway = F.sigmoid(self.conv_highway_1x1_2(highway, train))
        self.highway = highway
        return h * highway + x * (1.0 - highway)


class Highway_Fitnet4(nutszebra_chainer.Model):

    def __init__(self, category_num):
        super(Highway_Fitnet4, self).__init__()
        # conv
        modules = []
        modules += [('conv1_1', Highway_Conv(3, 32, 3, 1, 1))]
        modules += [('conv1_2', Highway_Conv(32, 32, 3, 1, 1))]
        modules += [('conv1_3', Highway_Conv(32, 32, 3, 1, 1))]
        modules += [('conv1_4', Highway_Conv(32, 48, 3, 1, 1))]
        modules += [('conv1_5', Highway_Conv(48, 48, 3, 1, 1))]
        modules += [('conv2_1', Highway_Conv(48, 80, 3, 1, 1))]
        modules += [('conv2_2', Highway_Conv(80, 80, 3, 1, 1))]
        modules += [('conv2_3', Highway_Conv(80, 80, 3, 1, 1))]
        modules += [('conv2_4', Highway_Conv(80, 80, 3, 1, 1))]
        modules += [('conv2_5', Highway_Conv(80, 80, 3, 1, 1))]
        modules += [('conv2_6', Highway_Conv(80, 80, 3, 1, 1))]
        modules += [('conv3_1', Highway_Conv(80, 128, 3, 1, 1))]
        modules += [('conv3_2', Highway_Conv(128, 128, 3, 1, 1))]
        modules += [('conv3_3', Highway_Conv(128, 128, 3, 1, 1))]
        modules += [('conv3_4', Highway_Conv(128, 128, 3, 1, 1))]
        modules += [('conv3_5', Highway_Conv(128, 128, 3, 1, 1))]
        modules += [('conv3_6', Highway_Conv(128, 128, 3, 1, 1))]
        modules += [('linear', BN_ReLU_Conv(128, category_num, filter_size=(1, 1), stride=(1, 1), pad=(0, 0)))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.category_num = category_num
        self.block = [5, 6, 6]
        self.name = 'highway_fitnet4_{}'.format(category_num)

    def weight_initialization(self):
        [link.weight_initialization() for _, link in self.modules]

    def count_parameters(self):
        return int(np.sum([link.count_parameters() for _, link in self.modules]))

    def __call__(self, x, train=False):
        for i in six.moves.range(len(self.block)):
            for ii in six.moves.range(1, self.block[i] + 1):
                x = self['conv{}_{}'.format(i + 1, ii)](x, train)
        batch, channels, height, width = x.data.shape
        h = F.reshape(F.average_pooling_2d(x, (height, width)), (batch, channels, 1, 1))
        return F.reshape(self.linear(h, train), (batch, self.category_num))

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
