import torch
from torch import nn
from torchsummary import summary
from torch.nn import functional as F



def my_lenet():
    # 与论文中的初始特征大小是32*32的，之后经过第一层卷积的变成28*28的；
    # 实际数据集是28*2的，所以这里稍微改了第一层的卷及参数，令其还是28*28的（不改的话，会变成24*24的）
    # 且输出层去掉了高斯激活，其余层的输出特征与参数与原论文大致相同
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2, stride=1), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, padding=0, stride=2),
        nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=1), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, padding=0, stride=2),
        # 如果将这里的np.Flatten与下一层nn.Linear的替换为一个卷积层的话，维数会出错？
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10))

    if torch.cuda.is_available():
        summary(net.cuda(), (1, 28, 28))
    else:
        summary(net, (1, 28, 28))
    return net


def my_alexnet():
    net = nn.Sequential(
        # 这里，我们使用一个11*11的更大窗口来捕捉对象。
        # 同时，步幅为4，以减少输出的高度和宽度。
        # 另外，输出通道的数目远大于LeNet
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 使用三个连续的卷积层和较小的卷积窗口。
        # 除了最后的卷积层，输出通道的数量进一步增加。
        # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(),
        # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
        nn.Linear(6400, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(),
        nn.Dropout(p=0.5),
        # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
        nn.Linear(4096, 10))

    if torch.cuda.is_available():
        summary(net.cuda(), (1, 224, 224))
    else:
        summary(net, (1, 224, 224))
    return net


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, padding=0, stride=2))
    return nn.Sequential(*layers)


def my_vgg(conv_arch, initial_channels=3):
    """

    :type conv_arch: object，为包含多个(块中卷积层数目，块最终输出通道数)二元组的元组
    """
    in_channels = initial_channels
    conv_blks = []
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    net = nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        # 因为是5个VGG块，每个块卷积不让特征维度改变，但是池化层令特征维度减半，故224/(2^5)=7
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

    if torch.cuda.is_available():
        summary(net.cuda(), (initial_channels, 224, 224))
    else:
        summary(net, (initial_channels, 224, 224))

    return net


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, stride=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, stride=1), nn.ReLU())


def my_nin():
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, padding=0, strides=4),
        nn.MaxPool2d(kernel_size=3, padding=0, stride=2),
        nin_block(96, 256, kernel_size=5, padding=2, strides=1),
        nn.MaxPool2d(kernel_size=3, padding=0, stride=2),
        nin_block(256, 384, kernel_size=3, padding=1, strides=1),
        nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
        nn.Dropout(0.5),
        # 标签类别数是10
        nin_block(384, 10, kernel_size=3, strides=1, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        # 将四维的输出转成二维的输出，其形状为(批量大小,10)
        nn.Flatten())

    if torch.cuda.is_available():
        summary(net.cuda(), (1, 224, 224))
    else:
        summary(net, (1, 224, 224))

    return net


class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1, padding=0, stride=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1, padding=0, stride=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1, stride=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1, padding=0, stride=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2, stride=1)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


def my_googlenet_v1():
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, padding=3, stride=2), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, padding=1, stride=2))

    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1), nn.ReLU(),
                       nn.Conv2d(64, 192, kernel_size=3, padding=1, stride=1), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),  # 输出通道为64+128+32+32=256
                       Inception(256, 128, (128, 192), (32, 96), 64),
                       nn.MaxPool2d(kernel_size=3, padding=1, stride=2))

    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                       Inception(512, 160, (112, 224), (24, 64), 64),
                       Inception(512, 128, (128, 256), (24, 64), 64),
                       Inception(512, 112, (144, 288), (32, 64), 64),
                       Inception(528, 256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                       Inception(832, 384, (192, 384), (48, 128), 128),  # 输出通道为384+384+128+128=1024
                       nn.AdaptiveAvgPool2d((1, 1)),  # 由于这里用了输出高宽为（1,1）的自适应平均池化，实际上相当于每个通道仅留1个信息
                       nn.Flatten())

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

    if torch.cuda.is_available():
        summary(net.cuda(), (1, 224, 224))
    else:
        summary(net, (1, 224, 224))

    return net


class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, padding=0, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


def my_resnet_18():
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 10))

    if torch.cuda.is_available():
        summary(net.cuda(), (1, 224, 224))
    else:
        summary(net, (1, 224, 224))

    return net


class MyRNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(MyRNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (
                torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device),
                torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
            )
