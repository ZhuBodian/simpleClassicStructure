import torch
from torch import nn
import utils
import my_net
#from d2l import torch as d2l
from torchsummary import summary
import math


def evaluate_loss_accuracy_gpu(net, data_iter, loss, device=None):  # @save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = utils.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            test_l = loss(y_hat, y)
            test_acc = utils.accuracy(y_hat, y)

    return test_l, test_acc


def train(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    print('开始训练'.center(100, '*'))

    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    timer, num_batches = utils.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        print(f'EPOCH{epoch + 1}'.center(75, '*'))
        timer2 = utils.MyTimer(f'epoch{epoch + 1}运行总时长')
        process_bar = utils.ProcessBar(len(train_iter))

        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            train_l = loss(y_hat, y)
            train_l.backward()
            optimizer.step()
            with torch.no_grad():
                train_acc = utils.accuracy(y_hat, y)
            timer.stop()

            process_bar.display(i, train_l=train_l, train_acc=train_acc)

        test_l, test_acc = evaluate_loss_accuracy_gpu(net, test_iter, loss)
        print(f'epoch{epoch + 1}：test_l：{test_l}，test_acc：{test_acc}')

        timer2.stop()

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{len(train_iter.dataset) * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')


def main_lenet():
    print('开始训练LeNet'.center(100, '*'))

    net = my_net.my_lenet()
    lr, num_epochs = 0.9, 10
    batch_size = 256
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size=batch_size)

    train(net, train_iter, test_iter, num_epochs, lr, utils.try_gpu())


def main_alexnet():
    print('开始训练AlexNet'.center(100, '*'))

    net = my_net.my_alexnet()
    lr, num_epochs = 0.01, 10
    batch_size = 128
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size=batch_size, resize=224)

    train(net, train_iter, test_iter, num_epochs, lr, utils.try_gpu())


def main_vgg_16():
    print('开始训练vgg16'.center(100, '*'))

    conv_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))  # 这个参数决定了是VGG16
    net = my_net.my_vgg(conv_arch, initial_channels=1)
    lr, num_epochs = 0.05, 10
    batch_size = 64

    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size=batch_size, resize=224)
    train(net, train_iter, test_iter, num_epochs, lr, utils.try_gpu())


def main_my_nin():
    print('开始训练nin'.center(100, '*'))

    net = my_net.my_nin()
    lr, num_epochs = 0.1, 10
    batch_size = 128
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size=batch_size, resize=224)

    train(net, train_iter, test_iter, num_epochs, lr, utils.try_gpu())


def main_my_googlenet_v1():
    print('开始训练googlenet_v1'.center(100, '*'))

    net = my_net.my_googlenet_v1()
    lr, num_epochs = 0.1, 10
    batch_size = 128
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size=batch_size, resize=224)

    train(net, train_iter, test_iter, num_epochs, lr, utils.try_gpu())


def main_my_resnet_18():
    print('开始训练resnet_18'.center(100, '*'))

    net = my_net.my_resnet_18()
    lr, num_epochs = 0.05, 10
    batch_size = 256
    train_iter, test_iter = utils.load_data_fashion_mnist(batch_size=batch_size, resize=224)

    train(net, train_iter, test_iter, num_epochs, lr, utils.try_gpu())


if __name__ == '__main__':
    torch.manual_seed(42)

    main_lenet()
    # main_alexnet()
    # main_vgg_16()
    # main_my_nin()
    # main_my_googlenet_v1()
    # main_my_resnet_18()
