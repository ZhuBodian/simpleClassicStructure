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


def predict_rnn(prefix, num_preds, net, vocab, device):
    """Generate new characters following the `prefix`.

    Defined in :numref:`sec_rnn_scratch`"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: utils.reshape(torch.tensor(
        [outputs[-1]], device=device), (1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def train_rnn_epoch(net, train_iter, loss, updater, device, use_random_iter):
    """Train a net within one epoch (defined in Chapter 8).

    Defined in :numref:`sec_rnn_scratch`"""
    state, timer = None, utils.Timer()
    metric = utils.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()
            else:
                # `state` is a tuple of tensors for `nn.LSTM` and
                # for our custom scratch implementation
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            utils.grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            utils.grad_clipping(net, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
        metric.add(l * utils.size(y), utils.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_rnn(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """Train a model (defined in Chapter 8).

    Defined in :numref:`sec_rnn_scratch`"""
    loss = nn.CrossEntropyLoss()
    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: utils.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_rnn(prefix, 50, net, vocab, device)
    # Train and predict
    print('开始训练RNN'.center(100, '*'))
    for epoch in range(num_epochs):
        print(f'EPOCH{epoch + 1}'.center(75, '*'))

        ppl, speed = train_rnn_epoch(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


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


def main_my_rnn():
    batch_size, num_steps = 32, 35
    train_iter, vocab = utils.load_data_time_machine(batch_size, num_steps)
    num_hiddens = 256
    rnn_layer = nn.RNN(len(vocab), num_hiddens)

    device = utils.try_gpu()
    net = my_net.MyRNNModel(rnn_layer, vocab_size=len(vocab))

    net = net.to(device)

    num_epochs, lr = 500, 1
    train_rnn(net, train_iter, vocab, lr, num_epochs, device)


if __name__ == '__main__':
    torch.manual_seed(42)

    # main_lenet()
    # main_alexnet()
    # main_vgg_16()
    # main_my_nin()
    # main_my_googlenet_v1()
    # main_my_resnet_18()
    main_my_rnn()
