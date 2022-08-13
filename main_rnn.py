import torch
from torch import nn
import utils
import my_net
#from d2l import torch as d2l
from torchsummary import summary
import math


def predict_rnn(prefix, num_preds, net, vocab, device):
    """Generate new characters following the `prefix`.
    prefix为前缀，预测的东西放这后面；num_preds为要预测多少词
    Defined in :numref:`sec_rnn_scratch`"""
    state = net.begin_state(batch_size=1, device=device)  # 毕竟一个一个字符遍历的，所以batch_size取1
    outputs = [vocab[prefix[0]]]
    # 将output的最后一个字符作为下一个预测的输入
    get_input = lambda: utils.reshape(torch.tensor([outputs[-1]], device=device), (1, 1))

    # 对prefix的先轮一遍取state（不在乎输出字符（因为没用），在乎隐状态）
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)  # t-1时刻的state与t时刻的输入与t-1时刻的state有关
        outputs.append(vocab[y])

    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))  # y为26个英文字母加上一个空格，一个unk，总共28维
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def train_rnn_epoch(net, train_iter, loss, updater, device, use_random_iter):
    """Train a net within one epoch (defined in Chapter 8).

    Defined in :numref:`sec_rnn_scratch`"""
    state, timer = None, utils.Timer()
    metric = utils.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling 因为随机不连续，所以上一个序列末的state不应该用到这里
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()  # 将该点的requires_grad设置为false，切断前面的计算图过程x→state→y，反向只计算到state→y
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


def train_rnn(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
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
    for epoch in range(num_epochs):
        print(f'EPOCH{epoch + 1}'.center(75, '*'))

        ppl, speed = train_rnn_epoch(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))

    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


def main_my_rnn():
    print('开始训练RNN'.center(100, '*'))

    batch_size, num_steps = 32, 35  # num_steps为每一次看一个多长的序列
    train_iter, vocab = utils.load_data_time_machine(batch_size, num_steps)
    num_hiddens = 256
    rnn_layer = nn.RNN(len(vocab), num_hiddens)

    device = utils.try_gpu()
    net = my_net.MyRNNModel(rnn_layer, vocab_size=len(vocab))

    net = net.to(device)

    num_epochs, lr = 500, 1
    train_rnn(net, train_iter, vocab, lr, num_epochs, device)


def main_my_gru():
    print('开始训练GRU'.center(100, '*'))

    batch_size, num_steps = 32, 35
    train_iter, vocab = utils.load_data_time_machine(batch_size, num_steps)

    vocab_size, num_hiddens, device = len(vocab), 256, utils.try_gpu()
    num_epochs, lr = 500, 1

    num_inputs = vocab_size
    gru_layer = nn.GRU(num_inputs, num_hiddens)
    model = my_net.MyRNNModel(gru_layer, len(vocab))
    model = model.to(device)
    train_rnn(model, train_iter, vocab, lr, num_epochs, device)


def main_my_lstm():
    print('开始训练LSTM'.center(100, '*'))

    batch_size, num_steps = 32, 35
    train_iter, vocab = utils.load_data_time_machine(batch_size, num_steps)

    vocab_size, num_hiddens, device = len(vocab), 256, utils.try_gpu()
    num_epochs, lr = 500, 1

    num_inputs = vocab_size
    lstm_layer = nn.LSTM(num_inputs, num_hiddens)
    model = my_net.MyRNNModel(lstm_layer, len(vocab))
    model = model.to(device)
    train_rnn(model, train_iter, vocab, lr, num_epochs, device)





if __name__ == '__main__':
    torch.manual_seed(42)

    # main_my_rnn()
    # main_my_gru()
    main_my_lstm()


