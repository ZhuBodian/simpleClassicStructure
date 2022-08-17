import torch
from torch import nn
import utils
import my_net
#from d2l import torch as d2l
from torchsummary import summary
import math
import collections



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


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = utils.MaskedSoftmaxCELoss()
    net.train()
    for epoch in range(num_epochs):
        print(f'EPOCH：{epoch + 1}'.center(75, '*'))

        timer = utils.Timer()
        metric = utils.Accumulator(2)  # 训练损失总和，词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()	# 损失函数的标量进行“反向传播”
            utils.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            print(f'loss: {float(l.sum())}')

    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = utils.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


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


def main_deep_rnn():
    print('开始训练DEEP RNN'.center(100, '*'))

    batch_size, num_steps = 32, 35
    train_iter, vocab = utils.load_data_time_machine(batch_size, num_steps)

    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    num_epochs, lr = 500, 2
    num_inputs = vocab_size
    device = utils.try_gpu()

    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
    model = my_net.MyRNNModel(lstm_layer, len(vocab))
    model = model.to(device)

    train_rnn(model, train_iter, vocab, lr, num_epochs, device)


def main_bi_rnn():
    print('开始训练bidirectional rnn（错误实例，用于不可预测未来）'.center(100, '*'))

    # 加载数据
    batch_size, num_steps, device = 32, 35, utils.try_gpu()
    train_iter, vocab = utils.load_data_time_machine(batch_size, num_steps)

    # 通过设置“bidirective=True”来定义双向LSTM模型
    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    num_inputs = vocab_size
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
    model = my_net.MyRNNModel(lstm_layer, len(vocab))
    model = model.to(device)

    # 训练模型
    num_epochs, lr = 500, 1
    train_rnn(model, train_iter, vocab, lr, num_epochs, device)


def main_seq2seq():
    print('开始训练seq2seq'.center(100, '*'))

    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, utils.try_gpu()

    train_iter, src_vocab, tgt_vocab = utils.load_data_nmt(batch_size, num_steps)
    encoder = my_net.Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = my_net.Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = my_net.EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')




if __name__ == '__main__':
    torch.manual_seed(42)

    # main_my_rnn()
    # main_my_gru()
    # main_my_lstm()
    # main_deep_rnn()
    # main_bi_rnn()
    main_seq2seq()



