import random
from re import X
from typing import override
import numpy as np
import torch

# 定义一个函数来获取词汇表
def get_vocab():
    import string
    a = string.ascii_lowercase
    vocab_x = ['<sos>', '<eos>', '<pad>']
    # 原代码中使用生成器表达式生成字符串 '{i}'，这会导致列表中添加的是 '{0}' 到 '{9}' 而不是数字 0 到 9 的字符串表示
    # 修改为将整数 i 转换为字符串 str(i) 并追加到 vocab_x 列表中
    vocab_x.extend(str(i) for i in range(0, 10))
    vocab_x.extend(a)
    # 创建一个将词汇映射到整数的字典
    # 原代码使用列表推导式的语法错误，应该使用字典推导式
    vocab_x = {word: i for i, word in enumerate(vocab_x)}
    # 创建一个将整数映射到词汇的字典
    vocab_xr = [k for k, v in vocab_x.items()]

    vocab_y = ['<sos>', '<eos>', '<pad>']
    vocab_y.extend(str(i) for i in range(0, 10))
    vocab_y.extend(a.upper())
    vocab_y = {word: i for i, word in enumerate(vocab_y)}
    vocab_yr = [k for k, v in vocab_y.items()]
    # 返回词汇表
    return vocab_x, vocab_xr, vocab_y, vocab_yr

# 定义一个函数来生成数据
def get_data(vocab_x, vocab_y, all_words):
    # 定义每个词被选中的概率，初始时每个词概率相等
    # 生成随机概率值
    word_probabilities = np.random.rand(len(all_words))
    # 归一化概率值，使其总和为 1
    word_probabilities /= word_probabilities.sum()

    # 随机选择 n 个词
    n = random.randint(5, 10)  # 可根据需要修改 n 的值
    x = np.random.choice(list(all_words.keys()), size=n, p=word_probabilities)
    x = x.tolist()

    def f(i):
        i = i.upper().replace('<SOS>', '<sos>').replace('<EOS>', '<eos>')
        if not i.isdigit():
            return i
        else:
            i = 9 - int(i)
            return str(i)
    
    y = [f(i) for i in x]
    y = y[::-1]

    x = ['<sos>'] + x + ['<eos>']
    y = ['<sos>'] + y + ['<eos>']
    
    x = x + ['<pad>'] * (12 - len(x)) 
    y = y + ['<pad>'] * (13 - len(y))
    x_e = [vocab_x[i] for i in x]
    y_e = [vocab_y[i] for i in y]
    x_e = torch.tensor(x_e)
    y_e = torch.tensor(y_e)
    # print(x_e.shape)
    # print(y_e.shape)
    return x, y, x_e, y_e


import torch.utils.data
class Dataset(torch.utils.data.Dataset):
    def __init__(self, vocab_x, vocab_y, all_words):
        super(Dataset, self).__init__()
    def __len__(self):
        return 10000
    def __getitem__(self, index):
        x, y, x_e, y_e = get_data(vocab_x, vocab_y, all_words)
        return x_e, y_e


# 定义一个函数来生成掩码
def mask_pad(data):
    mask = data == vocab_x['<pad>']
    mask = mask.reshape(-1, 1, 1, 12)
    mask = mask.expand(-1, 1, 12, 12)
    return mask

# 定义一个函数来生成掩码
def mask_tril(data):
    tril = 1 - torch.tril(torch.ones(1, 12, 12, dtype=torch.long)) # 1*12*12
    mask = data == vocab_y['<pad>'] # n*12
    mask = mask.unsqueeze(1).long() # n*1*12
    mask = mask + tril # n*1*12
    mask = mask > 0 # n*1*12
    mask = (mask == 1).unsqueeze(1)
    return mask

def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
    score = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(8) # n*1*12*12
    # 对 score 应用掩码
    if mask is not None:
        score = score.masked_fill_(mask, -float('inf'))
    score = torch.softmax(score, dim=-1)
    score = torch.matmul(score, V) # n*1*12*8
    score = score.permute(0, 2, 1, 3).reshape(-1, 12, 32) # n*12*8
    return score

class MultiHead(torch.nn.Module):
    def __init__(self):
        super(MultiHead, self).__init__()
        self.fc_Q = torch.nn.Linear(32, 32)
        self.fc_K = torch.nn.Linear(32, 32)
        self.fc_V = torch.nn.Linear(32, 32)
        self.fc_out = torch.nn.Linear(32, 32)
        self.norm = torch.nn.LayerNorm(32)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, Q, K, V, mask):
        b = Q.shape[0]
        clone_Q = Q.clone()
        Q = self.norm(Q)
        Q = self.fc_Q(Q)
        K = self.norm(K)
        K = self.fc_K(K)
        V = self.norm(V)
        V = self.fc_V(V)
        Q = Q.reshape(b, 12, 4, 8).permute(0, 2, 1, 3)
        K = K.reshape(b, 12, 4, 8).permute(0, 2, 1, 3)
        V = V.reshape(b, 12, 4, 8).permute(0, 2, 1, 3)
        score = attention(Q, K, V, mask)
        score = self.fc_out(score)
        score = self.dropout(score)
        score = score + clone_Q
        return score

# 定义一个类来实现位置编码，使用正弦和余弦函数
class PositionEmbedding(torch.nn.Module):
    def __init__(self):
        super(PositionEmbedding, self).__init__()
        def get_pe(pos, i, d_model):
            pe = pos / (1e4**(i / d_model))
            if i % 2 == 0:
                return np.sin(pe)
            else:
                return np.cos(pe)

        pe = torch.empty(12, 32)
        for pos in range(12):
            for i in range(32):
                pe[pos, i] = get_pe(pos, i, 32)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.embed = torch.nn.Embedding(39, 32)
        self.embed.weight.data.normal_(0, 0.1)

    # 定义前向传播函数
    # 输入 x 的形状为 (batch_size, seq_len)
    # 输出 pe 的形状为 (batch_size, seq_len, d_model)
    def forward(self, x):
        x = x.long()
        x = self.embed(x)
        x = x + self.pe
        return x

class FullConnectedOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.Dropout(0.1)
        )
        self.norm = torch.nn.LayerNorm(32)
    
    def forward(self, x):
        clone_x = x.clone()
        x = self.norm(x)
        x = self.fc(x)
        x = x + clone_x
        return x

class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head = MultiHead()
        self.ffn = FullConnectedOutput()

    def forward(self, x, mask):
        x = self.multi_head(x, x, x, mask)
        x = self.ffn(x)
        return x

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layers = torch.nn.ModuleList([EncoderLayer() for _ in range(3)])

    def forward(self, x, mask):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x

class DecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head1 = MultiHead()
        self.multi_head2 = MultiHead()
        self.ffn = FullConnectedOutput()

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.multi_head1(y, y, y, mask_tril_y) # 这里的 mask_tril_y 是在 Encoder 中计算得到的
        y = self.multi_head2(y, x, x, mask_pad_x)  # 这里的 mask_pad_x 是在 Decoder 中计算得到的
        y = self.ffn(y)
        return y

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_layers = torch.nn.ModuleList([DecoderLayer() for _ in range(3)])
        
    def forward(self, x, y, mask_pad_x, mask_tril_y):
        for layer in self.decoder_layers:
            y = layer(x, y, mask_pad_x, mask_tril_y)
        return y

class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_x = PositionEmbedding()
        self.embedding_y = PositionEmbedding()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc_out = torch.nn.Linear(32, 39)
    
    def forward(self, x, y):
        mask_pad_x = mask_pad(x)
        mask_tril_y = mask_tril(y)
        x = self.embedding_x(x)
        y = self.embedding_y(y)
        x = self.encoder(x, mask_pad_x)
        y = self.decoder(x, y, mask_pad_x, mask_tril_y)
        y = self.fc_out(y)
        return y

model = Transformer()

def predect(x):
    model.eval()
    mask_pad_x = mask_pad(x)
    target = [vocab_y['<sos>']] + [vocab_y['<pad>']] * 12
    target = torch.LongTensor(target).unsqueeze(0)
    x = model.embedding_x(x)
    x = model.encoder(x, mask_pad_x)
    for i in range(12):
        y = target
        mask_tril_y = mask_tril(y)
        y = model.embedding_y(y)
        y = model.decoder(x, y, mask_pad_x, mask_tril_y)
        out = model.fc_out(y)
        out = out[:, -1, :]
        out = torch.argmax(out, dim=-1).detach()
        target[:, i+1] = out
    return target

def train(dataloader: torch.utils.data.DataLoader):
    loss_func = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.5)
    for epoch in range(1):
        for i, (x, y) in enumerate(dataloader):
            pred = model(x, y[:, :-1])
            pred = pred.reshape(-1, 39)
            y = y[:, 1:].reshape(-1)
            select = y != vocab_y['<pad>']
            pred = pred[select]
            y = y[select]
            loss = loss_func(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if i % 10 == 0:
                print(f"epoch: {epoch}, step: {i}, loss: {loss.item()}")
        sched.step()
            

if True == True:
    vocab_x, vocab_xr, vocab_y, vocab_yr = get_vocab()
    all_words = {word:i for i, word in enumerate(vocab_x)}
    all_words.pop('<pad>')
    all_words.pop('<sos>')
    all_words.pop('<eos>')
    dataset = Dataset(vocab_x, vocab_y, all_words)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, drop_last=True)
    train(dataloader)
    model.save('model.pth')

