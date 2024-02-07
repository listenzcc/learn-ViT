"""
File: basic-transformer.py
Author: Chuncheng Zhang
Date: 2024-02-02
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Basic knowledge of how to transformer
    Pytorch API:
    - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-02-02 ------------------------
# Requirements and constants
import math
import torch
import random
import torch.nn as nn

from pathlib import Path
from tqdm.auto import tqdm
from rich import print, inspect
from loguru import logger

assert torch.cuda.is_available(), 'cuda is not available'
print('CUDA available: %s' % torch.cuda.is_available())


# %% ---- 2024-02-02 ------------------------
# Function and class
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        pe = pe.to('cuda')
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)
        self.pe = pe

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TaskModel(nn.Module):

    def __init__(self, d_model=128):
        super(TaskModel, self).__init__()

        # 定义词向量，词典数为10。我们不预测两位小数。
        n = 27
        self.embedding = nn.Embedding(num_embeddings=n, embedding_dim=128)
        # 定义Transformer。超参是我拍脑袋想的
        self.transformer = nn.Transformer(d_model,
                                          nhead=16,
                                          num_encoder_layers=4,
                                          num_decoder_layers=4,
                                          dim_feedforward=512,
                                          norm_first=True,
                                          batch_first=True
                                          )

        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout=0)

        # 定义最后的线性层，这里并没有用Softmax，因为没必要。
        # 因为后面的CrossEntropyLoss中自带了
        self.predictor = nn.Linear(128, n)

    def forward(self, src, tgt):
        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to('cuda')
        src_key_padding_mask = TaskModel.get_key_padding_mask(src).to('cuda')
        tgt_key_padding_mask = TaskModel.get_key_padding_mask(tgt).to('cuda')

        # 对src和tgt进行编码
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # 将准备好的数据送给transformer
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)

        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 2] = -torch.inf
        return key_padding_mask


class MyEnglishContent(object):
    allowed_chars = ' abcdefghijklmnopqrstuvwxyz'
    txt = ''
    names = dict()

    def __init__(self, path:Path=None):
        self.append(path)

    def append(self, path:Path, name:str=None):
        '''Append the txt by the contents of the path'''
        if path is None:
            logger.warning('path is None')
            return self.txt

        if name is None:
            name = path.name

        self.names[len(self.txt)] = name

        txt = self.read_path(path)
        self.txt += ' ' + txt.strip()
        self.n = len(self.txt)

        logger.debug(f'Appended path: {path} | {name}, {len(txt)} | {self.n}')

        return self.txt

    def rnd_fetch(self, length:int=40, return_name:bool=False) -> list:
        k = random.randint(0, self.n - length)
        patch = self.txt[k:k+length]

        if return_name:
            for i, v in self.names.items():
                if i > k:
                    break
                name = v
            return [self.c2i(c) for c in patch], name
        else:
            return [self.c2i(c) for c in patch]

    def translate(self, idx_list:list) -> str:
        return ''.join(self.i2c(i) for i in idx_list)

    def c2i(self, c:str) -> int:
        '''Convert char c to its idx'''
        return self.allowed_chars.index(c)

    def i2c(self, i:int) -> str:
        '''Convert idx i to its char'''
        return self.allowed_chars[i]

    def read_path(self, path:Path):
        '''Read and parse the txt in path'''
        def abc_only(string:str):
            # Remove any characters other than self.allowed_chars
            return ' '.join(''.join(f for f in ' '.join(f.lower() for f in string.split()) if f in self.allowed_chars).strip().split())

        txt = [abc_only(e) for e in open(path).read().split('\n') if len(e) > 40]
        txt = ' '.join(txt)

        return txt


# %% ---- 2024-02-02 ------------------------
# Play ground

content = MyEnglishContent()

for p in tqdm(list(Path('txt').iterdir()), 'Read files'):
    content.append(p)

logger.debug('Using content: %s' % content.txt[:20] + '...' + content.txt[-20:], content.n)

fetched, name = content.rnd_fetch(return_name=True)
src = torch.LongTensor(fetched)
print(f'Fetched: {name}: {content.translate(src)}, {fetched}')
print(content.names)

# %% ---- 2024-02-02 ------------------------
# Pending
# model = TaskModel()
# rnd_idx = content.rnd_fetch()
# src = torch.LongTensor([rnd_idx])
# tgt = torch.LongTensor([rnd_idx])
# src = torch.LongTensor([[0, 3, 4, 5, 6, 1, 2, 2]])
# tgt = torch.LongTensor([[3, 4, 5, 6, 1, 2, 2]])
# out = model(src, tgt)
# print(out.size())

# %%
def generate_random_batch(batch_size):
    src = []
    tgt = []
    tgt_y = []
    names = []
    for _ in range(batch_size):
        rnd_idx, name = content.rnd_fetch(length=80, return_name=True)
        names.append(name)
        src.append(rnd_idx[:40])
        tgt.append(rnd_idx[40:79])
        tgt_y.append(rnd_idx[41:])
        # # 随机生成句子长度
        # random_len = random.randint(1, max_length - 2)
        # # 随机生成句子词汇，并在开头和结尾增加<bos>和<eos>
        # random_nums = [0] + [random.randint(3, 9) for _ in range(random_len)] + [1]
        # # 如果句子长度不足max_length，进行填充
        # random_nums = random_nums + [2] * (max_length - random_len - 2)
        # src.append(random_nums)
    src = torch.LongTensor(src)
    tgt = torch.LongTensor(tgt)
    tgt_y = torch.LongTensor(tgt_y)

    # # tgt不要最后一个token
    # tgt = src[:, :-1]
    # # tgt_y不要第一个的token
    # tgt_y = src[:, 1:]

    # 计算tgt_y，即要预测的有效token的数量
    n_tokens = (tgt_y != -1).sum()

    # 这里的n_tokens指的是我们要预测的tgt_y中有多少有效的token，后面计算loss要用
    return src, tgt, tgt_y, n_tokens, names

src, tgt, tgt_y, n_tokens, names = generate_random_batch(10)
print(src.shape, tgt.shape, tgt_y.shape, n_tokens, names)
# print(generate_random_batch(10))

# %%
model = TaskModel().to('cuda')
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=3e-4)

# %%

total_loss = 0

step = 0
for i in range(100000):
    # 生成数据
    src, tgt, tgt_y, n_tokens, names = generate_random_batch(batch_size=40)
    src = src.to('cuda')
    tgt = tgt.to('cuda')
    tgt_y = tgt_y.to('cuda')

    # 清空梯度
    optimizer.zero_grad()
    # 进行transformer的计算
    out = model(src, tgt)
    # 将结果送给最后的线性层进行预测
    out = model.predictor(out)

    """
    计算损失。由于训练时我们的是对所有的输出都进行预测，所以需要对out进行reshape一下。
            我们的out的Shape为(batch_size, 词数, 词典大小)，view之后变为：
            (batch_size*词数, 词典大小)。
            而在这些预测结果中，我们只需要对非<pad>部分进行，所以需要进行正则化。也就是
            除以n_tokens。
    """
    a = out.contiguous().view(-1, out.size(-1))
    b = tgt_y.contiguous().view(-1)

    # selector = [not e[0].isnan() for e in a]
    # a = a[selector]
    # b = b[selector]

    loss = criteria(a, b) / n_tokens

    # print(out)
    # print(out.contiguous().view(-1, out.size(-1)))
    # print(tgt_y.contiguous())
    # print(loss)
    if loss.isnan():
        # print(f'Loss is nan at step {step}, breaking out.')
        # break
        # print(f'Loss is nan at step {step}')
        continue
    else:
        step += 1
        # 计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()

        total_loss += loss.item()

        # 每100次打印一下loss
        if i % 100 == 0:
            print("Step {}|{}, total_loss: {}, loss: {}".format(step, i, total_loss, loss))
            total_loss = 0


# %%
model_eval = model.eval()

def continue_writing(src:torch.Tensor, tgt:torch.Tensor):
    # 一个一个词预测，直到达到句子最大长度
    for i in range(40):
        # 进行transformer计算
        out = model_eval(src, tgt)
        # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
        predict = model.predictor(out[:, -1])
        # 找出最大值的index
        y = torch.argmax(predict, dim=1)
        # 和之前的预测结果拼接到一起
        tgt = torch.concat([tgt, y.unsqueeze(1)], dim=1)
    
    return dict(
        predicted_tgt=tgt,
        src_txt = [content.translate(e) for e in src],
        tgt_txt = [content.translate(e) for e in tgt]
    )

n = 10
src, tgt, tgt_y, n_tokens, names = generate_random_batch(n)
src = src.to('cuda')
tgt = torch.LongTensor(tgt[:, 0].unsqueeze(1)).to('cuda')
print(names, src.shape, tgt.shape)

res = continue_writing(src, tgt)
print(res)

for j in range(n):
    src_txt = res['src_txt'][j]
    tgt_txt = res['tgt_txt'][j]
    name = names[j]
    print(f'{j}, {name}\n{src_txt} -> {tgt_txt}')


# %%

# src, tgt, tgt_y, n_tokens, names = generate_random_batch(1)
# tgt = torch.LongTensor([[tgt[0][0]]])
# print(src.shape, tgt.shape)

# src = src.to('cuda')
# tgt = tgt.to('cuda')

# # 一个一个词预测，直到达到句子最大长度
# for i in range(40):
#     # 进行transformer计算
#     out = model_eval(src, tgt)
#     # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
#     predict = model.predictor(out[:, -1])
#     # 找出最大值的index
#     y = torch.argmax(predict, dim=1)
#     # 和之前的预测结果拼接到一起
#     tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)

# print(tgt)
# print(names[0])
# print(content.translate(src[0]))
# print(content.translate(tgt[0]))
# # print(''.join([content.i2c(e) for e in src[0]]))
# # print(''.join([content.i2c(e) for e in tgt[0]]))


# %% ---- 2024-02-02 ------------------------
# Pending
# src, tgt, tgt_y, n_tokens, names = generate_random_batch(10)
# src = src.to('cuda')
# tgt = torch.LongTensor(tgt[:, 0].unsqueeze(1)).to('cuda')
# print(src.shape, tgt.shape)

# out = model_eval(src, tgt)
# print(out.shape)

# predict = model.predictor(out[:, -1])
# y = torch.argmax(predict, dim=1)
# print(y)

# torch.concat([tgt, y.unsqueeze(1)], dim=1)

# transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
# src = torch.rand((10, 32, 512))
# tgt = torch.rand((20, 32, 512))
# out = transformer_model(src, tgt)
# print(out.shape)



# %%
