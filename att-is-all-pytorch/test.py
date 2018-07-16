import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
import math
def seq_padding_mask(seq):
    # seq: [batch * max_len],to a [batch * max_len_q * max_len_k]
    seq_mask = seq.ne(0).type(torch.FloatTensor)
    seq_mask = seq_mask.unsqueeze(2)
    seq_mask_ = seq_mask.transpose(1, 2)
    mask = torch.bmm(seq_mask, seq_mask_)
    return mask.eq(0)
def seq_dec_mask(seq):
    # seq:[batch * max_len * max_len]
    ones = torch.ones(seq.size())
    return (torch.from_numpy(np.triu(ones,1)) + seq.type(torch.FloatTensor)).ne(0)
    # return seq

def get_attn_padding_mask(seq_q, seq_k):
    ''' Indicate the padding-related part to mask '''
    # ？key 和每一个问题进行计算，所以 key PAD 的地方补0
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)   # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.'''
    # 得到下三角矩阵

    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask
f = nn.Softmax(dim=-1)
w_1 = nn.Conv1d(2,3,1)

a = torch.Tensor([[[1,23],[1,32]],[[1,2],[3,4]]]) #2*2*2
mask = torch.Tensor([[[1,1],[1,0]],[[0,0],[1,0]]]).type(torch.ByteTensor)
attn_weight = (a.masked_fill_(mask,-float('inf')) )
# print(attn_weight)
# print(w_1(attn_weight))
print((1-mask.type(torch.FloatTensor)))
print(a)
print(a * (1-mask.type(torch.FloatTensor)))

# dec_slf_attn_pad_mask = get_attn_padding_mask(a, a)
# dec_slf_attn_sub_mask = get_attn_subsequent_mask(a)
# dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)

# print(dec_slf_attn_pad_mask)
# print(dec_slf_attn_sub_mask)



# b = torch.Tensor([1.3]).type(torch.FloatTensor)
# c = 'v_accu_4553.chkpt'
# print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:.3f} , elapse: {elapse:3.3f} min'.format(ppl=2.353,accu=34.5433,elapse=43.45345,))

# d_vec = 3
#
# x = np.arange(1,d_vec+1).reshape(1,-1)
# x = np.array([pow(10000, xi // 2 * 2 / d_vec) for xi in x])
# x = x.repeat(5,axis=0)
# x0 = np.zeros((1,d_vec))
# x = np.array([pos/p for pos,p in enumerate(x)])
# x[1:,0::2] = np.sin(x[1:,0::2])
# x[1:,1::2] = np.cos(x[1:,1::2])
# x = np.vstack((x0,x))
# print(x)



