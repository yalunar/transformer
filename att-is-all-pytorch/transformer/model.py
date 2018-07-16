import torch
import torch.nn as nn
import torch.nn.functional as F
import constant
import numpy as np
from transformer.layers import EncoderLayer,DecoderLayer

def positional_encoding(n_position,d_positvec):
    """
    :param n_position: the max position is the seq length without padding. PAD is 0
    :param d_positvec: equal wordvec's dimension,
    :return: a matrix similar to embedding with size [n_position * d_positvec].
    """
    x = np.arange(1,d_positvec+1)
    x = np.array([pow(10000,xi//2*2/d_positvec) for xi in x]).reshape(1,-1)
    x = x.repeat(n_position,axis=0)
    x0 = np.zeros((1,d_positvec))
    x = np.array([pos/p for pos,p in enumerate(x)])
    x[1:, 0::2] = np.sin(x[1:, 0::2])
    x[1:, 1::2] = np.cos(x[1:, 1::2])
    position_embedding = nn.Embedding(n_position,d_positvec,padding_idx=constant._PAD_)
    position_embedding.weight.data = torch.from_numpy(x).type(torch.FloatTensor)
    return position_embedding

def seq_padding_mask(seq_q, seq_k):
    # # q: [batch * max_len],to a [batch * max_len_q * max_len_k]
    # q_mask = q.ne(constant._PAD_).type(torch.FloatTensor)
    # q_mask = q_mask.unsqueeze(2)
    # k_mask = k.ne(constant._PAD_).type(torch.FloatTensor)
    # k_mask = k_mask.unsqueeze(1)
    # mask = torch.bmm(q_mask,k_mask)
    # return mask.eq(0)
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(constant._PAD_).unsqueeze(1)  # bx1xsk
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k)  # bxsqxsk
    return pad_attn_mask

def seq_dec_mask(seq):
    # seq:[batch * max_len * max_len]
    ones = torch.ones(seq.size())
    return (torch.from_numpy(np.triu(ones, 1)) + seq.type(torch.FloatTensor)).ne(0)

class Encoder(nn.Module):
    def __init__(self,n_layers,n_head,d_wordvec,d_hidden,dv,dk,dq,n_src_tokens,max_seq_len):
        # d_wordvec = d_model
        super(Encoder,self).__init__()
        self.n_src_tokens = n_src_tokens
        self.max_seq_len = max_seq_len
        self.d_model = d_wordvec
        self.n_head = n_head
        self.n_layers = n_layers
        self.word_embedding = nn.Embedding(n_src_tokens, d_wordvec,padding_idx=constant._PAD_)
        self.positional_embedding = positional_encoding(max_seq_len, self.d_model)
        self.enc_layers = nn.ModuleList([EncoderLayer(n_head,self.d_model,d_hidden,dv,dk,dq) for _ in range(n_layers)])

    def forward(self, src_input,src_position):
        # input:[batch * max_length]
        inputi = src_input.type(torch.LongTensor)
        input_embedding = self.word_embedding(inputi)# / np.sqrt(self.d_model)
        position_embedding = self.positional_embedding(src_position)
        enc_input = input_embedding + position_embedding # [batch * max_length * d_wordvec]
        encoder_mask = seq_padding_mask(src_input,src_input)
        enc_output = enc_input
        for i,encoder in enumerate(self.enc_layers):
            enc_output = encoder(enc_output,encoder_mask)
        return enc_output

class Decoder(nn.Module):
    def __init__(self,n_layers,n_head,d_wordvec,d_hidden,dv,dk,dq,n_tgt_tokens,max_seq_len):
        super(Decoder,self).__init__()
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_model = d_wordvec
        self.n_tgt_tokens = n_tgt_tokens
        self.max_seq_len = max_seq_len
        self.word_embedding = nn.Embedding(n_tgt_tokens,d_wordvec,padding_idx=constant._PAD_)
        self.positional_embedding = positional_encoding(max_seq_len,self.d_model)
        self.dec_layers = nn.ModuleList([DecoderLayer(n_head,self.d_model,d_hidden,dv,dk,dq) for _ in range(self.n_layers)])

    def forward(self,src_input,tgt_positi,tgt_input,enc_output):
        # input:[batch * max_length]
        inputi = tgt_input.type(torch.LongTensor)
        input_embedding = self.word_embedding(inputi) / np.sqrt(self.d_model)
        position_embedding = self.positional_embedding(tgt_positi)
        dec_input = input_embedding + position_embedding  # [batch * max_length * d_wordvec]
        dec_mask = seq_dec_mask(seq_padding_mask(tgt_input, tgt_input))
        enc_dec_mask = seq_padding_mask(src_input, tgt_input)
        dec_output = dec_input
        for i,decoder in enumerate(self.dec_layers):
            dec_output = decoder(dec_output,dec_mask,enc_output,enc_dec_mask)

        return dec_output

class Transformer(nn.Module):
    def __init__(self,n_layers,n_head,d_hidden,dv,dk,dq,n_src_tokens,n_tgt_tokens,d_model=128,max_seq_len=35):
        super(Transformer,self).__init__()
        self.n_src_tokens = n_src_tokens
        self.n_tgt_tokens = n_tgt_tokens
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.dv = dv
        self.dk = dk
        self.dq = dq
        self.max_seq_len = max_seq_len

        self.encoder = Encoder(n_layers,n_head,d_model,d_hidden,dv,dk,dq,n_src_tokens,max_seq_len)
        self.decoder = Decoder(n_layers,n_head,d_model,d_hidden,dv,dk,dq,n_tgt_tokens,max_seq_len) #[batch * max_length * d_model]
        self.linear = nn.Linear(d_model,n_tgt_tokens)
        self.softmax = nn.Softmax(dim=-1)

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        enc_freezed_param_ids = set(map(id, self.encoder.positional_embedding.parameters()))
        dec_freezed_param_ids = set(map(id, self.decoder.positional_embedding.parameters()))
        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, src,tgt):
        src_input,src_positi = src
        tgt_input,tgt_positi = tgt
        enc_output = self.encoder(src_input,src_positi)
        dec_output = self.decoder(src_input,tgt_positi,tgt_input,enc_output)
        liner_output = self.linear(dec_output)
        output_logits = self.softmax(liner_output) #[batch * max_length * n_tgt_tokens]
        return output_logits

