import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init

class ScaledDotProductAttn(nn.Module):
    def __init__(self,dk):
        super(ScaledDotProductAttn,self).__init__()
        # ? divide 这项是否可以不加到 attribute 里面
        # self.d_model = d_model
        self.divide =  np.sqrt(dk)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,q,k,v,mask):
        """
        dq = dk ; dv
        ? max_seq_len_k = max_seq_len_v
        :param q: [batch * max_seq_len_q * dq]
        :param k: [batch * max_seq_len_k * dk]
        :param v: [batch * max_seq_len_v * dv]
        :param mask: [batch * max_seq_len_q * max_seq_len_k], type= torch.ByteTensor
        :return:
        """
        # batch,max_len,dk = k.size()
        attn_vau = torch.bmm(q,k.transpose(1,2)) / self.divide # [batch * max_seq_len_q * max_seq_len_k]
        attn_weight = self.softmax(attn_vau.masked_fill_(mask,-float('inf')))
        return torch.bmm(attn_weight,v) # [batch * max_seq_len_q * dv]

class LayerNormalization(nn.Module):
    def __init__(self,d_model,eps=1e-3):
        super(LayerNormalization,self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self,x):
        # x = [batch * max_seq_len_q * d_model]
        mean_x = torch.mean(x,dim=-1,keepdim=True)
        variance_x = torch.std(x,dim=-1,keepdim=True)
        out = (x - mean_x.expand_as(x)) / (variance_x.expand_as(x) + self.eps)
        out = out * self.gamma.expand_as(out) + self.beta.expand_as(out)

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,n_head,d_model,dv,dk,dq,dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        # d_k = d_q
        self.n_head = n_head
        self.d_model = d_model
        self.dv = dv
        self.dk = dk
        self.dq = dq
        self.w_v = nn.Parameter(torch.FloatTensor(n_head,d_model,dv))
        self.w_k = nn.Parameter(torch.FloatTensor(n_head,d_model,dk))
        self.w_q = nn.Parameter(torch.FloatTensor(n_head,d_model,dq))
        self.attn = ScaledDotProductAttn(dk)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)
        self.linear = nn.Linear(n_head * dv,d_model)

        init.xavier_normal_(self.w_q)
        init.xavier_normal_(self.w_k)
        init.xavier_normal_(self.w_v)

    def forward(self,v,k,q, mask):
        """
        :param q: [batch * max_len_q * d_model]
        :param k: [batch * max_len_k * d_model]
        :param v: [batch * max_len_v * d_model]
        :param mask:[batch * max_len_q * max_len_k]
        :return:
        """
        residual = k
        batch, max_len_q, d_model = q.size()
        batch, max_len_k, d_model = k.size()
        batch, max_len_v, d_model = v.size()
        q = q.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.d_model) # [n_head * (batch*max_len_q) * d_model], 8*(10*45)*512
        k = k.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.d_model) # [n_head * (batch*max_len_k) * d_model]
        v = v.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.d_model) # [n_head * (batch*max_len_v) * d_model]

        q_ = torch.bmm(q,self.w_q).view(-1,max_len_q,self.dq) # [(n_head*batch) * max_len_q * dq] 80*45*64 ?
        k_ = torch.bmm(k,self.w_k).view(-1,max_len_k,self.dk) # [(n_head*batch) * max_len_k * dk]
        v_ = torch.bmm(v,self.w_v).view(-1,max_len_v,self.dv) # [(n_head*batch) * max_len_v * dv]

        outputs = self.attn(q_,k_,v_,mask.repeat(self.n_head,1,1)) # [(n_head * batch) * max_seq_len_q * dv]
        outputs = torch.cat(torch.split(outputs,batch,dim = 0),dim = -1)  # [batch * max_seq_len_q * (n_head * dv)]
        outputs = self.linear(outputs) # [batch * max_seq_len_q * d_model]
        outputs = self.dropout(outputs)
        input = outputs + residual
        outputs = self.layer_norm(input)

        return outputs

class PositionwiseFFN(nn.Module):
    def __init__(self,d_model,d_hidden,dropout=0.1):
        super(PositionwiseFFN,self).__init__()
        self.w_1 = nn.Conv1d(d_model,d_hidden,1)  #input: batch * channel * len
        self.w_2 = nn.Conv1d(d_hidden,d_model,1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, x):
        # x:[batch * len * d_model]
        residual = x
        y = self.relu(self.w_1(x.transpose(1,2))) #[batch * d_hidden * len]
        y = self.w_2(y) #[batch * d_model * len]
        y = y.transpose(1,2) #[batch * len * d_model]
        y = self.dropout(y)
        output = self.layer_norm(y + residual)
        return output

class EncoderLayer(nn.Module):
    def __init__(self,n_head,d_model,d_hidden,dv,dk,dq):
        super(EncoderLayer,self).__init__()
        self.multiheadAttn = MultiHeadAttention(n_head,d_model,dv,dk,dq)
        self.positionwiseffn = PositionwiseFFN(d_model,d_hidden)
    def forward(self,x,mask):
        output = self.multiheadAttn(x,x,x,mask) #[batch * max_seq_len_q * d_model]
        enc_output = self.positionwiseffn(output) #[batch * max_seq_len_q * d_model]
        return enc_output

class DecoderLayer(nn.Module):
    def __init__(self,n_head,d_model,d_hidden,dv,dk,dq):
        super(DecoderLayer,self).__init__()
        self.mask_multiheadAttn = MultiHeadAttention(n_head,d_model,dv,dk,dq)
        self.multiheadAttn = MultiHeadAttention(n_head,d_model,dv,dk,dq)
        self.positionwiseffn = PositionwiseFFN(d_model,d_hidden)

    def forward(self,x,mask,enc_output,enc_dec_mask):
        output = self.mask_multiheadAttn(x,x,x,mask) #[batch * max_seq_len_q * d_model]
        output = self.multiheadAttn(enc_output,enc_output,output,enc_dec_mask)
        dec_output = self.positionwiseffn(output)
        return dec_output




