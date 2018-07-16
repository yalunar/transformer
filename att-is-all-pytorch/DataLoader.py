import constant
import torch
import numpy as np

class DataLoader(object):
    def __init__(self,data_src,data_tgt,batch_size,n_max_seq_len,n_samples,n_src_tokens,n_tgt_tokens):
        self.src = data_src
        self.tgt = data_tgt
        self.batch_size = batch_size
        self.n_max_seq_len = n_max_seq_len
        self.start_idx = 0
        self.end_idx = n_samples
        self.i_batch = 0
        self.n_batch = n_samples // batch_size
        self.n_src_tokens = n_src_tokens
        self.n_tgt_tokens = n_tgt_tokens
        self.stop = False

    def pad_2_longest_seq(self,lst):
        # max_seq_len = max([len(l) for l in lst])
        # print(lst)
        pad_seq = [lsti + [constant._PAD_] * (self.n_max_seq_len - len(lsti)) for lsti in lst]
        pos_seq = [[i+1 if pad_seq_ii != constant._PAD_ else 0 for i,pad_seq_ii in enumerate(pad_seq_i)] for pad_seq_i in pad_seq]
        return torch.from_numpy(np.array(pad_seq)),torch.from_numpy(np.array(pos_seq))

    def next(self):
        if self.i_batch <= self.n_batch:
            left_idx = self.batch_size * self.i_batch
            right_idx = self.batch_size * (self.i_batch + 1) if self.batch_size * (self.i_batch + 1) < self.end_idx else self.end_idx

            self.i_batch += 1
            if self.i_batch >= self.n_batch:
                self.stop = True

            batch_src = self.src[left_idx : right_idx]
            batch_pad_src,batch_pos_src = self.pad_2_longest_seq(batch_src)

            batch_tgt = self.tgt[left_idx: right_idx]
            batch_pad_tgt, batch_pos_tgt = self.pad_2_longest_seq(batch_tgt)
            return (batch_pad_src,batch_pos_src),(batch_pad_tgt,batch_pos_tgt)

        else:
            raise StopIteration()



