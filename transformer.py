''' ========== pytorch transformer ========== '''
''' ==== attention is all you need, 2017 ==== '''

# source codes from https://blog.floydhub.com/the-transformer-in-pytorch/

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

import math
import copy

# === module 1. input embedding ===

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embed(x)

# === module 2. positional encoding ===

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()

        self.d_model = d_model

        pos_enc = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pos_enc[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pos_enc[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('pos_enc', pos_enc)

# === module 3. (optional) masking ===

# === module 4. multihead attention ===

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.lastlinear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.k_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        #calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # heads concatenating and the final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.lastlinear(concat)

        return output

# === module 5. scaled dot-product attention ===

def attention(q, k, v, d_k, mask=None, dropout=None):
    # (a) matrix mutiplication of Q and K
    # (b) and then scaling by d_k
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # (c) (optional) masking
    if mask != None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # (d) softmaxing
    scores = F.softmax(scores, dim=-1)

    # (e) (optional) dropout
    if dropout != None:
        scores = dropout(scores)

    # (f) matrix multiplication of QK and V
    output = torch.matmul(scores, v)
    
    return output

# === module 6. feed forward networks ===

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)

        return x

# === module 7. add & normalization ===

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias

        return norm

# === encoder layer - decoder layer ===

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()

        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x, mask):
        # commented lines are from the source codes
        # x2 = self.norm_1(x)
        # x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        # x2 = self.norm_2(x)
        # x = x + self.dropout_2(self.ff(x2))

        # my codes refered to the paper
        # residuals connected to each norm layer
        x_tmp = self.dropout_1(self.attn(x, x, x, mask))
        x = x + self.norm_1(x_tmp)
        x_tmp = self.dropout_2(self.ff(x))
        x = x + self.norm_2(x_tmp)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()

        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).cuda()

    def forward(self, x, e_outputs, src_mask, trg_mask):
        # commented lines are from the source codes
        # x2 = self.norm_1(x)
        # x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        # x2 = self.norm_2(x)
        # x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        # x2 = self.norm_3(x)
        # x = x + self.dropout_3(self.ff(x2))

        # my codes refered to the paper
        # e_outputs are from encoder
        # residuals connected to each norm layer
        x_tmp = self.dropout_1(self.attn_1(x, x, x, trg_mask))
        x = x + self.norm_1(x_tmp)
        x_tmp = self.dropout_2(self.attn_2(x, e_outputs, e_outputs, src_mask))
        x = x + self.norm_2(x_tmp)
        x_tmp = self.dropout_3(self.ff(x))
        x = x + self.norm_3(x_tmp)

        return x

# === create N stacks of encoder and decoder layer ===

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# === encoder - decoder ===

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()

        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pos_enc = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pos_enc(x)

        # create N stacks of encoder layer
        for i in range(self.N):
            x = self.layers[i](x, mask)

        return self.norm(x) # 정규화 몇 번을 하는거야..


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()

        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pos_enc = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pos_enc(x)

        # crate N stacks of decoder layer
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        
        return self.norm(x)

# === transformer ===

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads=6):
        super().__init__()

        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_outputs = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_outputs)

        return output