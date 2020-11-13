import torch
import torch.nn as nn

from transformer_modules \
    import Embedder, PositionalEncoder, Norm, MultiHeadAttention, FeedForward

import copy

# ==== one layer of encoder - decoder ====

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()

        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        
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
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

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

# ==== encoder - decoder ====

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()

        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)

        # create N encoder layers
        for i in range(self.N):
            x = self.layers[i](x, mask)

        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()

        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)

        # create N decoder layers
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)

        return self.norm(x)

# ==== create N stacks of encoder - decoder ====

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# ==== transformer ====

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()

        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)

        return output

# === model loader ===

def get_model(opt, src_vocab, trg_vocab):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
       
    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    if opt.device == 0:
        model = model.cuda()
    
    return model