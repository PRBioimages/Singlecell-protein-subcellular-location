import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
import numpy as np

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class Transformer_ResNet50D(nn.Module):
    def __init__(self, model_name='resnet200d', out_features=19, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name.split('_')[-1], pretrained=pretrained, in_chans=4)
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.last_linear = nn.Linear(in_features=n_features, out_features=512)
        self.last_linear1 = nn.Linear(in_features=512, out_features=out_features)
        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(512)
        self.relu = nn.ReLU()

    def forward(self, x, cnt):
        features = self.model(x)
        pooled = nn.Flatten()(self.pooling(features))
        pooled = self.last_linear(pooled)
        cells_pooled = self.relu(pooled)
        viewed_pooled = cells_pooled.view(-1, cnt, cells_pooled.shape[-1])
        # print(viewed_pooled.shape)
        viewed_pooled = self.transformer(viewed_pooled)
        # print(viewed_pooled.shape)
        return self.last_linear1(self.dropout(cells_pooled)), viewed_pooled


    @property
    def net(self):
        return self.model


NUM_CLASSES = 19


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)

    def forward(self, inputs):
        return self.linear(inputs)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x len_q x d_k]
        # k: [b_size x len_k x d_k]
        # v: [b_size x len_v x d_v] note: (len_k == len_v)

        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale_factor
        # attn: [b_size x len_q x len_k]
        if attn_mask is not None:
            assert attn_mask.size() == attn.size()
            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        outputs = torch.bmm(attn, v)
        # outputs: [b_size x len_q x d_v]

        return outputs, attn


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(d_hid), requires_grad=True)
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True,)
        std = z.std(dim=-1, keepdim=True,)
        ln_out = (z - mean.expand_as(z)) / (std.expand_as(z) + self.eps)
        ln_out = self.gamma.expand_as(
            ln_out) * ln_out + self.beta.expand_as(ln_out)

        return ln_out


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout=0.1):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_k))
        self.w_k = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_k))
        self.w_v = nn.Parameter(torch.FloatTensor(n_heads, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_k, dropout)

        init.xavier_normal_(self.w_q)
        init.xavier_normal_(self.w_k)
        init.xavier_normal_(self.w_v)

    def forward(self, q, k, v, attn_mask):
        (d_k, d_v, d_model, n_heads) = (
            self.d_k, self.d_v, self.d_model, self.n_heads)
        b_size = k.size(0)

        q_s = q.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)
        # [n_heads x b_size * len_q x d_model]
        k_s = k.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)
        # [n_heads x b_size * len_k x d_model]
        v_s = v.repeat(n_heads, 1, 1).view(n_heads, -1, d_model)
        # [n_heads x b_size * len_v x d_model]

        q_s = torch.bmm(q_s, self.w_q).view(b_size * n_heads, -1, d_k)
        # [b_size * n_heads x len_q x d_k]
        k_s = torch.bmm(k_s, self.w_k).view(b_size * n_heads, -1, d_k)
        # [b_size * n_heads x len_k x d_k]
        v_s = torch.bmm(v_s, self.w_v).view(b_size * n_heads, -1, d_v)
        # [b_size * n_heads x len_v x d_v]

        # perform attention, result_size = [b_size * n_heads x len_q x d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_heads, 1, 1)
        outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)

        # return a list of tensors [b_size x len_q x d_v] (length: n_heads)
        return torch.split(outputs, b_size, dim=0), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.attention = _MultiHeadAttention(
            d_k, d_v, d_model, n_heads, dropout)
        self.proj = Linear(n_heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        residual = q
        # outputs: a list of tensors [b_size x len_q x d_v] (length: n_heads)
        outputs, attn = self.attention(q, k, v, attn_mask=attn_mask)
        # concatenate 'n_heads' multi-head attentions
        outputs = torch.cat(outputs, dim=-1)
        # project back to residual size,  [b_size x len_q x d_model]
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        return self.layer_norm(residual + outputs), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        outputs = self.relu(self.conv1(inputs.transpose(1, 2)))
        outputs = self.conv2(outputs).transpose(1, 2)
        # outputs: [b_size x len_q x d_model]
        outputs = self.dropout(outputs)

        return self.layer_norm(residual + outputs)


class EncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(
            d_k, d_v, d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, enc_inputs, self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, attn_mask=self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn


def get_attn_pad_mask(seq_q, seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # b_size x 1 x len_k
    pad_attn_mask = pad_attn_mask.expand(b_size, len_q, len_k)
    # b_size x len_q x len_k

    return pad_attn_mask


class Transformer(nn.Module):
    def __init__(self, fv='resnet50d', dropout=0.1, NUM_HEADS=8, NUM_LAYERS=1):
        super(Transformer, self).__init__()
        if fv == 'resnet50d':
            MODEL_DIM = 2048
        else:
            MODEL_DIM = 512
        # QUERY_DIM = 32
        KEY_DIM = 32 * 2
        VALUE_DIM = 32 * 2
        FF_DIM = MODEL_DIM * 1

        self.layers = nn.ModuleList(
            [EncoderLayer(KEY_DIM, VALUE_DIM, MODEL_DIM, FF_DIM, NUM_HEADS,
                          dropout) for _ in range(NUM_LAYERS)])

        self.proj = nn.Linear(MODEL_DIM, NUM_CLASSES)
        self.embeddings = Embeddings(MODEL_DIM)

    def forward(self, enc_inputs):
        enc_outputs = self.embeddings(enc_inputs)
        # enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attn_mask = None
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)

        out = enc_outputs[:, 0, :]
        out = self.proj(out)
        return out


class Embeddings(nn.Module):
    def __init__(self, hidden_size):
        super(Embeddings, self).__init__()
        self.hidden_size = hidden_size
        self.classifer_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x):
        bs = x.shape[0]
        cls_tokens = self.classifer_token.expand(bs, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x





