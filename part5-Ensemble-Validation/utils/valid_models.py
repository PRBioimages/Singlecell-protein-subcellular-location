import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
import numpy as np
import yaml
import json
import os
import glob


class Element:
    def __repr__(self):
        return ', '.join(['{}: {}'.format(k, v) for k, v in self.__dict__.items()])


class DPP(Element):
    def __init__(self, dict):
        self.nodes = 1
        self.gpus = 4
        self.rank = 0
        self.sb = True
        self.mode = 'train'
        self.checkpoint = None


class Basic(Element):
    def __init__(self, dict):
        self.seed = dict.get('seed', '233')
        self.GPU = str(dict.get('GPU', '0'))
        self.id = dict.get('id', 'unnamed')
        self.debug = dict.get('debug', False)
        self.mode = dict.get('mode', 'train')
        self.search = dict.get('search', False)
        self.amp = dict.get('amp', 'None')
        # if len(self.GPU) > 1:
        #     self.GPU = [int(x) for x in self.GPU]


class Experiment(Element):
    def __init__(self, dict):
        self.name = dict.get('name', 'KFold')
        self.random_state = dict.get('random_state', '2333')
        self.fold = dict.get('fold', 5)
        self.run_fold = dict.get('run_fold', 0)
        self.weight = dict.get('weight', False)
        self.method = dict.get('method', 'none')
        self.tile = dict.get('tile', 12)
        self.count = dict.get('count', 16)
        self.regression = dict.get('regression', False)
        self.scale = dict.get('scale', 1)
        self.level = int(dict.get('level', 1))
        self.public = dict.get('public', True)
        self.merge = dict.get('merge', True)
        self.n = dict.get('N', True)
        self.batch_sampler = dict.get('batch_sampler', False)
        # batch sampler
        #   initial_miu: 6
        #   miu_factor: 6
        self.pos_ratio = dict.get('pos_ratio', 16)
        self.externals = dict.get('externals', [])
        self.initial_miu = dict.get('initial_miu', -1)
        self.miu_factor = dict.get('miu_factor', -1)
        self.full = dict.get('full', False)
        self.preprocess = dict.get('preprocess', 'train')
        self.image_only = dict.get('image_only', True)
        self.skip_outlier = dict.get('skip_outlier', False)
        self.outlier = dict.get('outlier', 'train')
        self.outlier_method = dict.get('outlier_method', 'drop')
        self.file = dict.get('csv_file', 'none')
        self.smoothing = dict.get('smoothing', 0)


class Data(Element):
    def __init__(self, dict):
        self.cell = dict.get('cell', 'none')
        self.name = dict.get('name', 'CouldDataset')
        if os.name == 'nt':
            self.data_root = dict.get('dir_nt', '/')
        else:
            self.data_root = dict.get('dir_sv', '/')
        self.onlyclass = dict.get('onlyclass', 'all')
        # for aws,
        # /home/sheep/Bengali/data
        # to any user
        try:
            self.data_root = glob.glob('/' + self.data_root.split('/')[1] + '/*/' + '/'.join(self.data_root.split('/')[3:]))[0]
        except:
            self.data_root = 'REPLACE ME PLZ!'


class Model(Element):
    def __init__(self, dict):
        self.name = dict.get('name', 'resnet50')
        self.param = dict.get('params', {})
        # add default true
        if 'dropout' not in self.param:
            self.param['dropout'] = True
        self.from_checkpoint = dict.get('from_checkpoint', 'none')
        self.out_feature = dict.get('out_feature', 1)


class Train(Element):
    '''
      freeze_backbond: 1
      freeze_top_layer_groups: 0
      freeze_start_epoch: 1


    :param dict:
    '''
    def __init__(self, dict):
        self.dir = dict.get('dir', None)
        if not self.dir:
            raise Exception('Training dir must assigned')
        self.batch_size = dict.get('batch_size', 8)
        self.num_epochs = dict.get('num_epochs', 100)
        self.cutmix = dict.get('cutmix', False)
        self.mixup = dict.get('mixup', False)
        self.beta = dict.get('beta', 1)
        self.cutmix_prob = dict.get('cutmix_prob', 0.5)
        self.cutmix_prob_increase = dict.get('cutmix_prob_increase', 0)
        self.validations_round = dict.get('validations_round', 1)
        self.freeze_backbond = dict.get('freeze_backbond', 0)
        self.freeze_top_layer_groups = dict.get('freeze_top_layer_groups', 0)
        self.freeze_start_epoch = dict.get('freeze_start_epoch', 1)
        self.clip = dict.get('clip_grad', None)
        self.combine_mix = dict.get('combine_mix', False)
        self.combine_list = dict.get('combine_list', [])
        self.combine_p = dict.get('combine_p', [])


class Eval(Element):
    def __init__(self, dict):
        self.batch_size = dict.get('batch_size', 32)


class Loss(Element):
    def __init__(self, dict):
        self.name = dict.get('name')
        self.param = dict.get('params', {})
        # if 'class_balanced' not in self.param:
        #     self.param['class_balanced'] = False
        self.weight_type = dict.get('weight_type', 'None')
        self.weight_value = dict.get('weight_value', None)
        self.cellweight = dict.get('cellweight', 0.1)
        self.pos_weight = dict.get('pos_weight', 10)


class Optimizer(Element):
    def __init__(self, dict):
        self.name = dict.get('name')
        self.param = dict.get('params', {})
        self.step = dict.get('step', 1)


class Scheduler(Element):
    def __init__(self, dict):
        self.name = dict.get('name')
        self.param = dict.get('params', {})
        self.warm_up = dict.get('warm_up', False)


class Transform(Element):
    def __init__(self, dict):
        self.name = dict.get('name')
        self.val_name = dict.get('val_name', 'None')
        self.param = dict.get('params', {})
        self.num_preprocessor = dict.get('num_preprocessor', 0)
        self.size = dict.get('size', (137, 236))
        self.half = dict.get('half', False)
        self.tiny = dict.get('tiny', False)
        self.smaller = dict.get('smaller', False)
        self.larger = dict.get('larger', False)
        self.random_scale = dict.get('random_scale', False)
        self.random_margin = dict.get('random_margin', False)
        self.random_choice = dict.get('random_choice', False)
        self.shuffle = dict.get('shuffle', False)
        self.scale = dict.get('scale', [])
        self.gray = dict.get('gray', False)


class Config:
    def __init__(self, dict):
        self.param = dict
        self.basic = Basic(dict.get('basic', {}))
        self.experiment = Experiment(dict.get('experiment', {}))
        self.data = Data(dict.get('data', {}))
        self.model = Model(dict.get('model', {}))
        self.train = Train(dict.get('train', {}))
        self.eval = Eval(dict.get('eval', {}))
        self.loss = Loss(dict.get('loss', {}))
        self.optimizer = Optimizer(dict.get('optimizer', {}))
        self.scheduler = Scheduler(dict.get('scheduler', {}))
        self.transform = Transform(dict.get('transform', {}))
        self.dpp = DPP({})

    def __repr__(self):
        return '\t\n'.join(['{}: {}'.format(k, v) for k, v in self.__dict__.items()])

    def dump_json(self, file_path):
        with open(file_path, 'w') as fp:
            json.dump(self.param, fp, indent=4)

    def to_flatten_dict(self):
        ft = {}
        for k, v in self.param.items():
            for kk, vv in v.items():
                if type(vv) in [dict, list]:
                    vv = str(vv)
                ft[f'{k}.{kk}'] = vv
        return ft

    @staticmethod
    def load_json(file_path):
        with open(file_path) as fp:
            data = json.load(fp)
        return Config(data)

    @staticmethod
    def load(file_path):
        with open(file_path) as fp:
            data = yaml.load(fp)
        return Config(data)


def get_config(name):
    return Config.load(os.path.dirname(os.path.realpath(__file__)) + '/' + name)


def get_model(cfg: Config, pretrained='imagenet'):
    if cfg.model.name in ['resnet200d', 'resnet152d', 'resnet101d', 'resnet50d', 'resnet34d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a MILResNet, mdl_name: {cfg.model.name}, pool: {pool}')
        return JakiroResNet200D(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    if cfg.model.name in ['xception']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a MILXception, mdl_name: {cfg.model.name}, pool: {pool}')
        return JakiroXception(pretrained=cfg.model.param['pretrained'], model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    if cfg.model.name in ['SA_attention_xception','SA_attention_resnet200d', 'SA_attention_resnet152d', 'SA_attention_resnet101d', 'SA_attention_resnet50d', 'SA_attention_resnet34d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a SA_Attention-based MIL, mdl_name: {cfg.model.name}, pool: {pool}')
        return SA_attentionResNet(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    if cfg.model.name in ['SA_Max_xception','SA_Max_resnet200d', 'SA_Max_resnet152d', 'SA_Max_resnet101d', 'SA_Max_resnet50d', 'SA_Max_resnet34d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a SA_Max-based MIL, mdl_name: {cfg.model.name}, pool: {pool}')
        return SA_MaxResNet(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    if cfg.model.name in ['SA_attention_xception','SA_attention_resnet200d', 'SA_attention_resnet152d', 'SA_attention_resnet101d', 'SA_attention_resnet50d', 'SA_attention_resnet34d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a SA_Attention-based MIL, mdl_name: {cfg.model.name}, pool: {pool}')
        return SA_attentionResNet(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    if cfg.model.name in ['attention_xception','attention_resnet200d', 'attention_resnet152d', 'attention_resnet101d', 'attention_resnet50d', 'attention_resnet34d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a Attention-based MIL, mdl_name: {cfg.model.name}, pool: {pool}')
        return AttentionResNet200D(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    if cfg.model.name in ['Cell_resnet50d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a Cell-ResNet, mdl_name: {cfg.model.name}, pool: {pool}')
        return RANZCRResNet200D(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    if cfg.model.name in ['CLAM']:
        return CLAM_SB(n_classes=19)
    elif cfg.model.name in ['CLAM_MB']:
        return CLAM_MB(n_classes=19,dropout=True)
    if cfg.model.name in ['custonattention_xception','custonattention_resnet200d', 'custonattention_resnet152d', 'custonattention_resnet101d', 'custonattention_resnet50d', 'custonattention_resnet34d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a Custom_ATTResNet50D, mdl_name: {cfg.model.name}, pool: {pool}')
        return Custom_ATTResNet50D(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    elif cfg.model.name in ['transformer_xception','transformer_resnet200d', 'transformer_resnet152d', 'transformer_resnet101d', 'transformer_resnet50d', 'transformer_resnet34d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a transformer-based MIL, mdl_name: {cfg.model.name}, pool: {pool}')
        return Transformer_ResNet50D(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    elif cfg.model.name in ['Cell_resnet50d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a Cell-ResNet, mdl_name: {cfg.model.name}, pool: {pool}')
        return CELLRANZCRResNet200D(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0), out_dim=cfg.model.out_feature)
    elif cfg.model.name in ['Cell_xception']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a Cell-xception, mdl_name: {cfg.model.name}, pool: {pool}')
        return CELLRANZCRxception(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0), out_dim=cfg.model.out_feature)
    elif cfg.model.name in ['imp_transformer_xception','imp_transformer_resnet200d', 'imp_transformer_resnet152d', 'imp_transformer_resnet101d', 'imp_transformer_resnet50d', 'imp_transformer_resnet34d']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a transformer-based MIL, mdl_name: {cfg.model.name}, pool: {pool}')
        return imp_Transformer_ResNet50D(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))
    elif cfg.model.name in ['gattention_resnet50d', 'gattention_xception']:
        pool = cfg.model.param.get('pool', 'AdaptiveAvgPool2d')
        print(f'[ ! ] Init a GAttention-based MIL, mdl_name: {cfg.model.name}, pool: {pool}')
        return GAttentionResNet200D(pretrained=True, model_name=cfg.model.name, pool=pool, dropout=cfg.model.param.get('dropout', 0))


class GAttentionResNet200D(nn.Module):
    def __init__(self, model_name='gattention_resnet200d', out_features=19, pretrained=False, dropout=0.5,
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
        self.last_linear = nn.Linear(in_features=n_features, out_features=out_features)
        self.last_linear2 = nn.Linear(in_features=n_features, out_features=out_features)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attn_Net_Gated(n_features)

    def forward(self, x, cnt):
        features = self.model(x)
        pooled = nn.Flatten()(self.pooling(features))
        viewed_pooled = pooled.view(-1, cnt, pooled.shape[-1])
        # print(viewed_pooled.shape)
        # viewed_pooled = viewed_pooled.max(1)[0]
        viewed_pooled = self.attention(viewed_pooled)
        # print(viewed_pooled.shape)
        return self.last_linear(self.dropout(pooled)), self.last_linear2(self.dropout(viewed_pooled))


    @property
    def net(self):
        return self.model


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


class imp_Transformer_ResNet50D(nn.Module):
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
        self.transformer = imp_Transformer(512)
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

class imp_Transformer(nn.Module):
    def __init__(self, fv='resnet50d', dropout=0.1, NUM_HEADS=8, NUM_LAYERS=1):
        super(imp_Transformer, self).__init__()
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
        # enc_outputs = self.embeddings(enc_inputs)
        enc_outputs = enc_inputs
        # enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attn_mask = None
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)

        out = enc_outputs[:, -1, :]
        out = self.proj(out)
        return out

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


class CLAM_SB(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=1, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=True):
        super(CLAM_SB, self).__init__()
        self.model = timm.create_model('resnet50d', pretrained=True, in_chans=4)
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.size_dict = {"small": [2048, 512, 256], "big": [2048, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, cnt, label=None, instance_eval=True, return_features=False, attention_only=False):
        device = h.device
        h = self.model(h)
        h = nn.Flatten()(self.pooling(h))
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, h)
        logits = self.classifiers(M)
        # Y_prob = torch.sigmoid(logits, dim=1)

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            # inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                classifier = self.instance_classifiers[i]
                logit_cells = classifier(h)
                precell = torch.softmax(logit_cells, 1)
                all_preds.append(precell[:,1].view(1, -1))
            cell_result = torch.cat(all_preds)
            cell_result0 = torch.transpose(cell_result, 1, 0)
            return logits, cell_result0
        else:
            logit_cells = self.classifiers(h)
            precell = torch.sigmoid(logit_cells)
            return logits, precell


class CLAM_MB(CLAM_SB):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=1, n_classes=2,
                 instance_loss_fn=nn.BCEWithLogitsLoss(reduction='none'), subtyping=True):
        nn.Module.__init__(self)
        self.model = timm.create_model('resnet50d', pretrained=True, in_chans=4)
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.size_dict = {"small": [2048, 512, 256], "big": [2048, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.5))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in
                           range(n_classes)]  # use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        # self.last_linear2 = nn.Linear(in_features=2048, out_features=19)

    def forward(self, h, cnt, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        h = self.model(h)
        h = nn.Flatten()(self.pooling(h))

        viewed_pooled = h.view(-1, cnt, h.shape[-1])
        A, h = self.attention_net(viewed_pooled)  # NxK
        # cell_loss = self.instance_loss_fn(A.view(-1, A.shape[-1]), label)
        cell_results = torch.sigmoid(A)
        A = torch.transpose(A, 2, 1)  # KxN
        A = F.softmax(A, dim=2)  # softmax over N

        M = torch.bmm(A, h)
        logits = torch.empty(len(viewed_pooled), self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[:, c] = self.classifiers[c](M[:,c,:]).view(-1)

        return logits, cell_results


def load_matched_state(model, state_dict):
    model_dict = model.state_dict()
    not_loaded = []
    for k, v in state_dict.items():
        if k in model_dict.keys():
            if not v.shape == model_dict[k].shape:
                print('Error Shape: {}, skip!'.format(k))
                continue
            model_dict.update({k: v})
        else:
            # print('not matched: {}'.format(k))
            not_loaded.append(k)
    if len(not_loaded) == 0:
        print('[ √ ] All layers are loaded')
    else:
        print('[ ! ] {} layer are not loaded'.format(len(not_loaded)))
    model.load_state_dict(model_dict)


LBL_NAMES = ["Nucleoplasm", "Nuclear Membrane", "Nucleoli", "Nucleoli Fibrillar Center", "Nuclear Speckles", "Nuclear Bodies", "Endoplasmic Reticulum", "Golgi Apparatus", "Intermediate Filaments", "Actin Filaments", "Microtubules", "Mitotic Spindle", "Centrosome", "Plasma Membrane", "Mitochondria", "Aggresome", "Cytosol", "Vesicles", "Negative"]
INT_2_STR = {x: LBL_NAMES[x] for x in np.arange(19)}

class AttentionResNet200D(nn.Module):
    def __init__(self, model_name='attention_resnet200d', out_features=19, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name.split('_')[-1], pretrained=pretrained, in_chans=4)
        # self.model.conv1[0] = torch.nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.last_linear = nn.Linear(in_features=n_features, out_features=out_features)
        self.last_linear2 = nn.Linear(in_features=n_features, out_features=out_features)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(n_features)

    # def forward(self, x):
    #     bs = x.size(0)
    #     features = self.model(x)
    #     pooled_features = self.pooling(features).view(bs, -1)
    #     output = self.fc(self.dropout(pooled_features))
    #     return output
    def forward(self, x, cnt):
        features = self.model(x)
        pooled = nn.Flatten()(self.pooling(features))
        viewed_pooled = pooled.view(-1, cnt, pooled.shape[-1])
        # print(viewed_pooled.shape)
        # viewed_pooled = viewed_pooled.max(1)[0]
        viewed_pooled, att_x = self.attention(viewed_pooled)
        # print(viewed_pooled.shape)
        return self.last_linear(self.dropout(pooled)), self.last_linear2(self.dropout(viewed_pooled)), att_x


    @property
    def net(self):
        return self.model

class SA_attentionResNet(nn.Module):
    def __init__(self, model_name='SA_attention_resnet200d', out_features=19, pretrained=False, dropout=0.25,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name.split('_')[-1], pretrained=pretrained, in_chans=4)
        # self.model.conv1[0] = torch.nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.last_linear = nn.Linear(in_features=n_features, out_features=512)
        self.last_linear1 = nn.Linear(in_features=512, out_features=out_features)
        self.last_linear2 = nn.Linear(in_features=512, out_features=out_features)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(512, Hidden_features=256)
        self.self_att = SelfAttention(512)
        self.relu = nn.ReLU()
        # self.att_map = None

    # def forward(self, x):
    #     bs = x.size(0)
    #     features = self.model(x)
    #     pooled_features = self.pooling(features).view(bs, -1)
    #     output = self.fc(self.dropout(pooled_features))
    #     return output
    def forward(self, x, cnt):
        features = self.model(x)
        pooled = nn.Flatten()(self.pooling(features))
        pooled = self.last_linear(pooled)
        cells_pooled = self.relu(pooled)
        pooled = pooled.view(-1, cnt, pooled.shape[-1])
        pooled, att_map, _, _ = self.self_att(pooled)
        # print(viewed_pooled.shape)
        # viewed_pooled = viewed_pooled.max(1)[0]
        viewed_pooled = self.attention(pooled)[0]
        # print(viewed_pooled.shape)
        return self.last_linear1(self.dropout(cells_pooled)), self.last_linear2(self.dropout(viewed_pooled))


    @property
    def net(self):
        return self.model


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter((torch.ones(1)))
        self.softmax = nn.Softmax(dim=-1)
        self.gamma_att = nn.Parameter((torch.ones(1)))

    def forward(self, x):
        x = x.view(-1, x.shape[-2], x.shape[-1]).permute((0, 2, 1))
        bs, C, length = x.shape
        proj_query = self.query_conv(x).view(bs, -1, length).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(bs, -1, length)  # B X C x (*W*H)

        energy = torch.bmm(proj_query, proj_key)  # transpose check

        attention = self.softmax(energy)  # BX (N) X (N)

        proj_value = self.value_conv(x).view(bs, -1, length)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(bs, C, length)

        out = self.gamma * out + x
        return out.permute(0, 2, 1), attention, self.gamma, self.gamma_att

class Custom_ATTResNet50D(nn.Module):
    def __init__(self, model_name='resnet200d', out_features=19, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d', cat_features=512):
        super().__init__()
        self.model = timm.create_model(model_name.split('_')[-1], pretrained=pretrained, in_chans=4)
        # self.model.conv1[0] = torch.nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.conv = nn.Conv2d(n_features, cat_features, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                                     bias=False)
        self.last_linear = nn.Linear(in_features=cat_features*2, out_features=out_features)
        self.last_linear2 = nn.Linear(in_features=cat_features*2, out_features=out_features)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(cat_features*2)
        self.attention = Attention(cat_features*2)

    # def forward(self, x):
    #     bs = x.size(0)
    #     features = self.model(x)
    #     pooled_features = self.pooling(features).view(bs, -1)
    #     output = self.fc(self.dropout(pooled_features))
    #     return output
    def forward(self, x, cnt):
        features = self.model(x)
        features = nn.ReLU()(features)
        features = self.conv(features)
        x = torch.cat((nn.AdaptiveAvgPool2d(1)(features), nn.AdaptiveMaxPool2d(1)(features)), dim=1)
        x = x.view(x.size(0), -1)
        x = self.bn1(x)
        viewed_pooled = x.view(-1, cnt, x.shape[-1])
        # print(viewed_pooled.shape)
        viewed_pooled, att = self.attention(viewed_pooled)
        # print(viewed_pooled.shape)
        return self.last_linear(self.dropout(x)), self.last_linear2(self.dropout(viewed_pooled))


    @property
    def net(self):
        return self.model


class RANZCRResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d', out_dim=19, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name.split('_')[-1], pretrained=pretrained, in_chans=4)
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.fc = nn.Linear(n_features, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(self.dropout(pooled_features))
        return output

    @property
    def net(self):
        return self.model


class JakiroResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d', out_features=19, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.conv1[0] = torch.nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.last_linear = nn.Linear(in_features=n_features, out_features=out_features)
        self.last_linear2 = nn.Linear(in_features=n_features, out_features=out_features)
        self.dropout = nn.Dropout(dropout)

    # def forward(self, x):
    #     bs = x.size(0)
    #     features = self.model(x)
    #     pooled_features = self.pooling(features).view(bs, -1)
    #     output = self.fc(self.dropout(pooled_features))
    #     return output
    def forward(self, x, cnt):
        features = self.model(x)
        pooled = nn.Flatten()(self.pooling(features))
        viewed_pooled = pooled.view(-1, cnt, pooled.shape[-1])
        # print(viewed_pooled.shape)
        viewed_pooled = viewed_pooled.max(1)[0]
        # print(viewed_pooled.shape)
        return self.last_linear(self.dropout(pooled)), self.last_linear2(self.dropout(viewed_pooled))


    @property
    def net(self):
        return self.model


class JakiroXception(nn.Module):
    def __init__(self, model_name='xception', out_features=19, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=4)
        # self.model.conv1[0] = torch.nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.last_linear = nn.Linear(in_features=n_features, out_features=out_features)
        self.last_linear2 = nn.Linear(in_features=n_features, out_features=out_features)
        self.dropout = nn.Dropout(dropout)

    # def forward(self, x):
    #     bs = x.size(0)
    #     features = self.model(x)
    #     pooled_features = self.pooling(features).view(bs, -1)
    #     output = self.fc(self.dropout(pooled_features))
    #     return output
    def forward(self, x, cnt):
        features = self.model(x)
        pooled = nn.Flatten()(self.pooling(features))
        viewed_pooled = pooled.view(-1, cnt, pooled.shape[-1])
        # print(viewed_pooled.shape)
        viewed_pooled = viewed_pooled.max(1)[0]
        # print(viewed_pooled.shape)
        return self.last_linear(self.dropout(pooled)), self.last_linear2(self.dropout(viewed_pooled))


    @property
    def net(self):
        return self.model


class Attention(nn.Module):
    def __init__(self, in_features, Hidden_features=512):
        super(Attention, self).__init__()
        self.in_features = nn.Linear(in_features=in_features, out_features=Hidden_features)
        self.out_features = nn.Linear(in_features=Hidden_features, out_features=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        att_x = self.in_features(self.dropout(x))
        att_x = self.relu(att_x)
        att_x = self.out_features(self.dropout(att_x))
        att_x = torch.softmax(att_x, dim=1)
        x = (x * att_x).sum(1)
        return x, att_x


class SA_MaxResNet(nn.Module):
    def __init__(self, model_name='SA_attention_resnet200d', out_features=19, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name.split('_')[-1], pretrained=pretrained, in_chans=4)
        # self.model.conv1[0] = torch.nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.last_linear = nn.Linear(in_features=n_features, out_features=out_features)
        self.last_linear2 = nn.Linear(in_features=n_features, out_features=out_features)
        self.dropout = nn.Dropout(dropout)
        # self.attention = torch.max(1)
        self.self_att = SelfAttention(n_features)
        # self.att_map = None

    # def forward(self, x):
    #     bs = x.size(0)
    #     features = self.model(x)
    #     pooled_features = self.pooling(features).view(bs, -1)
    #     output = self.fc(self.dropout(pooled_features))
    #     return output
    def forward(self, x, cnt):
        features = self.model(x)
        pooled = nn.Flatten()(self.pooling(features))

        pooled = pooled.view(-1, cnt, pooled.shape[-1])
        pooled, att_map, _, _ = self.self_att(pooled)
        # print(viewed_pooled.shape)
        # viewed_pooled = viewed_pooled.max(1)[0]
        viewed_pooled = pooled.max(1)[0]
        # print(viewed_pooled.shape)
        return self.last_linear2(self.dropout(pooled)).view(-1, self.last_linear.out_features), self.last_linear2(self.dropout(viewed_pooled))


class SA_Max_CalibrationResNet(nn.Module):
    def __init__(self, model_name='SA_attention_resnet200d', out_features=19, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name.split('_')[-1], pretrained=pretrained, in_chans=4)
        # self.model.conv1[0] = torch.nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.last_linear = nn.Linear(in_features=n_features, out_features=out_features)
        self.last_linear2 = nn.Linear(in_features=n_features, out_features=out_features)
        self.dropout = nn.Dropout(dropout)
        # self.attention = torch.max(1)
        self.self_att = SelfAttention(n_features)
        # self.att_map = None

    # def forward(self, x):
    #     bs = x.size(0)
    #     features = self.model(x)
    #     pooled_features = self.pooling(features).view(bs, -1)
    #     output = self.fc(self.dropout(pooled_features))
    #     return output
    def forward(self, x, cnt):
        features = self.model(x)
        pooled = nn.Flatten()(self.pooling(features))

        pooled = pooled.view(-1, cnt, pooled.shape[-1])
        pooled, att_map, _, _ = self.self_att(pooled)
        # print(viewed_pooled.shape)
        # viewed_pooled = viewed_pooled.max(1)[0]
        viewed_pooled = pooled.max(1)[0]
        # print(viewed_pooled.shape)
        return att_map, self.last_linear2(self.dropout(viewed_pooled))


    @property
    def net(self):
        return self.model

class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        A = torch.softmax(A, dim=1)
        x = (A * x).sum(1)
        return x


class CELLRANZCRResNet200D(nn.Module):
    def __init__(self, model_name='resnet200d', out_dim=19, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name.split('_')[-1], pretrained=pretrained, in_chans=4)
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.fc = nn.Linear(n_features, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(self.dropout(pooled_features))
        return output

    @property
    def net(self):
        return self.model


class CELLRANZCRxception(nn.Module):
    def __init__(self, model_name='xception', out_dim=19, pretrained=False, dropout=0.5,
                 pool='AdaptiveAvgPool2d'):
        super().__init__()
        self.model = timm.create_model(model_name.split('_')[-1], pretrained=pretrained, in_chans=4)
        # if pretrained:
        #     pretrained_path = '../input/resnet200d-pretrained-weight/resnet200d_ra2-bdba9bf9.pth'
        #     self.model.load_state_dict(torch.load(pretrained_path))
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        if pool == 'AdaptiveAvgPool2d':
            self.pooling = nn.AdaptiveAvgPool2d(1)
        elif pool == 'gem':
            self.pooling = GeM()
        self.fc = nn.Linear(n_features, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(self.dropout(pooled_features))
        return output

    @property
    def net(self):
        return self.model
