import torch
from torch import nn
import timm
import torch.nn.functional as F
from torch.nn.parameter import Parameter


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
        return x


class SA_attentionResNet(nn.Module):
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
        self.attention = Attention(n_features)
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
        viewed_pooled = self.attention(pooled)
        # print(viewed_pooled.shape)
        return self.last_linear(self.dropout(pooled)).view(-1, self.last_linear.out_features), self.last_linear2(self.dropout(viewed_pooled))


    @property
    def net(self):
        return self.model


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter((torch.zeros(1)).cuda())
        self.softmax = nn.Softmax(dim=-1)
        self.gamma_att = nn.Parameter((torch.ones(1)).cuda())

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

        # out = self.gamma * out + x
        return out.permute(0, 2, 1), attention, self.gamma, self.gamma_att


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


