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


class AttentionResNet50D(nn.Module):
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
        viewed_pooled = self.attention(viewed_pooled)
        # print(viewed_pooled.shape)
        return self.last_linear(self.dropout(pooled)), self.last_linear2(self.dropout(viewed_pooled))


    @property
    def net(self):
        return self.model


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


class GAttentionResNet50D(nn.Module):
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
        self.l = 0.2

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
        viewed_pooled = self.attention(viewed_pooled)
        # print(viewed_pooled.shape)
        return self.last_linear(self.dropout(x)), self.last_linear2(self.dropout(viewed_pooled))


    @property
    def net(self):
        return self.model


